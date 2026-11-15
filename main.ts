// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// It reads the brief (the provider directory, the quality table, the workload
// manifest), probes every provider's catalog route, builds one catalog, then
// serves the client's twenty requests: each one goes to the best free model
// that meets its requirements, by the quality table's rating, failing over to
// the next best free model when a provider refuses (402, 429 or 5xx).
//
// The routing is rating-ordered with a per-request failover loop: a provider
// that refuses is skipped for the rest of that request, and a provider that
// refuses three calls in a row is treated as spent for the trial (a scheduled
// failure is never three in a row; a spent budget or an outage always is).
// An OpenAI request that asks for JSON mode never goes to an anthropic-wire
// provider, whose wire cannot express JSON mode and would fail the echo.
// Streaming OpenAI calls always ask for usage, so the client's usage check
// sees the provider's numbers.
//
// Every answer is normalised through lib/upstream.ts, so any provider on any
// wire can serve any client on either wire: the whole ANSWER text, the echo
// of the request, the stop reason and the usage numbers all carry through
// unchanged. Nothing here reads the clock or unseeded randomness, so a replay
// leaves the same state export.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIModelList,
  type ServedModel,
} from './lib/wire.ts';
import {
  anthropicJsonToNorm,
  anthropicStreamToNorm,
  anthropicToGemini,
  geminiJsonToNorm,
  geminiStreamToNorm,
  normToAnthropicJson,
  normToAnthropicStream,
  normToOpenAIJson,
  normToOpenAIStream,
  openAIJsonToNorm,
  openAIStreamToNorm,
  openAIToAnthropic,
  openAIToGemini,
  type NormAnswer,
} from './lib/upstream.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider call may take, in milliseconds. */
const PROVIDER_TIMEOUT_MS = 15000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;
/** Consecutive refused calls after which a provider is treated as spent. */
const EXHAUST_AFTER_STREAK = 3;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;
/** Refused calls in a row, per provider: reset by any success. */
const failStreak = new Map<string, number>();
/** Providers treated as spent for the rest of the trial. */
const exhaustedProviders = new Set<string>();

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, in the brief's provider order: what GET /v1/models lists. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Route {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/**
 * Every free model that meets one request's requirements, best rating first.
 * An OpenAI request in JSON mode never goes to an anthropic-wire provider:
 * its wire cannot express JSON mode, so the provider's echo would miss it.
 * Ties break on provider id, then model id: the order is deterministic.
 */
function candidatesFor(requires: Requirement, clientApi: 'openai' | 'anthropic'): Route[] {
  const out: Route[] = [];
  for (const probe of probes) {
    if (!probe.ok) continue;
    if (exhaustedProviders.has(probe.provider.id)) continue;
    if (clientApi === 'openai' && requires.json === true && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort((a, b) => b.rating - a.rating || (a.probe.provider.id < b.probe.provider.id ? -1 : a.probe.provider.id > b.probe.provider.id ? 1 : 0) || byId(a.model, b.model));
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

function sendText(res: ServerResponse, status: number, contentType: string, text: string): void {
  res.writeHead(status, { 'content-type': contentType, 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

interface UpstreamError {
  status: number;
  code: string;
}

type UpstreamResult = { ok: true; answer: NormAnswer } | { ok: false; error: UpstreamError };

/** The error code of a provider's refusal body, in its own wire's shape. */
function errorCodeOf(wire: string, body: unknown): string {
  if (!isRecord(body)) return '';
  if (wire === 'openai' && isRecord(body.error)) return typeof body.error.code === 'string' ? body.error.code : '';
  if (wire === 'anthropic' && isRecord(body.error)) return typeof body.error.type === 'string' ? body.error.type : '';
  if (wire === 'gemini' && isRecord(body.error)) return typeof body.error.status === 'string' ? body.error.status : '';
  return '';
}

/** The provider-native request body for one client request, on one provider's wire. */
function upstreamBody(route: Route, clientApi: 'openai' | 'anthropic', body: Json, stream: boolean): Json {
  const wire = route.probe.provider.wire;
  const model = route.model.id;
  if (wire === 'openai') {
    if (clientApi === 'openai') {
      const out: Json = { ...body, model };
      if (stream) out.stream_options = { include_usage: true };
      return out;
    }
    return anthropicToOpenAI(body, model);
  }
  if (wire === 'anthropic') {
    if (clientApi === 'anthropic') return { ...body, model };
    return openAIToAnthropic(body, model);
  }
  // gemini: the model rides in the URL path.
  return clientApi === 'openai' ? openAIToGemini(body) : anthropicToGemini(body);
}

/** The provider-native URL for one call, on one provider's wire. */
function upstreamUrl(route: Route, stream: boolean): string {
  const base = joinBase(ENV_URL, route.probe.provider.base);
  const wire = route.probe.provider.wire;
  if (wire === 'openai') return `${base}/v1/chat/completions`;
  if (wire === 'anthropic') return `${base}/v1/messages`;
  const model = encodeURIComponent(route.model.id);
  return stream ? `${base}/v1beta/models/${model}:streamGenerateContent?alt=sse` : `${base}/v1beta/models/${model}:generateContent`;
}

/**
 * One call to one provider, normalised. A refusal (402, 429, 5xx) or an
 * unreachable provider comes back as an error, never an exception, so the
 * caller can fail over to the next candidate.
 */
async function callUpstream(route: Route, clientApi: 'openai' | 'anthropic', body: Json, stream: boolean): Promise<UpstreamResult> {
  const provider = route.probe.provider;
  const wire = provider.wire;
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (wire === 'anthropic') headers['anthropic-version'] = '2023-06-01';
  if (stream) headers.accept = 'text/event-stream';
  let response: Response;
  try {
    response = await fetch(upstreamUrl(route, stream), {
      method: 'POST',
      headers,
      body: JSON.stringify(upstreamBody(route, clientApi, body, stream)),
      signal: AbortSignal.timeout(PROVIDER_TIMEOUT_MS),
    });
  } catch (err) {
    return { ok: false, error: { status: 502, code: (err as Error).name === 'TimeoutError' ? 'timeout' : 'unreachable' } };
  }
  if (!response.ok) {
    let code = '';
    try {
      code = errorCodeOf(wire, await response.json());
    } catch {
      // The refusal carried no JSON body: the status is enough.
    }
    return { ok: false, error: { status: response.status, code } };
  }
  try {
    if (!stream) {
      const data: unknown = await response.json();
      if (!isRecord(data)) return { ok: false, error: { status: 502, code: 'bad_answer' } };
      if (wire === 'openai') return { ok: true, answer: openAIJsonToNorm(data) };
      if (wire === 'anthropic') return { ok: true, answer: anthropicJsonToNorm(data) };
      return { ok: true, answer: geminiJsonToNorm(data) };
    }
    if (response.body === null) return { ok: false, error: { status: 502, code: 'no_stream' } };
    if (wire === 'openai') return { ok: true, answer: await openAIStreamToNorm(response.body) };
    if (wire === 'anthropic') return { ok: true, answer: await anthropicStreamToNorm(response.body) };
    return { ok: true, answer: await geminiStreamToNorm(response.body) };
  } catch (err) {
    return { ok: false, error: { status: 502, code: 'bad_answer' } };
  }
}

interface Served {
  route: Route;
  answer: NormAnswer;
  failedFirst: boolean;
}

/**
 * Serve one request: try the candidates in rating order until one answers.
 * A provider that refuses is skipped for the rest of this request. A
 * provider's failure streak grows on every refusal and resets on any success;
 * at three in a row it is treated as spent for the trial.
 */
async function serve(clientApi: 'openai' | 'anthropic', body: Json, requires: Requirement, stream: boolean): Promise<Served | null> {
  const candidates = candidatesFor(requires, clientApi);
  const skipped = new Set<string>();
  let failedFirst = false;
  for (const route of candidates) {
    const providerId = route.probe.provider.id;
    if (skipped.has(providerId)) continue;
    const result = await callUpstream(route, clientApi, body, stream);
    if (result.ok) {
      failStreak.set(providerId, 0);
      return { route, answer: result.answer, failedFirst };
    }
    failedFirst = true;
    skipped.add(providerId);
    const streak = (failStreak.get(providerId) ?? 0) + 1;
    failStreak.set(providerId, streak);
    if (streak >= EXHAUST_AFTER_STREAK) exhaustedProviders.add(providerId);
    log(`failover: ${providerId}/${route.model.id} answered HTTP ${result.error.status} (${result.error.code}); trying the next candidate`);
  }
  return null;
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by any provider's wire. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const stream = body.stream === true;
  const served = await serve('openai', body, requirementsFor(body), stream);
  if (served === null) return sendError(res, 'openai', 503, 'no_free_model', 'no free model on the directory answered this request');
  const model = served.route.model.id;
  if (stream) return sendText(res, 200, 'text/event-stream; charset=utf-8', normToOpenAIStream(served.answer, model));
  return sendJson(res, 200, normToOpenAIJson(served.answer, model));
}

/** POST /v1/messages: an Anthropic-shaped client, answered by any provider's wire. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const stream = body.stream === true;
  const served = await serve('anthropic', body, requirementsFor(body), stream);
  if (served === null) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model on the directory answered this request');
  const model = served.route.model.id;
  if (stream) return sendText(res, 200, 'text/event-stream; charset=utf-8', normToAnthropicStream(served.answer, model));
  return sendJson(res, 200, normToAnthropicJson(served.answer, model));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
