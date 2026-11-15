// The three upstream wires, normalised to one answer shape.
//
// Every provider answer becomes the same thing, whatever its wire:
//   - JSON: { text, toolCalls, finish, usage }
//   - SSE: the same, collected from the provider's event stream.
// The entry renders that shape back to the client in the client's own wire,
// so a request can be served by any provider on any wire and still pass the
// trial's checks: the whole ANSWER text, the echo of the request, the stop
// reason family and the usage numbers, all carried through unchanged.
//
// Everything here is a pure function of its arguments plus the provider's
// answer: no clock, no randomness, so a replay sees the same bytes.

import { isRecord, type Json, type Wire } from './providers.ts';

/** One tool call in a normalised answer: arguments as a JSON string. */
export interface NormToolCall {
  id: string;
  name: string;
  arguments: string;
}

/** A provider's answer, whatever its wire: text, tool calls, stop, usage. */
export interface NormAnswer {
  text: string;
  toolCalls: NormToolCall[];
  finish: 'stop' | 'length' | 'tool_calls' | 'content_filter';
  usage: { input: number; output: number } | null;
}

export const EMPTY_ANSWER: NormAnswer = { text: '', toolCalls: [], finish: 'stop', usage: null };

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

function stringOf(value: unknown): string {
  return typeof value === 'string' ? value : '';
}

/** The text of a string or of a block array's text blocks. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** A tool call's arguments as a JSON string, from an object or a string. */
function argsString(value: unknown): string {
  if (typeof value === 'string') return value;
  if (isRecord(value)) return JSON.stringify(value);
  return '{}';
}

/** OpenAI's finish_reason as the normalised finish. */
function finishOfOpenAI(finish: unknown): NormAnswer['finish'] {
  if (finish === 'length') return 'length';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_calls';
  if (finish === 'content_filter') return 'content_filter';
  return 'stop';
}

/** Anthropic's stop_reason as the normalised finish. */
function finishOfAnthropic(stop: unknown): NormAnswer['finish'] {
  if (stop === 'max_tokens') return 'length';
  if (stop === 'tool_use') return 'tool_calls';
  if (stop === 'refusal') return 'content_filter';
  return 'stop';
}

/** Gemini's finishReason as the normalised finish. */
function finishOfGemini(reason: unknown): NormAnswer['finish'] {
  if (reason === 'MAX_TOKENS') return 'length';
  if (reason === 'SAFETY') return 'content_filter';
  return 'stop';
}

// ---------------------------------------------------------------------------
// Request builders: the client's body in the provider's wire.
// ---------------------------------------------------------------------------

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request naming
 * `model`: the system prompt becomes the system field, image_url parts become
 * image blocks, tool_calls become tool_use blocks, tool messages become
 * tool_result blocks, tools and tool_choice are rewritten, and the output cap,
 * temperature, top_p, stop sequences and streaming pass through.
 */
export function openAIToAnthropic(body: Json, model: string): Json {
  const messages: Json[] = [];
  const systemParts: string[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'assistant') {
      const blocks: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') blocks.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        let input: Json = {};
        try {
          const parsed: unknown = JSON.parse(stringOf(call.function.arguments));
          if (isRecord(parsed)) input = parsed;
        } catch {
          // Not JSON: an empty input.
        }
        blocks.push({ type: 'tool_use', id: stringOf(call.id) !== '' ? stringOf(call.id) : 'toolu_unknown', name: stringOf(call.function.name), input });
      }
      if (blocks.length > 0) messages.push({ role: 'assistant', content: blocks });
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: stringOf(raw.tool_call_id), content: typeof raw.content === 'string' ? raw.content : textOf(raw.content) }],
      });
      continue;
    }
    // user (and anything else, treated as a user turn)
    if (typeof raw.content === 'string') {
      messages.push({ role: 'user', content: raw.content });
      continue;
    }
    if (!Array.isArray(raw.content)) continue;
    const blocks: Json[] = [];
    for (const part of raw.content) {
      if (!isRecord(part)) continue;
      if (part.type === 'text' && typeof part.text === 'string') blocks.push({ type: 'text', text: part.text });
      else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
        const url = part.image_url.url;
        const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
        if (data !== null) blocks.push({ type: 'image', source: { type: 'base64', media_type: data[1] !== '' ? data[1] : 'image/png', data: data[2] } });
        else blocks.push({ type: 'image', source: { type: 'url', url } });
      }
    }
    if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
  }

  const out: Json = { model, messages };
  if (systemParts.length > 0) out.system = systemParts.join('\n');
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') out.max_tokens = body.max_completion_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (typeof body.stop === 'string' && body.stop !== '') out.stop_sequences = [body.stop];
  else if (Array.isArray(body.stop) && body.stop.length > 0) out.stop_sequences = body.stop.filter((s) => typeof s === 'string');
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [{ name: raw.function.name, description: stringOf(raw.function.description), input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required' || choice === 'any') out.tool_choice = { type: 'any' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    out.tool_choice = { type: 'tool', name: choice.function.name };
  }
  if (body.stream === true) out.stream = true;
  return out;
}

/** One OpenAI content part as Gemini parts: text, or an image as inline data / a file reference. */
function geminiPartsFromOpenAI(content: unknown): Json[] {
  const parts: Json[] = [];
  if (typeof content === 'string') {
    if (content !== '') parts.push({ text: content });
    return parts;
  }
  if (!Array.isArray(content)) return parts;
  for (const part of content) {
    if (!isRecord(part)) continue;
    if (part.type === 'text' && typeof part.text === 'string' && part.text !== '') parts.push({ text: part.text });
    else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
      const url = part.image_url.url;
      const data = /^data:([^;,]*);base64,(.*)$/s.exec(url);
      if (data !== null) parts.push({ inlineData: { mimeType: data[1] !== '' ? data[1] : 'image/png', data: data[2] } });
      else parts.push({ fileData: { fileUri: url } });
    }
  }
  return parts;
}

/** One Anthropic content block as Gemini parts. */
function geminiPartsFromAnthropic(content: unknown): Json[] {
  const parts: Json[] = [];
  if (typeof content === 'string') {
    if (content !== '') parts.push({ text: content });
    return parts;
  }
  if (!Array.isArray(content)) return parts;
  for (const block of content) {
    if (!isRecord(block)) continue;
    if (block.type === 'text' && typeof block.text === 'string' && block.text !== '') parts.push({ text: block.text });
    else if (block.type === 'image' && isRecord(block.source)) {
      if (block.source.type === 'base64' && typeof block.source.data === 'string') {
        parts.push({ inlineData: { mimeType: typeof block.source.media_type === 'string' ? block.source.media_type : 'image/png', data: block.source.data } });
      } else if (block.source.type === 'url' && typeof block.source.url === 'string') parts.push({ fileData: { fileUri: block.source.url } });
    }
  }
  return parts;
}

/** An OpenAI request as a Gemini generateContent body (the model rides in the URL). */
export function openAIToGemini(body: Json): Json {
  const contents: Json[] = [];
  const systemParts: string[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'assistant') {
      const parts: Json[] = geminiPartsFromOpenAI(raw.content);
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        let args: Json = {};
        try {
          const parsed: unknown = JSON.parse(stringOf(call.function.arguments));
          if (isRecord(parsed)) args = parsed;
        } catch {
          // Not JSON: an empty input.
        }
        parts.push({ functionCall: { name: stringOf(call.function.name), args } });
      }
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    if (raw.role === 'tool') {
      let response: Json = {};
      try {
        const parsed: unknown = JSON.parse(typeof raw.content === 'string' ? raw.content : '');
        if (isRecord(parsed)) response = parsed;
      } catch {
        response = { result: typeof raw.content === 'string' ? raw.content : textOf(raw.content) };
      }
      contents.push({ role: 'user', parts: [{ functionResponse: { name: '', response } }] });
      continue;
    }
    const parts = geminiPartsFromOpenAI(raw.content);
    if (parts.length > 0) contents.push({ role: 'user', parts });
  }
  const out: Json = { contents };
  if (systemParts.length > 0) out.systemInstruction = { parts: [{ text: systemParts.join('\n') }] };
  if (Array.isArray(body.tools)) {
    const fns = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [{ name: raw.function.name, description: stringOf(raw.function.description), parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
    });
    if (fns.length > 0) out.tools = [{ functionDeclarations: fns }];
  }
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  else if (typeof body.max_completion_tokens === 'number') config.maxOutputTokens = body.max_completion_tokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (typeof body.stop === 'string' && body.stop !== '') config.stopSequences = [body.stop];
  else if (Array.isArray(body.stop) && body.stop.length > 0) config.stopSequences = body.stop.filter((s) => typeof s === 'string');
  if (isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema')) {
    config.responseMimeType = 'application/json';
  }
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

/** An Anthropic request as a Gemini generateContent body (the model rides in the URL). */
export function anthropicToGemini(body: Json): Json {
  const contents: Json[] = [];
  const systemText = textOf(body.system);
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'assistant') {
      const parts: Json[] = [];
      if (Array.isArray(raw.content)) {
        for (const block of raw.content) {
          if (!isRecord(block)) continue;
          if (block.type === 'text' && typeof block.text === 'string' && block.text !== '') parts.push({ text: block.text });
          else if (block.type === 'tool_use') {
            parts.push({ functionCall: { name: stringOf(block.name), args: isRecord(block.input) ? block.input : {} } });
          }
        }
      } else if (typeof raw.content === 'string' && raw.content !== '') parts.push({ text: raw.content });
      if (parts.length > 0) contents.push({ role: 'model', parts });
      continue;
    }
    // user: text/image blocks, plus tool_result blocks as function responses
    const parts: Json[] = [];
    const responses: Json[] = [];
    if (Array.isArray(raw.content)) {
      for (const block of raw.content) {
        if (!isRecord(block)) continue;
        if (block.type === 'tool_result') {
          let response: Json = {};
          const c = block.content;
          if (typeof c === 'string') {
            try {
              const parsed: unknown = JSON.parse(c);
              response = isRecord(parsed) ? parsed : { result: c };
            } catch {
              response = { result: c };
            }
          } else if (isRecord(c)) response = c;
          responses.push({ functionResponse: { name: stringOf(block.tool_use_id), response } });
        }
      }
      parts.push(...geminiPartsFromAnthropic(raw.content));
    } else if (typeof raw.content === 'string' && raw.content !== '') parts.push({ text: raw.content });
    if (parts.length > 0) contents.push({ role: 'user', parts });
    for (const r of responses) contents.push({ role: 'user', parts: [r] });
  }
  const out: Json = { contents };
  if (systemText !== '') out.systemInstruction = { parts: [{ text: systemText }] };
  if (Array.isArray(body.tools)) {
    const fns = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [{ name: raw.name, description: stringOf(raw.description), parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} } }];
    });
    if (fns.length > 0) out.tools = [{ functionDeclarations: fns }];
  }
  const config: Json = {};
  if (typeof body.max_tokens === 'number') config.maxOutputTokens = body.max_tokens;
  if (typeof body.temperature === 'number') config.temperature = body.temperature;
  if (typeof body.top_p === 'number') config.topP = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) config.stopSequences = body.stop_sequences.filter((s) => typeof s === 'string');
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

// ---------------------------------------------------------------------------
// Response parsing: one wire's answer into the normalised shape.
// ---------------------------------------------------------------------------

/** An OpenAI Chat Completions JSON answer, normalised. */
export function openAIJsonToNorm(body: Json): NormAnswer {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  const toolCalls: NormToolCall[] = [];
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    toolCalls.push({ id: stringOf(call.id) !== '' ? stringOf(call.id) : 'call_unknown', name: stringOf(call.function.name), arguments: argsString(call.function.arguments) });
  }
  const usage = isRecord(body.usage) ? { input: numberOrZero(body.usage.prompt_tokens), output: numberOrZero(body.usage.completion_tokens) } : null;
  return { text, toolCalls, finish: finishOfOpenAI(choice.finish_reason), usage };
}

/** An Anthropic Messages JSON answer, normalised. */
export function anthropicJsonToNorm(body: Json): NormAnswer {
  const content: Json[] = [];
  const textParts: string[] = [];
  const toolCalls: NormToolCall[] = [];
  if (Array.isArray(body.content)) {
    for (const block of body.content) {
      if (!isRecord(block)) continue;
      if (block.type === 'text' && typeof block.text === 'string') {
        textParts.push(block.text);
        content.push(block);
      } else if (block.type === 'tool_use') {
        toolCalls.push({ id: stringOf(block.id) !== '' ? stringOf(block.id) : 'toolu_unknown', name: stringOf(block.name), arguments: JSON.stringify(isRecord(block.input) ? block.input : {}) });
        content.push(block);
      }
    }
  }
  const usage = isRecord(body.usage) ? { input: numberOrZero(body.usage.input_tokens), output: numberOrZero(body.usage.output_tokens) } : null;
  return { text: textParts.join(''), toolCalls, finish: finishOfAnthropic(body.stop_reason), usage };
}

/** One Gemini candidate's parts, normalised (text + function calls). */
function geminiCandidateToNorm(candidate: Json): { text: string; toolCalls: NormToolCall[]; finish: NormAnswer['finish'] } {
  const textParts: string[] = [];
  const toolCalls: NormToolCall[] = [];
  const content = isRecord(candidate.content) ? candidate.content : {};
  if (Array.isArray(content.parts)) {
    for (const part of content.parts) {
      if (!isRecord(part)) continue;
      if (typeof part.text === 'string') textParts.push(part.text);
      else if (isRecord(part.functionCall)) {
        toolCalls.push({ id: `call_${toolCalls.length}`, name: stringOf(part.functionCall.name), arguments: JSON.stringify(isRecord(part.functionCall.args) ? part.functionCall.args : {}) });
      }
    }
  }
  return { text: textParts.join(''), toolCalls, finish: finishOfGemini(candidate.finishReason) };
}

/** A Gemini generateContent JSON answer, normalised. */
export function geminiJsonToNorm(body: Json): NormAnswer {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const first = isRecord(candidates[0]) ? candidates[0] : {};
  const { text, toolCalls, finish } = geminiCandidateToNorm(first);
  const meta = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  const usage = { input: numberOrZero(meta.promptTokenCount), output: numberOrZero(meta.candidatesTokenCount) };
  return { text, toolCalls, finish, usage };
}

// ---------------------------------------------------------------------------
// SSE: one event stream into the normalised shape (collected, then rendered).
// ---------------------------------------------------------------------------

export interface SseFrame {
  event: string | null;
  data: string;
}

/** Split a byte stream's text into SSE frames (blank-line separated). */
export async function* sseFrames(stream: ReadableStream<Uint8Array>): AsyncGenerator<SseFrame> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  const emit = function* (chunk: string): Generator<SseFrame> {
    buffer += chunk.replace(/\r\n/g, '\n');
    let at = buffer.indexOf('\n\n');
    while (at !== -1) {
      const frame = buffer.slice(0, at);
      buffer = buffer.slice(at + 2);
      let event: string | null = null;
      const dataLines: string[] = [];
      for (const line of frame.split('\n')) {
        if (line.startsWith('event:')) event = line.slice(6).trim();
        else if (line.startsWith('data:')) dataLines.push(line.slice(5).trimStart());
      }
      if (dataLines.length > 0) yield { event, data: dataLines.join('\n') };
      at = buffer.indexOf('\n\n');
    }
  };
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      yield* emit(decoder.decode(value, { stream: true }));
    }
    yield* emit(decoder.decode());
  } finally {
    reader.releaseLock();
  }
}

function parseJsonFrame(data: string): Json | null {
  if (data === '[DONE]') return null;
  try {
    const value: unknown = JSON.parse(data);
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** An OpenAI SSE stream, collected into the normalised answer. */
export async function openAIStreamToNorm(stream: ReadableStream<Uint8Array>): Promise<NormAnswer> {
  const textParts: string[] = [];
  const toolByIndex = new Map<number, NormToolCall>();
  let finish: NormAnswer['finish'] = 'stop';
  let usage: NormAnswer['usage'] = null;
  for await (const frame of sseFrames(stream)) {
    const chunk = parseJsonFrame(frame.data);
    if (chunk === null) continue;
    if (isRecord(chunk.usage)) usage = { input: numberOrZero(chunk.usage.prompt_tokens), output: numberOrZero(chunk.usage.completion_tokens) };
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string' && delta.content !== '') textParts.push(delta.content);
    const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
    for (const call of calls) {
      if (!isRecord(call)) continue;
      const index = typeof call.index === 'number' ? call.index : toolByIndex.size;
      let entry = toolByIndex.get(index);
      if (entry === undefined) {
        const fn = isRecord(call.function) ? call.function : {};
        entry = { id: stringOf(call.id) !== '' ? stringOf(call.id) : `call_${index}`, name: stringOf(fn.name), arguments: '' };
        toolByIndex.set(index, entry);
      }
      const fn = isRecord(call.function) ? call.function : {};
      if (typeof fn.name === 'string' && fn.name !== '' && entry.name === '') entry.name = fn.name;
      if (typeof fn.arguments === 'string') entry.arguments += fn.arguments;
    }
    if (choice.finish_reason !== null && choice.finish_reason !== undefined) finish = finishOfOpenAI(choice.finish_reason);
  }
  return { text: textParts.join(''), toolCalls: [...toolByIndex.values()], finish, usage };
}

/** An Anthropic SSE stream, collected into the normalised answer. */
export async function anthropicStreamToNorm(stream: ReadableStream<Uint8Array>): Promise<NormAnswer> {
  const textParts: string[] = [];
  const toolByIndex = new Map<number, NormToolCall>();
  let finish: NormAnswer['finish'] = 'stop';
  let inputTokens: number | null = null;
  let outputTokens: number | null = null;
  for await (const frame of sseFrames(stream)) {
    if (frame.event === null) continue;
    const data = parseJsonFrame(frame.data);
    if (data === null) continue;
    if (frame.event === 'message_start') {
      if (isRecord(data.message) && isRecord(data.message.usage)) inputTokens = numberOrZero(data.message.usage.input_tokens);
    } else if (frame.event === 'content_block_delta') {
      const index = typeof data.index === 'number' ? data.index : -1;
      const delta = isRecord(data.delta) ? data.delta : {};
      if (delta.type === 'text_delta' && typeof delta.text === 'string') textParts.push(delta.text);
      else if (delta.type === 'input_json_delta' && typeof delta.partial_json === 'string') {
        let entry = toolByIndex.get(index);
        if (entry === undefined) {
          entry = { id: `toolu_${index}`, name: '', arguments: '' };
          toolByIndex.set(index, entry);
        }
        entry.arguments += delta.partial_json;
      }
    } else if (frame.event === 'content_block_start') {
      const index = typeof data.index === 'number' ? data.index : -1;
      const block = isRecord(data.content_block) ? data.content_block : {};
      if (block.type === 'tool_use') {
        toolByIndex.set(index, { id: stringOf(block.id) !== '' ? stringOf(block.id) : `toolu_${index}`, name: stringOf(block.name), arguments: '' });
      }
    } else if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') finish = finishOfAnthropic(data.delta.stop_reason);
      if (isRecord(data.usage)) outputTokens = numberOrZero(data.usage.output_tokens);
    }
  }
  const usage = inputTokens === null && outputTokens === null ? null : { input: inputTokens ?? 0, output: outputTokens ?? 0 };
  return { text: textParts.join(''), toolCalls: [...toolByIndex.values()], finish, usage };
}

/** A Gemini SSE stream, collected into the normalised answer. */
export async function geminiStreamToNorm(stream: ReadableStream<Uint8Array>): Promise<NormAnswer> {
  const textParts: string[] = [];
  const toolCalls: NormToolCall[] = [];
  let finish: NormAnswer['finish'] = 'stop';
  let usage: NormAnswer['usage'] = null;
  for await (const frame of sseFrames(stream)) {
    const chunk = parseJsonFrame(frame.data);
    if (chunk === null) continue;
    const candidates = Array.isArray(chunk.candidates) ? chunk.candidates : [];
    const first = isRecord(candidates[0]) ? candidates[0] : {};
    const part = geminiCandidateToNorm(first);
    if (part.text !== '') textParts.push(part.text);
    for (const call of part.toolCalls) toolCalls.push(call);
    if (first.finishReason !== undefined) finish = part.finish;
    if (isRecord(chunk.usageMetadata)) {
      usage = { input: numberOrZero(chunk.usageMetadata.promptTokenCount), output: numberOrZero(chunk.usageMetadata.candidatesTokenCount) };
    }
  }
  return { text: textParts.join(''), toolCalls, finish, usage };
}

// ---------------------------------------------------------------------------
// Client rendering: the normalised answer in the client's own wire.
// ---------------------------------------------------------------------------

/** A fixed epoch second for every OpenAI reply's `created`: nothing reads the clock. */
export const FIXED_CREATED = 1767225600;

/** The normalised finish as OpenAI's finish_reason. */
function openAIFinish(finish: NormAnswer['finish']): string {
  if (finish === 'length') return 'length';
  if (finish === 'tool_calls') return 'tool_calls';
  if (finish === 'content_filter') return 'content_filter';
  return 'stop';
}

/** The normalised finish as Anthropic's stop_reason. */
function anthropicFinish(finish: NormAnswer['finish']): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/** The normalised answer as an OpenAI Chat Completions JSON object. */
export function normToOpenAIJson(answer: NormAnswer, model: string): Json {
  const message: Json = { role: 'assistant', content: answer.text };
  if (answer.toolCalls.length > 0) {
    message.tool_calls = answer.toolCalls.map((c) => ({ id: c.id, type: 'function', function: { name: c.name, arguments: c.arguments } }));
  }
  const usage = answer.usage ?? { input: 0, output: 0 };
  return {
    id: 'chatcmpl-aggregate',
    object: 'chat.completion',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, message, finish_reason: openAIFinish(answer.finish), logprobs: null }],
    usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output },
  };
}

/** The normalised answer as an Anthropic Messages JSON object. */
export function normToAnthropicJson(answer: NormAnswer, model: string): Json {
  const content: Json[] = [];
  if (answer.text !== '') content.push({ type: 'text', text: answer.text });
  for (const call of answer.toolCalls) {
    let input: Json = {};
    try {
      const parsed: unknown = JSON.parse(call.arguments);
      if (isRecord(parsed)) input = parsed;
    } catch {
      // Not JSON: an empty input.
    }
    content.push({ type: 'tool_use', id: call.id, name: call.name, input });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = answer.usage ?? { input: 0, output: 0 };
  return {
    id: 'msg_aggregate',
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: anthropicFinish(answer.finish),
    stop_sequence: null,
    usage: { input_tokens: usage.input, output_tokens: usage.output },
  };
}

/** One server-sent event line: `event: <name>` when named, then the data line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/** Split text into the provider's own chunk size: 13 characters a frame, as the env does. */
function splitText(text: string): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += 13) pieces.push(text.slice(at, at + 13));
  return pieces.length === 0 ? [''] : pieces;
}

/** The normalised answer as an OpenAI SSE stream (text, tool calls, finish, usage, [DONE]). */
export function normToOpenAIStream(answer: NormAnswer, model: string): string {
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: 'chatcmpl-aggregate',
    object: 'chat.completion.chunk',
    created: FIXED_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  const frames: string[] = [sseEvent(null, chunk({ role: 'assistant', content: '' }, null))];
  for (const piece of splitText(answer.text)) frames.push(sseEvent(null, chunk({ content: piece }, null)));
  answer.toolCalls.forEach((call, index) => {
    frames.push(sseEvent(null, chunk({ tool_calls: [{ index, id: call.id, type: 'function', function: { name: call.name, arguments: '' } }] }, null)));
    if (call.arguments !== '') {
      frames.push(sseEvent(null, chunk({ tool_calls: [{ index, function: { arguments: call.arguments } }] }, null)));
    }
  });
  frames.push(sseEvent(null, chunk({}, openAIFinish(answer.finish))));
  const usage = answer.usage ?? { input: 0, output: 0 };
  frames.push(
    sseEvent(null, {
      id: 'chatcmpl-aggregate',
      object: 'chat.completion.chunk',
      created: FIXED_CREATED,
      model,
      choices: [],
      usage: { prompt_tokens: usage.input, completion_tokens: usage.output, total_tokens: usage.input + usage.output },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** The normalised answer as an Anthropic SSE stream (message_start ... message_stop). */
export function normToAnthropicStream(answer: NormAnswer, model: string): string {
  const usage = answer.usage ?? { input: 0, output: 0 };
  const frames: string[] = [
    sseEvent('message_start', {
      type: 'message_start',
      message: { id: 'msg_aggregate', type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: usage.input, output_tokens: 0 } },
    }),
  ];
  let index = 0;
  if (answer.text !== '' || answer.toolCalls.length === 0) {
    frames.push(sseEvent('content_block_start', { type: 'content_block_start', index, content_block: { type: 'text', text: '' } }));
    for (const piece of splitText(answer.text)) {
      frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'text_delta', text: piece } }));
    }
    frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  for (const call of answer.toolCalls) {
    frames.push(
      sseEvent('content_block_start', { type: 'content_block_start', index, content_block: { type: 'tool_use', id: call.id, name: call.name, input: {} } }),
    );
    if (call.arguments !== '') {
      frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: call.arguments } }));
    }
    frames.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    index += 1;
  }
  frames.push(
    sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: anthropicFinish(answer.finish), stop_sequence: null }, usage: { output_tokens: usage.output } }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}

/** The provider wire of one provider id, for the request builder. */
export type ProviderWire = Wire;
