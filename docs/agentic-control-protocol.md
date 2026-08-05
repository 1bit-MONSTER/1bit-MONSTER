# Agentic Control Protocol (ACP)

Control protocol for agent harnesses (pi, Claude, custom agents) to drive
[Abacus.AI RouteLLM](https://abacus.ai/help/developer-platform/route-llm/) —
the API surface of ChatLLM / Super Assistant. Gives an agent access to every
major LLM (OpenAI, Anthropic, Google, xAI, Meta, DeepSeek, Qwen, ...) plus
image gen, TTS, and vision/PDF input through **one OpenAI-compatible
endpoint**, with automatic model routing (`route-llm`) and automatic
failover.

Spec source: abacus.ai/help/developer-platform/route-llm/* (2025-08 fetch).

---

## 1. Transport & Auth

| | |
|---|---|
| Base URL (self-serve) | `https://routellm.abacus.ai/v1` |
| Base URL (enterprise) | `https://<workspace>.abacus.ai/v1` |
| Auth | `Authorization: Bearer <api_key>` (all requests) |
| Key source | https://abacus.ai/app/route-llm-apis (part of ChatLLM subscription) |

Key handling rules for agents:
- **Never** echo the key into prompts, logs, or commits. Read from env
  (`ABACUS_API_KEY` / `ACP_API_KEY`) or `~/.config/abacus/route-llm.key`.
- If a call returns 401, report the failure to the user; do not retry with
  the same key.

### Login-auth alternative (no API key)

The `abacusai` CLI authenticates with the account login (device-code OAuth)
and is a full ChatLLM client. It accepts `-m route-llm` (auto-route) and
reports credits per call. Use it when no API key is configured; the HTTP API
itself accepts Bearer keys only — a login session cannot be substituted.

## 2. Endpoints

All three share base URL, auth, and credit accounting. Use **Chat
Completions** unless the task needs the Responses API agentic tooling.

| Endpoint | Path | Format | Use for |
|---|---|---|---|
| Chat Completions | `/v1/chat/completions` | OpenAI | default; every text model, tools, multimodal, JSON schema |
| Responses API | `/v1/responses` | OpenAI Responses | reasoning/agentic workflows (GPT-5 family, o-series, Grok) |
| Anthropic Messages | `/v1/messages` | Anthropic | Claude models via Anthropic SDK |
| Model catalog | `/v1/models` | OpenAI | live model list + pricing; also `GET /v1/models` for image models |

## 3. Model Semantics — the auto-route feature

- `model` is **optional**. Omitted or `"route-llm"` → the router picks the
  best model for the request (complexity, cost, speed). This is the default
  and the recommended setting for most agent work.
- Pinning: pass an exact model id (e.g. `gpt-5.4`, `claude-sonnet-4-6`,
  `gemini-3.1-pro`, `o3-mini` for reasoning) to force a specific model.
  Use `GET /v1/models` for the live catalog — display names differ from API
  ids (e.g. `flux-2-pro` vs `flux2_pro`).
- The response `model` field reports the model actually served (differs from
  the requested id under `route-llm`). Record it for cost/quality tracking.
- Unlisted models are proxied to their respective providers.

Routing policy for agents:
1. Default: `route-llm`. Don't pin unless the task demands it.
2. Pin a reasoning model (`o3-mini`, `o3`) for math/planning/verification.
3. Pin a specific model when output must be reproducible across runs.
4. On 429/5xx from a pinned model, retry with `route-llm` (failover).

## 4. Chat Completions Request Schema

```json
{
  "model": "route-llm",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "max_tokens": 2048,
  "temperature": 0.2,
  "stream": false
}
```

Message `content`: string (text) or array of parts for multimodal
(`{"type":"text","text":...}`, `{"type":"image_url",...}`, file parts).
Roles: `system` | `user` | `assistant` | `tool`.

Optional params: `top_p`, `stop` (≤4 sequences), `presence_penalty`,
`frequency_penalty`, `response_format`, `tools`, `tool_choice`,
`modalities`, `audio`.

Useful defaults for agent work:
- `temperature`: 0.0–0.3 for factual/extraction/structured output; 0.7+ for
  creative writing.
- `max_tokens`: always set — prevents runaway long responses.

### Structured output
- `response_format: {"type": "json_object"}` → valid JSON; **must** also
  instruct the model in a system/user message.
- `response_format: {"type": "json_schema", "json_schema": {name, schema,
  strict}}` → schema-enforced; no prose instruction needed. Use
  `strict: true` whenever output is parsed programmatically.

### Multimodal input
- Images: OpenAI-style `image_url` content parts.
- PDFs: `{"type":"file","file":{"filename":"doc.pdf","file_data":"<https url or base64>"}}`.
- Audio: `gpt-4o-audio-preview` / Gemini TTS models (see RouteLLM audio ref).

### Image / audio generation
- `modalities: ["image"]` with an image model id (`flux-2-pro`, `dall-e`,
  `seedream`, ...) → generates images over the same endpoint.
- `modalities: ["text","audio"]` + `audio` object → TTS.

## 5. Tool Calling Protocol

**Stateless.** The server never executes tools and never persists
tool-call state. The agent is the executor:

1. Send `messages` + `tools` (OpenAI function schema) + `tool_choice: "auto"`.
2. If the response has `message.tool_calls` and `finish_reason: "tool_calls"`:
   - execute each call client-side (real tool, local script, MCP, shell),
   - append the assistant message **with its `tool_calls`** verbatim,
   - append one `{"role":"tool","tool_call_id":"...","content":"<result>"}`
     per call,
   - resend the full history **with the same `tools`**.
3. Repeat until `finish_reason: "stop"` (or `"length"` → raise max_tokens).
4. Multiple tool calls can arrive in one response — execute them in
   parallel, then batch the results back.
5. Streaming: tool call `arguments` arrive fragmented across chunks;
   aggregate per `index` before parsing JSON.

Example loop (wire-level):

```
→ {"model":"route-llm","messages":[user], "tools":[get_weather], "tool_choice":"auto"}
← {"choices":[{"message":{"content":null,"tool_calls":[{id,function:{name,arguments}}]}, "finish_reason":"tool_calls"}]}
→ {"model":"route-llm",
     "messages":[user, {assistant w/ tool_calls}, {role:"tool", tool_call_id, content:"{...}"}],
     "tools":[get_weather]}
← {"choices":[{"message":{"content":"It's 72°F..."}},"finish_reason":"stop"]}
```

## 6. Streaming

`stream: true` → SSE chunks (`data: {...}` deltas, terminal `data: [DONE]`).
Chunks carry `choices[0].delta.content`; usage arrives in the final chunk.
Aggregate deltas; never trust a partial chunk as final text.

## 7. Errors & Retry

| Status | Meaning | Agent action |
|---|---|---|
| 400 | invalid request | fix payload; do not retry as-is |
| 401 | bad/missing key | stop, surface to user |
| 429 | rate limit | exponential backoff, retry ≤3 |
| 5xx | server error | retry ≤3 w/ backoff; then fall back to `route-llm` or local models |

Error body: `{"error":{"message","type","code"}}`.

## 8. Operations

- Credits: ChatLLM subscription includes 20,000 credits/mo; proprietary LLMs
  billed at provider list price, open-weight at best-available price.
  RouteLLM stays usable past the monthly credit cap (accounting only).
- Track the served model id and token usage from every response for cost
  accounting; log per-call `usage` when auditing.

## 9. Implementation

- Reference client: `tools/acp.py` (Python stdlib only — no pip deps).
- Agent skill: `~/.pi/agent/skills/abacus-routellm/SKILL.md` — when to use
  ACP, how to invoke the client, routing policy.
- Verify connectivity: `python3 tools/acp.py check`.

## 10. Open questions / pending verification (need a live API key)

- [ ] Exact behavior of `route-llm` on tool-calling requests (router choice
      vs pinned fallback).
- [ ] Streaming `usage` chunk shape (standard OpenAI or custom).
- [ ] `GET /v1/models` response fields for pricing (id, cost per MTok).
- [ ] Whether Responses API tools execute server-side.
