# Lemonade Compatibility

**Status:** Active · targeting Lemonade **v11.5.0+** (tracked as of 2026-07-22)

`1bit.systems` NPU and GPU engines are compatible with [Lemonade](https://github.com/lemonade-sdk/lemonade) via the OpenAI-compatible `/v1/chat/completions` endpoint, the Anthropic-compatible `/v1/messages` endpoint, and the Ollama-compatible API surface Lemonade exposes.

## How it works

`1bit-halo-server` (port `:8180` — NPU engine) and the ZINC GPU engine both expose the standard OpenAI chat completions API. Lemonade (`lemond`, default port `:13305`, base URL `http://localhost:13305/api/v1`) can route omni-modal requests to either backend as a sub-process or a remote OpenAI-compatible server, including as a candidate in a **`collection.router`** model.

```yaml
# lemonade config.yaml
models:
  - name: qwen3-npu
    server:
      url: http://127.0.0.1:9090  # FLM proxy or 1bit-halo-server
      type: openai
```

## Quick start

```bash
# 1. Start the 1bit NPU daemon
./npu_engine_all model.q4nx 16

# 2. Lemond auto-discovers or point manually
lemond launch --model qwen3-npu
```

## What changed in Lemonade v11.x (sync notes)

Tracked against the [v11.5.0 release](https://github.com/lemonade-sdk/lemonade/releases/tag/v11.5.0). Items that affect the `1bit.systems` integration:

- **⚠️ Breaking — CORS default removed.** The server no longer sends
  `Access-Control-Allow-Origin: *` by default. Non-loopback browser origins are
  rejected with `403` unless listed in `LEMONADE_ALLOWED_ORIGINS`, and the
  WebSocket same-origin fallback was removed. Any remote/LAN browser UI pointed
  at a Lemonade instance fronting a 1bit backend must now set
  `LEMONADE_ALLOWED_ORIGINS`. Loopback (`127.0.0.1`) is unaffected.
- **Lemonade Router (`collection.router`).** Requests can be steered across
  candidates by rule / classifier / semantic-similarity / LLM-as-router policy,
  with per-candidate decision traces. A 1bit OpenAI-compatible endpoint can be
  registered as a router candidate with zero changes on the 1bit side.
- **`POST /v1/classify`** — new text-classification endpoint (onnxruntime
  backend), used by the router's classifier policy.
- **`/health` now reports `is_busy` and `is_streaming`** ([#2720](https://github.com/lemonade-sdk/lemonade/pull/2720)) — useful for external health/liveness checks against a routed 1bit backend.
- **Server-side job engine** (`/jobs` — post/pause/interrupt/resume/delete
  multi-step recipes).
- **`lemond` as an MCP client host** — connects to external stdio MCP servers via
  admin-gated `/internal/mcp/*` endpoints.
- **CLI HTTPS/TLS** — `lemonade` honors `http(s)://` schemes in `LEMONADE_HOST`,
  so a remote 1bit-fronting endpoint can be reached over TLS.
- **ModelScope search** alongside Hugging Face in the Model Manager.

## Agent-client compatibility (Claude Code / Codex)

Two fixes that harden Lemonade's Anthropic/`Responses` paths for agent clients
routing to 1bit backends are staged on our fork and **pending upstream** (not yet
in v11.5.0):

- **[#2662](https://github.com/lemonade-sdk/lemonade/issues/2662)** — Anthropic
  `/v1/messages` now folds inline `role:"system"` messages (as Claude Code sends
  them) into the system prompt instead of silently dropping them, and maps
  `thinking.type: "adaptive"` to `enable_thinking`. Previously produced empty
  responses.
- **[#2674](https://github.com/lemonade-sdk/lemonade/issues/2674)** — the SSE
  streaming proxy now surfaces a backend non-200 as a terminated error event
  (`data: {error…}` + `[DONE]`) instead of leaking a raw JSON blob mid-stream,
  so clients (e.g. Codex CLI) stop retrying against a stream that never
  completes.

Until these land upstream, prefer non-streaming or a simple Chat Completions
client when driving a routed 1bit backend from strict-template reasoning models.

## `*_bin` config keys

Lemonade v10.3+ (PR [#1713](https://github.com/lemonade-sdk/lemonade/pull/1713))
accepts `builtin` / `latest` / version tag / local path values for every backend
binary. The `1bit-systems` packaging can supply `ryzenai.server_bin` as `latest`
or a pinned tag; this contract is unchanged in v11.5.0.

## First-party contribution status

- `bong-water-water-bong` is credited as a **reviewer on the Lemonade Router
  milestone** in the [v11.5.0 release notes](https://github.com/lemonade-sdk/lemonade/releases/tag/v11.5.0).
- Previously a [listed contributor](https://github.com/lemonade-sdk/lemonade/releases/tag/v10.9.0)
  in v10.9.0 (test/documentation: PR [#2447](https://github.com/lemonade-sdk/lemonade/pull/2447)).

## Version compatibility table

| 1bit.systems | Lemonade | Status |
|---|---|---|
| v2026.07.20+ | v11.5.0 | ✅ Verified (Jul 2026) — note CORS breaking change |
| v2026.07+ | v10.9.0 | ✅ Compatible (basic chat + Anthropic/OpenAI) |
| v2026.04+ | v10.3 | ✅ Compatible (API stable) |
| v2026.04+ | v10.0+ | ✅ Basic chat |

## Updating

When a new Lemonade release ships:

1. Check the [Lemonade releases](https://github.com/lemonade-sdk/lemonade/releases) for breaking API changes (esp. CORS/auth/origin and endpoint-shape changes).
2. Smoke-test with `1bit-halo-server` on port `:9090` (OpenAI chat, Anthropic `/v1/messages`, streaming).
3. Update the sync notes + version table above and the site wiki.

## MCP support

Both projects expose MCP servers. As of v11.x, `lemond` can also act as an MCP
*client host*. See:
- [Lemonade MCP docs](https://github.com/lemonade-sdk/lemonade)
- 1bit.systems packaging docs
