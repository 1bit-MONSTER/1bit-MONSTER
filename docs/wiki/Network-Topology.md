# Network Topology

> This page exists because issue #1 linked to it before it was written. It
> describes how the pieces talk to each other at runtime.

## Components

Every server on this list is the **same single binary** (`build/1bit`),
dispatched by subcommand (`1bit zaya`, `1bit unified`, `1bit jarvis`, ...).
There is no separate daemon process or router binary — the old Python
`daemon/npu-gpu-cpud.py` proxy was replaced by the native engine and has
been removed from the repo. In production, each model gets its own
systemd unit running `1bit zaya` (or `1bit unified`) on its own port:

```
 client (OpenAI-compatible)
        │  POST /v1/chat/completions
        ▼
 ┌───────────────────────────────────────────────────────────┐
 │ one or more `1bit` processes, each a systemd unit          │
 │  zaya-npu.service    → 1bit zaya    :8088 (FLM/NPU)        │
 │  zaya-qwen06.service → 1bit zaya    :8089 (NPU2)           │
 │  zaya-gpu8b.service  → 1bit zaya    :8090 (HIP 1BP)        │
 │  jarvis.service      → 1bit jarvis  :8081 (voice loop, UI) │
 │  flm-whisper.service → FLM whisper  :8496 (STT for Jarvis) │
 └───────────────────────────┬─────────────────────────────────┘
             │ each unit picks its own backend at startup
   ┌─────────┼───────────────┬───────────────┐
   ▼         ▼               ▼               ▼
  NPU       GPU             CPU        HIP 1BP (ternary)
 (XDNA2)  (ROCm/Vulkan)   (fallback)   (engine/npu, engine/gpu)
```

Ports above match the current production fleet on the reference Strix Halo
box (see `docs/journey.md` UPDATE 33); a single-model dev setup only needs
one of these, e.g. `./build/1bit unified` on its default port.

## Ports & endpoints

| Endpoint                     | Default            | Purpose                          |
|------------------------------|--------------------|----------------------------------|
| `/v1/chat/completions`       | `127.0.0.1:8088`   | OpenAI-compatible chat API       |
| `/v1/models`                 | `127.0.0.1:8088`   | list available models            |
| `/health`                    | `127.0.0.1:8088`   | liveness check                   |

By default each server binds to **loopback only** (`127.0.0.1`).

## Exposing the API on the network

There is **no built-in auth by default** (JARVIS's WS voice endpoint is the
exception — it supports a bearer token, see
[`docs/mobile/RUNBOOK.md`](../mobile/RUNBOOK.md)). To let clients on other
machines reach an API server you have two supported options:

1. **Reverse proxy you control** (nginx/caddy/`onebitd`) — the recommended
   path. The proxy terminates TLS/auth and forwards to the loopback-bound
   engine, so the engine itself never faces the network.
2. **Explicit network bind + optional bearer token.** The API servers accept
   `--host ADDR` to bind off-loopback (e.g. `--host 0.0.0.0`), and
   `--api-token TOKEN` to require `Authorization: Bearer <token>` on every
   request (also settable via env: `UNIFIED_API_TOKEN`, `VISION_API_TOKEN`,
   `IMAGE_API_TOKEN`). Binding off-loopback prints a loud startup warning;
   without a token the server is unauthenticated, so only do this behind a
   firewall/VPN you trust.

| Server (`build/1bit <sub>` / binary) | Flags | Default bind |
|------------------------------|-------|--------------|
| `unified` | `--host`, `--api-token` | `127.0.0.1:8088` |
| `vision` (`vision_server`) | `--host`, `--api-token` | `127.0.0.1:8089` |
| `image_server` (standalone binary, built with diffusion) | `--host`, `--api-token` | `127.0.0.1:8089` |
| `router` (`unified_router`) | `--bind` | `127.0.0.1:18181` |
| `onebitd` | `--host` | `127.0.0.1:13305` |
| `jarvis` gateway | `JARVIS_BIND_ADDR`, `WS_STREAM_BIND` | `127.0.0.1` |

> Note: `image_server` historically hardcoded `0.0.0.0` (network-reachable
> with no auth). It now defaults to loopback like every other server; pass
> `--host 0.0.0.0 --api-token <token>` to expose it deliberately.

## Backend selection

Each `1bit zaya` / `1bit unified` process picks a backend (NPU/GPU/CPU) per
request or at startup, per its `--strategy`/model config — see
`tools/unified_router.cpp` (run via `1bit router`) and
[`docs/guides/architecture.md`](../guides/architecture.md). There is no
separate `unified-router.py` — it was rewritten in C++ and folded into the
single binary.

## See also

- [Installation](Installation.md)
- [Getting Started](../guides/getting-started.md)
