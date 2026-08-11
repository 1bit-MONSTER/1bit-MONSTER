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

By default each server binds to **loopback only** (`127.0.0.1`). Expose it on a
LAN only behind a reverse proxy you control; there is no built-in auth
(JARVIS's WS voice endpoint is the exception — it supports a bearer token,
see [`docs/mobile/RUNBOOK.md`](../mobile/RUNBOOK.md)).

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
