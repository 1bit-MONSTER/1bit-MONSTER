# JARVIS — Local Private AI Agent

A fully local, private AI agent server built on the `1bit-systems` inference
engine. All core inference runs on-device against the local NPU (`npu_xrt`
via the FLM bridge) and/or GPU (ROCm, via Ollama) backends on AMD Radeon
hardware — no cloud dependency, no external API calls for any core function.

**The running server is now `jarvis-rs/` (Rust)** — a full port of the
original Python implementation, kept API-compatible so 1bit Mobile and the
`/chat` web UI need no changes. See `jarvis-rs/` for the server itself.
This directory now holds only what stays Python: `voice/record.py` and
`voice/train.py` (offline voice-clone data collection/training — never
imported by the running server) and `voice/codec.py` (the `AudioCodec`
architecture those two scripts share; the server's own decode-only
inference path is a separate `candle-nn` port at `jarvis-rs/src/voice.rs`).

## Capabilities

- **Retrieval-augmented generation (RAG)** — full-text search over a local
  markdown knowledge base (`jarvis-rs/src/rag.rs`). Documents can be
  uploaded via API and are automatically injected as context for relevant
  queries.
- **Local multi-turn memory** — each conversation (`session_id`) is
  persisted server-side to `conversations/<session_id>.md` and recalled on
  every subsequent request to that session, independent of whatever history
  a given client happens to send. This means a phone client, a curl script,
  and a desktop UI hitting the same session all share continuity.
- **Tool invocation** (`jarvis-rs/src/tools.rs`) — the model can call
  `search_knowledge`, `get_time`, `list_models`, or `add_note` via a
  `TOOL_CALL: {...}` directive in its output; the server parses, executes,
  and feeds the result back for a grounded final answer.
- **Multi-step task planning** (`jarvis-rs/src/planner.rs`) — complex
  requests are decomposed into an ordered list of subtasks by a fast local
  model, each subtask is routed to whichever model in the local roster
  actually fits it (vision → `qwen3vl`, heavy reasoning → the larger model,
  everything else → the fast default), and a synthesis pass combines the
  results — grounded on the actual tool outputs, not just each subtask
  model's own paraphrase of them. Because the underlying engine can hold
  multiple of these models resident at once, steps don't pay a cold-load
  penalty between them.
- **Permission & privacy control** — tools are classed `safe` (read-only,
  always runs) or `sensitive` (mutates local state); sensitive tools only
  execute if the request explicitly passes `allow_write: true`. Every tool
  call — allowed or denied — is appended to a local-only audit log
  (`<knowledge_dir>/tools/audit.log`) that is never transmitted anywhere.
- **Voice** — local speech-to-text (`whisper-rs`, ggml "tiny" model) and
  text-to-speech (Piper subprocess, plus a `candle-nn` voice-cloning codec
  decoder), all on-device. The physical-speaker mirror
  (`jarvis-rs/src/audio_out.rs`) plays every TTS reply out through any
  non-onboard ALSA playback device attached to the box, and the LAN beacon
  (`jarvis-rs/src/beacon.rs`) lets 1bit Mobile auto-discover the server
  without typing in an IP address.

## Architecture

```
Client (curl / 1bit Mobile app / web UI)
        │  HTTP (OpenAI-compatible /v1/chat/completions, plus /v1/agent/plan)
        ▼
jarvis-rs (Rust, axum) ── session memory (rag.rs) ── knowledge base (rag.rs)
        │                                              ▲
        ├─ tool-call loop (tools.rs) ───────────────────┘  (search_knowledge, add_note)
        ├─ multi-step planner (planner.rs) ── routes subtasks across models
        ▼
jarvis-rs/src/routing.rs
        ├─ NPU backend      → npu_xrt engine via FLM bridge  (AMD XDNA 2 NPU)
        ├─ unified backend  → tools/unified_server.cpp        (Zyphra family, etc.)
        └─ GPU backend      → Ollama /api/chat                (AMD Radeon GPU, ROCm)
```

## Setup

Requires a running local backend:

- **NPU backend**: the `1bit-systems` NPU engine / FLM bridge listening on
  `NPU_URL` (default `http://127.0.0.1:52625`).
- **GPU backend**: [Ollama](https://ollama.com) listening on `OLLAMA_URL`
  (default `http://127.0.0.1:11434`) with whichever models from
  `jarvis-rs/src/routing.rs`'s `MODEL_ROUTING` table you want to use pulled
  locally (`ollama pull qwen3.5:9b`, etc.).
- **STT model**: `jarvis-rs/models/ggml-tiny.bin` (whisper.cpp GGML format —
  `curl -L -o jarvis-rs/models/ggml-tiny.bin https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin`),
  or set `WHISPER_MODEL_PATH` to point elsewhere.
- **TTS (Piper fallback)**: a `piper` binary on `PATH` (or at
  `$JARVIS_VENV/bin/piper`) plus a voice model under `$PIPER_VOICES_DIR`
  (default `~/piper-voices`), e.g. `en_US-lessac-medium.onnx`.

### Run

```bash
cd 1bit-systems/jarvis-rs
cargo build --release
JARVIS_PORT=8080 ./target/release/jarvis-rs
# → http://localhost:8080/chat
```

Environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `NPU_URL` | `http://127.0.0.1:52625` | NPU/FLM backend (compiled in, see `routing.rs`) |
| `OLLAMA_URL` | `http://127.0.0.1:11434` | GPU/Ollama backend (compiled in) |
| `JARVIS_PORT` | `8080` | HTTP listen port |
| `JARVIS_KNOWLEDGE_DIR` | `~/jarvis/data/knowledge` | RAG + memory + audit log storage |
| `WHISPER_MODEL_PATH` | `models/ggml-tiny.bin` | STT model path |
| `PIPER_VOICES_DIR` | `~/piper-voices` | Piper TTS voice models |
| `JARVIS_VENV` | `~/jarvis-env` | venv checked first for a `piper` binary |

`--no-beacon` disables the LAN auto-discovery UDP broadcast.

## API

- `POST /v1/chat/completions` (OpenAI-compatible) — body accepts the usual
  `model`/`messages`/`stream`/`max_tokens`/`temperature`, plus:
  - `session_id` (str, default `"default"`) — which memory thread to use
  - `rag` (bool, default `true`) — inject knowledge-base context
  - `tools` (bool, default `true` for non-streaming requests) — enable tool calling
  - `allow_write` (bool, default `false`) — permit sensitive tools (`add_note`)
- `POST /v1/agent/plan` — `{"request": "...", "session_id": "...", "allow_write": false}`
  → decomposes and executes a multi-step plan, returns `{"plan": [...], "steps": [...], "answer": "..."}`
- `POST /v1/knowledge/search`, `POST /v1/knowledge/upload`, `GET /v1/knowledge` — RAG
- `POST /v1/audio/transcriptions`, `POST /v1/audio/speech` — STT/TTS
- `GET /v1/audio/devices` — local ALSA playback device discovery (physical-speaker mirror status)
- `GET /v1/voice/packs`, `POST /v1/voice/packs`, `POST /v1/voice/activate` — voice-clone pack management
- `GET /v1/models` — list routable models
- `GET /health`, `GET /live` — liveness

## Model roster

See `jarvis-rs/src/routing.rs`'s `MODEL_ROUTING` for the full table. The
engine can hold multiple of these resident at once, which is what the
multi-step planner relies on to route different subtasks to different
models without a reload between steps.

## Voice cloning (offline training)

Still Python, unchanged, and not part of the running server:

```bash
python3 jarvis/voice/record.py --name bcloud --session 1
python3 jarvis/voice/train.py --samples ./voice/samples/bcloud/ --name bcloud
```

Produces a `.voice` pack (`decoder.pt` + `speaker.pt` + `metadata.json`,
tar.gz'd) that `jarvis-rs` loads directly via `candle`'s PyTorch pickle
reader — no format conversion needed.
