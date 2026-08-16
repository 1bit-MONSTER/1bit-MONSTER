# JARVIS — The Flagship Voice Pipeline

JARVIS is the reference **end-to-end application** built on the 1bit engine: a fully local voice assistant where every stage — listening, understanding, thinking, and speaking — runs on the same binary, on your own hardware. It's the clearest demonstration of what the engine is *for*.

> JARVIS is an application of the engine, not a separate product.

> **Ecosystem:** [max-xdna-backend](https://github.com/1bit-systems/max-xdna-backend) is a secondary experiment documenting the XDNA 2 NPU work for the upstream/funding story — the engine itself runs the same NPU natively, no MAX involved. If you only want raw inference, you never need it — but it shows the whole stack working together: STT + LLM + TTS + voice cloning, no cloud, no Python in the hot path.

## The pipeline

```
mic ─▶ VAD ─▶ STT (Whisper) ─▶ router ─▶ LLM ─▶ TTS (codec) ─▶ voice ─▶ speaker
        │                          │        │        │            │
     silence                   picks the  planner  persona    voice-clone
     detection                 best model + RAG +   /style     adapter
                               per request  tools
```

| Stage | What it does | Backed by |
|-------|--------------|-----------|
| **Capture / VAD** | Mic streaming + voice-activity detection (silence gating) | `tools/jarvis/audio_stream.*`, `vad.*` |
| **STT** | Speech → text | [Whisper V3 Turbo](model-families/whisper.md) via FLM's whisper HTTP endpoint (`:8496`, override `JARVIS_STT_URL`) — replaced the earlier whisper.cpp+ffmpeg fork/exec path |
| **Route** | Pick the best model/backend per request | `tools/jarvis/routing.*` → `unified_server` `/v1/chat/completions` |
| **Reason** | Multi-step planning, retrieval, tool calls | `planner.*`, `rag.*`, `tools.*`, `context.*` |
| **LLM** | Generate the response | any catalog model — auto-backend-selected |
| **Persona** | Voice/character + response style | `persona.*`, `personas/*.json` |
| **TTS** | Text → speech (streaming) | `codec_tts.*`, `tts.*` |
| **Voice clone** | Custom cloned voice | RVQ-VAE codec + QLoRA adapter + ONNX decoder (`zaya_audio/`) |
| **Playback** | Stream audio out | `audio_out.*` |

The **router** sends every catalog model through `unified_server`'s own auto-backend-selecting endpoint (NPU/GPU/CPU chosen automatically); unknown model ids fall through to Ollama. The **planner** decomposes a request into 2–5 subtasks, routes each to the model that fits it best, runs each with its own tool-call sub-loop, then synthesizes one grounded final answer.

## Running it

```bash
# build the engine (single binary)
cmake -B build && cmake --build build --target onebin

# start the inference server the pipeline routes to (default port 8088 —
# jarvis's router defaults to http://127.0.0.1:8088, override with UNIFIED_URL
# if you pick a different port here)
./build/1bit unified

# run the JARVIS voice loop (subcommand of the same binary)
./build/1bit jarvis
```

## Voice cloning

Cloned voices are produced offline by the `zaya_audio` pipeline and then loaded as a voice pack at runtime:

```bash
# record → train codec → extract embeddings → train adapter → export
python -m zaya_audio.pipeline --mode all --voice-name my_voice
```

Stages can be run independently and resumed. The exported adapter + ONNX decoder is what the TTS stage streams at inference time.

## Personas

Personas (`personas/zaya_default.json`, `personas/zaya_professional.json`) set the assistant's character, system prompt, and voice/style. Swap or add your own JSON to change how JARVIS sounds and behaves.

---

## Connecting to the frontend (web UI)

JARVIS ships a built-in browser chat UI. No setup, no key — just open it:

```bash
1bit jarvis                    # start the agent (or it runs as a service)
# then open in your browser:
#   http://localhost:8080/chat      ← the chat UI (also served at /)
#   http://localhost:8080/dashboard ← usage/status dashboard
```

The UI is **local-only by design** (zero trust): the browser sends no API
key, so the server only serves it to loopback connections. Remote devices
use the keyed API instead (next section) — or pair the phone app, which
finds the server automatically via the UDP beacon (`:13305`).

### Desktop install
Run `1bit jarvis` in a terminal, then open `http://localhost:8080/chat`.
That's it.

### ISO / appliance (headless)
The ISO installs JARVIS as a service (starts at boot, no terminal needed).

- **On the appliance itself** (if it has a display/browser): open
  `http://localhost:8080/chat`.
- **From another machine**: the UI is not reachable remotely (loopback
  bind + no key path in the browser — that's intentional). Connect over
  the API instead:
  1. On the appliance, get a pairing code:
     `curl -X POST http://localhost:8080/v1/pair/start`
  2. Open the returned `claim_url` on your device → your `sk_live_...` key
  3. Use it as `Authorization: Bearer <key>` (see API section below)
- To let LAN devices reach JARVIS at all, set `JARVIS_BIND_ADDR=0.0.0.0`
  in the service environment. The web UI still needs a key from remote —
  this only opens the API/ports. Remote UI access is deliberately not
  supported: use the API or the mobile app.

---

## Connecting to the API (after install)

Two processes, two ports:

| Process | Command | Port | Purpose |
|---------|---------|------|---------|
| inference | `1bit unified` | **8088** | model backend (JARVIS routes here) |
| JARVIS | `1bit jarvis` | **8080** | agent API: chat, voice, RAG, personas, pairing |

### 1. Pair a device (one-time, QR zero-trust)

At startup JARVIS prints a **QR code** + one-time code on the console (5-min
TTL, single-use). Scan it with a phone on the same network → it claims the
code and returns a per-device API key. To generate a fresh code any time:

```bash
curl -X POST http://127.0.0.1:8080/v1/pair/start
# → {"code":"AB3...", "claim_url":"http://192.168.1.5:8080/v1/pair/claim?code=AB3..."}
# open claim_url (or scan its QR) → get your sk_live_... key
```

Set `JARVIS_PUBLIC_URL=https://your.tunnel.url` when running `1bit jarvis`
so the QR/claim/base URLs use your HTTPS tunnel instead of the LAN IP.

### 2. Talk to it

**Same machine (loopback):** local requests are trusted — no key needed.

```bash
curl http://127.0.0.1:8080/v1/models
curl -X POST http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"auto","messages":[{"role":"user","content":"hello"}],"max_tokens":256}'
```

**Any other device (LAN/remote):** send the key you got from pairing.

```bash
KEY=sk_live_xxx
curl -X POST http://192.168.1.5:8080/v1/chat/completions \
  -H 'Authorization: Bearer '"$KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"auto","messages":[{"role":"user","content":"hello"}]}'
```

OpenAI-compatible, so any OpenAI SDK works: `base_url=http://<host>:8080/v1`,
`api_key=<your key>`.

### Endpoints

| Method | Path | Auth | Purpose |
|--------|------|------|---------|
| GET | `/v1/models` | public | model list |
| POST | `/v1/chat/completions`, `/api/chat` | Bearer | chat (streaming via `stream:true`) |
| POST | `/v1/audio/transcriptions` | Bearer | STT (multipart `file`) |
| POST | `/v1/audio/speech` | Bearer | TTS |
| POST | `/v1/audio/chat` | Bearer | voice-in/voice-out round trip |
| POST | `/v1/knowledge/upload`, `/v1/knowledge/search` | Bearer | RAG |
| GET | `/v1/persona`; POST `/v1/persona` | Bearer | persona control |
| POST | `/v1/agent/plan` | Bearer | planner (multi-step, tools) |
| POST | `/v1/api-key/create`; `/v1/api-key/revoke`; GET `/v1/api-key/list` | Bearer | key management |
| GET | `/v1/usage` | Bearer | quotas |
| GET | `/health`, `/live` | public | liveness |

Revoke a device: `curl -X POST http://127.0.0.1:8080/v1/api-key/revoke -d '{"key":"<the-key>"}'`
(loopback trusted; from a device use its own Bearer key).

---

**See also:** [Zyphra family](model-families/zyphra.md) (the LLM/TTS/voice models) · [Whisper](model-families/whisper.md) (STT) · [architecture](guides/architecture.md)
