# JARVIS — the voice assistant that ships with 1bit

```
mic → VAD → STT (libwhisper) → LLM (engine) → TTS (piper) → speaker
```

JARVIS is the reference app for the engine: every stage is in-process,
pure C++, and (for the LLM stage) runs on any engine backend — NPU, GPU, CPU.

## Default stack: Zyphra (the crown jewel)

JARVIS's default experience is the **Zyphra ecosystem** — the complete
MIT open-source pipeline (ZR1 routing → ZAYA / BlackMamba / Zamba2 LLMs →
codec voice). When started without `--model`, JARVIS picks the first Zyphra
model found in the weights dir (preference: ZAYA1-8B → ZAYA1-74B →
BlackMamba-2.8B → BlackMamba-1.5B → Zamba2-7B/2.7B/1.2B → ZR1-1.5B).

Clients can bypass the default entirely: `jarvis --model <any>` loads
their own model for a different experience. Install with
`bash install.sh --with-jarvis` to build JARVIS, get a `jarvis` launcher
+ config at `~/.config/1bit/jarvis.env`, and an optional systemd user
unit to start it at login.

## The rebuild (2026-08-06)

JARVIS v1 was a C++ port of the deleted Python `jarvis/` package — and it
carried every fork that ever lived inside it:

| Gutted | Was | Why |
|--------|-----|-----|
| `auth.cpp/h` | API-key auth for the SaaS | voice-cloning product, gone |
| `billing.cpp/h` | Stripe metering | voice-cloning product, gone |
| `usage.cpp/h` | per-user usage caps | voice-cloning product, gone |
| `beacon.cpp/h` | 1bit-Mobile pairing | agent stack, gone |
| `persona.cpp/h`, `planner.cpp/h` | agent personas/planner | agent stack, gone |
| `rag.cpp/h`, `tools.cpp/h` | RAG + tool calling | agent stack, gone |
| `routing.cpp/h`, `context.cpp/h` | agent routing/memory | engine routers own this now |
| `audio_stream.cpp/h` | WebSocket side-server (own port) | two servers for one voice loop |
| `codec_tts.cpp/h`, `voice_cli.py` | ONNX codec TTS + Python CLI | voice-cloning (personal quest — kept out of the repo; the codec decoder can return as a stock voice later) |
| `jarvis_server.cpp` | 1761-line HTTP agent server (port 8080 → talks to unified_server :8088) | the fork condenser itself |
| `zaya_audio/` | Python training toolkit (codec, voice packs) | personal quest, gutted |

Rebuilt as **JARVIS v2**:

- **One process.** In-process `BackendManager` — the old JARVIS talked to a
  second server over HTTP. The engine IS the app.
- **Pure C++.** No Python anywhere. The only subprocesses are `arecord`,
  `aplay`, and `piper` — the same fork/exec idiom as the engine's own NPU
  worker.
- **Thin.** `tools/jarvis/` is 9 files, ~1300 lines: `audio` (capture +
  playback), `stt` (libwhisper wrapper), `tts` (piper bridge), `vad`
  (energy-based, kept from v1), `jarvis_app` (the pipeline).

## Build & run

```bash
cmake --build build --target jarvis_app    # build/jarvis
# or inside the one binary:
./build/1bit jarvis --model "Qwen3-0.6B" --text        # text chat
./build/1bit jarvis --model "Qwen3-0.6B" \
    --whisper models/whisper-tiny.gguf \
    --piper-model ~/piper/en_US-lessac-medium.onnx    # voice
```

Flags: `--model` (required), `--weights-dir`, `--text`, `--whisper`,
`--piper` / `--piper-model`, `--mic DEVICE`, `--system`, `--max-tokens`,
plus fleet mode: `--mesh-dispatch`, `--mesh-name`, `--port`, `--mesh-port`.

## Fleet mode — JARVIS with a DSH brain on the mesh

`--mesh-dispatch` gives JARVIS **DSH awareness**: no local model, no engine
init — JARVIS becomes a thin fleet node that dispatches every LLM turn to
the sibling install that serves the requested model.

```
user speaks → mic → VAD → STT → [dispatch over the mesh] → TTS → speaker
                                   │
                discovers: who serves what? (UDP multicast)
                decides:    local → best model match → any chat peer
                posts:      OpenAI-compatible /v1/chat/completions
```

```bash
# a fleet node serving the model:
./build/mesh_peer --name alice --port 18088 --stub-chat --models "Qwen3-4B:stub"
# JARVIS with no local model, brain on the mesh:
./build/1bit jarvis --mesh-dispatch --model Qwen3-4B --port 18081
curl -X POST localhost:18081/v1/jarvis/turn -d '{"text":"what can you do?"}'
```

- JARVIS **announces itself** on the mesh (`/v1/mesh/*` + `/v1/jarvis/*`),
  so sibling installs and DSH brains can find it and route to it.
- `POST /v1/jarvis/turn` is the DSH brain's socket: text in → dispatched
  reply out (spoken if a piper voice is loaded).
- The DSH brain (`integrations/dsh/jarvis-brain.js`) does the capability
  routing: `--say "..." --model ZAYA1-74B` dispatches to the machine that
  serves that model. Run it as a DSH skill (`jarvis-fleet-brain`) to make
  DSH the heart and soul of the pipeline.
- Cold start: JARVIS retries dispatch briefly until neighbors are
  discovered. Test: `Testing/jarvis_mesh_smoke.sh` (4/4).

**→ [Mesh protocol](mesh-protocol.md)** · **→ [DSH brain](integrations/dsh/README.md)**

## Status & known limits (honest)

| Stage | Status |
|-------|--------|
| VAD | ✅ energy-based, 20 ms frames, lookback/ramp-down |
| LLM | ✅ in-process engine, any backend, history (last 6 turns); default = Zyphra stack |
| TTS | ✅ piper (fork/exec, 22050 Hz); codec voice = P2 |
| STT | ✅ GPU-accelerated (src/whisper_hip.hip) when a HIP device is present; scalar CPU fallback; `WHISPER_GPU=0` forces scalar. Verify on hardware: `whisper_demo model.gguf audio.wav --check-gpu` |
| Barge-in | 🔲 utterances during a reply are dropped (P1) |

## Roadmap

- **P1 — STT on the engine.** GPU path landed 2026-08-06
  (`src/whisper_hip.hip`, runtime-detected, `WHISPER_GPU=0` to force
  scalar). Remaining: route whisper through an engine backend for NPU.
  Verify on hardware first: `whisper_demo model.gguf audio.wav --check-gpu`
- **P1 — sentence streaming.** Piper per sentence while the LLM keeps
  generating (the old codec decoder's 13 ms framing was built for this;
  the clean pipeline makes it a queue, not a server).
- **P1 — barge-in.** VAD detects speech during playback → stop TTS, new
  turn.
- **P2 — codec voice returns.** The RVQ codec decoder (`src/codec_decoder`
  pattern) as a *stock* voice via ONNX Runtime — no training, no voice
  packs, no per-user anything.
- **P2 — ALSA native audio.** `arecord`/`aplay` pipes → `snd_pcm` when
  latency tuning matters (see `ponytail:` note in `audio.h`).
