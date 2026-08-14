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

**See also:** [Zyphra family](model-families/zyphra.md) (the LLM/TTS/voice models) · [Whisper](model-families/whisper.md) (STT) · [architecture](guides/architecture.md)
