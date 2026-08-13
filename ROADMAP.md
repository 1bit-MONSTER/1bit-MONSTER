# Roadmap — 1bit: the inference engine, JARVIS out of the box

```
engine (NPU + GPU + CPU, one binary)  →  JARVIS (voice assistant, reference app)
```

Everything else was cut. Voice cloning (training, voice packs, marketplace,
billing) was a personal quest — gutted from the repo. The agent stack
(coding agent, RAG, planners, personas) is AMD Gaia's turf — gutted. The
repo now has one through-line: **the engine, and the app that proves it.**

## Pillars

1. **The engine** — one MIT C++23 binary, every backend:
   - NPU: XDNA 2 DPU kernels, Q4NX/1BP formats, 64 MB SRAM + 64 MB aperture
     (see `research/` for the RE notes that made it possible)
   - GPU: HIP + Vulkan (ZINC), fused MoE shaders
   - CPU: scalar + OpenMP fallbacks
   - 1BP as the single canonical format; importers for GGUF/ONNX/Q4NX/H1B
2. **JARVIS** — `mic → VAD → STT → LLM → TTS → speaker`, pure C++,
   in-process with the engine. See `docs/jarvis.md`.
   Default experience: the **Zyphra stack** (ZR1 → ZAYA/BlackMamba/Zamba2
   → codec voice) — the crown jewel; `--model` bypasses it.
3. **One router** (WS-09) — per-token, cost-ranked dispatch across
   backends; the input signal for MoE expert staging (WS-07/WS-11).

## Now (P0)

- [ ] WS-11: instrument the NPU weight path — where bytes are copied per
      token on the Qwen3-0.6B q4nx decode
- [ ] WS-09: land the single router (retire `cascade` + `unified-router.py`)
- [ ] JARVIS P1: whisper on the engine (`whisper_kernels.hip` wired into
      the forward pass) — the blocker for JARVIS as a daily driver

## Next (P1)

- [ ] WS-11: NVMe expert streaming — 1-bit MoE past the 8B NPU ceiling
      (heat-table scheduler, double-buffered aperture DMA)
- [ ] WS-07: MoE decode + self-speculative staging on staged experts
- [ ] JARVIS: sentence-streaming TTS, barge-in

## Later (P2)

- [ ] WS-08: MLA KV — long-context memory on the 63 GB iGPU
- [ ] JARVIS: stock codec voice (ONNX, no training), native ALSA audio
- [ ] Packaging: installers, `models` catalog refresh, CI on real NPU
      hardware

## What was cut (and why it's not coming back)

| Cut | Reason |
|-----|--------|
| Voice cloning: codec training, voice packs, `zaya_audio/` | personal quest — kept out of the repo; stock codec voice may return as an engine artifact (P2) |
| SaaS: auth, billing, usage, beacon | product layer, no product |
| Agent stack: onebit agent, RAG, planner, personas, tools, watchdog | AMD Gaia owns that space; the engine serves it via Lemonade |
| JARVIS v1's HTTP hop + WebSocket side-server | one process, one pipeline |

*Engine first. Everything else is a tenant, not a co-owner.*
