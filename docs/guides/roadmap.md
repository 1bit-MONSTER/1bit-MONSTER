# Roadmap

The engine roadmap. This is the **single source of truth** for where 1bit is headed. (Aspirational product/business ideas built *on* the engine live under [docs/goals/](../goals/README.md), not here.)

## Recently completed (2026)

- [x] Reverse-engineered AMD's closed FastFlowLM NPU stack → open C++23 (`libnpu_engine_universal.so`)
- [x] FLM v0.9.46 model extraction: 37 models, 209 xclbins
- [x] 1BP format (Q4NX 4-bit dense · TQ2 2-bit ternary) with auto architecture detection
- [x] Single-binary `build/1bit` — every server + CLI dispatched by subcommand
- [x] Mamba1 GPU backend (BlackMamba 79.4 tok/s), Mamba2/Zamba2 hybrid
- [x] GGML-Vulkan (llama.cpp) backend — 662 tok/s peak (SmolLM2-135M)
- [x] Per-group INT8 quantization, incremental K/V attention, NPU fused engine
- [x] Canonical [models](../wiki/models.md) and [performance](../wiki/performance.md) SSOTs
- [x] Packaging: deb, snap, tarball, docker, ollama, AUR
- [x] Image & video generation (`image_server`, ComfyUI nodes)

## Engine phases

### Phase 1 — INT8 NPU inference ✅
5 INT8 xclbins NPU-verified (QKV/O/GU/D/KV), per-tensor symmetric INT8, batched prefill (20 ms/tok), OpenAI-compatible HTTP server.

### Phase 2 — Speculative decode 📋
Draft model (KQV-only / 1-layer Qwen3-0.6B) → batched INT8 verification → token acceptance. Target <50 ms/tok effective. No new xclbins needed (reuses INT8 GEMMs at M=N).

### Phase 3 — GGUF + model-agnostic 📋
GGUF Q8_0 loading, direct Q8_0→INT8 BO packing (no intermediate dequant), multi-model via xclbin parameterization, NPU attention dispatch for high context (>32 tokens).

### Phase 4 — 1-bit / BitNet 🔮
BitNet b1.58 ternary loading, ternary GEMV kernel, hybrid precision (BF16 attention + ternary weights). Target <25 ms/tok on Strix Halo NPU.

### Phase 5 — Productionization ✅ (mostly)
OpenAI-compatible server, Ollama/LangChain/Open WebUI compatibility, Docker/AUR/snap/deb packaging. Remaining: Windows support via AMD's XDNA 2 driver.

## Application milestone — JARVIS voice pipeline

The [JARVIS pipeline](../jarvis.md) is the reference end-to-end application (STT → LLM → TTS → voice cloning) that exercises the whole engine locally. Engine-side work that supports it:

- [x] Local voice loop: VAD → Whisper STT → routed LLM → codec TTS → playback
- [x] Persona system + multi-step planner + RAG + tool calls
- [ ] Whisper STT on NPU end-to-end (currently GPU HIP)
- [ ] Streaming voice codec decoder to ONNX for GPU/NPU/CPU
- [ ] Sub-second end-to-end voice latency

## Model coverage

- [ ] Complete Kimi (Gated MLA MoE) integration
- [ ] Reverse-engineer DeepSeek V4 Flash 0731 from open weights
- [ ] Vulkan port of the Mamba1 selective scan
- [ ] Vision encoder on Vulkan (Qwen-VL)
