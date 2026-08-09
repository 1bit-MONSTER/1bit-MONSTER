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

## Format policy (measured, 2026-08-09 — see [models/catalog/QUALITY.md](../../models/catalog/QUALITY.md))

- **Q8_0 / INT8 = quality default for <7B models** (near-lossless vs f16).
- **Q4_K_M / Q4NX = INT4 for ≥7B** where it is genuinely lossless; NOT for the sub-4B catalog (2.1× PPL on 0.6B).
- **1BP (TQ2 ternary) = size tier only** — NPU pool / edge / disk-constrained targets.
- **Never requantize** — always from f16/bf16 sources (Q4→TQ2 and Q8→Q4 both measured destructive).
- **Sparsity: out of scope** on Strix Halo (no 2:4 HW, memory-bound). Revisit only for data-center GPUs.

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

**Voice-cloning runbook (all scripts exist and parse — verified 2026-08-05; the only missing input is the ~30 min sample recording, which is the user's step):**
```
# 1. Record ~30 min (VAD + loudness-normalized, 24 kHz)
python zaya_audio/record.py --duration 1800 --speaker_name <name> --output_dir ./voice_samples
# 2. Train the RVQ-VAE codec (5.87M params)
python zaya_audio/train_codec.py --data_dir ./voice_samples --output_dir ./codec_out
# 3. Train the text->codec-token adapter (QLoRA, ROCm)
python zaya_audio/train_adapter.py --codec ./codec_out --data_dir ./voice_samples --output_dir ./adapter_out
# 4. Export ONNX + assemble the .voice pack (~25 MB)
python zaya_audio/export_onnx.py --checkpoint ./codec_out --output ./codec.onnx
python zaya_audio/voice_pack.py --codec ./codec.onnx --adapter ./adapter_out --speaker <name> --output ./<name>.voice
```

- [ ] Sub-second end-to-end voice latency

## Model coverage

- [ ] Complete Kimi (Gated MLA MoE) integration
- [ ] Reverse-engineer DeepSeek V4 Flash 0731 from open weights
- [ ] Vulkan port of the Mamba1 selective scan
- [ ] Vision encoder on Vulkan (Qwen-VL)
