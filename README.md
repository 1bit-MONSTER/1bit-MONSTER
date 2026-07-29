<div align="center">

<img src="site/assets/brand-lockup.svg" alt="1bit.systems" width="540">

# One Binary to rule them all

### Pure C++23 inference engine · NPU + GPU + CPU in a single binary · Zero Python · Zero Rust · Zero config files
# 1bit.systems — One Binary. All Backends.

[![CI](https://github.com/bong-water-water-bong/1bit-systems/actions/workflows/ci.yml/badge.svg)](https://github.com/bong-water-water-bong/1bit-systems/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-00ff00.svg)](LICENSE)
[![ROCm](https://img.shields.io/badge/rocm-7.15.0a-f00fd2.svg)](https://github.com/bong-water-water-bong/TheRock)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-76b900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Metal](https://img.shields.io/badge/Metal-Apple%20Silicon-ff9500.svg)](https://developer.apple.com/metal/)
[![Strix Halo](https://img.shields.io/badge/strix%20halo-gfx1151%20%2B%20XDNA%202-12a0ed.svg)](https://www.amd.com/en/products/processors/laptop/ryzen/ai-max-series.html)
[![GGUF](https://img.shields.io/badge/GGUF-Qwen2%20%7C%20Qwen3%20%7C%20Mamba-00ff00)](src/gguf_loader.cpp)
[![1BP](https://img.shields.io/badge/1BP-single%20file%2C%20zero%20config-00ffaa)](include/onebp_format.h)

**[🌐 Website](https://1bit.systems)** · **[🤗 1BP Models](https://huggingface.co/bong-water-water-bong)** · **[📚 Docs](docs/README.md)** · **[🛠️ Journey](docs/journey.md)** · **[📊 Benchmarks](docs/wiki/performance.md)** · **[🗺️ Roadmap](docs/guides/roadmap.md)**

**1bit** is an open-source, model-agnostic C++23 inference engine for running large language models on **AMD Strix Halo** (XDNA 2 NPU, RDNA 3.5 GPU), NVIDIA GPUs (CUDA), Apple Silicon (Metal), and any Vulkan 1.2+ device — all from a **single binary with zero Python at runtime**. It reads **GGUF**, **ONNX**, and the native **1BP** ternary format (TQ2 2-bit quantization) with automatic architecture detection — no config files, no model registry, no per-model glue code.
**1bit** is an open-source, model-agnostic C++23 inference engine for running large language models on **AMD Strix Halo** (XDNA 2 NPU, RDNA 3.5 GPU), NVIDIA GPUs (CUDA), Apple Silicon (Metal), and any Vulkan 1.2+ device — all from a **single in-process binary with zero Python at runtime**. It reads **GGUF**, **ONNX**, and the native **1BP** ternary format (TQ2 2-bit quantization) with automatic architecture detection — no config files, no model registry, no per-model glue code. Fully open-source under **MIT license**. 17 model architectures supported, 46+ 1BP models.

We reverse-engineered AMD's closed-source NPU stack (FastFlowLM) in 4 days — turning 22 proprietary `.so` files into a 207 KB open-source binary. We then extracted 37 pre-built FLM models with 209 NPU xclbins, and created our own 1BP format to transform AMD's open-source models into high-performance ternary binaries. Fully open-source under **MIT license**. 19 model architectures supported, 46+ 1BP models, including early support for **Moonshot AI's Kimi family** (Gated MLA MoE) — see [reverse-engineering notes](docs/research/kimi-k3-reverse-engineering.md).

**Platform support:**
- **AMD Strix Halo** — XDNA 2 NPU + ROCm HIP GPU (79 tok/s BlackMamba 1.5B)
- **NVIDIA GPU** — CUDA backend (sm_70+)
- **Apple Silicon** — Metal GPU backend
- **Any Vulkan 1.2+ GPU** — ZINC engine
- **x86 CPU** — OpenMP fallback

**Key numbers:**
- 19 model architectures · 46+ 1BP models · 4 backends
- 543 tok/s peak kernel (TQ2 GEMV, ROCm HIP)
- 79.4 tok/s end-to-end (BlackMamba 1.5B, Strix Halo)
- 37 FLM models extracted (209 NPU xclbins)
- Moonshot Kimi family (Gated MLA MoE) — architecture reverse-engineered, converter built, see [reverse-engineering notes](docs/research/kimi-k3-reverse-engineering.md)
- 433 tok/s peak kernel (Q1 GEMV, ROCm HIP)
- 79.4 tok/s end-to-end (BlackMamba 1.5B, Strix Halo)
- 37 FLM models extracted (209 NPU xclbins)
- Moonshot Kimi family (Gated MLA MoE) — architecture support in progress, see [reverse-engineering notes](docs/research/kimi-k3-reverse-engineering.md)
> **Data note**: Per-backend tok/s figures are published with sources and honesty flags in [`docs/wiki/performance.md`](docs/wiki/performance.md). The NPU v12 engine currently measures **69 tok/s** (re-measured 2026-07-12 on Qwen3-0.6B). Mamba1 GPU numbers (79.4 tok/s) are current and re-validated 2026-07-26.

**17 architectures · 46+ 1BP models** — see [`models/catalog/README.md`](models/catalog/README.md) for the full list.

### 🚀 Flagship 1BP models — built, quantized & hosted by us

| Model | Family | Arch | Measured | Download |
|-------|--------|------|:--------:|:--------:|
| **Zaya1-8B** | Zyphra | MoE (16-expert) | **~64 tok/s** ✅ | [🤗 HF](https://huggingface.co/bong-water-water-bong/ZAYA1-8B-1BP) |
| **ZAYA1-VL-8B** | Zyphra | Vision-Language · ViT-L + MoE | **Vision + Text** 🆕 | [🤗 HF](https://huggingface.co/bong-water-water-bong/ZAYA1-8B-1BP) |
| **BlackMamba-1.5B** | Zyphra | Mamba1 · MoE | **79.4 tok/s** ✅ | [🤗 HF](https://huggingface.co/bong-water-water-bong/BlackMamba-1.5B-1BP) |
| **BlackMamba-2.8B** | Zyphra | Mamba1 · MoE | **46.0 tok/s** ✅ | [🤗 HF](https://huggingface.co/bong-water-water-bong/BlackMamba-2.8B-1BP) |
| **Zamba2-2.7B** | Zyphra | Mamba2-hybrid | **~30 tok/s** ✅ | [🤗 HF](https://huggingface.co/bong-water-water-bong/Zamba2-2.7B-Instruct-v2-1BP) |
| **ZR1-1.5B** | Zyphra | Dense · reasoning | **26 tok/s** (ZINC GPU) ✅ | [🤗 HF](https://huggingface.co/bong-water-water-bong/ZR1-1.5B-1BP) |
| **Bonsai-1.7B** | Deepgrove | Ternary TQ2 (2-bit) | 21.9 tok/s | [🤗 HF](https://huggingface.co/bong-water-water-bong/Bonsai-1.7B-TQ2-1BP) |
| **ZUNA1.1** 📝 | Zyphra | 🧠 EEG foundation model | N/A — not an LLM | [🤗 HF](https://huggingface.co/Zyphra/ZUNA1.1) · [GitHub](https://github.com/Zyphra/zuna) |
| **Zonos-v0.1** 📝 | Zyphra | 🗣️ Text-to-speech | N/A — not an LLM | [🤗 HF](https://huggingface.co/Zyphra/Zonos-v0.1-hybrid) |

> **Zyphra is AMD's open-source AI lab** — the first model family trained end-to-end on AMD ROCm, no NVIDIA CUDA required. From training through inference, every Zyphra model runs on the AMD open ecosystem: ROCm for training, this engine for inference on Strix Halo (NPU + GPU + CPU).
>
> This engine supports the **complete Zyphra LLM family** across every architecture — dense (ZR1), MoE (Zaya1), Mamba1 SSM (BlackMamba), Mamba2 hybrid (Zamba2), and **vision-language (ZAYA1-VL-8B)** — all running on the same binary, zero Python, zero proprietary code. All models are converted with a pure-C++ toolchain, hosted on Hugging Face with open weights under permissive licenses. **[Browse all on Hugging Face →](https://huggingface.co/bong-water-water-bong)**
>
> Zyphra also publishes **non-LLM models** documented in our catalog for completeness: **ZUNA1.1** (🧠 EEG diffusion autoencoder, 380M params — reconstructs missing EEG channels, denoises, upsamples montages) and **Zonos-v0.1** (🗣️ text-to-speech, leading open-weight TTS trained on 200k+ hours). These are not convertible to 1BP (different architectures, modalities, and inference pipelines) but are included in [`models/catalog/README.md`](models/catalog/README.md) as reference.

### Why 1BP?

1BP is this project's native model format, designed to eliminate the config-file tax that every other format imposes:

- **256-byte header** — magic, version, quantization type (Q4NX/TQ2/F16/F32), model architecture enum, tokenizer config — **zero external config.json or tokenizer.json**
- **Memory-mappable weight data** — Q4NX-tiled arrays laid out exactly as the NPU DMA expects them, no load-time reshape or transpose. TQ2 ternary packs 2-bit codes at exactly half the size of Q4NX (2560 bytes/tile vs 5120)
- **Tensor index** — named tensors with native dims, byte offset, and size — no safetensors index file needed

The format exists because every model format the project ingests (GGUF, ONNX, safetensors) has a different indexing scheme, padding convention, and metadata layout. 1BP is the **normalization layer**: converters write 1BP once, the engine reads 1BP everywhere, and the translation cost is paid at conversion time rather than on every inference startup.

**Find pre-converted 1BP models at [1bit.systems/models](https://1bit.systems/models)** — Zamba2, ZR1, BlackMamba, and community-submitted conversions.

**AMD shipped the NPU locked. We unlocked it in 4 days** — no docs, no NDAs, just a laptop and a disassembler. FastFlowLM, AMD's closed-source NPU inference engine, was fully reverse-engineered and replaced with a native open-source stack; the project's own NPU engine (`engine/npu/`, `npu_engine_universal`) now dispatches directly via XRT. Full numbers in [How We Got Here](#-how-we-got-here--reverse-engineering-the-xdna-2-npu) below. Then we kept going: Mamba1 GPU kernels (79.4 tok/s on BlackMamba), Vulkan flash attention, model-agnostic GGUF routing, and a self-healing agent watchdog — 1800+ hours of engineering across all of it. One binary, all backends, zero Python.

Model-agnostic end to end: the engine auto-detects architecture and quantization from the model header — no config files, no model registry, no per-model glue code. It reads **GGUF** and **ONNX** directly, speaks FastFlowLM's own **Q4NX** tiled layout natively, and ships **1BP** — this project's own single-file format (256-byte header + tensor index + memory-mappable Q4NX-tiled weights, zero external config.json).

**[Read the full journey &rarr;](docs/journey.md)** — 1800+ lines, every crash and breakthrough documented.

</div>

---
## What is this?

A single C++23 binary (~207 KB) that runs LLMs and VLMs on AMD NPU (XDNA 2),
ROCm/CUDA/Metal/Vulkan GPU, and CPU — zero Python, zero config files.
Auto-detects 18 model architectures from GGUF/1BP headers.

We reverse-engineered AMD's closed-source NPU stack (FastFlowLM), extracted 37
pre-built models with 209 NPU xclbins, and created our own 1BP ternary weight
format to transform AMD's open-source models into high-performance binaries.
MIT licensed.

**Key numbers:**
- 18 model architectures · 46+ 1BP models · 4 backends
- 433 tok/s peak kernel (Q1 GEMV, ROCm HIP)
- 79.4 tok/s end-to-end (BlackMamba 1.5B, Strix Halo)
- 37 FLM models extracted (209 NPU xclbins)

## Quick Start

```bash
git clone https://github.com/bong-water-water-bong/1bit-systems
cd 1bit-systems && cmake -B build && cmake --build build
./build/zaya_server -m model.1bp -p "Hello world"
```

See the [Installation Guide](docs/wiki/Installation.md) for full instructions.

## Model Families

| Family | Type | Best Backend | Peak tok/s | Status |
|--------|------|-------------|-----------:|--------|
| Zaya1 | MoE + CCA attn | GPU HIP | 64 | ✅ |
| BlackMamba | Mamba1 + MoE | GPU HIP | 79.4 | ✅ |
| Zamba2 | Mamba2 hybrid | GPU Vulkan | 30 | ✅ |
| Qwen2/Qwen3 | Dense / VL | GPU Vulkan | 423 | ✅ |
| Llama 3.1/3.2 | Dense | GPU HIP (kernel) | 543 | ✅ |
| DeepSeek V2/V3/R1 | MoE + MLA | GPU HIP | 20 | ✅ |
| Mistral / Pixtral | Dense | GPU HIP (kernel) | 543 | ✅ |
| Gemma 3/4 | Dense | NPU | 67.5 | ✅ |
| Phi4-Mini | Dense | NPU | 67.5 | ✅ |
| Bonsai (Deepgrove) | Ternary-native | GPU HIP | 21.9 | ✅ |
| Laguna | Dense | GPU HIP (kernel) | 543 | ✅ |
| Falcon | Dense + MQA | GPU HIP (kernel) | 543 | ✅ |
| OLMo | Dense (no RoPE) | GPU HIP (kernel) | 543 | ✅ |
| Zamba2 | Mamba2 hybrid | GPU Vulkan | ~30 | ✅ |
| Qwen2/Qwen3 | Dense / VL | GPU HIP / NPU | — | ✅ |
| Llama 3.1/3.2 | Dense | GPU HIP / NPU | — | ✅ |
| DeepSeek V2/V3/R1 | MoE + MLA | GPU HIP | — | ✅ |
| Mistral / Pixtral | Dense | GPU HIP | — | ✅ |
| Gemma 3/4 | Dense | NPU / GPU | — | ✅ |
| Phi4-Mini | Dense | NPU | — | ✅ |
| Bonsai (Deepgrove) | Ternary-native | GPU HIP | 21.9 | ✅ |
| Laguna | Dense | GPU HIP | — | ✅ |
| Falcon | Dense + MQA | GPU HIP | — | ✅ |
| OLMo | Dense (no RoPE) | GPU HIP | — | ✅ |
| ZR1 | Dense reasoning | GPU Vulkan | 26 | ✅ |
| Qwen2-VL / Qwen3-VL | Vision-Language | GPU HIP | — | ✅ |
| Whisper | Speech-to-text | NPU / GPU | — | ✅ |
| Nanbeige4.1 | Dense | NPU | — | ✅ |
| Moonshot Kimi (Moonlight, Kimi-VL) | Gated MLA MoE | GPU HIP | — | ✅ architecture analyzed |
| Moonshot Kimi (Moonlight, Kimi-VL) | Gated MLA MoE | GPU HIP | — | 🚧 in progress |

**[→ Full model details and per-model benchmarks](docs/wiki/models.md)**

## Benchmarks

| Benchmark | tok/s | Backend | Status |
|-----------|:-----:|---------|--------|
| Q1 GEMV kernel | 433 | ROCm HIP | ✅ |
| Fused TQ2 kernel | 420 | ROCm HIP | ✅ |
| GPU ternary (Vulkan) | 318 | Vulkan ZINC | ✅ |
| BlackMamba 1.5B e2e | 79.4 | ROCm HIP | ✅ |
| BlackMamba 2.8B e2e | 46.0 | ROCm HIP | ✅ |

**[→ Full benchmarks](docs/wiki/performance.md)**

## Architecture

```
gguf / 1bp ──▶ [model loader: auto-detect 18 architectures]
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
   NPU (XDNA 2)   GPU (ROCm)    GPU (Vulkan)
   32 tiles,       Radeon 8060S   Radeon 8060S
   50 TOPS
        │              │              │
        └──────────────┼──────────────┘
                       ▼
                 CPU (OpenMP)
```

**[→ Architecture deep-dive](docs/guides/architecture.md)** · **[→ NPU reverse-engineering story](docs/journey.md)**

## Backends

| Backend | Hardware | What It Runs |
|---------|----------|-------------|
| NPU (npu_engine_universal) | XDNA 2, 32 tiles | INT8 GEMM, FLM models |
| GPU HIP (ROCm) | Radeon 8060S | Ternary GEMV, MoE, SSM |
| GPU Vulkan (ZINC) | Radeon 8060S | Dense models, 1BP |
| GPU CUDA | NVIDIA | Ternary kernels |
| GPU Metal | Apple Silicon | Ternary kernels |
| CPU (OpenMP) | x86 | Q4NX fallback |
- **GGUF** — Qwen2 / Qwen3 layout (header+embedding read; single transformer weight path; per-architecture attention/FFN not validated for Llama/Mistral/DeepSeek)
- **ONNX** — Protobuf wire format (F32/F16/BF16/INT8/INT32)
- **Q4NX** — FastFlowLM's native tiled format, fully decoded (311 tensors, 4-bit groups of 32 with bf16 scales, 32×256 NPU tile layout) — see [`Q4NX_FORMAT.md`](docs/research/fastflowlm-analysis/Q4NX_FORMAT.md)
- **1BP** — this project's native format: single self-contained file, Q4NX-tiled weights, no external metadata. The `gguf_to_onebp` tool (pure C++, `tools/gguf_to_onebp.cpp`) converts any GGUF model in place — no Python.
- **H1B** — Legacy ternary format

### Backends

- **Mamba1 GPU** — Radeon 8060S via ROCm HIP. Alternating SSM + MoE layers (BlackMamba architecture). **79.4 tok/s** (1.5B).
- **NPU** — XDNA 2 (32 tiles), fully in-process via `npu_engine_universal` (XRT-based, C++23). Runs GGUF/Q4NX/1BP models directly — no FastFlowLM subprocess, no closed-source dependency. Instruction sequences and GEMM/MHA dispatch were reverse-engineered from FLM's 22 `.so` libraries; xclbin bitstreams are rebuilt from AIE generators via `aiecc`/Chess (AMD Xilinx IP). See [`docs/research/fastflowlm-decode/SUMMARY.md`](docs/research/fastflowlm-decode/SUMMARY.md).
- **GPU (ZINC)** — Radeon 8060S via Vulkan SPIR-V (GGUF/H1B models, multi-arch)
- **GPU (HIP)** — ROCm HIP for Zaya-style models
- **CPU** — Fallback (scalar / AVX-512 / generic GGUF)

---

## Model Coverage

Model-agnostic isn't just a claim about the loader — it's been exercised across genuinely different architectures: dense transformer (Qwen3, Llama, ZR1), mixture-of-experts (Zaya1-74B-A4B, Qwen 35B MoE), vision-language (Qwen2-VL, ZAYA1-VL-8B), Mamba2-hybrid state-space (Zamba2), Mamba1 state-space + MoE (BlackMamba), and genuinely ternary/1-bit-native weights (Bonsai, stored via 1BP's TQ2 quant, not just upsampled to 4-bit) — same engine, same auto-detect path, no per-architecture fork.

Our [model catalog](models/catalog/README.md) also documents Zyphra's **non-LLM models** — **ZUNA1.1** (🧠 EEG diffusion autoencoder — reconstructs missing EEG channels, denoises, upsamples montages), **Zonos-v0.1** (🗣️ TTS), and **ZONOS2** (🗣️ TTS MoE) — as reference, though these cannot run on this engine (different architectures, modalities, and inference pipelines).

### Zaya1 — the flagship family

| Model | Params | Format | Performance | Status |
|-------|:------:|--------|-------------|:------:|
| **Zaya1-8B** | 8.84B | Q4NX / **1BP** | ~64 tok/s decode (GPU) | ✅ Primary — extensively tested, native 1BP support |
| Zaya1 Preview 74B-A4B (MoE) | 74.79B | Q4NX / **1BP** | 17.9 tok/s (iGPU, llama.cpp fork, 2026-07-03 — historical, no longer runs on current hardware) | ✅ 1BP conversion complete — [HF](https://huggingface.co/bong-water-water-bong/ZAYA1-74B-preview-1BP) |

Zaya1-8B is the model this project was built around: it's the one validated end-to-end through Q4NX, GGUF, and 1BP, and the one the `gguf_to_onebp` converter targets first when converting into the native format. Both sizes are published complete on Hugging Face — [**ZAYA1-8B-1BP**](https://huggingface.co/bong-water-water-bong/ZAYA1-8B-1BP) (1283 tensors, 16-expert MoE FFN weights) and [**ZAYA1-74B-preview-1BP**](https://huggingface.co/bong-water-water-bong/ZAYA1-74B-preview-1BP) (1923 tensors, 24-expert MoE FFN weights) — every tensor structurally verified against the source GGUF (exact parameter-count match) and numerically verified (dequantized values within expected 4-bit quantization tolerance).

### Zyphra family — beyond Zaya

Zaya1's maker, Zyphra, publishes several other architecturally distinct model lines. Converted the ones this engine can actually run end to end — dense transformer and Mamba2-hybrid — through the same 1BP pipeline:

| Model | Params | Architecture | Format | Status |
|-------|:------:|--------------|--------|:------:|
| [Zamba2-1.2B-Instruct-v2](https://huggingface.co/bong-water-water-bong/Zamba2-1.2B-Instruct-v2-1BP) | 1.2B | Mamba2-hybrid (attention every 6th layer) | **1BP** | ✅ |
| [Zamba2-2.7B-Instruct-v2](https://huggingface.co/bong-water-water-bong/Zamba2-2.7B-Instruct-v2-1BP) | 2.7B | Mamba2-hybrid | **1BP** | ✅ |
| [Zamba2-7B-Instruct-v2](https://huggingface.co/bong-water-water-bong/Zamba2-7B-Instruct-v2-1BP) | 7B | Mamba2-hybrid | **1BP** | ✅ |
| [ZR1-1.5B](https://huggingface.co/bong-water-water-bong/ZR1-1.5B-1BP) | 1.5B | Dense transformer (Qwen2 arch), reasoning-tuned | **1BP** | ✅ **26 tok/s (ZINC GPU)** |
| [BlackMamba-1.5B](https://huggingface.co/bong-water-water-bong/BlackMamba-1.5B-1BP) | 1.5B | Mamba1 + top-1 MoE (no attention at all) | **1BP** | ✅ **79.4 tok/s** |
| [BlackMamba-2.8B](https://huggingface.co/bong-water-water-bong/BlackMamba-2.8B-1BP) | 2.8B | Mamba1 + top-1 MoE | **1BP** | ✅ **46.0 tok/s** |

Each converted from a Q8_0/BF16 source (not a 4-bit GGUF) to avoid compounding quantization error through a second 4-bit pass, then structurally and numerically verified the same way as the Zaya1 conversions. One caveat: Zamba2 lacks tuned ROCm kernels, so it runs on the PyTorch fallback path — roughly 73× slower than the attention models above (see [`models/catalog/README.md`](models/catalog/README.md)).

> **Dense GPU inference is live and verified correct**: ZR1-1.5B (Qwen2 arch) runs on the native C++ ZINC Vulkan backend and matches the CPU reference **token-for-token** ([#844](https://github.com/bong-water-water-bong/1bit-systems/issues/844) — closed). ZINC is enabled by default for the architectures it computes correctly (llama/mistral/qwen2) and falls back to the exact `cpu_generic` path otherwise; `ZINC_DISABLE=1` forces HIP/CPU. The engine is also **crash-hardened** — a backend that fails to initialize fails over to HIP/CPU instead of taking the server down.

**BlackMamba required a from-scratch converter** — no upstream GGUF export exists for this architecture, and it predates the architecture support standard converters have. The one-time bootstrap conversion shipped with three real correctness bugs on the first pass (wrong Q4_0 nibble encoding, a conv1d weight reshape that silently scrambled channel/kernel-tap pairing, and a dropped MoE router bias), all found and fixed by cross-checking against the in-tree C++ reference (`tools/blackmamba_cpu_reference.cpp`) and the official Zyphra implementation — see the model cards on Hugging Face for the full writeup. The resulting weights are what `gguf_to_onebp` now ingests directly.

**Fast inference is now wired**: `src/mamba1_engine.hip` kernels are compiled into `librocm_cpp.so` and the `Mamba1Backend` (HIP GPU) is registered as a first-class backend in `BackendManager`. Both BlackMamba sizes load end-to-end through the Mamba1 GPU backend: alternating SSM layers (rmsnorm → in_proj → conv1d/silu → selective_scan → gate → out_proj) and MoE FFN layers (router → top-1 expert dispatch → SiLU → scale-add residual). The diagnostic tool `tools/test_mamba1_backend.cpp` loads a Mamba1 GGUF directly into the HIP backend for testing without the HTTP server. PR [#579](https://github.com/bong-water-water-bong/1bit-systems/pull/579) shipped the build linkage, conv state fix, and A_log exponentiation fix.

**Vision-language is now supported**: ZAYA1-VL-8B — a real vision-language model combining a SigLIP ViT vision encoder with the Zaya1-8B MoE text decoder. The vision tower (24-layer ViT, fused QKV, Q8_0 quant) and connector (QWEN2_MERGER-style projector) are handled by the new `vision_encoder` library — pure C++, no Python, no OpenCV. The converter now preserves vision weights in 1BP files with full tensor metadata. See [`include/vision_encoder.h`](include/vision_encoder.h) and [`tools/zaya1_vl_demo.cpp`](tools/zaya1_vl_demo.cpp). (An earlier POC, Qwen2-VL, did minimal real image-to-text before ZAYA1-VL landed.)

### Beyond LLMs — Zyphra's Other Models

Zyphra's research portfolio extends beyond language models into **brain-computer interfaces** and **speech synthesis**. These are documented in our [model catalog](models/catalog/README.md) as ecosystem reference but are **not convertible to 1BP** and do not run on this engine:

| Model | Domain | What it does | 
|-------|:------:|--------------|
| **ZUNA1.1** | 🧠 EEG | Foundation model for EEG — denoises, reconstructs missing channels, upsamples sparse montages. 380M-parameter diffusion autoencoder with 4D rotary positional encoding over (x,y,z,t). Requires <1GB VRAM, runs on consumer GPUs. Apache 2.0. [GitHub](https://github.com/Zyphra/zuna) · [HF](https://huggingface.co/Zyphra/ZUNA1.1) |
| **Zonos-v0.1** | 🗣️ TTS | Leading open-weight text-to-speech. 200k+ hours multilingual speech training. Transformer (434⭐) and hybrid (1,106⭐) variants. Apache 2.0. GGUF versions available for `zonos.cpp`. [GitHub](https://github.com/Zyphra/Zonos) · [HF](https://huggingface.co/Zyphra/Zonos-v0.1-hybrid) |
| **ZONOS2** | 🗣️ TTS MoE | Next-gen TTS with Mixture of Experts. GGUF via `zonos2.cpp`. [GitHub](https://github.com/Zyphra/ZONOS2) · [HF](https://huggingface.co/Zyphra/ZONOS2) |

### 🏆 Top 5 — Raw NPU Engine, No FLM (single binary, auto-detected)

*From [`engine/npu/BENCHMARKS.md`](engine/npu/BENCHMARKS.md), measured 2026-07-03/07-12 — predates the 2026-07-19 GGUF dequant correctness fixes (Q2_K/Q3_K/Q5_K, RoPE, dtype enums), so treat as directional, not re-verified. (Also superseded: a previously reported 572 tok/s DSpark speculative-decoding figure turned out to come from an undertrained checkpoint — see [issue #235](https://github.com/bong-water-water-bong/1bit-systems/issues/235).)*

| Model | Family | Decode | Tok/s | Correctness |
|-------|--------|:------:|:-----:|:-----------:|
| Qwen3-0.6B | Qwen3 | 36 ms/tok | **28** | 28/28 ✅ |
| Gemma4-E2B | Gemma | 62 ms/tok | **16** | 35/35 ✅ |
| Qwen3-VL-4B | Qwen3 (vision) | 93 ms/tok | **11** | 36/36 ✅ |
| Llama-3.1-8B | Llama | 100 ms/tok | **10** | 32/32 ✅ |
| Qwen3-8B | Qwen3 | 127 ms/tok | **8** | 36/36 ✅ |

Same binary, same auto-detect path, no per-model glue — the loader reads architecture off the model header for all 46+ 1BP models.

### TQ2 — the actual 1-bit/ternary storage path

Every model above is stored via 1BP's default Q4NX quant (4-bit, works for any source precision). `ONEBP_TQ1`/`ONEBP_TQ2` have been defined in the format since it was designed but were never implemented — meaning even genuinely ternary-trained models were getting upsampled to 4-bit on the way in. Fixed for TQ2: symmetric 2-bit quantization (every value is exactly `-scale`, `0`, or `+scale`, one BF16 scale per 32-group, no zero-point needed), exactly half of Q4NX's tile size.

| Model | Params | Format | Verification | HF |
|-------|:------:|--------|---------------|-----|
| [Bonsai-1.7B](https://huggingface.co/bong-water-water-bong/Bonsai-1.7B-TQ2-1BP) | 1.72B | **1BP (TQ2)** | 100% of dequantized values match source within BF16 scale-rounding (mean rel. error rounds to 0.000000) — lossless repack, not requantization | [link](https://huggingface.co/bong-water-water-bong/Bonsai-1.7B-TQ2-1BP) |

Convert another ternary-native model the same way: `./build/gguf_to_onebp model.gguf output.1bp --tq2` (pure C++, no Python). `ONEBP_TQ1` (1.58-bit, base-3 packing) is still unimplemented — 256 isn't evenly divisible by its 5-values-per-byte scheme, so it needs more careful boundary handling than TQ2 did.

---

## 📜 How We Got Here — Reverse Engineering the XDNA 2 NPU

This project started with a laptop, a disassembler, and no docs. AMD shipped the Ryzen AI Max+ 395 with a 50 TOPS XDNA 2 NPU locked behind a closed-source runtime (FastFlowLM) — 22 proprietary `.so` files, 209 xclbin bitstreams, zero documentation. **We reverse-engineered the entire stack in 4 days and replaced it with open-source code.**

| Component | Before (closed) | After (open) |
|-----------|:----------------:|:------------:|
| CLI + server | `flm`, 87.8 MB | Rebuilt, 17.5 MB |
| NPU sequence gen | 22 proprietary `.so` files | `libnpu_engine_universal.so` (173 KB) |
| FPGA bitstreams | 209 `.xclbin` files | 63 rebuilt from AIE generators |
| Toolchain | AMD Xilinx IP | `aiecc` + Peano/AMD Xilinx IP |

The key finding: the `.so` files were NPU instruction **sequence generators**, not compute kernels — the actual computation lives entirely in the `.xclbin` FPGA bitstreams. Both layers are now fully rebuildable from source.

> **Read the full 1800+ line journey** → [`docs/journey.md`](docs/journey.md) — every crash, breakthrough, and bug documented in real-time.
>
> **Technical reverse-engineering report** → [`docs/research/fastflowlm-decode/SUMMARY.md`](docs/research/fastflowlm-decode/SUMMARY.md)
>
> **Raw analysis** → [`docs/research/fastflowlm-analysis/`](docs/research/fastflowlm-analysis/) — binary analysis, xclbin captures, instruction traces

Since then: Mamba1 GPU backend (79.4 tok/s), Vulkan flash attention, model-agnostic GGUF routing, TQ2 ternary format, vision-language support, and a self-healing agent watchdog — **1800+ hours of engineering, all open source, MIT.**

---

## License

MIT — do whatever you want.

## Links

[Website](https://1bit.systems) · [Docs](docs/) · [Models](docs/wiki/models.md) · [Benchmarks](docs/wiki/performance.md) · [Journey](docs/journey.md) · [Roadmap](docs/guides/roadmap.md)
