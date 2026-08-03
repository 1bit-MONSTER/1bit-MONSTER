<div align="center">

<img src="site/assets/brand-lockup.svg" alt="1bit.systems" width="540">

# One Binary to rule them all

### Pure C++23 inference engine · NPU + GPU + CPU in a single binary · Zero Python · Zero Rust · Zero config files

[![CI](https://github.com/1bit-systems/1bit-systems/actions/workflows/ci.yml/badge.svg)](https://github.com/1bit-systems/1bit-systems/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-00ff00.svg)](LICENSE)
[![Site](https://img.shields.io/badge/site-1bit.systems-12a0ed.svg)](https://1bit.systems)
[![ROCm](https://img.shields.io/badge/rocm-7.15.0a-f00fd2.svg)](https://github.com/ROCm/TheRock)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-76b900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Metal](https://img.shields.io/badge/Metal-Apple%20Silicon-ff9500.svg)](https://developer.apple.com/metal/)
[![Strix Halo](https://img.shields.io/badge/strix%20halo-gfx1151%20%2B%20XDNA%202-12a0ed.svg)](https://www.amd.com/en/products/processors/laptop/ryzen/ai-max-series.html)
[![GGUF](https://img.shields.io/badge/GGUF-Qwen2%20%7C%20Qwen3%20%7C%20Mamba-00ff00)](src/gguf_loader.cpp)
[![1BP](https://img.shields.io/badge/1BP-single%20file%2C%20zero%20config-00ffaa)](include/onebp_format.h)
[![Tests](https://img.shields.io/github/actions/workflow/status/1bit-systems/1bit-systems/ci.yml?branch=main&label=tests)](https://github.com/1bit-systems/1bit-systems/actions/workflows/ci.yml)

**[🌐 Website](https://1bit.systems)** · **[🤗 1BP Models](https://huggingface.co/bong-water-water-bong)** · **[📚 Docs](docs/README.md)** · **[🛠️ Journey](docs/journey.md)** · **[📊 Benchmarks](docs/wiki/performance.md)** · **[🗺️ Roadmap](docs/guides/roadmap.md)**

**1bit runs AI models on your own hardware, fast — including on a chip nothing else can touch.** One C++ binary, MIT-licensed, no Python or Rust in the engine, no config files: build it, point it at a model, run. It does LLM inference on AMD's Strix Halo NPU (AMD's own runtime for that chip is closed-source — we reverse-engineered it from scratch), on AMD/NVIDIA/Apple GPUs, or on plain CPU. It's an inference engine, not a chat app — bring your own frontend. 47 models run out of the box, 135M to 74B parameters. **No installer yet** — today it's build-from-source (below).

Two examples of what that buys you: **Zyphra is the flagship pipeline** — LLM, TTS, and voice cloning running end-to-end from the same binary, not three stacks glued together. And when a major model drops, we don't wait for vendor support — **Kimi K3** (Moonshot's 2.8T-param model, released July 27, 2026) is already being reverse-engineered from the open weights, architecture worked out within days of release. **DeepSeek V4 Flash 0731** (released August 2, 2026) is next on the list — same pure-C++ reverse-engineering process, no Python involved.

<details>
<summary>Formats, backends, and reverse-engineering scope</summary>

Reads **GGUF**, **ONNX**, and the native **1BP** format (Q4NX 4-bit for dense models, TQ2 2-bit ternary for ternary-native checkpoints like Bonsai) with automatic architecture detection — no model registry, no per-model glue code. (Per the [1BP format policy](docs/wiki/models.md#1bp-format-policy-2026-07-31-verdict-ppl-measured), TQ2 of dense models is quality-destructive — Q4NX is the dense-model format.)

We reverse-engineered AMD's closed-source NPU stack (FastFlowLM) in 4 days — turning 22 proprietary `.so` files into a 1.5 MB open-source binary, the zaya_server NPU core (1,578,576 B raw / 1,302,736 B stripped); every server + CLI now ships as one ~67 MB ELF at `build/1bit` (both auto-tracked in [site/numbers.json](site/numbers.json)). We then extracted 37 pre-built FLM models with 209 NPU xclbins, and created our own 1BP format to transform AMD's open-source models into high-performance ternary binaries. 19 model architectures supported, 47 1BP models, including early support for Moonshot AI's Kimi family — see [reverse-engineering notes](docs/research/kimi-k3-reverse-engineering.md).

</details>

**Platform support:**
- **AMD Strix Halo** — XDNA 2 NPU + ROCm HIP GPU + **GGML-Vulkan (llama.cpp)**
- **NVIDIA GPU** — CUDA backend (sm_70+)
- **Apple Silicon** — Metal GPU backend
- **Any Vulkan 1.2+ GPU** — ZINC engine + **GGML-Vulkan (llama.cpp)**
- **x86 CPU** — OpenMP fallback

**Key numbers** (re-measured 2026-08-01, Radeon 8060S / Strix Halo, GGML-Vulkan):
- 19 model architectures · 47 1BP models · **5 backends** (NPU, HIP, ZINC, **GGML-Vulkan**, CPU)
- **662 tok/s** peak end-to-end (SmolLM2-135M, **GGML-Vulkan**)
- **373 tok/s** (Qwen3-0.6B Q4_K, **GGML-Vulkan**) — up from 344
- **100 tok/s** (Qwen2.5-VL-3B, **GGML-Vulkan**)
- **65 tok/s** (Qwen3.5-4B, **GGML-Vulkan**)
- **43.2 TFLOPS** INT8 prefill (WMMA) · **201 GB/s** TQ1 GEMV
- **video-lora** pure-C++ Vulkan backend built in (conv2d/group_norm/silu/elementwise/attention/lora_merge — GPU-verified)
- 37 FLM models extracted (209 NPU xclbins)
- Moonshot Kimi family (Gated MLA MoE) — architecture reverse-engineered, converter built

</div>

---

## Quick Start

```bash
git clone https://github.com/1bit-systems/1bit-systems
cd 1bit-systems && cmake -B build && cmake --build build
./build/1bit zaya -m model.1bp -p "Hello world"
```

**One binary, every entry point.** `build/1bit` is a single ELF holding every
server and the CLI — dispatched by subcommand (`zaya`, `unified`, `router`,
`jarvis`, `vision`, `chat`, `pull`, `list`, …) or by legacy symlink name
(`zaya_server`, `unified_server`, `jarvis_server`, `vision_server`, …).
Packages ship `1bit` plus symlinks; nothing else is needed at runtime.

See the [Installation Guide](docs/wiki/Installation.md) for full instructions.

## Model Families

We group models by **architecture family** — each entry shows 1BP file size, supported backends, and real measured performance. Families with an end-to-end pipeline (LLM + voice + vision) are marked with a pipeline badge.

**Legend:** ✅ = validated · ⚙️ = optimized · 🏁 = end-to-end · 🔄 = in progress

---

### 🧬 Zyphra Ecosystem — Complete End-to-End Pipeline

Zyphra's model portfolio spans the entire AI stack: **EEG → LLM (dense, MoE, Mamba) → TTS → Voice cloning**. We support all of them — 11 models in 1BP format, plus EEG and TTS pipelines documented for ecosystem completeness.

| Model | Params | 1BP Size | Backend(s) | Pipeline | Perf |
|-------|:------:|:--------:|------------|:--------:|:----:|
| **ZAYA1-8B** | 8.8B | 6.6 GB¹ | ZINC / HIP / NPU | 🧠🗣️ | 64 tok/s HIP |
| **ZAYA1-74B-preview** | 74B | 739 MB² | ZINC / HIP | 🧠🗣️ | — |
| **ZAYA1-VL-8B** | 8.8B | — | ZINC (vision) | 👁️🧠🗣️ | — |
| **ZR1-1.5B** | 1.5B | 781 MB | ZINC / NPU | 🧠🗣️ | 26 tok/s ZINC |
| **BlackMamba-1.5B** | 1.5B | 970 MB | Mamba1 HIP | 🧠🗣️ | **79.4 tok/s** 🏁 |
| **BlackMamba-2.8B** | 2.8B | 1.8 GB | Mamba1 HIP | 🧠🗣️ | 46.0 tok/s 🏁 |
| **Zamba2-1.2B-v2** | 1.2B | 1.1 GB | ZINC ✅ / NPU | 🧠 | 30 tok/s ZINC |
| **Zamba2-2.7B-v2** | 2.7B | 2.4 GB | ZINC ✅ / NPU | 🧠 | — |
| **Zamba2-7B-v2** | 7B | 6.6 GB | ZINC ✅ / NPU | 🧠 | — |
| **Zamba-7B-v1** | 7B | 4.3 GB | Mamba1 HIP | 🧠 | — |

**Pipeline depth:**
- 🧠 **LLM** — Zaya (MoE+CCA), ZR1 (dense reasoning), BlackMamba (Mamba1+MoE), Zamba (Mamba1/2 hybrid)
- 👁️ **Vision** — ZAYA1-VL-8B (built-in vision encoder)
- 🗣️ **Voice** — Voice cloning pipeline (RVQ-VAE codec + QLoRA adapter + ONNX decoder + streaming + persona system), see [`tools/jarvis/`](tools/jarvis/)
- 🧠 **EEG** — ZUNA1.1 / ZUNA (diffusion autoencoder, not 1BP) → [Zyphra/ZUNA1.1](https://huggingface.co/Zyphra/ZUNA1.1) · ⭐ 320
- 🗣️ **TTS** — Zonos-v0.1-hybrid / ZONOS2 (neural audio codec + MoE, not 1BP) → [Zyphra/Zonos-v0.1-hybrid](https://huggingface.co/Zyphra/Zonos-v0.1-hybrid) · ⭐ 1,106

> ¹ ZAYA1-8B 1BP is ~6.6 GB full-weight — the 149 MB entry on HF is MoE-expert-stripped; use the complete file. ² 1BP is extremely small (TQ2 ternary quantization on a 74B MoE).

---

### 🏗️ Qwen Family — Dense + VL + MoE + Speech

The most versatile ecosystem: dense models from 0.5B to 72B, vision-language variants, DeepSeek-distilled derivatives, and Whisper speech-to-text. Strongest on GPU Vulkan (ZINC) with 423 tok/s peak.

| Model | Params | 1BP Size | Backend(s) | Perf |
|-------|:------:|:--------:|------------|:----:|
| **Qwen2.5-0.5B** | 0.5B | 328 MB | ZINC / NPU | — |
| **Qwen3-0.6B** | 0.6B | 356 MB | **GGML-Vulkan** / ZINC / NPU / HIP | **373 tok/s** |
| **Qwen3-4B** | 4B | 2.2 GB | ZINC / NPU / HIP | — |
| **Qwen3-8B** | 8B | 4.1 GB | ZINC / NPU / HIP | 423 tok/s ZINC |
| **Qwen2-VL-2B** | 2B | 781 MB | ZINC (vision) | — |
| **Qwen3-VL-4B** | 4B | 2.2 GB | ZINC (vision) | — |
| **Qwen2-VL-7B** | 7B | 3.9 GB | ZINC (vision) | — |
| **DeepSeek-R1-Distill-Qwen-7B** | 7B | 3.8 GB | ZINC / NPU / HIP | — |
| **Whisper (speech-to-text)** | — | — | NPU / GPU | 🔄 |

**Pipeline depth:** Dense LLM + Vision-Language (text inference ✅, vision encoder 🔄) + Whisper speech-to-text

---

### 📐 Dense Transformers

General-purpose dense transformer models — Llama-derived, Mistral, Gemma, Phi, Falcon, OLMo, Granite. Each supports 3+ backends and auto-detects from GGUF/1BP headers.

| Model | Params | 1BP Size | Backend(s) | Peak tok/s |
|-------|:------:|:--------:|------------|:----------:|
| **SmolLM2-135M** | 135M | 101 MiB | **GGML-Vulkan** / ZINC / CPU | **662** 🏆 |
| **SmolLM2-360M** | 360M | 259 MiB | **GGML-Vulkan** / ZINC / CPU | **389** |
| **SmolLM2-1.7B** | 1.7B | 1007 MiB | **GGML-Vulkan** / ZINC / CPU | **167** |
| **Llama-3.2-1B** | 1B | 581 MB | **GGML-Vulkan** / ZINC / NPU | — |
| **Llama-3.2-3B** | 3B | 1.7 GB | **GGML-Vulkan** / ZINC / NPU / HIP | — |
| **Llama-3.1-8B** | 8B | 4.1 GB | **GGML-Vulkan** / ZINC / NPU / HIP | — |
| **TinyLlama-1.1B** | 1.1B | 328 MB | **GGML-Vulkan** / ZINC / NPU | — |
| **Mistral-7B-v0.3** | 7B | 4.3 GB | **GGML-Vulkan** / ZINC / NPU / HIP | — |
| **Ministral-8B** | 8B | 4.7 GB | **GGML-Vulkan** / ZINC / NPU / HIP | — |
| **Gemma-2-2B** | 2B | 1.2 GB | **GGML-Vulkan** / ZINC / NPU / HIP | — |
| **Gemma-3-1B** | 1B | 447 MB | **GGML-Vulkan** / ZINC / NPU | — |
| **Gemma-3-4B** | 4B | 1.9 GB | **GGML-Vulkan** / ZINC / NPU / HIP | — |
| **Phi-3-mini** | 3.8B | 2.3 GB | **GGML-Vulkan** / ZINC / NPU / HIP | — |
| **Phi-3.5-mini** | 3.8B | 2.3 GB | **GGML-Vulkan** / ZINC / NPU / HIP | — |
| **Phi-4-mini** | 3.8B | 1.9 GB | **GGML-Vulkan** / ZINC / NPU / HIP | — |
| **Falcon3-1B** | 1B | 675 MB | **GGML-Vulkan** / ZINC / NPU / HIP | — |
| **Falcon3-3B** | 3B | 1.4 GB | **GGML-Vulkan** / ZINC / NPU / HIP | — |
| **Falcon3-7B** | 7B | 4.0 GB | **GGML-Vulkan** / ZINC / NPU / HIP | — |
| **Falcon3-10B** | 10B | 5.7 GB | **GGML-Vulkan** / ZINC / NPU / HIP | — |
| **OLMo-2-7B** | 7B | 3.9 GB | **GGML-Vulkan** / ZINC / NPU / HIP | — |
| **OLMo-2-13B** | 13B | 7.6 GB | **GGML-Vulkan** / ZINC / NPU / HIP | — |
| **Granite-3.2-2B** | 2B | 1.5 GB | **GGML-Vulkan** / ZINC / NPU / HIP | — |
| **Granite-3.2-8B** | 8B | 4.8 GB | **GGML-Vulkan** / ZINC / NPU / HIP | — |
| **Nanbeige4.1** | — | — | NPU | — |

**[→ Full per-model benchmarks](docs/wiki/models.md)**

---

### 🔀 Specialized Architectures

Mixture-of-Experts, ternary, and other non-standard architectures — MoE for sparse throughput, TQ2 ternary for extreme compression, and reverse-engineered architectures.

**MoE & Sparse**

| Model | Params | 1BP Size | Backend(s) | Perf |
|-------|:------:|:--------:|------------|:----:|
| **DeepSeek-V2/V3/R1** | — | — | GPU HIP | 20 tok/s |
| **DeepSeek-R1-Distill-Llama-8B** | 8B | 4.1 GB | **GGML-Vulkan** / ZINC / NPU / HIP | **44 tok/s** |
| **Mixtral-8x7B** | 46.7B | 27.8 GB | **GGML-Vulkan** / ZINC / NPU / HIP | — |
| **Laguna-S-2.1** | 48×256ex | 73.5 GB | ZINC / NPU / HIP | — |
| **Laguna-XS-2.1** | 40×256ex | 20.9 GB | ZINC / NPU / HIP | — |
| **Laguna-S-dflash (draft)** | 6L dense | 665 MB | ZINC / NPU / HIP | — |
| **Moonshot Kimi (Moonlight, Kimi-VL)** | Gated MLA MoE | — | GPU HIP | 🔄 arch analyzed |

**Ternary & 1-bit (TQ2)** — Native ternary models using TQ2 2-bit quantization

| Model | Params | 1BP Size | Backend | Perf |
|-------|:------:|:--------:|---------|:----:|
| **Bonsai-1.7B** | 1.7B | 841 MB | HIP GPU | 21.9 tok/s |
| **Bonsai-4B** | 4B | 2.2 GB | HIP GPU | — |
| **Bonsai-8B** | 8B | 4.1 GB | HIP GPU | — |
| **Bonsai-27B** | 27B | 15 GB | HIP GPU | — |

See the **[Benchmarks](#benchmarks)** section below and the **[performance SSOT](docs/wiki/performance.md)** for full per-model and kernel numbers.

---

**[→ Full model catalog with 1BP conversion status](models/catalog/README.md)** · **[→ Per-model benchmarks](docs/wiki/models.md)**

## Benchmarks

End-to-end decode re-measured **2026-08-01** on **AMD Ryzen AI MAX+ 395 (Radeon 8060S, 32 GB UMA)** — all layers on Vulkan (`-ngl 999`), via `llama-cli -st`. Canonical data (with kernel microbenchmarks) lives in the **[performance SSOT](docs/wiki/performance.md)** / [`site/benchmarks.json`](site/benchmarks.json).

| Model | Prompt t/s | Gen t/s (e2e) | Backend | Status |
|-------|:-----:|:-----:|---------|--------|
| SmolLM2-135M Q4_K_M | 3,646 | **662** | **GGML-Vulkan** | 🏆 peak |
| SmolLM2-360M Q4_K_M | 3,299 | **389** | **GGML-Vulkan** | ✅ |
| SmolLM2-1.7B Q4_K_M | 1,469 | **167** | **GGML-Vulkan** | ✅ |
| Qwen3-0.6B Q4_K_M | 1,138 | **373** | **GGML-Vulkan** | ✅ |
| Qwen2.5-VL-3B Q4_K_M | 775 | **100** | **GGML-Vulkan** | ✅ |
| Qwen3.5-4B Q4_K_M | 56 | **65** | **GGML-Vulkan** | ✅ |
| DeepSeek-R1-Distill-Llama-8B Q4_K_M | — | **44** | **GGML-Vulkan** | ✅ |
| BlackMamba-1.5B | — | **79.4** | Mamba1 HIP | ✅ |
| BlackMamba-2.8B | — | **46.0** | Mamba1 HIP | ✅ |
| Q1 GEMV / Fused TQ2 / TQ2 GEMV (kernel) | — | 433 / 420 / 367 | ROCm HIP | ✅ |
| GPU ternary (kernel) | — | 318 | Vulkan ZINC | ✅ |
| Prefill INT8 WMMA | — | **43.2 TFLOPS** | INT8 WMMA | ✅ |

Vision pipeline (1BP, CPU): Mage-ViT → Mage-VL-4B through `vision_server` — image in, description out, end-to-end on the 1BP path.

**[→ Full benchmarks (SSOT)](docs/wiki/performance.md)**

## Architecture

```
gguf · onnx · q4nx · 1bp · h1b ──▶ [model loader: auto-detect 19 architectures]
                                    │
                                    ▼
                            [BackendManager
                             profile + select
                             fastest backend]
                                    │
                    ┌───────────────┼────────────────┐
                    ▼               ▼                 ▼
              NPU (XDNA 2)    GPU (ROCm HIP)   GPU (Vulkan ZINC)
              32 tiles,        Radeon 8060S     Radeon 8060S
              50 TOPS
                    │               │                 │
                    └───────┬───────┼──────┬──────────┘
                            │       │      │
                     ┌──────▼────┐  │  ┌───▼────────┐
                     │ GPU CUDA  │  │  │ GPU Metal   │
                     │ NVIDIA    │  │  │ Apple Silicon│
                     │ sm70+     │  │  └─────────────┘
                     └───────────┘  │
                                     ▼
                               CPU (OpenMP)
                               x86 fallback
```

**[→ Architecture deep-dive](docs/guides/architecture.md)** · **[→ NPU reverse-engineering story](docs/journey.md)**

## Backends

| Backend | Hardware | What It Runs |
|---------|----------|-------------|
| NPU (npu_engine_universal) | XDNA 2, 32 tiles | INT8 GEMM, FLM models |
| GPU HIP (ROCm) | Radeon 8060S | Ternary GEMV, MoE, SSM |
| GPU Vulkan (ZINC) | Radeon 8060S | Dense models, 1BP |
| GPU CUDA | NVIDIA sm70+ | Ternary kernels |
| GPU Metal | Apple Silicon | Ternary kernels |
| GGML-Vulkan (llama.cpp) | Any Vulkan 1.2+ | GGUF dense/MoE, flash-attn |
| Mamba1 (HIP) | Radeon 8060S | State-space (BlackMamba, Zamba) |
| Zamba2 (hybrid) | Radeon 8060S | Mamba2 + attention hybrids |
| CPU (OpenMP) | x86 | Q4NX fallback |

## Image & Video Generation (ComfyUI)

`image_server` adds Stable Diffusion-family image and video generation on top of the same backends, via the [stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) submodule, exposed as an OpenAI-style API and as ComfyUI custom nodes.

- **Image — ✅ shipped.** Text-to-image and image-to-image (SD, SDXL, FLUX, Qwen-Image, Z-Image, + LoRA) — `POST /v1/images/generations`, `POST /v1/images/edits`.
- **Video — ✅ shipped.** Text-to-video and image-to-video (Wan, LTX, Hunyuan), WebM/AVI encode — `POST /v1/video/generations`. Verified end-to-end with Wan2.1 T2V 1.3B on ROCm (2m38s vs. 8m48s CPU).
- **ComfyUI nodes**: `1BP Image Generate`, `1BP Video Generate`, plus existing LLM/VLM/TTS nodes — see [`integrations/comfyui/`](integrations/comfyui/README.md).

Build with `-DUSE_DIFFUSION=ON` (requires the `stable-diffusion.cpp` submodule); run `./build/image_server -p 8089` alongside `unified_server`.

## 📜 How We Got Here — Reverse Engineering the XDNA 2 NPU

This project started with a laptop, a disassembler, and no docs. AMD shipped the Ryzen AI Max+ 395 with a 50 TOPS XDNA 2 NPU locked behind a closed-source runtime (FastFlowLM) — 22 proprietary `.so` files, 209 xclbin bitstreams, zero documentation. **We reverse-engineered the entire stack in 4 days and replaced it with open-source code.**

| Component | Before (closed) | After (open) |
|-----------|:----------------:|:------------:|
| CLI + server | `flm`, 87.8 MB | Rebuilt, 1.5 MB (zaya_server) |
| NPU sequence gen | 22 proprietary `.so` files | `libnpu_engine_universal.so` (open-source C++23) |
| NPU bitstreams | 209 `.xclbin` files | 287 xclbins (63 rebuilt from AIE generators + 209 FLM-extracted + 15 BF16 perf) |
| Toolchain | AMD Xilinx IP | `aiecc` + Peano/AMD Xilinx IP |
| Model extraction | N/A | 37 pre-built FLM models extracted, 47 1BP models published |

The key finding: the `.so` files were NPU instruction **sequence generators**, not compute kernels — the actual computation lives entirely in the `.xclbin` FPGA bitstreams. Both layers are now fully rebuildable from source.

> **Read the full 1900+ line journey** → [`docs/journey.md`](docs/journey.md) — every crash, breakthrough, and bug documented in real-time.
>
> **Technical reverse-engineering report** → [`docs/research/fastflowlm-decode/SUMMARY.md`](docs/research/fastflowlm-decode/SUMMARY.md)
>
> **Raw analysis** → [`docs/research/fastflowlm-analysis/`](docs/research/fastflowlm-analysis/) — binary analysis, xclbin captures, instruction traces

Since then: Mamba1 GPU backend (79.4 tok/s), Vulkan flash attention, model-agnostic GGUF routing, TQ2 ternary format, **37 FLM models extracted and integrated** (DeepSeek-R1, Gemma4, Qwen3.5 Omni, Whisper, 20+ more — all running on NPU), **per-group INT8 quantization** (+3 tok/s quality fix, #1074), **NPU runtime instruction generator** (#1054), **incremental K/V attention** (#1053), **Moonshot Kimi family** reverse-engineering (Gated MLA MoE), **NPU fused engine** (oplayer=30/31/32), and a self-healing agent watchdog — **1900+ hours of engineering, all open source, MIT.**

---

## License

MIT — do whatever you want.

## Links

[Website](https://1bit.systems) · [Docs](docs/) · [Models](docs/wiki/models.md) · [Benchmarks](docs/wiki/performance.md) · [Journey](docs/journey.md) · [Roadmap](docs/guides/roadmap.md)
