<div align="center">

<img src="site/assets/brand-lockup.svg" alt="1bit.systems" width="540">

# Run AI on your own hardware — fast

### An open-source, pure-C++ inference engine. NPU + GPU + CPU in one binary. Zero Python. MIT.

[![CI](https://github.com/1bit-systems/1bit-systems/actions/workflows/ci.yml/badge.svg)](https://github.com/1bit-systems/1bit-systems/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-00ff00.svg)](LICENSE)
[![Site](https://img.shields.io/badge/site-1bit.systems-12a0ed.svg)](https://1bit.systems)
[![Strix Halo](https://img.shields.io/badge/strix%20halo-gfx1151%20%2B%20XDNA%202-12a0ed.svg)](https://www.amd.com/en/products/processors/laptop/ryzen/ai-max-series.html)
[![Models](https://img.shields.io/badge/models-19%20architectures%20%C2%B7%2047%201BP-00ffaa)](docs/model-families/README.md)

**[🌐 Website](https://1bit.systems)** · **[📚 Docs](docs/README.md)** · **[🧬 Model Families](docs/model-families/README.md)** · **[🗣️ JARVIS Pipeline](docs/jarvis.md)** · **[🛠️ The Story](docs/journey.md)** · **[🗺️ Roadmap](docs/guides/roadmap.md)**

</div>

---

## The story

It started with a laptop, a disassembler, and no docs.

AMD shipped the Ryzen AI Max+ 395 with a 50 TOPS XDNA 2 NPU — and locked it behind a closed-source runtime: 22 proprietary `.so` files, 209 bitstreams, zero documentation. Nothing else could touch that chip.

**We reverse-engineered the whole stack in 4 days and replaced it with open C++** — turning 22 proprietary libraries into one 1.5 MB open-source binary. That reverse-engineering effort grew into **1bit**: one MIT-licensed C++ engine that runs LLMs on the NPU, on AMD / NVIDIA / Apple GPUs, or on plain CPU.

**→ [Read the full journey](docs/journey.md)** — every crash, breakthrough, and bug, documented in real time.

## What it is

1bit is an **inference engine** — the thing that actually runs the model. It's not a chat app; bring your own frontend.

- **One binary, every backend.** `build/1bit` holds every server and the CLI, dispatched by subcommand (`zaya`, `unified`, `jarvis`, `vision`, `chat`, …).
- **Point it at a model and run.** Reads GGUF, ONNX, and the native 1BP format, auto-detecting **19 architectures** — no config files, no per-model glue.
- **47 models out of the box**, 135M to 74B parameters, across NPU + GPU + CPU.
- **Zero Python in the engine.** Pure C++23, MIT.

## Quick start

```bash
git clone https://github.com/1bit-systems/1bit-systems
cd 1bit-systems && cmake -B build && cmake --build build
./build/1bit zaya -m model.1bp -p "Hello world"
```

> No installer yet — today it's build-from-source. See the [Installation Guide](docs/wiki/Installation.md).

## Featured: the Zyphra ecosystem

Zyphra is the one family that spans the **entire stack** — EEG → LLM (dense, MoE, Mamba) → TTS → voice cloning — all running end-to-end on the same binary. It's what the [JARVIS pipeline](docs/jarvis.md) is built on.

| Model | Params | 1BP Size | Backend(s) | Perf |
|-------|:------:|:--------:|------------|:----:|
| **ZAYA1-8B** | 8.8B | 6.6 GB | ZINC / HIP / NPU | 64 tok/s HIP |
| **ZAYA1-74B-preview** | 74B | 739 MB | ZINC / HIP | — |
| **ZR1-1.5B** | 1.5B | 781 MB | ZINC / NPU | 26 tok/s ZINC |
| **BlackMamba-1.5B** | 1.5B | 970 MB | Mamba1 HIP | **79.4 tok/s** 🏁 |
| **BlackMamba-2.8B** | 2.8B | 1.8 GB | Mamba1 HIP | 46.0 tok/s 🏁 |
| **Zamba2-1.2B / 2.7B / 7B** | 1.2–7B | 1.1–6.6 GB | ZINC / NPU | 30 tok/s ZINC |

**→ [Full Zyphra breakdown](docs/model-families/zyphra.md)**

## Model families

19 architectures, each with its **own page** and a full breakdown (params, 1BP size, backends, measured perf):

[**Zyphra**](docs/model-families/zyphra.md) · [Qwen](docs/model-families/qwen.md) · [Llama](docs/model-families/llama.md) · [Mistral](docs/model-families/mistral.md) · [Gemma](docs/model-families/gemma.md) · [Phi](docs/model-families/phi.md) · [Falcon](docs/model-families/falcon.md) · [OLMo](docs/model-families/olmo.md) · [Granite](docs/model-families/granite.md) · [SmolLM](docs/model-families/smollm.md) · [DeepSeek](docs/model-families/deepseek.md) · [GPT-OSS](docs/model-families/gpt-oss.md) · [Laguna](docs/model-families/laguna.md) · [Kimi](docs/model-families/kimi.md) · [BitNet / Bonsai](docs/model-families/bitnet-bonsai.md) · [Whisper](docs/model-families/whisper.md)

**→ [All families, indexed](docs/model-families/README.md)** · **→ [Combined support SSOT](docs/wiki/models.md)**

## Flagship pipeline: JARVIS

JARVIS is the reference **end-to-end application** — a fully local voice assistant where every stage runs on the engine:

```
mic → VAD → STT (Whisper) → router → LLM → TTS (codec) → cloned voice → speaker
```

It ties the whole engine together (speech-to-text, any catalog LLM, streaming TTS, voice cloning, personas, planning + RAG + tools) with no cloud and no Python in the hot path.

**→ [How the JARVIS pipeline works](docs/jarvis.md)**

## Benchmarks

Headline end-to-end decode, re-measured **2026-08-01** on AMD Ryzen AI MAX+ 395 (Radeon 8060S, 32 GB UMA), GGML-Vulkan:

| Model | Gen tok/s (e2e) | Backend |
|-------|:---------------:|---------|
| SmolLM2-135M | **662** 🏆 | GGML-Vulkan |
| Qwen3-0.6B | **373** | GGML-Vulkan |
| Qwen2.5-VL-3B | **100** | GGML-Vulkan |
| Qwen3.5-4B | **65** | GGML-Vulkan |
| DeepSeek-R1-Distill-Llama-8B | **44** | GGML-Vulkan |
| BlackMamba-1.5B | **79.4** | Mamba1 HIP |

Plus **43.2 TFLOPS** INT8 prefill (WMMA). Full per-model and kernel numbers live in the **[performance SSOT](docs/wiki/performance.md)**.

## Platforms & backends

- **AMD Strix Halo** — XDNA 2 NPU + ROCm HIP GPU + GGML-Vulkan
- **NVIDIA GPU** — CUDA (sm_70+) · **Apple Silicon** — Metal · **Any Vulkan 1.2+ GPU** — ZINC + GGML-Vulkan · **x86 CPU** — OpenMP

**5 backends** (NPU, HIP, ZINC, GGML-Vulkan, CPU). It also does Stable-Diffusion-family image & video generation via `image_server`.

**→ [Architecture deep-dive](docs/guides/architecture.md)** · **→ [Backends & internals](docs/README.md)** · **→ [Image/video generation](integrations/comfyui/README.md)**

## Learn more

- 📚 **[Documentation index](docs/README.md)** — start here
- 🚀 **[Getting started](docs/guides/getting-started.md)** · **[Build guide](docs/guides/building.md)**
- 🧬 **[Model families](docs/model-families/README.md)** · 📊 **[Benchmarks](docs/wiki/performance.md)**
- 🛠️ **[The engineering journey](docs/journey.md)** · 🗺️ **[Roadmap](docs/guides/roadmap.md)**

> Ideas for products built *on* the engine live under [docs/goals/](docs/goals/README.md) — separate from the engine itself.

## License

MIT — do whatever you want.
