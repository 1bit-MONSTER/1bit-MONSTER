<div align="center">

<img src="site/assets/banner.svg" alt="1bit.MONSTER — One engine. Any model. Zero Python." width="820">

[![CI](https://github.com/1bit-MONSTER/1bit-MONSTER/actions/workflows/ci.yml/badge.svg)](https://github.com/1bit-MONSTER/1bit-MONSTER/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**[Website](https://1bit.monster)** · **[Community (Fluxer)](https://fluxer.gg/7wqCREKi)** · **[Join Discord](https://discord.gg/Qy38d4Xu2h)** · **[Docs](docs/README.md)** · **[Model families](docs/model-families/README.md)** · **[Benchmarks](docs/wiki/performance.md)** · **[JARVIS](docs/jarvis.md)** · **[The story](docs/journey.md)** · **[Roadmap](docs/guides/roadmap.md)**

pure C++26 · zero Python at runtime · MIT

</div>

---

**One engine. Any model. Zero Python.**

A model-agnostic, hardware-agnostic inference engine in a single C++26 binary. Point it at a model file — GGUF, 1BP, ONNX, H1B, safetensors — and it auto-detects the architecture and runs on whatever hardware you have: AMD XDNA 2 NPU, GPU (HIP, CUDA, Metal, Vulkan), or CPU. No config files, no per-model glue, no Python interpreter anywhere.

## What you get

- **One binary** — `build/1bit` is busybox-style: every server and CLI in a single ELF, dispatched by subcommand (`1bit zaya`, `unified`, `router`, `jarvis`, `vision`, …).
- **Any model** — 552 architecture tokens mapping 1,774 HuggingFace arch strings; 317,310 / 317,310 text-generation checkpoints on the hub (100%) land on an engine token.
- **Any hardware** — NPU (XDNA 2, reverse-engineered in 4 days — [the story](docs/journey.md)), GPU (HIP, CUDA, Metal, Vulkan), CPU (AVX-512/scalar). Auto-routed per model.
- **Zero Python** — pure C++26 at runtime. No virtualenv, no interpreter, no runtime stack to babysit.

## Quick start

```bash
git clone https://github.com/1bit-MONSTER/1bit-MONSTER
cd 1bit-MONSTER && cmake -B build && cmake --build build
./build/1bit zaya -m model.1bp -p "Hello world"
```

That's the whole install. Full build guide: [docs/guides/building.md](docs/guides/building.md).

## Model families

1bit auto-detects [16+ model families](docs/model-families/README.md) with zero per-model code — Zyphra (Zaya, Zamba2, BlackMamba), Qwen, Llama, Mistral, Gemma, Phi, Falcon, OLMo, Granite, SmolLM, DeepSeek, GPT-OSS, Laguna, Kimi, BitNet/Bonsai, Whisper. The [Zyphra family](docs/model-families/zyphra.md) is the flagship: a full stack from EEG → LLM → TTS on one binary.

Also in the box: **JARVIS** — a fully-local voice pipeline (mic → STT → LLM → TTS → speaker) that proves the engine end-to-end ([docs/jarvis.md](docs/jarvis.md)).

## Docs

- [Docs index](docs/README.md) · [Getting started](docs/guides/getting-started.md) · [Architecture](docs/guides/architecture.md) · [Benchmarks](docs/wiki/performance.md) · [Lemonade compat](docs/guides/Lemonade-Compat.md) · [The Mesh](docs/mesh-protocol.md) · [Roadmap](docs/guides/roadmap.md)

## Community

- **Discord** → https://discord.gg/Qy38d4Xu2h
- **Fluxer** (official support) → https://fluxer.gg/7wqCREKi
- **Issues & feature requests** → https://github.com/1bit-MONSTER/1bit-MONSTER/issues

## License

MIT — do whatever you want.
