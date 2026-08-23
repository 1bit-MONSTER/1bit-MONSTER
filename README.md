<div align="center">

<img src="site/assets/banner.png" alt="1bit.MONSTER — One engine. Any model. Zero Python." width="820">

[![CI](https://github.com/1bit-MONSTER/1bit-MONSTER/actions/workflows/ci.yml/badge.svg)](https://github.com/1bit-MONSTER/1bit-MONSTER/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

**[Website](https://1bit.monster)** · **[Docs](docs/README.md)** · **[Benchmarks](docs/wiki/performance.md)** · **[The story](docs/journey.md)** · **[Roadmap](docs/guides/roadmap.md)**

pure C++23 · no Python · MIT

</div>

---

## Why this exists

**AMD's XDNA 2 NPU shipped closed.** 22 proprietary `.so` files, 209 xclbin bitstreams, zero public documentation. One person reverse-engineered the entire stack in four days — no docs, just a disassembler, `strace`/`ftrace`, and a C++ compiler — and kept building from there. Every crash and breakthrough since is logged in the open, ~1800+ hours in. **[Read the story →](docs/journey.md)**

**One engine. Every model.** 100% of HuggingFace's architecture-bearing text-generation checkpoints (317,310 of them) map to a token this engine knows how to run — Llama, Qwen, DeepSeek, GLM, Mamba/SSM, MoE, vision, ternary/1-bit, all of it. Reads GGUF, ONNX, and native 1BP. Same binary, no config file, on NPU, GPU, or CPU. **[Model families →](docs/model-families/README.md)**

**JARVIS, out of the box.** A fully local voice assistant ships with the engine, not bolted on: mic → VAD → STT → LLM → TTS → speaker, one process, pure C++, zero cloud calls. `./build/1bit jarvis` and it's listening. **[JARVIS docs →](docs/jarvis.md)**

## Quick start

```bash
git clone https://github.com/1bit-MONSTER/1bit-MONSTER
cd 1bit-MONSTER && cmake -B build && cmake --build build
./build/1bit zaya -m model.1bp -p "Hello world"
```

That's the whole install. No runtime, no virtualenv, no Python.

## What it is

An **inference engine** — the thing that actually runs the model. Not a chat app; bring your own frontend.

- **Model agnostic.** Reads GGUF, ONNX, and native 1BP. Detects the architecture, picks a kernel path. No config files.
- **Hardware agnostic.** Same binary, same command, on any silicon: AMD Strix Halo NPU + GPU, NVIDIA (CUDA — [needs testers](https://github.com/1bit-MONSTER/1bit-MONSTER/issues/1703)), Apple Silicon, any Vulkan 1.2+ GPU, x86 CPU. No rebuild to move a model ([architecture](docs/guides/architecture.md)).
- **One binary, every server.** `build/1bit` holds the CLI and all servers, dispatched by subcommand (`zaya`, `unified`, `jarvis`, `vision`, `chat`, …). OpenAI-compatible `POST /v1/chat/completions`, speculative decoding, image/video/voice generation.

## Benchmarks

Headline end-to-end decode on an AMD Ryzen AI MAX+ 395 (Radeon 8060S, 32 GB UMA):

| Model | Gen tok/s (e2e) | Backend |
|-------|:---------------:|---------|
| SmolLM2-135M | **662** | GGML-Vulkan |
| Qwen3-0.6B | **373** | GGML-Vulkan |
| BlackMamba-1.5B | **79.4** | Mamba1 HIP |
| BlackMamba-2.8B | **46.0** | Mamba1 HIP |
| DeepSeek-R1-Distill-Llama-8B | **44** | GGML-Vulkan |
| ZAYA1-74B-preview | **17.6** | GGML-Vulkan |

Plus **38.84 TFLOPS** INT8 prefill. Measured, not projected — full numbers and methodology: **[docs/wiki/performance.md](docs/wiki/performance.md)**.

## Also in the box

- **The Mesh** — installs self-discover on the LAN and federate: shared routing, model exchange, load sharing. Zero config. ([docs/mesh-protocol.md](docs/mesh-protocol.md))

## Learn more

- **[Documentation index](docs/README.md)** · **[Getting started](docs/guides/getting-started.md)** · **[Build guide](docs/guides/building.md)**
- **[Model families](docs/model-families/README.md)** · **[Benchmarks](docs/wiki/performance.md)** · **[Roadmap](docs/guides/roadmap.md)**
- **[The engineering journey](docs/journey.md)** · **[Contributing](CONTRIBUTING.md)**

## License

MIT — do whatever you want.
