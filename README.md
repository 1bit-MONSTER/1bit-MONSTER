<div align="center">

<img src="site/assets/banner.png" alt="1bit.MONSTER — One engine. Any model. Zero Python." width="820">

[![1bit.MONSTER](https://raw.githubusercontent.com/1bit-MONSTER/1bit-MONSTER/main/site/assets/badges/badge-amber.svg)](https://github.com/1bit-MONSTER)

## One engine to rule them all

### 100% HF model coverage. Any hardware. One open-source, pure-C++ inference engine — NPU + GPU + CPU. Zero Python.

[![CI](https://raw.githubusercontent.com/1bit-MONSTER/1bit-MONSTER/main/site/assets/badges/badge-ci-pass.svg)](https://github.com/1bit-MONSTER/1bit-MONSTER/actions/workflows/ci.yml)
[![License: MIT](https://raw.githubusercontent.com/1bit-MONSTER/1bit-MONSTER/main/site/assets/badges/badge-license.svg)](LICENSE)
[![Site](https://raw.githubusercontent.com/1bit-MONSTER/1bit-MONSTER/main/site/assets/badges/badge-site.svg)](https://1bit.monster)

**[Website](https://1bit.monster)** · **[Docs](docs/README.md)** · **[Benchmarks](docs/wiki/performance.md)** · **[The story](docs/journey.md)** · **[Roadmap](docs/guides/roadmap.md)**

`engine online` · MIT · pure C++23 · no Python

</div>

---

## Quick start

```bash
git clone https://github.com/1bit-MONSTER/1bit-MONSTER
cd 1bit-MONSTER && cmake -B build && cmake --build build
./build/1bit zaya -m model.1bp -p "Hello world"
```

That's the whole install. No runtime, no virtualenv, no Python.

## What it is

An **inference engine** — the thing that actually runs the model. Not a chat app; bring your own frontend.

- **Model agnostic.** Reads GGUF, ONNX, and native 1BP. Detects the architecture, picks a kernel path. No config files. 100% of HuggingFace text-gen architectures map to an engine token ([model families](docs/model-families/README.md)).
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

Plus **38.84 TFLOPS** INT8 prefill. Full numbers: **[docs/wiki/performance.md](docs/wiki/performance.md)**.

## Also in the box

- **JARVIS** — a fully local voice assistant: mic → VAD → STT → LLM → TTS → speaker. No cloud. ([docs/jarvis.md](docs/jarvis.md))
- **The Mesh** — installs self-discover on the LAN and federate: shared routing, model exchange, load sharing. Zero config. ([docs/mesh-protocol.md](docs/mesh-protocol.md))

## Learn more

- **[Documentation index](docs/README.md)** · **[Getting started](docs/guides/getting-started.md)** · **[Build guide](docs/guides/building.md)**
- **[Model families](docs/model-families/README.md)** · **[Benchmarks](docs/wiki/performance.md)** · **[Roadmap](docs/guides/roadmap.md)**
- **[The engineering journey](docs/journey.md)** · **[Contributing](CONTRIBUTING.md)**

## License

MIT — do whatever you want.
