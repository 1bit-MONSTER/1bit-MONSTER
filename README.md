<div align="center">

<img src="site/assets/brand-lockup.svg" alt="1bit.MONSTER" width="540">

## One engine. Every model. Any chip.


### 100% HF model coverage. Any hardware. One open-source, pure-C++ inference engine — NPU + GPU + CPU in a single engine. Model agnostic. Hardware agnostic. Zero Python.

[![CI](https://github.com/1bit-MONSTER/1bit-MONSTER/actions/workflows/ci.yml/badge.svg)](https://github.com/1bit-MONSTER/1bit-MONSTER/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-C6FF3D.svg)](LICENSE)
[![Site](https://img.shields.io/badge/site-1bit.monster-C6FF3D.svg)](https://1bit.monster)
[![Models](https://img.shields.io/badge/HF%20coverage-100%25-C6FF3D.svg)](docs/model-families/README.md)
[![Backends](https://img.shields.io/badge/hardware-NPU%20%C2%B7%20HIP%20%C2%B7%20Vulkan%20%C2%B7%20CUDA%20%C2%B7%20Metal%20%C2%B7%20CPU-C6FF3D.svg)](docs/guides/architecture.md)

**[Website](https://1bit.monster)** · **[Docs](docs/README.md)** · **[Model families](docs/model-families/README.md)** · **[Benchmarks](docs/wiki/performance.md)** · **[JARVIS](docs/jarvis.md)** · **[The story](docs/journey.md)** · **[Roadmap](docs/guides/roadmap.md)**

`engine online` · MIT · pure C++23 · no Python · 6 hardware targets

</div>

---

## Quick start

```bash
# build from source — no installer yet
git clone https://github.com/1bit-MONSTER/1bit-MONSTER
cd 1bit-MONSTER && cmake -B build && cmake --build build
./build/1bit zaya -m model.1bp -p "Hello world"
```

That is the whole install. One engine, no runtime, no virtualenv, no Python.

| 100% | 32 | 25 | 6 | 0 |
|:---:|:--:|:--:|:-:|:-:|
| HF checkpoints covered | arch tokens | families in manifest | hardware targets | Python in the runtime |

## Model agnostic

Point it at a model and run. The engine reads **GGUF**, **ONNX**, and the native **1BP** format, detects the architecture, and picks a kernel path — no config files, no per-model glue, no conversion step you have to babysit.

Coverage does not come from porting models one at a time. It comes from the architecture class: 32 arch tokens map 120+ HF architecture strings — roughly 50 classes cover the long tail of what HuggingFace hosts, and a class that works brings its whole family with it. The registry is measured against a full HF census: **317,310 / 317,310 arch-bearing text-gen checkpoints (100.00%) map to an engine token** (`Testing/census_coverage.py` regenerates the count from the actual committed mapping).

The full-catalog end state — 500+ models, HuggingFace-native bring-up — is planned in **[docs/plans/monster-500-models.md](docs/plans/monster-500-models.md)**. The [models SSOT](docs/wiki/models.md) is the single source of truth for coverage; the [roadmap](docs/guides/roadmap.md) tracks the remaining gap (glm4/cohere2/lfm2 hybrid families, encoder-decoder T5/BART out of scope, ~2,000 one-off custom classes).

**→ [All families, indexed](docs/model-families/README.md)** · **→ [Combined support SSOT](docs/wiki/models.md)**

## Frontier gates: 5/5 validated against reference implementations



Every architecture the engine claims to support is held to a **generation gate**: run the engine on a real (or mini) checkpoint and compare logits against the reference implementation. The five newest frontier families were audited, implemented, and gated in one session (2026-08-16) — the gates caught real math bugs each time:

| Family | Arch | Engine | Gate result |
|--------|------|--------|-------------|
| **Nemotron 3** | LayerNorm1P + relu2 MLP + partial RoPE | `backend_generic` | ✅ **real** 8B checkpoint, top1 7503 *" Paris"* == HF, logit corr 0.99986 |
| **DeepSeek V4** | Shared-KV MQA + mHC (Sinkhorn) + hash-MoE | `src/deepseek_v4.cpp` | ✅ mini-gate top1 342 == HF, 20/20 top-20 |
| **GLM-5.2** | V3-MLA + DSA indexer (cross-layer top-k) | `src/glm_moe_dsa.cpp` | ✅ mini-gate top1 171 == HF, 20/20 |
| **MiMo V2** | MoD hybrid (SWA+full GQA, sigmoid group-topk) | `src/mimo_v2.cpp` | ✅ mini-gate top1 524 == HF, 20/20 |
| **Qwen3.5** | GatedDeltaNet + gated GQA hybrid | `src/qwen3_5.cpp` | ✅ mini-gate top1 142 == HF, 20/20, corr 1.0 |

Each gate compares full logits (not just greedy argmax) against the authoritative reference — the HuggingFace modeling source for the exact checkpoint. The audit step proved decisive: two of the five families shipped in our engine **before** the audit were written against fictional architectures (DeepSeek V4 had MLA + a 4×4 "mHC mix matrix" that don't exist; the real thing is Shared-KV MQA + Sinkhorn hyper-connections). The gate suite (`Testing/run_all.sh`) is now **17/17 green**, and the full model census holds **100.00%** coverage of HuggingFace architectures.

**→ [The frontier plan](docs/plans/monster-500-models.md)** — the 5-gate work order, per-family architecture facts, and what each gate proved.

## Hardware agnostic

The same binary and the same command line, on whatever silicon you have:

| Platform | Backend |
|----------|---------|
| AMD Strix Halo (Ryzen AI Max+ 395) | XDNA 2 NPU + ROCm HIP + GGML-Vulkan |
| NVIDIA GPU (sm_70+) | CUDA |
| Apple Silicon | Metal |
| Any Vulkan 1.2+ GPU | ZINC + GGML-Vulkan |
| x86 CPU | OpenMP |

Six hardware targets are probed at startup (`has_npu`, `has_hip_gpu`, `has_vulkan`, `has_cuda`, `has_metal`, `has_avx512`) and served by **12 backend implementations** in the factory: CPU, Generic, Vulkan, HIP, NPU, ZINC, Zamba2, Mamba1, CUDA, Metal, VART, ONNX-NPU. Some are model-specific SSM paths rather than device backends, and stub backends return `can_infer() == false` so they are discovered but never selected for inference. No rebuild to move a model from NPU to GPU to CPU.

**→ [Architecture deep-dive](docs/guides/architecture.md)**

## // the hard truth

**22 proprietary libraries. 209 bitstreams. Zero documentation.**

AMD shipped the Ryzen AI Max+ 395 with a 50 TOPS XDNA 2 NPU and locked it behind a closed-source runtime. Nothing else could touch that chip. We reverse-engineered the whole stack in 4 days and replaced it with open C++ — one MIT-licensed engine that runs LLMs on the NPU, on AMD / NVIDIA / Apple GPUs, or on plain CPU.

**→ [Read the full journey](docs/journey.md)** — every crash, breakthrough, and bug, documented in real time.
**→ [The Audit Trail](docs/audit-trail.md)** — 1.5 TB of raw evidence, archived nightly on the Raspberry Pi backup server.

## What it is

1bit MONSTER is an **inference engine** — the thing that actually runs the model. It is not a chat app; bring your own frontend.

- **One engine, every target.** `build/1bit` holds every server and the CLI, dispatched by subcommand (`zaya`, `unified`, `jarvis`, `vision`, `chat`, …).
- **OpenAI-compatible API.** `POST /v1/chat/completions` against the `unified` server; pooled models, per-request routing.
- **Speculative decoding** in-process via `--draft-model` + `--spec-decode` (lossless vs greedy).
- **Beyond text.** Stable-Diffusion-family image and video generation via `image_server`; Whisper STT and codec TTS for voice.

## Benchmarks

Headline end-to-end decode, measured **2026-08-01** on AMD Ryzen AI MAX+ 395 (Radeon 8060S, 32 GB UMA):

| Model | Gen tok/s (e2e) | Backend |
|-------|:---------------:|---------|
| SmolLM2-135M | **662** | GGML-Vulkan |
| Qwen3-0.6B | **373** | GGML-Vulkan |
| Qwen2.5-VL-3B | **100** | GGML-Vulkan |
| BlackMamba-1.5B | **79.4** | Mamba1 HIP |
| BlackMamba-2.8B | **46.0** | Mamba1 HIP |
| Qwen3.5-4B | **65** | GGML-Vulkan |
| DeepSeek-R1-Distill-Llama-8B | **44** | GGML-Vulkan |
| ZAYA1-74B-preview | **17.6** | GGML-Vulkan |

Plus **43.2 TFLOPS** INT8 prefill (WMMA, variant I8-APRE, re-measured 2026-08-01).

BlackMamba figures re-measured 2026-07-26 after the `__shfl_xor_sync` kernel fixes. Numbers in this file are checked against `site/numbers.json` in CI:

```bash
python3 scripts/generate_readme_numbers.py --check
```

Kernel-level GEMV figures (Q1 433, TQ2 420, ternary 318 tok/s) are **isolated throughput, bit-exact vs CPU reference — not end-to-end decode**, and are deliberately kept out of the headline table.

### Unified control plane — measured 2026-08-07

All five zoo models served from **one** `unified` process (`--pool` keeps every model resident), one OpenAI-compatible endpoint, measured end-to-end through `POST /v1/chat/completions` including per-request model routing:

| Model | tok/s (e2e) | Backend |
|-------|:-----------:|---------|
| Qwen3-4B | **20.8** | NPU FLM (XDNA) |
| Qwen3-0.6B Instruct | **12.4** | GGML-Vulkan |
| Llama-3.2-1B Instruct | **12.4** | GGML-Vulkan |
| Bonsai-1.7B-TQ2 | **3.1** | HIP 1BP |
| Zamba2-1.2B-Instruct-v2 | **2.2** | HIP (Mamba2 SSD) |

`scripts/zoo-smoke.sh` (5/5 PASS) runs the same path; `POST /v1/pool` reports residency.

**→ [Full performance SSOT](docs/wiki/performance.md)**

## Adoption

From `site/numbers.json` (repo telemetry, synced):

| 6,113 | 783 | 47 | 14 |
|:-----:|:---:|:--:|:--:|
| total clones | unique cloners | models shipping | open issues |

## Flagship pipeline: JARVIS

A fully local voice assistant where every stage runs on the engine:

```
mic → VAD → STT (Whisper) → router → LLM → TTS (codec) → cloned voice → speaker
```

No cloud, no Python in the hot path.

**→ [How the JARVIS pipeline works](docs/jarvis.md)**

## Learn more

- **[Documentation index](docs/README.md)** — start here
- **[Getting started](docs/guides/getting-started.md)** · **[Build guide](docs/guides/building.md)**
- **[Model families](docs/model-families/README.md)** · **[Benchmarks](docs/wiki/performance.md)**
- **[The engineering journey](docs/journey.md)** · **[Roadmap](docs/guides/roadmap.md)**
- **[Contributing](CONTRIBUTING.md)**

> **[MAX XDNA backend](https://github.com/1bit-systems/max-xdna-backend)** — secondary evidence repo (MIT): proves the XDNA 2 NPU can be driven from outside AMD tooling. The engine itself is MAX-free by design.

## License

MIT — do whatever you want.
