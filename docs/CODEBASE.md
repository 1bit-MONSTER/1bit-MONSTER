# 1bit.MONSTER — Codebase Reference

A code-level map of the `1bit-MONSTER` repository: directory layout, build
targets, core abstractions, module-by-module breakdown, and how the pieces
fit together. Written from the source tree itself — the companion to
[`docs/guides/architecture.md`](guides/architecture.md) (pipeline-level view)
and [`docs/README.md`](README.md) (documentation index).

> **TL;DR** — One pure-C++23 binary (`build/1bit`) that loads any HuggingFace
> model (GGUF / ONNX / 1BP), auto-detects the architecture, and routes to a
> hardware backend (AMD NPU, AMD/NVIDIA GPU via HIP/CUDA, Apple Metal, Vulkan,
> or CPU). ~155k lines of first-party C++/HIP, 131 files in `src/`, a 109 KB
> CMakeLists, 26 `backend_*` sources (~20 inference paths), and a GitNexus
> index of **24,011 symbols / 44,265 relationships / 216 execution flows**.

---

## 1. Repository facts

| Metric | Value |
|---|---|
| Version | `2026.08.04` (date-based, from `VERSION`) |
| License | MIT (`LICENSE`) + `LICENSE-SHERRY.md` (SHERRY ternary format license) |
| Language | C++23 + HIP, zero Python at runtime |
| First-party code | ~155.5k lines (`.cpp`/`.h`/`.hpp`/`.hip` excl. `third_party/`) |
| Total code (all langs) | ~204k lines |
| Source files | 131 in `src/` + 45 headers in `include/` |
| Symbol index | 24,011 symbols, 44,265 edges, 759 communities, 216 execution flows (`.gitnexus/`) |
| Architecture coverage | 552 engine arch tokens ← 1,774 HF `architectures` strings |
| Primary platform | AMD Strix Halo (Ryzen AI MAX+, XDNA 2 NPU + Radeon GPU) |
| Domain | `1bit.monster` — static site served from `site/` (Cloudflare Pages) |

**Index caveat:** the GitNexus symbol numbers above were snapshotted 2026-08-21
on a feature branch — refresh with `node .gitnexus/run.cjs analyze` (see
`AGENTS.md`) before treating them as current. Also note `VERSION` (2026.08.04)
lags `CHANGELOG.md` head (2026.08.10): versioning is date-based per release
tag, not a monotonic “current” marker.

### Top-level layout

```
1bit-MONSTER/
├── CMakeLists.txt            # 109 KB — every target, platform branches
├── src/                      # core engine: backends, loaders, engines, servers
├── include/                  # public headers (common.h, backend interfaces)
├── engine/                   # hardware engines: npu/ gpu/ fusion/ reference/
├── kernels/                  # standalone .hip kernels (GEMV/GEMM, quant)
├── tools/                    # one-off CLIs, converters, benchmarks, demos
├── tests/  Testing/          # unit/e2e tests + model-census harness
├── docs/                     # guides, wiki, research, model families
├── benchmarks/               # dated results + site/benchmarks.json recorder
├── site/                     # static marketing/docs site (Cloudflare Pages)
├── mobile/                   # Flutter companion app (Android/iOS)
├── npu-infer/                # standalone NPU inference (C++ + Rust)
├── spec-decode/              # speculative-decoding research (Eagle3, Medusa)
├── ck-prefill/               # Composable Kernel prefill GEMMs (ROCm)
├── integrations/             # comfyui, dsh, vllm-toolbox adapters
├── packaging/                # deb/appimage/docker/aur/snap/ollama/service
├── .github/workflows/        # 18 CI/CD workflows
├── third_party/              # git submodules (stable-diffusion.cpp, audio.cpp,
│                             #   FastFlowLM, llama.cpp fork)
├── experimental/             # bit1_mlx (MLX experiment) + 1BIT_README.md
├── hackathon/                # demo scripts, spec docs, submission checklists
├── fastflowlm_analysis/      # RE raw captures: FLM ISA, Q4NX format, secrets
├── themes/                   # 1bit.json — pi/agent color theme
├── bin/                      # checked-in binaries: 1bit, zaya-up
├── pixi.toml / pixi.lock     # pixi package-manager environment
└── scripts/                  # ops + model-prep shell/Python helpers
```

---

## 2. The single binary (`build/1bit`)

Everything ships as **one ELF** (busybox-style), assembled by the `onebin`
target (`CMakeLists.txt` ~line 1358) with `OUTPUT_NAME "1bit"`.
`tools/onebin.cpp` dispatches on `argv[0]` (symlink names) **or** subcommand:

| Subcommand | Aliases | Entry point | What it runs |
|---|---|---|---|
| `zaya` | `zaya-server`, `server` | `zaya_server_main` | Multi-backend inference server (HIP/NPU/CPU) |
| `unified` | `unified-server` | `unified_server_main` | Multi-backend + embedded Lemonade core |
| `router` | `unified-router` | `unified_router_main` | NPU/GPU content-aware routing proxy |
| `lemonade` | `lemond` | `unified_server_main --lemonade` | Lemonade-compatible server |
| `jarvis` | `voice`, `jarvis_server` | `jarvis_app_main` | Voice pipeline server (mic → STT → LLM → TTS) |
| `vision` | `vl`, `vision_server` | `vision_server_main` | Vision-language server |
| `zuna` | — | `zuna_main` | ZUNA EEG-autoencoder port |
| `onebitd` | `daemon` | `onebitd_main` | HTTP-proxying inference daemon |

Symlink names (`zaya_server`, `unified_server`, `unified_router`, `onebitd`,
`jarvis_server`, `vision_server`) also dispatch directly so existing
exec/pkill sites in tools keep working unchanged.

Quick start:

```bash
cmake -B build && cmake --build build
./build/1bit zaya -m model.1bp -p "Hello world"
```

---

## 3. Core abstractions

### 3.1 `BackendType` / `ModelConfig` (`include/common.h`)

`BackendType` enumerates 15 hardware paths:

`HIP_GPU`, `VULKAN`, `NPU_XRT`, `CPU_AVX512`, `CPU_SCALAR`, `GENERIC`,
`ZAMBA2`, `ZAMBA2_GPU`, `ZINC_GPU`, `Q4NX_FUSION`, `CUDA_GPU`, `METAL_GPU`,
`VART` (Versal/Zynq DPU), `ONNX_NPU` (VitisAI EP).

`ModelConfig` is the canonical model description: deprecated short-name fields
(`hidden`, `n_heads`, …) are aliases for long-name fields (`hidden_size`,
`num_heads`, …) that must be kept in sync via `set_dim()`; plus MoE knobs
(`num_experts_top`, `n_shared_experts`, `first_k_dense`, `expert_groups`,
`routed_scaling`, `correction_bias`), attention variants (`attention_multiplier`
for granite, `norm_is_layernorm` for OLMo, `nemotron_layernorm1p`), and
`ModelFormat` (`GGUF | H1B | Q4NX | SAFETENSORS | RAW_BIN | ONEBP`).

### 3.2 The `Backend` interface (`src/backend.h`)

Every backend implements: `init`, `reset`, `forward` (token → hidden),
`forward_embed` (multimodal splice point), `lm_head`, `generate`, `benchmark`,
`destroy` — plus optional extensions:

- **Pilot cross-layer prefetch** — `set_pilot` / `preload_layer` overlap weight
  I/O with compute (see `include/pilot.h`).
- **Text-level generation** — `generate_text` / `continue_text` for backends
  that tokenize internally (FLM NPU subprocess), with multi-turn KV reuse.
- **Speculative-decode primitives** — `decode_one`, `verify_batch`, KV-truncate.
- **Logit access** — `last_logits()` for sampling/validation.

### 3.3 `BackendManager` (`src/backend_manager.cpp`, 1,708 lines)

The registry and init orchestrator. Discovers hardware, builds candidate
backends, and runs a **try-until-one-succeeds** init loop over
`preferred_ids` (per-model route order, set by the router). Also owns
`backend_monitor` (health), `backend_detect` (hardware probing), and the
`backend_plugin` mechanism. `include/backend_manager.h` exposes the canonical
interface used by servers and the C ABI.

### 3.4 Model routing & discovery

- **`src/model_discovery.cpp`** — parses model files/configs to fill a
  `ModelConfig`; maps HF `architectures` strings via `rcpp_arch_from_string`
  (552 engine tokens, from `include/rocm_cpp/bitnet_model.h`).
- **`src/model_router.cpp`** — static policy table deciding backend *order*
  per architecture: MoE → `hip_gpu`+`cpu_scalar`; qwen3 → `npu_xrt`+`cpu_generic`;
  zamba2 → `zamba2_gpu`+`cpu_generic`; mamba/zamba → `hip_gpu` (per-layer MoE
  routing for BlackMamba); GGUF/H1B → `zinc_gpu`+`cpu_generic`; fallthrough →
  `hip_gpu`+`cpu_generic`. The proprietary FLM-subprocess path was removed —
  this project ships **zero proprietary code**.
- **`src/dynamic_router.cpp` / `src/strategy_engine.cpp`** — runtime routing
  policy. `strategy_engine` implements per-token routing strategies (e.g.
  `CascadeConfig`: after N tokens always route to the large backend) as pure,
  lock-free `route()` functions — the agent watchdog only mutates shared state
  via atomics. `dynamic_router` is the prefill/decode dispatch counterpart.

### 3.5 The C ABI (`include/onebit_c.h` → `libonebit.so`)

The **only** surface Mojo/MAX and external tools touch: opaque `OneBitHandle`,
exception-safe, single-threaded per handle. Functions: `onebit_create/destroy`,
`onebit_init(weights_dir, model_name)`, backend enumeration/selection,
`onebit_forward/generate/lm_head`, logits access, and `onebit_version`.
Built by the `onebit_engine` SHARED target (`src/onebit_c.cpp`).

---

## 4. Model loaders & formats

| Loader | Source | Formats |
|---|---|---|
| `gguf_loader.cpp` / `gguf_reader.cpp` | `src/` | GGUF (all common quants; shared parser, no HIP dep) |
| `gguf_zamba2_loader.cpp` | `src/` | Zamba2 GGUF specialization |
| `onebp_model.cpp` + `include/onebp_format.h` | `src/` + `include/` | Native **1BP** (binary 1-bit pack) |
| `q4nx_reader.cpp` | `src/` | Q4NX (4-bit, per-group BF16 scales) |
| `h1b_loader.cpp` (924 lines) | `src/` | H1B (SHERRY ternary hybrid) |
| `safetensors_reader.cpp` | `src/` | HF safetensors (for 1BP conversion) |
| `onnx_loader.cpp` (752 lines) | `src/` | ONNX graphs (INT8 NPU path) |
| `engine/npu/src/onebp_loader.cpp` (535 lines) | `engine/npu/` | 1BP on the NPU engine |

**1BP format family** (see `docs/research/block-scaled-ternary-format.md`):
- **TQ2** — block-scaled ternary (SHERRY) weights; `tq2_to_q4nx.cpp` converts.
- **Q4NX** — 4-bit with per-group BF16 scales, the NPU INT8 pipeline input.
- Conversion toolchain: `gguf_to_onebp.cpp` (+ `.mojo`/`.py`), `hf_to_onebp.py`,
  `convert_zaya_safetensors_to_gguf.py`, `qwen3_to_onebp.py`, `tq2_to_q4nx.cpp`,
  `onebp_to_trg.cpp`.

---

## 5. Backend catalog (`src/backend_*.cpp`)

| File | Backend | Notes |
|---|---|---|
| `backend_hip.cpp` | HIP GPU (ROCm) | Generic HIP kernels + MoE (CCA) |
| `backend_hip_1bp.cpp` | HIP GPU 1BP | 1BP weights on GPU (`hip_1bp_kernels.hip`) |
| `backend_cuda.cpp` / `cuda_engine.cu` | CUDA | NVIDIA — needs testers (#1703) |
| `backend_metal.mm` | Metal | Apple Silicon |
| `backend_vulkan.cpp` / `backend_vulkan_hpp.cpp` | Vulkan | Portable GPU path |
| `backend_ggml_vulkan.cpp` | GGML-Vulkan | llama.cpp ggml integration (MIT) |
| `backend_zinc.cpp` | ZINC GPU | Vulkan compute IR runtime (`engine/gpu/zinc_cpp`) |
| `backend_npu.cpp` (706 L) | NPU XRT | Native XDNA 2 via XRT/xclbins |
| `backend_npu_flm.cpp` (539 L) | NPU FLM bridge | Router route **removed** (zero proprietary code) — file kept for whisper STT (`:8496`) + multi-turn KV-reuse paths |
| `backend_fused.cpp` / `backend_fused_npu.cpp` | Fused | `engine/fusion` Q4NX CPU + NPU fused paths |
| `backend_mamba1.cpp` (600 L) | Mamba1 | SSM + per-layer MoE (BlackMamba) |
| `backend_zamba2.cpp` / `backend_zamba2_vulkan.cpp` | Zamba2 | Mamba2 hybrid SSD |
| `backend_laguna.cpp` (832 L) | Laguna | Qwen-ish family (`laguna_*.hip`) |
| `backend_cpu.cpp` (915 L) | CPU | AVX-512 + scalar reference |
| `backend_generic.cpp` (2,660 L) | Generic CPU | The universal GGUF CPU fallback — largest backend |
| `backend_onnx.cpp` / `backend_vart.cpp` | ONNX / VART | ONNX Runtime + VitisAI EP, Versal DPU |
| `backend_frontier.cpp` | Frontier | Adapter wrapping the standalone engines (`deepseek_v4`, `glm_moe_dsa`, `mimo_v2`, `qwen3_5`) behind the canonical `Backend` interface — validated 17/17 vs HF references (2026-08-16) |
| `backend_manager.cpp` (1,708 L) | — | Registry + init (see §3.3) |
| `backend_monitor.cpp`, `backend_plugin.cpp`, `backend_factory.cpp`, `backend_detect_win.cpp` | — | Health, plugin ABI, factory, Windows detection |

> 26 `backend_*` sources total (25 `.cpp` + `backend_metal.mm`); **~20 are
> inference-facing** — `manager`, `monitor`, `plugin`, `factory`, `detect_win`
> are infrastructure, not hardware backends.

---

## 6. Model engines (`src/*_engine.cpp`)

Architecture-specific inference implementations layered over the backends:

- **`zaya_engine.cpp` (2,052 L)** — the flagship: Zaya1/Zaya2 (Zamba2-based
  hybrid + MoE), with `zaya_codec.cpp` and `zaya_moe_launcher.hip`.
- **MoE engines** — `afmoe_engine.cpp`, `cohere2moe_engine.cpp`,
  `ernie45moe_engine.cpp`, `exaonemoe_engine.cpp`, `granitemoehybrid_engine.cpp`,
  `jetmoe_engine.cpp`, `lfm2moe_engine.cpp`, `phimoe_engine.cpp`,
  `glm_moe_dsa.cpp` (GLM-4-MoE/DeepSeek-style gating), `minimax_engine.cpp`,
  `minimaxm2_engine.cpp`, `mellum_engine.cpp`, `nemotron_h_engine.cpp`,
  `hyv3_engine.cpp`, `qwen3next_engine.cpp` (567 L).
- **SSM engines** — `mamba1_engine.hip` (with `mamba2_kernels.cpp/.hip/.h`),
  `zamba2_engine.cpp` + `zamba2_engine_hip.hip` + `zamba2_hybrid_gpu.hip`,
  `falconmamba_engine.cpp`, `rwkv_engine.cpp`.
- **Other** — `deepseek.cpp` / `deepseek_v4.cpp` (MLA, DSA), `falconh1_engine.cpp`,
  `strategy_engine.cpp`, `kv_rotorquant.cpp` (KV-cache quantization),
  `lora.cpp` (adapter merge), `afmoe_engine.cpp`.

Support code: `tokenizer.cpp` (840 L) + `simple_tokenizer.cpp`, `kv_cache_attn*.hip`
(attention kernels), `prefill_dispatcher.cpp`, `model_discovery.cpp`,
`unified_pool.cpp` (buffer pooling).

---

## 7. Hardware engines (`engine/`)

### 7.1 NPU — XDNA 2 native engine (`engine/npu/`)

The crown jewel: a **native AMD XDNA 2 NPU engine** built by
reverse-engineering the proprietary stack (see `docs/journey.md`).

- `src/npu_engine_universal.cpp` (3,464 L) — the universal NPU engine
  (INT8 GEMM pipeline, BO management, per-layer weight upload).
- `src/npu_engine_fused.hip` (1,189 L) / `npu_engine_overlap.hip` (1,126 L) —
  fused multi-GEMM and overlap variants.
- `src/zaya_decode.cpp` (566 L) — fused decode launch for Zaya-class models.
- `src/npu_dims.h`, `src/npu_engine_i8ctx_inc.h` (859 L) — dims + INT8 context.
- `generators/` — xclbin generation (MLIR/AIE2 tooling).
- `kernel/`, `fused_insts/`, `pool/`, `xclbins/`, `tokenizer/`, `tools/` —
  kernels, fused instruction sets, buffer pools, prebuilt xclbins, tokenizer
  and instruction tooling.
- Design docs in-repo: `zaya74b_npu_design.md`, `zamba2_npu_design.md`,
  `BENCHMARKS.md`, `ZYPHRIA_XCLBINS.md`, `AIE2P-FACTS.md`.

Key verified facts (from `docs/guides/architecture.md`): 4 GEMM xclbin contexts
(QKV/O/GU/D, INT8, group_id_B=4) alive at once on NPU2; K-interleaving bug
fixed via `dataReuse` broadcast annotations; BFP16 precision collapse fixed by
switching to INT8 quantization; 8+ concurrent `hw_context`s OK on firmware
1.1.2.65.

### 7.2 GPU — ZINC runtime (`engine/gpu/`)

- `zinc_cpp/` — the C++ ZINC backend: Vulkan compute inference over an IR
  graph, multi-quant/multi-arch, no per-model specialization.
- `src/zinc_rt/` — runtime (dispatch, shader management).
- `src/shaders/` — GLSL/Vulkan compute shaders.
- `src/server/` — ZINC serving layer.
- Plus the ggml-vulkan integration (`src/backend_ggml_vulkan.cpp`) which
  currently drives headline decode numbers.

### 7.3 Fusion — Q4NX CPU fused engine (`engine/fusion/`)

`cpu_layer.cpp/.h`, `cpu_q4nx_loader.h`, fused shaders, `zero_copy/`
(NPU↔GPU shared-buffer substrate), `dspark.h` (speculative draft), and a
HIP GPU-verify path (`gpu_verify.hip`). Targets the `Q4NX_FUSION` backend.

### 7.4 Reference (`engine/reference/max-ports/`)

Reference HIP implementations **disassembled from AMDGPU ELFs of MAX 26.5.0
production kernels** (gfx1151 / Strix Halo) and verified on this box —
explicitly **reference only, not wired into the build** (see
`max_recipes_port_guide.md`): `gemm_wmma.cu`, `gemm_wmma_lds.cu`,
`gemm_fp8_wmma.cu`, `gemv_dpp.cu`, `rmsnorm_dpp.cu`.

---

## 8. Kernels (`kernels/` + `src/*.hip`)

**81** standalone HIP kernel files (47 in `kernels/`, 34 in `src/`):

- **Ternary / 1-bit**: `ternary_gemv*.hip` (phase5, halo, twla, wmma, sherry,
  q1_0, block-scaled), `bonsai_*.hip` (Q1 + TQ2 1024-wide GEMV), `chain_gemv_tq2_1024.hip`,
  `fused_gemv_tq2_1024.hip`, `prefill_chain_tq2_1024.hip`, `iq_gemv.hip` (IQ1S).
- **Quant**: `oscar_quant.hip` (INT2), `q4k_gemv.hip`, `rotorquant_pack.hip`,
  `tq2_bwopt.hip`, `twla_int4_gemv.hip`, `twla_tq2_int4.hip`, `fp8_wmma_gemv.hip`.
- **Attention**: `kv_cache_attn*.hip` (FD/rotor/i8 variants), `medusa_tree_attn.hip`,
  `vitallm_sparse_attn.hip`, `v_interleave_kernel.hip`.
- **Fused ops**: `fused_qkv_gemv.hip`, `fused_norm_gemv.hip`, `lm_head_fused.hip`,
  `argmax_kernel.hip`, `hadamard_rotate_butterfly.hip`, `deepseek_mla.hip`.

---

## 9. Servers, daemons & networking

### 9.1 HTTP servers (`src/server/`, `tools/`)

- `server.cpp` / `rest_handler.cpp` — boost::beast + asio HTTP server
  (FastFlowLM-derived), OpenAI-compatible endpoints: `GET /v1/version`,
  `GET /v1/models`, `POST /v1/chat/completions`, `POST /v1/completions`,
  `POST /v1/embeddings`, `POST /v1/audio/transcriptions`, plus Ollama-style
  `/api/ps`, `/api/tags`, `/api/show`. NPU access serialized via
  `g_npu_in_use` / `g_npu_active_requests` (see `server.hpp`).
- `tools/unified_server.cpp` — the unified multi-backend server
  (subcommand `unified`), with embedded Lemonade core (`--lemonade`).
- `tools/zaya_server.cpp` — legacy zaya server entry (`1bit zaya`).
- `tools/vision_server.cpp`, `tools/image_server.cpp` — vision-language serving.
- `tools/unified_router.cpp` — content-aware NPU/GPU routing proxy.
- `tools/onebitd.cpp` — inference daemon (spawns backend, proxies HTTP).

### 9.2 JARVIS voice pipeline (`tools/jarvis/`, `src/audio_bridge.cpp`)

Fully local voice assistant: mic → VAD → STT (NPU-FLM whisper HTTP endpoint
`:8496`) → LLM → TTS → speaker. Pure C++, no cloud. See `docs/jarvis.md`.

### 9.3 The Mesh (`src/mesh/`)

Self-discovering LAN federation: `peer_discovery` (mDNS/UDP),
`peer_api` (handlers for `GET /v1/mesh/me`, `GET /v1/mesh/peers`,
`POST /v1/mesh/handshake`, `POST /v1/mesh/ask`, `POST /v1/mesh/answer`),
`dispatch` (shared routing), `mesh_agent` (cross-node Q&A), `node_identity`.
See `docs/mesh-protocol.md`.

---

## 10. Mobile, NPU-infer & speculative decoding

### 10.1 Mobile (`mobile/`) — Flutter companion app

Android + iOS app (`lib/`: `audio`, `screens`, `state`, `ws`), talks to the
engine over WebSocket. `docs/mobile/` holds the design docs.

### 10.2 NPU-infer (`npu-infer/`) — standalone NPU stack

Independent C++ (`src/`: `engine.cpp`, `ffi_bridge.cpp`, `flm_bridge.cpp`,
`main.cpp`, `model.c`) + **Rust** bindings (`rust/` with `build.rs`) + `1bit/`
legacy (deprecated) + `bf16_kernel_dev/` + `docs/` + `tests/` + `tools/`.
Runs 1BP models directly on the NPU with zero Python.

### 10.3 Speculative decoding (`spec-decode/`)

Research + implementation of Eagle3 / Medusa draft-model decoding:
`draft/` (draft models), `engine/`, `eval/`, `train_draft.py`,
`train_eagle3.py`, `bench/`, `configs/`. DSpark full-pipeline integration
(Eagle3 draft on NPU/CPU + GPU verify) is built in `CMakeLists.txt` and uses
`engine/fusion/dspark.h`.

### 10.4 CK prefill (`ck-prefill/`)

Composable Kernel (ROCm) prefill GEMMs: WMMA fp16, packed-int4,
ternary-as-packed-int4 variants (`gemm_wmma_*.cpp`, `common.hpp`) — the
**38.84 TFLOPS INT8 prefill** path.

---

## 11. Tools & scripts

### 11.1 Format converters (`tools/`)

`gguf_to_onebp.{cpp,py}`, `gguf_to_tilefuse.cpp`, `gguf_to_h1b.cpp`,
`gguf_to_onnx{.py,_int8.py}`, `convert_zaya_safetensors_to_gguf.py`,
`convert_zamba2_safetensors_to_gguf.py`, `convert_qwen36_moe_q4nx.py`,
`hf_to_onebp.{py,mojo}`, `qwen3_to_onebp.{py,mojo}`, `safetensors_to_onnx_int8.*`,
`tokenizer_json_to_htok.*`, `tq2_to_q4nx.cpp`, `onebp_to_trg.cpp`,
`export_codec_gguf.py`, `convert_float32_bins_to_q4nx.py`.

### 11.2 Dev tools & demos

`backend_demo.cpp`, `bitnet_decode.cpp`, `bitnet_tui.cpp`, `bitnet_debate.cpp`,
`onebin.cpp`, `onebit.cpp`, `spec_decode.cpp`, `scan_1bp.cpp`, `verify_1bp.cpp`,
`gguf_htok.cpp`, `make_gate.cpp`, `mesh_peer.cpp`, `lora_merge.cpp`,
`zaya_logit_check.cpp`, `vl_logit_check.cpp`, `whisper_demo.cpp`,
`zaya1_vl_demo.cpp`, `dspark_full.cpp`, `dspark_gpu_bench.cpp`,
`vision_qwen2vl_poc.cpp`, `vl_pipeline_test.cpp`, `hadamard_export.cpp`,
`oscar_calib.cpp`, `gen_npu_insts.cpp`, `qwen36_gdn_probe.cpp`,
`qwen36_moe_probe.cpp`, `unified_router.cpp`, `unified_server.cpp`.

### 11.3 Ops & prep (`scripts/`)

`build_*.sh` (1BP/GPU/NPU), `demo.sh`, `download_*`, `setup-therock.sh`,
`setup_npu_xclbins.sh`, `npu-driver-reapply.sh`, `model-cache.sh`,
`jarvis-gateway.service`, `zaya-npu.service`, `ws_session_*` (WebSocket
smoke tests), `push_to_hub.sh`, `upload_models.py`, `merch/` (logo/brand
tooling: `restyle_*.py`, `rollout_logo.py`), `generate_readme_numbers.py`,
`sync-version.sh`, `wait-pr-merge.sh`.

---

## 12. Tests & validation

- **`tests/`** — CMake-registered tests: `test_backend`, `test_gguf_reader_dequant`,
  `test_npu_flm_delta`, `test_moe_fused_math`, `test_tokenizer_logprobs`,
  `test_generic_attention`, `zaya_batch_check`, `issue1527_loader_check`,
  `test_lora_merge`, `test_medusa_skeleton`, `test_ck_gemm`, `test_standalone`,
  `test_vulkan_gemv`, `test_cca_attn`, `zaya_gpu_decode` (bench), plus
  `download_and_run.sh` smoke runner that records into `site/benchmarks.json`.
- **`Testing/`** — the model-census harness: `census_*.py`/`.json` sweep all
  HF architectures to measure coverage; self-check probes
  (`arch_mapping_selfcheck.cpp`, `arch_mapper_probe.cpp`, `discovery_selfcheck.cpp`,
  `dedup_loader_check.cpp`); differential oracle comparisons
  (`cmp_deepseek_v4.cpp`, `cmp_glm_moe_dsa.cpp`, `cmp_qwen3_5.cpp`,
  `cmp_mimo_v2.cpp`, `cmp_instella*.cpp`); `e2e_afmoe.cpp`; AIE2P self-checks.
- **Benchmarks** — `benchmarks/` (dated `RESULTS-*.md`, `record.sh`,
  `data/`, `latest.json`) feeding the **single source of truth**
  `site/benchmarks.json` (read by README, `docs/wiki/performance.md`, site).
- **CI** — `.github/workflows/ci.yml` (build+test), `bench.yml`,
  `validate-benchmarks.yml`, `validate-claims.yml` (audit-trail checker),
  `end-to-end-smoke.yml`, `codeql.yml`, `scope-guard.yml`, `release.yml`,
  `deploy.yml`, `pr-agent.yml`, `npu-reset.yml`, `video-lora-ci.yml`, etc.

---

## 13. Build system

Single giant `CMakeLists.txt` (`project(1bit-monster LANGUAGES C CXX HIP)`,
min CMake 3.21) + `CMakeLists_windows.txt` / `cmake/windows_build.cmake` for
Windows. Notable targets (section markers in the file):

| Target | Output | Purpose |
|---|---|---|
| `onebin` | `build/1bit` | **The** single binary (all servers) |
| `librocm_cpp` | shared | C API for Mojo/MAX (`onebit_c`) |
| `libgguf_reader` | static | shared GGUF parser (no HIP dep) |
| `gguf_to_onebp` / `gguf_to_tilefuse` / `tf_gemv` | exe | format converters |
| `libonebp_model`, `libvl_image`, `libvision_encoder`, `libwhisper`, `libpixtral` | static | model/vision/audio libs |
| `zinc_cpp` | static | ZINC Vulkan backend |
| `libbackend_manager` | static | Backend abstraction layer |
| `ppl_generic`, `gguf_htok`, `make_gate` | exe | perplexity gates |
| `onebit_engine` | `libonebit.so` | C ABI seam (Mojo) |
| `mesh` | exe | mesh substrate + `/v1/mesh/*` |
| `npu_engine_hybrid`, `qwen36_moe_probe`, `gen_npu_insts` | exe | NPU tooling |
| `bench_*`, `test_*`, `zaya_gpu_decode`, `vl_pipeline_test` | exe | benches/tests |

Platform notes: HIP detection via `project(LANGUAGES HIP)`; TheRock SDK
discovery under Python site-packages; ggml-vulkan imported from the vendored
`third_party/llama.cpp` fork; submodules: `stable-diffusion.cpp`,
`audio.cpp`, `FastFlowLM` (ROCm, vendored analysis in
`docs/vendored-fastflowlm.md`), `llama.cpp` fork
(`docs/llama.cpp-fork.md`).

---

## 14. Docs, site & packaging

- **Docs** (`docs/`): index `README.md`, guides (`architecture`, `building`,
  `getting-started`, `launch`, `windows`, `roadmap`), wiki (models SSOT,
  performance SSOT, NPU architecture, boot config, network topology,
  decisions), research deep-dives, `model-families/` (per-family pages,
  Zyphra featured), `journey.md` (the 4-day NPU reversal story), `jarvis.md`,
  `mesh-protocol.md`, `audit-trail.md` (1.5 TB of evidence),
  `plans/` (mojo-fold, monster-500, one-heap-pivot),
  `agentic-control-protocol.md` + `agent-role-prompts.md` (agent
  orchestration), `archive/` (superseded/blocked plans, kept for provenance).
  Top-level governance/tracking docs live outside `docs/`:
  `ROADMAP.md`, `AUDIT_ISSUES.md` (open-issues ledger), `TODO_TRACKING.md`,
  `AGENTS-WORKTREES.md` (worktree conventions), `CHANGELOG.md`.
- **Site** (`site/`): static Cloudflare Pages site (`wrangler.toml`,
  `_headers`, `_redirects`, `functions/`), generated HTML pages
  (`1bit-*.html`), `benchmarks.json`, `numbers.json`, badges.
- **Packaging** (`packaging/`): deb, appimage, docker, aur, binary, snap,
  ollama, systemd services (`zaya-npu.service`, `jarvis-gateway.service`,
  `npu-companion.service`), `install.sh`.
- **Mobile build** — `mobile/` Flutter (pubspec, Android/iOS).

---

## 15. Key execution flows (GitNexus)

The `.gitnexus/` index (24k symbols, snapshot 2026-08-21) recognizes 216
execution flows — refresh with `node .gitnexus/run.cjs analyze` if stale. The
highest-traffic ones to start from when reading code:

1. **Boot → backend selection** — `main` (onebin) → `*_server_main` →
   `BackendManager::init` → `select_backend_route` → backend `init` →
   loader → `ModelConfig`.
2. **Decode loop** — `forward(token)` → embed → per-layer norm/attn/MLP →
   `lm_head` → sample (with optional speculative verify via `verify_batch`).
3. **Prefill** — `prefill_dispatcher` → CK (`ck-prefill/`) or NPU prefill
   xclbins → KV cache fill.
4. **NPU launch** — `npu_engine_universal` → per-layer BO upload → GEMM
   launches (QKV/O/GU/D) → readback → host sync.
5. **Serving** — REST `/v1/chat/completions` → handler → backend
   `generate_text`/token loop → SSE streaming → `/v1/mesh/*` federation.
6. **Conversion** — GGUF/safetensors → 1BP (TQ2/Q4NX) → NPU/GPU load.

---

## 16. How to extend the engine

### Add a new model family

1. Map the HF `architectures` string → an engine token in `rcpp_arch_from_string`
   (`include/rocm_cpp/bitnet_model.h`), or add the mapping if absent.
2. Fill the `ModelConfig` fields in the loader path (`src/model_discovery.cpp`
   + the format loader in use — GGUF/1BP/ONNX/H1B).
3. Implement the math as a standalone engine (`src/*_engine.cpp`) or extend
   `backend_generic.cpp`; validate against the HF BF16 reference using the
   `Testing/cmp_*.cpp` oracle pattern (all 17/17 families were verified this way).
4. Expose it behind the `Backend` interface — directly, or via a
   `backend_frontier`-style adapter if it's a standalone engine the router
   can't reach otherwise.
5. Add a route order in `src/model_router.cpp`.
6. Add a family page in `docs/model-families/` and a row in `docs/wiki/models.md`.

### Add a new hardware backend

1. Create `src/backend_<name>.cpp` implementing the `Backend` interface
   (`src/backend.h`).
2. Register it in `src/backend_factory.cpp` and `BackendManager`
   (`include/backend_manager.h`).
3. Add a `BackendType` value + `backend_name()` string in `include/common.h`.
4. Add a route in `src/model_router.cpp` and a CMake target/conditional in
   `CMakeLists.txt`.
5. Add a smoke test under `tests/` (mirror `test_backend.cpp`).

### Add a new tool / CLI

- Compile it into `onebin` (add a subcommand in `tools/onebin.cpp`) so it ships
  inside `build/1bit` — the convention for everything reachable from the CLI —
  or keep it standalone with its own `add_executable` (e.g. converters/benches).

---

## 17. Quick navigation cheat-sheet

| I want to… | Start at |
|---|---|
| Understand the inference pipeline | `docs/guides/architecture.md` |
| See all hardware backends | `src/backend_*.cpp` + `include/common.h` (`BackendType`) |
| Add a new model family | [§16 How to extend](#16-how-to-extend-the-engine) + `src/model_router.cpp`, `src/model_discovery.cpp`, `include/rocm_cpp/bitnet_model.h` |
| Understand NPU internals | `engine/npu/` + `docs/wiki/npu-architecture.md` |
| See benchmark numbers | `docs/wiki/performance.md` + `site/benchmarks.json` |
| Know which models run | `docs/wiki/models.md` + `docs/model-families/` |
| Build from scratch | `docs/guides/building.md` |
| Run a server | `docs/guides/launch.md` + `tools/unified_server.cpp` |
| Contribute | `CONTRIBUTING.md`, `AGENTS.md`, `.github/pull_request_template.md` |
