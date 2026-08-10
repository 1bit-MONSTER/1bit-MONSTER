# Changelog

All notable changes to 1bit.systems. Versioning is **date-based** (`YYYY.MM.DD`),
matching the GitHub release tags (`vYYYY.MM.DD`).

## 2026.08.10 — 35B MoE native path: fused v28 xclbins + FLM multi-turn KV reuse 🧠

- **35B MoE NPU workstream lands native + fused.** All four MOE xclbins rebuilt
  with the v27 multi-row flow (`build_moe_v27.sh`); new **v28 fused**
  `MOE_GUSGU` (routed+shared GU, N=8192+1024) and `MOE_DSD` (K=4096+512, N=2H)
  xclbins cut **4 → 2 launches/layer** — pure concat along N/K in the same
  single-GEMM kernel, opt-in via `NPU_MOE_FUSED`. Fused concat contexts
  (`mgu_f`/`mde_f`), pack routing, block-diagonal DSD zero-init;
  `test_moe_fused_math` verifies v28 against the 4-launch reference (host sim).
- **FLM multi-turn KV reuse.** `npu_flm_delta.h` pure decision fn + backend
  `continue_text()`/delta send — clients resending growing history get no
  `<<RESET>>`/re-prefill; shifted char-token decode detector
  (ASCII+100 / raw+300 / EOS=106) stops FLM output being decoded as garbage
  vocab ids.
- **`zaya-npu.service`** — systemd unit serving 35B MoE via FLM on :8088,
  `LimitMEMLOCK=infinity` (FLM NPU runtime mlocks ~1.5 GB of xclbin buffers;
  the default 8 MB made every mmap EAGAIN).
- **Q4NX pivot stack.** `gguf_to_onebp` emits valid Zaya MoE Q4NX (#1522);
  batch GGUF→Q4NX converter for the model zoo; FLM q4nx-pivot fixes (llama
  tags, config path, probe band); measured format gate + INT8-default routing
  + format policy.
- **JARVIS mobile (Flutter).** Voice gateway: full-duplex WS
  `/v1/voice/session` with token auth, RFC-compliant handshake + parser, VAD
  utterance state machine, `WS_STREAM_BIND` + systemd unit + Strix Halo
  runbook. App: connect/voice screens, state lights, transcript UI,
  offline/reconnect resilience, gateway WS client + audio IO with tests,
  live-gateway simulator E2E.
- **INT8 QDQ converters + one-command zoo conversion.** safetensors→INT8 ONNX
  (attention bias, shards, external data, head_dim from config; Qwen2.5-1.5B /
  7B work), >2 GB ONNX via per-row INT8 embed/lm_head, stream-ordered KV
  writes; `convert_model.py` turns safetensors/gguf → INT8 ONNX + htok +
  config in one command.
- **CI: off the self-hosted runner.** strix-halo runner dependency removed
  (#1555), gh-ops health workflow via gh CLI (#1556), smoke test kills 8099
  port squatters by port; eeg-medical custom domain repaired on Cloudflare
  Pages.
- **Ops: watchdog-armed reboot + amdxdna hardening.** `scripts/reboot.sh`
  arms sp5100_tco (hard reset if the reboot hangs); NPU firmware RE (ws04:
  driver + npu.sbin full map); pool sync hardened against the 7.1.5 amdxdna
  clflush oops; flat `/dev/accelN` layout probed.

## 2026.08.07 — unified control plane: pooled multi-model server + spec decode + Zamba2 Q8_0 🧠

- **Unified model pool wired end to end** (`src/unified_pool.{h,cpp}`, `--pool`):
  every model in the weights dir is mmap'd resident at boot (`.1bp` parsed,
  `.gguf`/`.q4nx` generic); `POST /v1/pool` reports residency; `/v1/models`
  tags pooled models. Measured on strix: **11 slots / ~6 GB resident**, all
  five zoo models served from one process, one OpenAI-compatible API.
- **In-server speculative decoding** (`--draft-model` + `--spec-decode`):
  `Backend` gains `decode_one` / `verify_batch` / `rollback` (ggml-vulkan
  impl); lossless greedy-consistent loop with batched verification and KV
  rollback. Deterministic 3/3 identical outputs; falls back to the normal
  loop when the target backend can't batch-verify. ggml-vulkan `reset()`
  now truly clears the KV per sequence (was a no-op).
- **Zamba2-1.2B-Instruct-v2 Q8_0** (1.84 GB): llama.cpp-fork arch alias +
  kv-name fallback, quantized tensor-exact vs HF checkpoint; chat API
  answers `4.0<|im_end|>`.
- **Model zoo end-to-end** (`scripts/zoo-smoke.sh`, 5/5 PASS): Llama-3.2-1B
  Instruct, Qwen3-0.6B Instruct, Bonsai-1.7B-TQ2, Zamba2 Q8_0, Qwen3-4B
  (NPU FLM) — one command, one server.
- **Spec-decode demo** (`tools/spec_decode.cpp`, `tools/spec_decode_README.md`):
  lossless vs greedy, cross-model draft/target pair.
- Fresh e2e numbers through the unified API in `README.md` (NPU Qwen3-4B
  20.8 tok/s; GGUF 1B ~12.4 tok/s; Zamba2 2.2 tok/s HIP).
- AMD Vitis/Unified SDI 2026.1 toolchain installed on strix
  (`/home/bcloud/Xilinx/2026.1`) for the FPGA-side roadmap.

## 2026.08.04 — memory campaign: leak root-caused + top-1 backend init 🧠

- **#1428 root-caused: not a leak — glibc arena fragmentation.** `FusedBackend::generate()`
  allocated a VOCAB-sized (608 KB) host vector *every token*; freed-but-never-trimmed,
  the blocks trip glibc's dynamic mmap-threshold into the brk arena, interleaving with
  the per-token HIP allocations so top-of-heap never returns to the OS. heaptrack proved
  zero unfreed bytes on the generate path. Buffers are now reusable members: mt=128 creep
  decays 17 MB → 20 kB per 10 reqs and plateaus (~160× less); the `MALLOC_*` env band-aid
  on `1bit-systems.service` was removed.
- **#1427 memory baseline cut ~10 GB.** Dead host f32 weight copies in FusedBackend freed
  after GPU upload/NPU pack (RSS 6768 → 5163 MB); `init_in_order()` now loads the top
  accelerator + one CPU fallback only, everything else inits lazily via the existing
  failover path (instance GTT 8.2 → 2.6 GB). Per-token cross-backend routing was
  KV-incoherent anyway (private KV per backend), so no routing value was lost.
- **Deployed** to `1bit-systems.service` (v2026.08.04 binary), verified: RSS ~5.3 GB flat,
  fused active + cpu fallback, hip_1bp/vulkan lazy.
- **10-bug audit resolved** (#1429–#1438) — SSRF, unkillable-server, OOB/SIGFPE/bad_alloc
  families, loader/backend hardening.
- **Local work artifacts gitignored** (model conv sources, appimage build, heaptrack dumps,
  tool state) — 31 untracked files cleared from status.
- Docs: engineering journey updated through 08-03 (UPDATE 29); packaging manifests synced
  to `2026.08.04` (issue #117 check green).

## 2026.08.03 — ws05 ppl gates + fused prefill chain, RVQ-VAE codec, narrative purge 🧹

- **ws05: per-vocab perplexity gates for every family** (#1243) — ppl gates now
  run for non-Qwen families too, with a truncation guard, single-instance lock,
  and gated-source swaps; partial-tile dequant stride fix + gemma arch aliases.
- **Kernel: fused O→residual→RMSNorm→Gate/Up chain (TQ2_1024)** — decode and
  prefill (M=128) variants cut the post-GEMM tail into one fused pass.
- **Codec: C++ RVQ-VAE decoder + GGUF export** (#1368) — codec now ships in the
  C++ stack with GGUF export support.
- **Narrative purge** (#1412 + follow-up) — stale TheRock-era and Rust-era
  claims removed from docs, CI, hackathon scripts, and packaging docs; proxy,
  agent CLI, and inference engine are all described as C++ now.
- Packaging manifests re-synced to `2026.08.03` (issue #117 check green).

## 2026.08.02 — one ELF to rule them all: every server + CLI in `1bit` 🚀

- **Single binary is now literal** — `build/1bit` embeds zaya_server,
  unified_server, unified_router, **jarvis_server** (TTS/voice, whisper +
  optional onnxruntime codec decoder) and **vision_server** (VL), plus the
  agent CLI: `1bit zaya|unified|router|jarvis|vision|chat|pull|list`.
  Dispatch by subcommand or legacy symlink name (argv[0]); the standalone
  targets still build for dev/CI.
- **`1bit pull` / `1bit list` in pure C++** — model registry + HTTPS download
  via httplib (no curl/bash); replaces the packaged bash launcher.
- **Packaging ships one binary** — deb/tarball stage now contains `usr/bin/1bit`
  + legacy-name symlinks (+ optional `1bit-npu` / `video_lora_vk_cli` sidecars).
  The 296-line bash launcher (`packaging/1bit`) was deleted; `install.sh` no
  longer requires Node.js — it installs the release tarball's single binary.
- Sizes: 67.2 MB raw / 64.4 MB stripped; `site/numbers.json` and the landing
  page bind the real size.
- Voice-clone endpoint now logs honestly: training is an optional Python
  (PyTorch) pipeline; serving/TTS stays pure C++.

## 2026.08.02 — llama.cpp fork synced to b10015-76-g0807d70be + build-hint fix 🔧

- **llama.cpp fork synced** — `third_party/llama.cpp` moved from the paged-KV PR
  head (`329cb3241`) to the documented sync point `0807d70be` (merge of PR #2 +
  873-commit upstream catch-up; +93 commits incl. Vulkan fixes #24362
  (FA mask_opt off on GCN), #25240 (submission threshold for small AMD GPUs),
  #25351/#25432 (SET_ROWS f16)). Rebuilt `-DGGML_VULKAN=ON -DBUILD_SHARED_LIBS=OFF`,
  Vulkan verified on Strix Halo (Radeon 8060S RADV, KHR_coopmat); `unified_server`
  relinked against the synced statics. Sync record: `docs/llama.cpp-fork.md`.
- **Build-hint fix** — the ggml-vulkan import warning in `CMakeLists.txt` now
  includes `-DBUILD_SHARED_LIBS=OFF`; without it ggml builds shared libs and the
  static import check silently fails even after a successful build.

## 2026.08.01 — Pure-C++ video-lora backend + fresh benchmark sweep + scope guard 🚀

- **Lemonade SDK embedded** — `unified_server` links Lemonade's server core
  (github.com/lemonade-sdk/lemonade, pinned at v11.5.1 in `third_party/lemonade`):
  all 14 Lemonade backends (llamacpp, flm, whispercpp, sd-cpp, kokoro,
  ryzenai-llm, vllm, ...) + the policy-based Router run in-process.
  `unified_server --lemonade` hands off to Lemonade's full server. The NPU
  backend's flm binary comes from the fastflowlm .deb / TheRock dist instead of
  the submodule build (submodule kept as last-resort fallback). `unified_router`
  routes NPU/GPU through Lemonade's RoutingPolicyEngine (deterministic keyword
  classifier; semantic/LLM classifiers are the upgrade path). Vendored-tree
  patches: `CMAKE_SOURCE_DIR` → `CMAKE_CURRENT_SOURCE_DIR` and a PUBLIC
  include-dir propagation on `lemonade-server-core` (both no-ops upstream).

- **TheRock-only builds** — release container no longer ships system ROCm
  (ubuntu:24.04 base; `rocm/dev-ubuntu-24.04:7.2.4-complete` removed) and CI
  never falls back to apt ROCm 7.2.4; all published benchmark claims say
  TheRock 7.15.0a. FastFlowLM .deb (v0.9.46) preferred over submodule build.

- **video-lora pure-C++ Vulkan backend** — `tools/video-lora/vulkan/`: complete
  Vulkan compute engine (conv2d, group_norm, silu, elementwise, attention with
  full softmax, lora_merge) linked into the single `zaya_server` binary. All ops
  GPU-verified vs CPU reference on Radeon 8060S. GLSL compiled at build time
  (glslc or glslangValidator).

- **PR agent hardened** — The-PR-Agent/pr-agent v0.41 (upstream) + DeepSeek +
  GitNexus impact reports; auto-review on every PR open and push; scope guard CI
  (engine + Jarvis) as a required check on main.

- **Fresh benchmark sweep (2026-08-01, Strix Halo / Radeon 8060S, TheRock ROCm 7.15.0a)**:
  - Qwen3-0.6B Q4_K GGML-Vulkan: **373 tok/s** decode (was 337)
  - SmolLM2-135M Q4_K GGML-Vulkan: **662 tok/s** decode (was 598)
  - Prefill INT8 WMMA (I8-APRE): **43.2 TFLOPS** (was 39.4)
  - Prefill 4h variant: 30.4 TFLOPS · TQ1 GEMV: 201 GB/s · Sherry: 157 GB/s
  - zaya_server: 1,578,576 B raw / 1,302,736 B stripped (video-lora linked)

## 2026.07.30 — GGML-Vulkan backend + CI smoke test fixed + stale PRs cleared 🏋️

- **GGML-Vulkan backend** — llama.cpp's Vulkan backend integrated as a new
  inference backend for GGUF/H1B models (MIT). Registered at T2_GPU tier with
  auto-failover → ZINC GPU → CPU.
  - Qwen3-0.6B Q4_K_M: **337 tok/s** (3.0 ms/tok), Init: 335 ms
  - SmolLM2-135M Q4_K_M: **598 tok/s** (1.7 ms/tok), Init: 110 ms
  - Standalone bench: `build/bench_ggml_vk <model.gguf> <tokens> <warmup>`

- **CI smoke test fixed** — 3 root causes resolved after 50+ consecutive failures:
  1. `libhipblas.so.3` not found → auto-detected TheRock SDK lib path at CMake
     configure time, added to `unified_server` RUNPATH
  2. `--model "Llama 3.2 1B Instruct"` didn't match `Llama-3.2-1B-Instruct.1bp`
     → hyphens/underscores normalized to spaces before name matching
  3. `npu_flm` always selected as active backend regardless of model format
     → backend selection respects `route.backend_ids_in_order` instead of
       global priority
  4. Models/ directory empty on CI (files git-ignored) → download
     SmolLM2-135M-Instruct-Q4_K_M.gguf (~104 MB) before server start

- **Backend selection fix** — active backend now chosen from the model route's
  ordered list (not global priority), so GGUF/H1B models route through
  ggml_vulkan first instead of being hijacked by npu_flm.

- **Model name normalization** — `-` and `_` in model names normalized to spaces
  before matching, so `--model "Qwen3 0.6B"` matches filenames like
  `Qwen3-0.6B.Q4_K_M.gguf`.

- **DynamicRouter integration** — ggml_vulkan (BackendType::GENERIC) participates
  in the existing per-token routing infrastructure via FASTEST strategy.

- **Stale PRs cleared** — #1148 closed (superseded), #1138 merged
  (validation-gaps.md restore + GGUF tooling), #1195 merged (1BP build
  pipeline scripts: build_1bp.sh, build_all_1bp.sh, bench_1bp_cpu.cpp).

- **Model catalog** — docs/wiki/models.md restored from base64 corruption (992
  lines of validated performance data). docs/validation-gaps.md restored with
  confirmed bugs + engineering blockers.

- **New tools added:** dump_gguf_meta.cpp, gguf_to_zaya_bins.cpp, run_gguf.cpp
  (minimal CPU inference runner), bench_1bp_cpu.cpp.

## 2026.07.29 — FastFlowLM integration + full benchmark sweep 🏋️

- **ROCm/FastFlowLM submodule added** — official MIT-licensed NPU engine at
  `third_party/FastFlowLM/`. Replaced reverse-engineered `npu_utils` files with
  official copies (bit-identical verifed). NPU kernel binaries now fully free
  for any use, including commercial.
- **16 new kernel benchmarks** measured live on Strix Halo (ROCm HIP): Q1 GEMV
  fused=431 tok/s, TQ2 standard=543 tok/s, Tile8 Zaya1-8B=57 tok/s, TWLA
  int4=3,009 tok/s, Prefill WMMA I8=40.66 TFLOPS, Mamba2 decode=1,293 tok/s,
  KV cache FD L=2048=57.3 GB/s, Sherry GEMV=155 GB/s.
- **Live `llama-bench` results** — 5 model configs across 3 families benchmarked
  on Vulkan ROCm and CPU: Qwen2.5-0.5B up to 15,853/423 tok/s (pp/tg),
  Qwen2.5-1.5B at 5,091/222 tok/s, Qwen3-0.6B at 12,818/273 tok/s.
- **NPU inference validated** — FLM native engine on real XDNA 2 hardware:
  Qwen3-0.6B at 67.5 tok/s sustained decode, 17.6 tok/s first-request.
- **1BP model catalog** — 37 models across 15 families documented with HF links.
  Qwen3-0.6B.1bp (355 MB) and BlackMamba-1.5B.1bp (970 MB) validated on hardware.
- **Green board** — All 19 model families show 🟢 across all 4 backends with
  ✅ validated status.
- **1BP → NPU integration** — `onebp_infer.cpp` ready, needs NPU engine pipeline
  wiring (~50 lines).
- **Zyphra family tracked** — Zaya1-8B 1BP model needs NDK xclbin compilation
  for full NPU support.

## 1.0.0 — 2026.07.26 — First Stable Release 🎉

# Changelog

All notable changes to 1bit.systems. Versioning is **date-based** (`YYYY.MM.DD`),
matching the GitHub release tags (`vYYYY.MM.DD`).

## 1.0.0 — 2026.07.26 — First Stable Release 🎉

- **CUDA + Metal GPU backends — cross-platform inference unlocked.** The single
  C++ inference engine now runs on NVIDIA (CUDA), AMD (HIP/ROCm), and Apple (Metal)
  GPUs from one CMake build. All three backends share the same ternary kernel
  library, weight loader, and server frontend (#858).
- **DeepSeek complete family — MLA + MoE architectures.** Full implementation of
  Multi-Head Latent Attention (MLA) and Mixture-of-Experts routing for the
  DeepSeek-v2/v3 model family, integrated into the dense GPU and CPU backends.
- **Vision-language pipelines: Qwen3-VL + ZAYA1-VL-8B.** End-to-end VL inference
  with ViT vision encoder, multimodal projector, and 1BP format support. Text
  decoder handles both Qwen3 and ZAYA1 architectures.
- **GPU Whisper kernels + Pixtral + FLUX feasibility.** Speech-to-text via native
  HIP GPU kernels (FFT, STFT) with whisper.cpp integration, plus Pixtral connector
  and FLUX diffusion model assessment.
- **Jarvis C++ port (Phases 1–3).** Complete C++ rewrite of the Jarvis agent:
  RAG, tool-calling, planner, routing, beacon (Phase 1); STT via whisper.cpp
  (Phase 2); TTS via piper with USB-speaker mirror (Phase 3) (#898, #902, #904).
- **NPU ternary pipeline.** Full TQ2 symmetric-ternary quant path in C++
  (`gguf_to_onebp --tq2`), block-vectorized `mac_8x8_8x8T` NPU kernel, IQ1_M
  and IQ1_S GPU dequant from llama.cpp, and NPU bridge wiring (#812, #887, #892).
- **GPU Render Engine — 2.6× faster prefill.** All GPU operations fused onto a
  single stream with pipelined async copies, eliminating kernel launch overhead
  for the prefill phase (#945).
- **6 paper-based performance improvements.** Integrated research techniques
  across backends for measurable throughput gains on all supported hardware (#917).
- **Bug fix sweep: 33 fixes across the stack.** Highlights:
  - Zamba2 `attention_forward` processed only **1 of 32 heads** — now fixed (#946)
  - MoE tensor shape validation + architecture guard (#947)
  - Server-side generation timeout + `RLIMIT_AS` OOM safety net (#948)
  - OOB token bounds check in Mamba1 forward() + server stability (#935)
  - Tolerate invalid UTF-8 in completion JSON responses (#944)
  - Serialize backend compute calls to stop watchdog/generate() race (#914)
  - HIP context bound on every Mamba1 entry point (#927)
  - NPU engine SIGPIPE ignored so dead workers don't crash the server (AUDIT #3)
  - unified_server data race fix (AUDIT #2)
  - 29 kernel `__shfl_xor_sync` fixes for NPU stability (#954-#962)
- **Windows MSVC build config.** Full CMake + MSVC toolchain for Windows builds,
  including `scripts/build_windows.cmd` and `CMakeLists_windows.txt`.
- **SEO overhaul.** Sitemap, robots.txt, meta tags, Cloudflare Pages Functions,
  auth worker, Web Analytics beacon across all 40 HTML pages (#926).
- **Repo-wide cleanup.** Dead file removal, stale benchmark corrections, honest
  claim validation, untracked build binary cleanup (#890, #892, #906, #908).

| Metric | Value |
|---|---|
| Commits since v2026.07.24 | 88 |
| Features | 16 |
| Bug fixes | 33 |
| Performance | 1 |

**Full changelog**: [v2026.07.24...v2026.07.26](https://github.com/1bit-systems/1bit/compare/v2026.07.24...v2026.07.26)

---

**Monthly cadence from here.** Next: `v1.1.0` — expected ~2026-08-26.

## 2026.07.24

- **Dense GPU inference through the C++ ZINC Vulkan backend — unlocked.** The
  reverse-engineered ZINC stack now runs dense GGUF transformers (Qwen2/ZR1)
  end-to-end on the Radeon 8060S and matches the CPU reference **token-for-token**
  at **~26 tok/s** (vs ~6 on CPU). Twelve bugs fixed to get there — the last was
  a KV cache sized from `context_length` (131072) that overran RADV's 4 GiB
  `maxStorageBufferRange`, silently zeroing cached-V reads and collapsing
  attention (#844, #847, #851, #852, #854). ZINC is enabled by default for
  the architectures it computes correctly (llama/mistral/qwen2) and falls back
  to `cpu_generic` otherwise (#856); `ZINC_DISABLE=1` forces HIP/CPU.
- **~8.5× faster dense CPU decode.** Parallelized the generic-backend matmul
  with OpenMP — ZR1-1.5B 1.5 → ~12 tok/s, deterministic, output unchanged (#849).
- **Engine crash-hardening.** A backend that fails to initialize (e.g. missing
  Vulkan shaders) now fails over to HIP/CPU instead of taking the server down;
  all backend init/benchmark/generate paths are exception-safe (#846, #847).
- **Security**: image-fetch curl restricted to http/https (SSRF/LFI hardening).
- **Pure-C++ 1BP toolchain, end to end.** Ported the `--tq2` symmetric-ternary
  quant path into `tools/gguf_to_onebp.cpp` (was Q4NX-only) and registered it as
  a first-class CMake target — the advertised `gguf_to_onebp model.gguf out.1bp
  --tq2` is now a real C++ binary, zero Python in the convert path. Verified:
  TQ2 output is exactly half of Q4NX and losslessly round-trips ternary input.
- **Engine health confirmed on-device**: BlackMamba-1.5B at **74.8 tok/s** live
  on Strix Halo (Radeon 8060S, gfx1151), three clean runs, no hangs.
- **Landing page**: new flagship 1BP model showcase (measured perf + direct
  Hugging Face download links) and a second-level **Zyphra** + **Poolside Laguna**
  family showcase. "One binary to rule them all" badge restored.
- **Repo hygiene**: untracked 431 committed `build_cmake/` artifacts + stray
  `Desktop/` that had been polluting every diff; reorganized 50 flat docs into
  `docs/{archive,marketing}/` + a navigation index; README marketing refresh.
- **Compiler warnings cleared** (#827 dead watchdog stores, #828 unused params,
  #829 dead `decode_ternary_word`/`qkv_dim`).

## 2026.07.20

- **Mamba1 GPU backend fully wired and fixed.** `backend_mamba1.cpp` + `mamba1_engine.hip` now compile as a first-class backend in `libbackend_manager.a`. Three critical correctness bugs fixed: conv state buffer overflow (shift loop out-of-bounds write), A_log never exponentiated (SSM scan used raw A_log instead of `A = -exp(A_log)`), and HIP device stub linkage (kernel launches wrapped in `extern "C"` helpers). BlackMamba 1.5B runs at **79.8 tok/s**, BlackMamba 2.8B at **46.4 tok/s** on Strix Halo (ROCm HIP, 15+15 MoE layers alternating). Diagnostic tool `tools/test_mamba1_backend.cpp` added for direct backend testing (#579).
- BlackMamba 1.5B and 2.8B GGUF files converted from HF cache (F16, 438/525 tensors) and benchmarked.

## 2026.07.19

- **FastFlowLM fully reverse-engineered and replaced as the default NPU path.** 22 closed-source `.so` libraries disassembled, 209 xclbin bitstreams traced to their AIE generators, whole stack rebuilt from source (#499, #500). `model_router.cpp` now routes qwen3-architecture models to the native, open-source `npu_xrt` engine first, with the FastFlowLM subprocess kept only as a fallback (#567) — its single-core GEMM kernels are correctness-verified on real hardware (`docs/research/GEMM-KERNEL-CORRECTNESS-CONFIRMED.md`), though throughput is currently lower until the 8-core multi-tile path lands.
- **Model-agnostic engine, broadened further**: GGUF architecture support 2→8 (LLAMA, MISTRAL, QWEN2, GEMMA, PHI, ZAMBA2), quant support 4→13 (Q4_1/Q5_0/Q5_1 legacy + full K-quant family), HIP backend now takes runtime `ModelConfig` instead of hardcoded dims, GGUF parsing consolidated into one shared, verified module (#436, #474, #488, #489, #494).
- **1BP ternary format**: fixed the converter/loader silently dropping norms and MoE expert weights (91% of Zaya1-8B was missing) (#528). TQ2 — real 2-bit symmetric ternary quantization, the format's actual "1-bit" storage path — implemented end-to-end (converter, loader, on-disk layout), verified lossless against source GGUF on a real ternary-trained model.
- **Vision**: Qwen2-VL support, minimal POC — real image-to-text end to end (#491, #492). Lightweight image preprocessing (stb_image, no OpenCV dependency) added.
- **Model catalog**: full Zyphra family showcase (Zamba2, ZR1, Zaya1-74B-preview) plus their 1BP conversions, all uploaded to Hugging Face (#529, #526).
- **NPU toolchain**: switched from Peano/LLVM-AIE to AMD Xilinx IP (Chess) (#527). 8-core INT8 GEMM correctness work reconciled across divergent branches (#344).
- **colibri int4 quantized matmul kernels + PILOT cross-layer prefetch** (#449). **A2A (Agent-to-Agent) protocol v1.0** support added to `zaya_server` (#345).
- Large correctness/security audit sweeps: ~60 numbered issues closed across #362, #415, #417, #436, #495, #496, #498, plus today's fixes (OSCAR attention cross-warp race, NPU worker pipe I/O timeout, concurrent HTTP handler state race).
- **Landing page**: removed a headline "tok/s" claim that the site's own data-integrity quarantine (`benchmarks/latest.json._unverified`) explicitly flagged as having no source — was still driving `<title>`/meta/OG tags and a JS bug that hardcoded it into the meta description on every load, bypassing the quarantine guard entirely.

## 2026.07.16

- feat(hardware-aware): auto-dispatch policy defaulting to N+G pathway
- fix(backend_manager): load_plugins now infers tier from plugin type instead of hardcoding T2_GPU
- doc: fixed stale paths in SECURITY.md, ROCm repo inconsistencies, CI pipeline table
- security: redacted exposed Stripe credentials from ROADMAP.md and site/store/index.html

## 2026.07.15

- fix(backend_manager): rank_backends now uses benchmark score as primary sort when FASTEST strategy
- fix(backend_manager): select_best respects strategy (FASTEST, LOWEST_POWER, ROUND_ROBIN, MANUAL)
- fix(backend_manager): benchmark_all keeps all benchmarked instances alive so re_evaluate can switch
- fix(backend_manager): generate() updates info.score with EMA so live latency feeds routing
- fix(backend_manager): set_strategy triggers re_evaluate for automatic strategies
- fix(backend_manager): added re_evaluate() — re-ranks and re-selects active backend
- fix(unified_server): removed post-benchmark re-init hack, now uses re_evaluate()
- chore(benchmarks): re-measured kernel benchmarks on Strix Halo (2026-07-15)

## 2026.07.15 (full benchmark sweep)

- feat(benchmarks): full kernel microbenchmark suite on Strix Halo
  - sherry GEMV: 153.0 GB/s | tq1 GEMV: 191.6 GB/s | halo GEMV: 162.8 GB/s
  - prefill 4h: 21.77 TFLOPS | prefill I8-APRE: 38.89 TFLOPS
  - bonsai full-model decode: 425 tok/s (Q1_0 1024-block), 358 tok/s (TQ2_1024)
  - fused TQ2 QKV+GU: 413 tok/s (1.15x speedup over individual launches)
  - KV Flash-Decoding: 12.65× speedup at seq_len=2048
  - RotorQuant PQ3: 9224 tok/s at seq_len=2048
- feat(bonsai): end-to-end real model decode verified on Bonsai 1.7B TQ2
  - Model load: PASS (hs=2048, is=6144, L=28, nh=16, nkv=8, V=151669)
  - Forward pass: coherent logits (argmax=76213, max=327007, min=-396679)
- chore(binary sizes): zaya_server=282KB, unified_server=1.2MB, bitnet_decode=688KB (historical)
- doc(benchmarks): published full results to benchmarks/RESULTS-2026-07-15.md

## [0.2.1] — 2026-06-26

### Bug fixes & robustness
- `install.sh`: Fixed `$1` unbound-variable crash when running without arguments
  under `set -euo pipefail`
- `env.sh`: Added `$LINK_DIR/build` to `PATH` so CLI tools (`bitnet_decode`,
  `bench_prefill_variants`, etc.) are discoverable after `source env.sh`
- `rust/src/main.rs`: Added `Drop` impl on `AppState` that kills the backend
  child process on server shutdown / panic (no more orphan zombies)
- `h1b_loader.cpp`: Added `f.fail()` checks after every `f.read()` to catch
  truncated or corrupt `.h1b` files early with a clear error message
- `tokenizer.cpp/.h`: [redacted]
  field is the *merged token id*, not the rank; rank is derived from insertion
  order. Added infinite-loop guard in BPE merge loop, empty-input early-return,
  and null-ids validation
- `prefill_dispatcher.cpp`: Added variant-index bounds and null-function-pointer
  checks before dispatch
- `CMakeLists.txt`: Removed `src/ck_gemm.cpp` from the HIP language property set
  (compiled as C++17 via CK's host-only path); removed `src/prefill_dispatcher.cpp`
  from the HIP source set (was duplicating `target_sources` entry)
- `.gitignore`: Removed duplicate `/rust/target` entry; added editor swap files
  and `ck-prefill/build/`
- `prim_kernels.hip`: Added `<cstdlib>` include for `std::abs` / `std::round`
  portability

### Documentation
- `tokenizer.h`: [redacted]
  is `new_id` (merged token id), not `rank`

## [0.2.0] — 2026-06-23
- Full benchmark on TheRock 7.15.0a (Ubuntu 24.04)
- Prefill 4h kernel: 21.94 TFlops (73% of rocBLAS, 2.9x per-byte)
- Decode halo: 27.01 µs (7.8x rocBLAS)
- Auto-tuner with 7 prefill variants
- CI: headers check + ShellCheck

## [0.1.0] — 2026-04-30
- Initial release on TheRock ROCm 7.13
- BitNet-2B-4T end-to-end decode at 82 tok/s
- Prefill 30.15 TFlops at 1.02x TheRock rocBLAS
- Decode GEMV 4.9-7.2x rocBLAS
