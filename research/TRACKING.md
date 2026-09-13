# Workstream Tracking

> Single source of truth for workstream/task status. Legend: 🔲 not started · 🔄 in progress · ✅ done · ⛔ blocked · ❌ killed.
> Table rows below last walked **2026-08-29**; the dated delta underneath supersedes them where the two disagree.

## Status delta — 2026-09-12 (dsh, strixhalo) — read this before the tables

Fourteen days of work were never folded back into the rows below. This section is
the correction, with the artifact that carries the evidence. It is deliberately
split into *verified* and *open*.

### Verified

| Item | State | Evidence |
|------|-------|----------|
| P0.1 NPU IO_PAGE_FAULT path | **not blocking any more** — NPU attention runs end-to-end at 1k ctx (attn 150 ms vs 14.0 s CPU reference) | `engine/npu/generators/FK3-STATUS-2026-09-12.md` rounds 20–26 (branch `goal/runlist-decode-wire`) |
| WS-00 Baseline & measurement | harness landed: `benchmarks/flm_parity.sh` with `FLM_PARITY_TRUE_NATIVE=1` (without it the "native" column drives FLM's own libs) | `benchmarks/RESULTS-flm-parity-harness-2026-09-09.md` + `benchmarks/FLM-PARITY-DATA-SOURCES.md` |
| WS-01 NPU fused attention | superseded on the prefill path by a **generated long-context attention ELF** (FLM's exported `gen_mha_engine_seq` + `aiebu`) — no in-engine fused-attention kernel was needed | FK3-STATUS rounds 15–26 |
| WS-02/WS-03 native quantized + ternary AIE | still not started as scoped; the AIE effort went into the fk-1..fk-3 fused-layer PoC instead (`n1_fk3*.py`, `build_fk3.sh`) | FK3-STATUS §fk-3 PoC |
| WS-07 MoE decode & spec (35B-A3B runlist) | **proven dead end** — see "Settled negative results" below | goal `mtusoiy1-cfdhqr`; `benchmarks/RESULTS-runlist-decode-35b-moe-2026-09-10.md` (on branch `goal/runlist-decode-wire`) |
| WS-11 NPU weight path | dense-Qwen3 @1k prefill now beats FLM's published bar on all four models (+39…43%) — **but see "Open questions" before quoting it** | FK3-STATUS rounds 21–26 |
| #2199 fused int4 `.data` placement | **fixed and merged** (`f3825fbb6`, PR #2282); xclbin rebuilt 67,306 → 75,040 B, provenance manifest regenerated | `bash engine/npu/tests/check_kernel_bss.sh` → `bss=0 / RESULT: PASS` on `f3825fbb6` (re-run 2026-09-12) |
| Census | full HF sweep refreshed | `051d93e8d` (#2255) |
| HRX `reset()` context loss (#2203) | fixed: re-imports `HRX_STATE_FILE` after reset, guards resumed ctx against `HRX_MAX_CTX_TOKENS` | `docs/issue-campaign/1942-triage.md` §1.2 |

### Open questions (do not treat as settled)

1. **Does the @1k dense-Qwen3 prefill win survive?** The rounds-21–26 numbers
   (0.6B 2087.7 tok/s official harness, +40% vs FLM's published 1494) came from
   runs that never printed `Bf16Mm: long-context attention ELF loaded` — i.e.
   `run_attn` was serving the embedded 256-token ELF. After the ELF-search fix
   (`npu_engine_bf16_mm.h`, uncommitted on `goal/runlist-decode-wire` at the time
   of writing) the 1k ELF does load, and the first post-fix run measured
   **227 s** attention — but **two `npu_engine_qwen3_0_6b` runs were executing
   concurrently on the single NPU**, so that measurement is invalid. Needs one
   clean serial re-run before the claim is quoted or committed.
2. **Zaya1-8B fused-MoE correctness red flag (2026-09-09).** The in-engine L1
   probe compared NPU MoE output against a CPU fp32 reference and got
   **corr −0.001552** (expected 0.998267), identical on both fused and split
   launches — deterministic but wrong, and the `.q4nx` had been re-converted
   after the last known-good run. `benchmarks/RESULTS-zaya1-8b-rebaseline-2026-09-09.md`.
   Not resolved as of this note.
3. **`flm_parity.sh` decode column.** It prints `2` because the decode
   invocation feeds the whole prompt through the per-token `NPU_RUNLIST=1` path
   and the parser then reads the prefill's ms/tok instead of the final
   `=== Z ms/tok (W tok/s) ===` line. Owned by the NPU thread (the file only
   exists on `goal/runlist-decode-wire`).

### Settled negative results (don't re-litigate without new evidence)

- **35B-A3B single-launch runlist decode.** Two independent causes: (a) the
  per-ctx 35B layer ELFs produce NaNs — the region-B int4 generator is closed
  source; (b) cross-xclbin runlist batching is infeasible because the MoE xclbins
  carry different ERT `group_id`s (R98). Net: 0.57 tok/s (CPU MoE, 80
  launches/token) or 0.32 tok/s (NPU fused M=1) — both below the ~0.7 tok/s
  baseline and far from the dense-Qwen3 one-submit class. Results:
  `benchmarks/RESULTS-runlist-decode-35b-moe-2026-09-10.md` (branch
  `goal/runlist-decode-wire`). Goal `mtusoiy1-cfdhqr` stopped 2026-09-11.
- **Native single-launch zero-h2-DMA MoE fusion** (2026-08-28) remains blocked by
  the iron ObjectFifo + 2-input-DMA constraint; p1/p2 two-launch (h2 via DDR) is
  the production path. See `docs/AGENT-COORDINATION.md`.

## Phase 0 — Stabilize the floor

| ID | Item | Status | Notes |
|----|------|:------:|-------|
| P0.1 | NPU exec fault path (IO_PAGE_FAULT per exec, ~10 s/layer) | 🔄 | Fix staged: amd_iommu=off in grub (backup grub.bak-20260731-1418); reboot + validate_npu_after_reboot.sh | Diagnosed: not a hang — 1000x-slow faulting exec; engine works e2e at 0.1 tok/s; see P01-DIG-FINDINGS.md |
| P0.2 | One router, retire the other two | 🔄 | cascade vs `tools/token_router.cpp` vs `unified-router.py` — those two tools are no longer in this tree (2026-08-29 triage); remaining work: failover order now follows the model route (`BackendManager::fallback_order()`, G1a), DynamicRouter per-token strategy still separate |
| P0.3 | 40-column decision in writing | 🔲 | NPU2-40 compiler or formally closed |
| P0.4 | Re-baseline raw numbers (HIP 113, DSpark 0.8, fusion 291) | 🔲 | After WS-00 harness lands |
| P0.5 | Kill CPU attention stub (8.4 ms/layer) | ✅ | Swap done: 53-248x kernel-level (FINDINGS.md); e2e measurable now (240s run reaches decode) |

## Workstreams

| WS | Name | P0 | P1 | P2 | Status |
|----|------|:--:|:--:|:--:|:------:|
| WS-00 | Baseline & measurement | 🔄 | 🔲 | — | runner done, ppl wiring next |
| WS-01 | NPU fused attention | 🔄 | 🔲 | 🔲 | swap done (53-248x, FINDINGS.md); e2e blocked on NPU IOMMU hang (P0.1) |
| WS-02 | XDNA quantized GEMM/GEMV | 🔲 | 🔲 | 🔲 | gated on P0.3 |
| WS-03 | Native ternary AIE microkernel | 🔲 | 🔲 | 🔲 | design exists (npu-ternary-roadmap.md) |
| WS-04 | CPU ternary kernel sweep | ✅ | 🔲 | 🔲 | sweep done: fairy 54-57x, vnni 45-62x, lut 31-41x vs scalar @ ~41 GB/s (FINDINGS.md) |
| WS-05 | 1BP v2 (ExTernD) | ✅ | 🔲 | 🔲 | probe done + full-matrix confirmation → FINDINGS.md |
| WS-06 | Precision-profile router | 🔲 | 🔲 | 🔲 | gets correction-planes option from WS-05 |
| WS-07 | MoE decode & spec | 🔲 | 🔲 | 🔲 | issue #938 entry gate |
| WS-08 | MLA & KV cache | 🔄 | 🔲 | 🔲 | gauge probe done; QK-normed MLA next |
| WS-09 | Router unification | 🔲 | 🔲 | 🔲 | gated on P0.2 |
| WS-10 | Metal/M5 + MLIR toolchain | 🔲 | 🔲 | 🔲 | — |
| WS-13 | Arch-gap closure (V4/V4.1, Mamba-3) | ✅ | 🔄 | 🔲 | P0 done (oracle ≤1e-8, compressor ≤5.4e-07, indexer ≤1.3e-08, compressed attention exact 7.451e-09, rope-theta control, shape-agnostic 1.080e-07); P1: GGUF route CLOSED (no V4 converter; no compressor/mhc tensors) → engine-native quantisation, real-checkpoint ingest open (WS-07/WS-11); P2: V4.1 modules + Mamba-3 specified, not implemented — ws13/FINDINGS.md + SPEC-v41-modules.md |
| WS-12 | HRX/Loom platform transition | ✅ | 🔲 | 🔲 | re-vendored 7953d7f + native `HRX_GPU` backend + decode-time failover (commits 43b38b4e, cc4fd23d, 2026-08-29) |

## Task detail

### ws13-arch-gap-closure
- [ ] P0: tiny-config oracle (V4.1 + Qwen3Mamba3) + compressor/indexer maths spec + runtime-format decision
- [ ] P1: CSA/HCA compressors + Lightning Indexer; shape-agnostic loader + GGUF aliases; real V4-Flash e2e identity gate
- [ ] P2: V4.1 deltas (engram, gate.bias_vl, candidate blocks, MTP) · vision scope decision · Mamba-3 SSM lane

### ws12-hrx-loom
- [x] P0: Re-vendor lemonade e1b31683 → 7953d7f (hrx backend arrives) — verified onebin registers `llamacpp-hrx` on gfx1151 (2026-08-29)
- [x] P0: Smoke-test HRX recipe end-to-end on gfx1151 — `Qwen3-30B-A3B-Instruct-2507-HRX` loaded + chat completed ("Paris"; prompt 130.8 tok/s, gen 35.2 tok/s) via `1bit unified --lemonade` (2026-08-29)
- [x] P1: Track llama.cpp RFC #27218 / ggml-hrx upstream — **PR #27218 "ggml-hrx: add AMD ROCm HRX native ggml backend" exists upstream** (2026-08-29); Vulkan's GET_ROWS CPU-fallback (PR #26854) validates the hybrid pattern. Re-benchmark when it lands.
- [ ] P1: Audit hrx-v2 branch (179 commits ahead) — decide fork track: HRX vs HIP/Vulkan
- [x] P1: Benchmark HRX vs HIP baseline on gfx1151 (RFC claims 30–50% prefill) — **NOT reproduced**: HIP wins large prefill (1227–1313 tok/s); HRX fails closed on `GET_ROWS` for large prompts; HRX wins warm decode (~175 vs ~70 tok/s). See `ws12-hrx-loom/BENCHMARK.md` (2026-08-29)
- [x] P2: Native `HRX_GPU` backend in the engine — `BackendType::HRX_GPU` + `backend_hrx.h/.cpp` (subprocess HRX llama-server, LSE-style), wired into `backend_manager` + `backend_factory`, `hrx_gpu` first in the GGUF/H1B + qwen3-GGUF routes, `src/backend_hrx.cpp` in `UNIFIED_SERVER_SOURCES`. Verified: selfcheck 10/10 + engine selects `hrx_gpu` functional on gfx1151. **Decode-time failover closed 2026-08-29**: HRX decode fail-closed (GET_ROWS) now re-routes to another backend (`BackendManager::generate_text` failover + `DynamicRouter::generate` retry/pick_backend_excluding) instead of 500. See `docs/research/hrx-backend.md` (2026-08-29)
- [ ] P2: Evaluate Loom (`loomc` C API) as an authoring surface for 1bit kernels
- [x] P2: HRX fallback quality — a model HRX can't fuse falls back to whatever backend is next in the DynamicRouter (often `cpu_generic`), and output quality can be poor for that model. **FIXED 2026-08-29 (G1a):** failover now follows the model route — `BackendManager::fallback_order()` (route order, then registration as last resort) drives `failover()` and unified_server's token-loop fallback, so HRX fail-closed lands on `ggml_vulkan` → `zinc_gpu` → `cpu_generic`, never on a registration-order NPU lane. Built + graph-analyzed (risk low); see `docs/research/hrx-engine-goal.md`.
- [x] P2: Lemonade × HRX data step (P3, 2026-08-29) — `third_party/lemonade/tools/gen_hrx_model_entries.py` + **43 `*-HRX` entries** in server_models.json (44 total; excludes embeddings/reranking + vision models — HRX chat-only). Validated: parses, recipe pinned, no dups. **lemond ModelManager cache 150 → 193 total (+43, zero errors); full pull+serve PASS on Qwen3-30B-A3B-Instruct-2507-HRX ("Paris", hrx-b59 bundle spawned)**. Engine-native path already routes all GGUF HRX-first.
- [x] P2: HF coverage tool (P5, 2026-08-29) — `tools/hf_coverage.py`: HF model id → coverage-lane verdict (L1 llama.cpp 263 archs extracted live from the vendored converter, L2 engine-specialized, L3 FLM, L4 lemonade) or the 5-step add-model checklist. Verified: Qwen3 → COVERED (L1), Zamba2 → COVERED (L2), fake arch → BLOCKED + checklist.
- [x] P2: In-process HRX engine (fork A, 2026-08-29) — `src/hrx_inprocess.{h,cpp}` (dlopen `libllama.so` RTLD_DEEPBIND, token-level decode on HRX0, 30B MoE GGUF served through `1bit unified`, `backend: hrx_gpu`). Engine fixes: MoE-GGUF route (MoE GGUFs no longer misrouted to the CCA/HIP path), router-registration refinement (only npu_flm excluded), router-exhaust recovery (fall through to manager failover with on-demand route init). **Bench: in-process ~80–87 tok/s warm decode — ~2× the same-day fresh subprocess (38.2 tok/s); beats HIP (~70)**. See `docs/research/hrx-engine-goal.md` P2.
- [x] P2: The native `HRX_GPU` backend and lemonade's `llamacpp-hrx` recipe are two independent HRX paths — **documented 2026-08-29 (engine-native primary; lemonade entries = UX complement); see `docs/research/hrx-engine-goal.md` §Deployment preference.**
- [ ] P2: Context7 library health — `context7.json` / `context7-config-full.json` show two duplicate `/1bit-monster/1bit-monster` entries (5,985 vs 5,441 snippets) and `parseFailures: 37`; reconcile the duplicates and review the failing docs.
- [x] **COMPLIANCE (HIGH): lemonade re-vendor provenance violates LOCAL-ONLY.** The `third_party/lemonade` re-vendor to `7953d7f6` (HRX backend, ~2026-08-29, commit 43b38b4e) was pulled from **upstream** `github.com/lemonade-sdk/lemonade` — explicitly forbidden by the LOCAL-ONLY rule ("never clone/pull from https://github.com/lemonade-sdk/lemonade"). The local lemonade source (`/home/bcloud/1bit-lemonade-v1170`, branch `chore/lemonade-v11.7.0`) does **NOT** contain commit `7953d7f`, so the vendored HRX backend was not reproducible from local-only source. **RESOLVED 2026-08-29 (option a: port + re-vendor local-only):** the local `1bit-lemonade-v1170/third_party/lemonade` snapshot was synced up to v11.8.1 + full HRX backend (backends/hrx, resources +44 `llamacpp-hrx` entries, backend_versions hrx-b59 pin, `gen_hrx_model_entries.py`, CMake LEMON_BACKENDS) so it is the authoritative LOCAL-ONLY source; the repo's `third_party/lemonade` content is now byte-identical to it, and both `UPSTREAM.md` docs source locally. Commits: local `8dd9b51f` (chore/lemonade-v11.7.0), repo `aa16c2ac`. No upstream contact made.
- [ ] P2: `hrx-engine-goal.md` open items — pull+serve the remaining representative model set (dense + MoE + long-context) recording tok/s / fail-closed per family; re-run benchmark recipe #4 when the bundle or PR #27218 moves.
- [x] P2: `research/ws12-hrx-loom/README.md` was stale (README items shown `[ ]` but actually DONE) — **reconciled 2026-08-29**: README P0/P1 re-vendor, register, smoke-test, track-RFC, and benchmark items marked `[x]`. Closing this tracker entry so it doesn't hang open.

### WS-00 — Baseline & measurement
- [x] P0: `run_benchmarks.sh` — runs all `build/bench_*` binaries → JSON + tagged summary (tested 2026-07-31: bench_kv_fd, 30.1 GB/s fd vs 4.4 fp16 at L=1024)
- [ ] P0: Perplexity harness (`ppl-harness.md` written; needs 1BP ppl mode + tokenizer parity check)
- [ ] P1: CI smoke bench on every commit
- [ ] P1: Single-source-of-truth benchmark table; kill stale numbers

### WS-01 — NPU fused attention
- [x] P0: GPU flash-decoding swap — isolated bench 53-248x (23.7 -> 0.2 ms @ ctx 1024); engine patched + builds (build/npu_engine_overlap_fd); e2e blocked on pre-existing NPU IOMMU hang (P0.1)
- [ ] P1: STEEL-style fused attention on XDNA (`attn.xclbin` ABI)
- [ ] P2: SANTA sampled-value attention for 32k+ contexts

### WS-02 — XDNA quantized GEMM/GEMV
- [ ] P0: Per-shape perf catalog of 4×I8 xclbin GEMM (M=1 vs M≥16)
- [ ] P1: Native int4-pack GEMV microkernel (W4A16/Q4NX class)
- [ ] P1: MLIR-LLM dialect structure into mlir-aie flow (P0.3 evidence)
- [ ] P2: W8A16 variant for sensitive layers

### WS-03 — Native ternary AIE microkernel
- [ ] P0: Finish `mm_ternary_tq2.cc` LUT unpack + ping-pong; verify on hardware
- [ ] P1: 2:4 structured sparsity on ternary (Sparse-BitNet)
- [ ] P2: RSR redundant-segment accumulation vs LUT unpack

### WS-04 — CPU ternary kernel sweep
- [x] P0: ternary_cpu_sweep.cpp — FairyFuse (pext) / Litespark (VNNI) / T-MAC-LUT vs scalar; all correct (<=1e-4); GU-like 37.7us @16T; ~41 GB/s
- [ ] P1: Publish comparison; adopt winner
- [ ] P2: Block-scaled ternary on CPU

### WS-05 — 1BP v2 (ExTernD)
- [x] P0: ExTernD probe — monotone decrease validated; TQ2+correction-planes = −15% @ +0.51 b/w (FINDINGS.md)
- [ ] P1: ppl confirmation on Bonsai-1.7B; 1BP v2 converter + C++ decoder
- [ ] P2: ship/kill vs GGUF Q2_K on 4 models

### WS-06 — Precision-profile router
- [ ] P0: Per-layer precision profile for one model (extend `tools/mr_gptq_rotate.py`)
- [ ] P1: Kernel dispatch by profile (`zaya_gpu_router.hip`)
- [ ] P2: MXSens-style sensitivity ranking; WS-05 correction planes as the "sensitive layer" option

### WS-07 — MoE decode & spec
- [ ] P0: Instrument draft/target agreement (explain 0% acceptance, issue #938)
- [ ] P1: DraftExpert-style expert-aware draft sizing
- [ ] P1: PagedWeight-style runtime expert quantization (63 GB iGPU)
- [ ] P2: AngelSpec drafter selection; stream-loading prefill (>12k tokens)

### WS-08 — MLA & KV cache
- [x] P0: Codec-Gauge probe — 2.96× int8 MSE gain, 1.57× int4 (FINDINGS.md)
- [ ] P0: QK-Normed MLA absorption on Kimi K3 gated-MLA decode
- [ ] P1: JoLT-style tensor decomposition for KV; rerun gauge probe on real KV dumps
- [ ] P2: Lynx-style progressive KV handoff

### WS-09 — Router unification
- [ ] P0: Land P0.2 (one router, five strategies)
- [ ] P1: Live throughput probe + health-check fallback
- [ ] P1: DOPS-style weight-layout awareness
- [ ] P2: Workload-shape-aware dispatch

### WS-10 — Metal/M5 + MLIR toolchain
- [ ] P0: BaseRT kernel patterns review vs Metal backend; port MoE GEMM variant
- [ ] P1: Reproduce BaseRT M5 benchmarks on our M5 box
- [ ] P1: AIE4ML structure onto mlir-aie flow (P0.3 evidence)
- [ ] P2: MLIR-LLM two-dialect layout for 40-column target
