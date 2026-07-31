# Workstream Tracking

> Single source of truth for workstream/task status. Legend: 🔲 not started · 🔄 in progress · ✅ done · ⛔ blocked · ❌ killed. Updated: 2026-07-31.

## Phase 0 — Stabilize the floor

| ID | Item | Status | Notes |
|----|------|:------:|-------|
| P0.1 | NPU exec fault path (IO_PAGE_FAULT per exec, ~10 s/layer) | 🔄 | Diagnosed: not a hang — 1000x-slow faulting exec; engine works e2e at 0.1 tok/s; see P01-DIG-FINDINGS.md |
| P0.2 | One router, retire the other two | 🔲 | cascade vs `tools/token_router.cpp` vs `unified-router.py` |
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
| WS-04 | CPU ternary kernel sweep | 🔲 | 🔲 | 🔲 | — |
| WS-05 | 1BP v2 (ExTernD) | ✅ | 🔲 | 🔲 | probe done + full-matrix confirmation → FINDINGS.md |
| WS-06 | Precision-profile router | 🔲 | 🔲 | 🔲 | gets correction-planes option from WS-05 |
| WS-07 | MoE decode & spec | 🔲 | 🔲 | 🔲 | issue #938 entry gate |
| WS-08 | MLA & KV cache | 🔄 | 🔲 | 🔲 | gauge probe done; QK-normed MLA next |
| WS-09 | Router unification | 🔲 | 🔲 | 🔲 | gated on P0.2 |
| WS-10 | Metal/M5 + MLIR toolchain | 🔲 | 🔲 | 🔲 | — |

## Task detail

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
- [ ] P0: Port FairyFuse + Litespark variants as microbenchmarks (vs 417 tok/s Q1 GEMV)
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
