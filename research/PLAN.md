# 1bit-monster Research → Implementation Plan

**Date:** 2026-07-31 · **Status:** Draft v1 — scaffolding + first P0 probes complete (WS-05, WS-08 ✅)
**Inputs:** 65-paper archive (`~/research-papers/`, see `RESEARCH-BRIEF-2026-07-31.md` + `SYNTHESIS.md`), stack audit (`~/STATE-OF-THE-STACK-2026-07-14.md`), project research notes (`docs/research/*`, `validation-gaps.md`), live kernel state (2026-07-25 NPU benchmarks).
**Backup:** mirror at `~/research-papers/backup/` (this dir was wiped once by an unknown process on 2026-07-31 — keep the mirror in sync).

**How to use:** Workstreams are independent tracks with shared prerequisites. Phase 0 must go first — it fixes the floor. Within a workstream, P0 = do now, P1 = next, P2 = if the P0/P1 bets pay off. Every task ends with a *validation* step; no task is "done" without a number in the benchmark table with an honesty tag.

---

## Phase 0 — Stabilize the floor (from stack audit + validation gaps)

| # | Item | Evidence | Owner |
|---|------|----------|-------|
| P0.1 | **Un-break NPU fusion end-to-end** (fix `DRM_IOCTL_AMDXDNA_CREATE_HWCTX` EINVAL — hypothesis: 4-5 simultaneous HW contexts collide on column budget) | audit §8 task 2; issues #939/#940 | NPU |
| P0.2 | **Pick one router, retire the other two** (cascade vs `tools/token_router.cpp` vs `unified-router.py`) — fold spec-decode + content routing in as strategies | audit §3, §8 task 1 | Router |
| P0.3 | **One written 40-column decision** — NPU2-40 compiler is in progress, or closed as blocked | audit §4, `docs/superpowers/specs/2026-06-28-40column-npu2-compiler-design.md` vs `STEP5-INT8-32TILE-PLAN.md` | NPU |
| P0.4 | **Re-baseline every raw/reported number** — ROCm HIP 113, DSpark 0.8, NPU v12 97, fusion 291 (unreproducible) | audit §7 | Bench |
| P0.5 | **Kill the CPU attention stub** (8.4 ms/layer × 28 = 235 ms/token) — issue #936 Critical | `docs/research/npu/benchmarks-2026-07-25.md` | NPU |

> P0.5 is the single highest-leverage engineering task: fixing it targets 40-60 tok/s on the fused engine (vs 5.0 today). WS-01 exists to make the *next* version NPU-native.

---

## Workstream map

```
WS-00 Baseline & measurement ── feeds every validation below
   │
   ├─► WS-01 NPU fused attention (STEEL)          [P0.5 prerequisite]
   ├─► WS-02 XDNA quantized GEMM/GEMV (TileFuse)
   ├─► WS-03 Native ternary AIE microkernel (RSR-core, Sparse-BitNet)
   ├─► WS-04 CPU ternary kernel sweep (FairyFuse, Litespark)
   ├─► WS-05 1BP v2 format (ExTernD) — P0 probe DONE, see FINDINGS.md
   ├─► WS-06 Precision-profile router (MXSens, TWLA, ParetoQ)
   ├─► WS-07 MoE decode & spec (DraftExpert, PagedWeight, AngelSpec)
   ├─► WS-08 MLA + KV cache (QK-Normed MLA, JoLT, Lynx) — P0 gauge probe DONE
   ├─► WS-09 Router unification (DOPS)
   └─► WS-10 Metal/M5 + MLIR toolchain (BaseRT, AIE4ML)
```

Dependencies: WS-01 needs P0.5 → WS-02/03 need the 40-column decision (P0.3) + WS-00 harness → WS-05 P1 needs ppl harness (WS-00) → WS-07 needs the router decision (P0.2) → WS-09 is P0.2's follow-through.

---

## WS-00 — Baseline & measurement harness

**Papers:** NPU-Bottlenecks 2607.05475 (PowerBench methodology), llm.npu 2407.05858.

- P0 ✅ **Benchmark runner** (`run_benchmarks.sh`) — runs all `build/bench_*` binaries, JSON + tagged summary. Tested: bench_kv_fd captured live (30.1 GB/s fd vs 4.4 GB/s fp16 at L=1024, 6.79×).
- P0 🔲 **Perplexity harness** — wiring doc written (`ppl-harness.md`); needs 1BP ppl mode in engine + tokenizer parity check.
- P1: CI smoke bench; single-source-of-truth table; purge stale numbers (291 tok/s).

## WS-01 — NPU fused attention

**Papers:** STEEL 2607.09385, SANTA 2605.01910, TurboQuant 2504.19874 (cross-ref).
- P0: GPU flash-decoding kernel kills the 8.4 ms/layer stub (issue #936) → 40-60 tok/s fused.
- P1: STEEL-style fused attention on XDNA (`attn.xclbin` ABI, sparsity-aware).
- P2: SANTA sampled-value attention at 32k+ contexts.

## WS-02 — XDNA quantized GEMM/GEMV

**Papers:** TileFuse 2606.11357, MLIR-LLM 2607.15865, AIE4ML 2512.15946.
- P0: Per-shape perf catalog of the 4×I8 xclbin GEMMs (M=1 vs M≥16).
- P1: Native int4-pack GEMV microkernel (W4A16 = Q4NX class); MLIR dialect structure into mlir-aie flow.
- P2: W8A16 variant for sensitive layers.

## WS-03 — Native ternary AIE microkernel

**Papers:** RSR-core 2603.27462, Sparse-BitNet 2603.05168, FairyFuse 2604.20913 (cross-ref). Design: `docs/research/npu-ternary-roadmap.md`.
- P0: Finish `mm_ternary_tq2.cc` LUT unpack + ping-pong; verify `ternary_tq2_gemv` on hardware.
- P1: 2:4 structured sparsity on ternary (Sparse-BitNet).
- P2: RSR redundant-segment accumulation vs LUT unpack.

## WS-04 — CPU ternary kernel sweep

**Papers:** FairyFuse 2604.20913, Litespark 2605.06485, 1-bit AI Infra 2410.16144.
- P0: Port FairyFuse + Litespark variants as isolated microbenchmarks vs Q1 GEMV 417 tok/s.
- P1: Publish comparison; adopt winner.
- P2: Block-scaled ternary on CPU (5 B/16 elems).

## WS-05 — 1BP v2: expanded-rank ternary format

**Papers:** ExTernD 2607.13511, NativeTernary 2604.03336, NanoQuant 2602.06694, ParetoQ 2502.02631, CAT-Q 2606.26650.
- P0 ✅ **ExTernD probe** (`externd_probe.py`, `FINDINGS.md`): core claim validated (monotone decrease, μ=3 → 0.062); direct TQ2 wins at equal bitrate (0.22 vs 0.54); **actionable result: TQ2 + 64 correction planes = −15% error at +0.51 b/w**.
- P1: ppl harness confirmation on Bonsai-1.7B; 1BP v2 converter + C++ decoder (correction-plane extension, versioned header).
- P2: ship/kill decision vs GGUF Q2_K on 4 models.

## WS-06 — Precision-profile router

**Papers:** MXSens 2607.17733, TWLA 2606.13054, MX 2310.10537, LiquidGEMM 2509.01229, INT4-for-transformers 2301.12017, NVFP4-vs-MXFP4 deep-read. Design: `docs/research/hybrid-w4a8-router.md` + `block-scaled-ternary-format.md`.
- P0: Per-layer precision profile for one model (extend `tools/mr_gptq_rotate.py`).
- P1: Kernel dispatch by profile (`zaya_gpu_router.hip`, 1 byte/sub-layer).
- P2: MXSens-style sensitivity ranking to auto-generate profiles.
- **New (from WS-05 probe):** sensitive layers can take TQ2+correction-planes instead of full FP8 — cheaper than a second format.

## WS-07 — MoE decode & spec

**Papers:** DraftExpert 2607.24434, PagedWeight 2607.16184, CPU-GPU MoE SLO 2606.10493, AngelSpec 2607.25852, MLA-Draft-Functional 2607.27269, 35B-on-6GB 2606.24031.
- P0: Instrument draft/target agreement (explain 0% acceptance, issue #938).
- P1: DraftExpert expert-aware draft sizing; PagedWeight runtime expert quantization (63 GB iGPU).
- P2: AngelSpec drafter selection; stream-loading prefill (>12k tokens).

## WS-08 — MLA & KV cache

**Papers:** QK-Normed MLA 2606.16310, MLA-Bottleneck 2607.23054, MLA-Draft-Functional 2607.27269, JoLT 2607.12550, Lynx 2607.01831, Codec-Gauge 2607.20538, TurboQuant 2504.19874, TriRoute 2607.06601.
- P0 ✅ **Codec-Gauge probe** (`codec_gauge_probe.py`, `FINDINGS.md`): orthogonal channel gauge → **2.96× lower MSE at int8, 1.57× at int4**; must pair with per-channel scales; rerun on real KV dumps next.
- P0 🔲 QK-Normed MLA absorption on Kimi K3 gated-MLA decode.
- P1: JoLT-style tensor decomposition for KV; P2: Lynx progressive KV handoff.

## WS-09 — Router unification

**Papers:** DOPS 2607.25498, AutoScale-NPU 2607.16488 (P2).
- P0: Land P0.2 (one router, five strategies: passthrough/cascade/spec/content/performance).
- P1: Live throughput probe + health-check fallback; DOPS weight-layout awareness.
- P2: Workload-shape-aware dispatch.

## WS-10 — Metal/M5 + MLIR toolchain

**Papers:** BaseRT 2607.19438, AIE4ML 2512.15946, MLIR-LLM 2607.15865.
- P0: BaseRT kernel patterns review vs Metal backend; port MoE GEMM variant.
- P1: Reproduce BaseRT M5 benchmarks on our M5 box; AIE4ML structure onto mlir-aie (P0.3 evidence).
- P2: MLIR-LLM two-dialect layout for 40-column target.

---

## Ideas parking lot → `IDEAS.md`

TriRoute, Recover-LoRA W2 gate/up, PolyQ channel-permutation, BITEMBED, BitDistill, Spectra, LUT-LLM, TOM/VitaLLM ASIC patterns, GSQ 2-3 bpp, Compression-MoE survey, CAT-Q PTQ ternarization, SANTA, NVLLM/AME-PIM (dead end), AutoScale-NPU.

---

## Success metrics (end of Q3 2026)

1. NPU fused engine ≥ 40 tok/s, validated, tagged (from 5.0).
2. Native TQ2 AIE microkernel ships in an xclbin: Bonsai-4B batch=1 decode ≥ 2× bridge path.
3. Spec-decode acceptance > 30%; BlackMamba 2.8B > 46.4 tok/s.
4. 1BP v2 decision: ship or kill, with ppl data.
5. One router, one benchmark command, zero stale numbers in docs.
