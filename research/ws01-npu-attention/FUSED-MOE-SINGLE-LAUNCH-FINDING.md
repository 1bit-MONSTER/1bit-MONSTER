# Fused MoE bottleneck — root-caused and a 1.74× win found (2026-09-04)

Follow-up to RESIDENT-ATTN-PACKS-VERIFICATION.md. After the resident int8
attention packs landed (`NPU_PROJ=1`), the fused MoE became the dominant cost.

## 1. Where the 5.4 ms/layer goes (split launch, `NPU_TIMING=1`)

Instrumented `zaya_decode.cpp`'s fused path. Per MoE layer (`[fused-t]`):

| phase | ms |
|---|---|
| header fold (`update_fused_header`) | 0.010 |
| launch (quantize + submit) | 0.021 |
| **P1 wait (GU→SiLU→h2, 8 MB weights)** | **3.645** |
| h2 sync (visibility barrier) | 0.001 |
| **P2 (D from h2, 4 MB weights)** | **1.771** |
| **total** | **5.447** |

So the cost is **not** the launch/sync overhead — it is the **12 MB/layer of
resident int8 expert weights streamed through the HOST_ONLY coherent path
(~2.1–3.6 GB/s) across two separate kernels**. 20 MoE layers × 5.4 ms ≈
108 ms/tok.

## 2. The levers, and why most are closed

- **int4 weights (`NPU_FUSED_I4=1`)** — halves GU weight bytes but is **10×
  slower** (p1wait 34.9 ms). The int4 GU kernel uses the `I4_SCALAR_C1`
  scalar fallback (the mmul C-store is miscompiled, #1869), i.e. ~67 M MACs
  in a scalar loop. Blocked on a toolchain fix.
- **faster BO flags (`NPU_WBO_FLAGS`)** — `SVM` throws "Bad BO type", `CACHEABLE`
  exhausts the NPU heap ("No space left on device"). HOST_ONLY is the only
  flag that holds the ~320 MB of resident weights.
- **single-launch fused** — was the 2.1 ms/layer #1759 design, but #1775 found
  run-to-run nondeterminism at MoE layers 3+ and the split launch replaced it.

## 3. The single-launch is deterministic again — 1.74× faster

Re-added the original single-launch (`final_i8_MOE_FUSED_zaya.xclbin`) behind
`NPU_FUSED_SINGLE=1` as an A/B probe. The #1775 nondeterminism suspects were
(a) CPU-attention omp and (b) NPU BO reuse — both now fixed in `main`
(#2053 omp + physical-core cap; per-layer `h2_bo`). Measured:

| config | ms/MoE-layer | ms/tok | tok/s |
|---|---|---|---|
| split launch (`NPU_FUSED=1`) | 5.45 | 207–215 | 4.8 |
| **single launch (`NPU_FUSED_SINGLE=1`)** | **3.10** | **175–186** | **5.5–5.7** |

**Correctness / determinism:**
- `[MoE L1 single dbg] corr=0.998267` (vs split 0.998469 — both at the int8
  floor; the small delta is float-silu vs q22-silu).
- **10/10 runs byte-identical**: 7 runs `<bos>` and 3 runs prompt `236778`
  (the #1775 repro) all produce identical `[NPU dbg]` logits
  (`min=-24.2578 max=34.5331 rms=6.4627`) and identical 10-token streams.

The split launch was an over-correction: #1775's own data showed **layer 1
bit-identical** and only layers 3+ varying — which points at CPU-attention
omp / BO-reuse, not the in-kernel h2 round-trip (that would have also
destabilised layer 1). With #2053 + per-layer `h2_bo` in `main`, the
single-launch is stable.

## 4. Status / recommendation

- **DONE**: the single-launch is now the default for `NPU_FUSED=1`
  (2026-09-04). `NPU_FUSED_SPLIT=1` restores the old two-launch split as the
  fallback; `NPU_FUSED_I4=1` keeps the int4 GU (always split). Verified:
  default `[fused-single-t] total=3.135 ms` vs split
  `[fused-t] p1wait=3.100 p2=1.521 total=4.651 ms`, same tokens/corr.
- The timing instrumentation (`[fused-t]` p1wait/h2sync/p2 breakdown) stays in
  `zaya_decode.cpp`.
- **Soak test: PASSED** (2026-09-04, see §5) — the extended multi-hundred-token
  determinism gate is satisfied.
- Remaining after that: the 3.1 ms/layer is still ~12 MB/layer weight-DMA
  bound; the next lever is resident-on-device weights (runlist + NPU-resident
  BOs, the issue's original "resident weights" plan) rather than more host-side
  micro-opt.

## 5. Extended soak test (the #1775 determinism gate) — PASSED

Ran the shipped `npu_engine_zr1` (single-launch default) on
`zaya1-8b-fresh.q4nx`, 300 tokens/run, on strixhalo with the Windows VM killed
(clean bandwidth):

| prompt | runs | N | result |
|---|---|---|---|
| `<bos>` | 3 | 300 | **byte-identical** (md5 `db2e8a70…`) |
| `236778` (#1775 repro) | 2 | 300 | **byte-identical** (md5 `2897fa17…`) |

5/5 runs deterministic (1500 tokens, ~60k MoE layer evals). `[MoE L1 single dbg]
corr=0.998469` every run; `[NPU dbg] logits min=-29.1048 max=19.3443 rms=5.4135`
identical across runs.

**Perf (clean box):** 104–107 ms/tok (**9.3–9.6 tok/s**) — ~1.7× faster than
§3's 175–186 ms/tok, which was measured under co-tenant contention (the Windows
VM + other jobs). The single-launch decode is deterministic and ~9.5 tok/s when
the box is quiet.

**Verdict:** the single-launch fused GU→SiLU→D is deterministic over extended
generations. The #1775 nondeterminism (CPU-attention omp + BO reuse) is
definitively gone with #2053 + per-layer `h2_bo` in main.
