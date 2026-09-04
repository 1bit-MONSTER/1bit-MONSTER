# Resident int8 attention packs — verified + re-benchmarked (2026-09-04)

Closes the remaining #1776 scope: "Zaya resident-weight int8 attention packs —
attention projections as NPU GEMMs with resident int8 packs (packB-style, like
the MoE path) so per-token weight streaming disappears; re-benchmark against the
post-#2053 CPU baseline first."

## What was measured

The `NPU_PROJ=1` path already in `main` (commit `3ced5bdb`, attention milestone
#1892): the 4 attention projections `wq/wk/wv1/wv2` are packed **once at startup**
into a fused QKV-concat int8 BO (`[H=2048, qd+kd+hv2+hv2=1536]`), `wo` into an
`o_proj` int8 BO (`[qd=1024, H=2048]`), both reusing the
`final_i8_MOE_D_zaya_m16.xclbin` (K=2048, N=2048) GEMM machinery. Per token only
the 2048-dim activation is quantized + launched + dequantized — the ~20 MB/layer
of float attention weights is no longer streamed through the CPU GEMV path.

Model: `~/models/zaya1-8b-fresh.q4nx` (the re-converted, fully-decoded Q4NX).
NPU: Strix Halo (AI MAX+ 395), XRT 2.21.75, firmware 1.1.2.65.

## Correctness (NPU vs the CPU float projection reference, layer 0, pos 0)

| projection | corr |
|---|---|
| q_proj (`wq`) | 0.999709 |
| k_proj (`wk`) | 0.999965 |
| v_proj_current (`wv1`) | 0.999367 |
| v_proj_delayed (`wv2`) | 0.998680 |
| o_proj (`wo`) | 0.999867 |

These are the int8 per-column-quantization floor (identical in kind to the MoE
path's 0.9993–0.9996). No packing/transpose bug — a layout error would show
corr ≪ 0.99, not 0.9987+. The QKV concat section ordering (q | k | v1 | v2) and
the o_proj `[qd→H]` transpose are correct.

## Re-benchmark — post-#2053 CPU baseline vs resident packs

Same prompt (bos only), `NPU_FUSED=1` (fused MoE on NPU), `N_GEN=8`, same window:

| config | ms/tok | tok/s |
|---|---|---|
| CPU attention projections (baseline, float embed) | 354 | 2.8 |
| CPU attention projections (baseline, `NPU_EMB_INT8=1`) | 333 | 3.0 |
| **`NPU_PROJ=1` resident int8 packs (`NPU_EMB_INT8=1`)** | **207–209** | **4.8** |

**~1.6×** faster than the post-#2053 CPU baseline. The fused MoE launch is
`~5.4 ms/layer` (`[fused-t] hdr=0.010 launch=0.021 wait+deq=5.394`) — the split
P1/P2 launch from the #1775 determinism fix — so 20 MoE layers ≈ 108 ms/tok is
now the dominant cost, and the attention projections add ~40–80 ms/tok instead of
the ~125 ms/tok the CPU GEMVs cost.

## Caveats (why absolute numbers, not tokens, are the bar)

- **Token streams diverge** (CPU `2733 47004 …` vs NPU_PROJ `22775 30122 …`).
  This is the *near-flat-logits* Zaya base model (word-salad, §15 of
  ZAYA-CCA-CPU-PORT.md), not a projection bug — the per-projection corr 0.9987+
  is the same quantization the MoE path ships, and the argmax flips on ~1e-3
  logits deltas. The established bar for this model is corr, not token parity.
- **Absolute ms/tok is co-tenant-sensitive.** This box is shared
  (memory-bandwidth contention, see the issue body); the 333 ms/tok CPU figure
  is higher than the pre-#2053 180 ms/tok record partly because the fused MoE is
  now the split 2-launch path (5.4 vs the issue's 2.1 ms/layer) and partly
  because of contention. The A/B (NPU_PROJ vs CPU, same window) is the signal.

## Status

Resident int8 attention packs: **done + verified**. Remaining (not this issue's
scope): the fused MoE is now the bottleneck (5.4 ms/layer split launch) — a
return to a deterministic single-launch fused GU→SiLU→D, or batching the two
per-layer projection launches + MoE into one runlist, is the next lever.
