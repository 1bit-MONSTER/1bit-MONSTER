---
bug_id: BUG-005
status: fixed
severity: high
scope: engine/npu/generators/n1_core_fused_gu_silu_d_iron.py (design dataflow)
title: D GEMM distributed K AND N across cores, but the cascade only sums K-partitions of the SAME column
---

# BUG-005: K+N cross-distribution (the root cause of wrong/nothing C2)

## Symptom

The D GEMM `C2 = h2(8×K) @ B_d(K×N_D)` produced:
- `N_D=1024` (single-pass) → **nothing** (C2 untouched), and
- `N_D=2048` (2 cg2) → only cg0, a plausible-but-**wrong** value.

Both looked like separate bugs but share one root cause.

## Root cause

The design distributed **both K and N** across the 8 cores:
- **K**: each core held 1/8 of h2's K-dimension,
- **N**: each core read its OWN column's B_d slice (per-column `of_b[c]`).

But the **hardware cascade only reduces the K-partitions of a SINGLE column.**
So the cascade summed 8 *different* columns' partials — mathematically wrong.
This is why "single-pass gives nothing" AND "2-pass gives cg0" both happened.

## The correction (validated)

The single-pass cascade must feed every core the SAME output column:
1. Each core reads **only its OWN h2 K-slices** — `ki = cg*8 + col` (the GU's
   h2 is valid ONLY there; the old code read `ki = 0..n_k`, 28/32 garbage).
2. **B_d is full-width, same columns for all cores**.
3. **ONE cascade pass** over the (8×N_D) partial via a wide-N mm
   (`matmul_i8_i32_wide`) + `cascade_reduce_{first,mid,last}_i32_wide`.

## Verification

`N_D=128`, `--no-gu --h2-const=1`, `B_d=1` →
**`C2=2048` everywhere = the exact expected value**
(each core: 4 k-slices × 64 = 256; 8-core cascade sum → 2048).

## Status

**FIXED + silicon-verified** (the D side). This is the real, durable win of
the session. The remaining GU-side blockers are BUG-006/BUG-007.
