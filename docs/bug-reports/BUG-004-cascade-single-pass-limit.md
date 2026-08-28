---
bug_id: BUG-004
status: open
severity: high
scope: engine/npu/generators/mm_kernel_reference.cc (AIE2P cascade hardware)
title: AIE2P cascade may be called only ONCE per launch (2nd call stalls at get_scd)
---

# BUG-004: AIE2P cascade is single-pass-per-launch

## Symptom

A second `cascade_reduce` call in the same core launch **stalls at
`get_scd`**. Isolated via an 8-core test with 2 D col-groups (2 drains):

- col-group 0 → `C[896]=688128` ✅ (the FIRST cascade pass works)
- col-group 1 → **untouched** (2nd cascade pass never returns)

## Reproduction

8-core cascade with `n_cg_d=2` (2 cascade_reduce calls per core). The 2nd
call hangs.

## Root cause

The AIE2P cascade stream cannot be re-armed within one launch — it's a
continuous one-shot stream. The 2nd `get_scd` never completes.

## Design consequence

The D GEMM MUST be **one cascade pass** over the full (8×N_D) partial. So the
per-core partial (h2_core @ B_d) must fit L1. For N_D=2048 that's 64 KB (too
big); N_D=128/256 fit. This is an architectural constraint, not a kernel bug.

## Status

OPEN (upstream hardware/toolchain). Worked around by the single-pass design —
but that collides with BUG-006/BUG-007 (the GU side), which is why the
single launch is blocked.
