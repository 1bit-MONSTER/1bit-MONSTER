---
bug_id: BUG-003
status: open
severity: high
scope: engine/npu/generators/mm_kernel_reference.cc (aie2p backend)
title: aie2p backend miscompiles the V[4] register-array cascade mmul form
---

# BUG-003: aie2p backend miscompiles the V[4] register-array cascade mmul form

## Symptom

The original cascade GEMM form — `int32 V[4]` register array + 4×-mmul
j-group + nested inner get_scd/put_mcd — produces a **deterministic** wrong
result on silicon:

- blocks 4–15 **unwritten**,
- block 2 **row-corrupted**.

## Reproduction

Run the cascade`_d_first/mid/last_i8_i32` kernel in that register-array form
against a CPU reference; the C2 output mismatches in the block pattern above
(deterministic, not noise).

## Root cause

The aie2p backend mis-compiles the `V[4]` register-array + nested
get/put_mcd form. Confirmed on the PEANO toolchain (`--no-xchesscc`); the
aie2p cascade intrinsics (`get_scd`/`put_mcd`) are peano-only anyway.

## Fix / workaround (in repo)

Use the **single-pass form**: no `V[4]` array, A-pointer reset per block,
**one mmul per block**, no nested inner get/put. This form is
**silicon-correct**. (`cascade_reduce_first/mid/last_i32` is the workaround.)

## Status

OPEN (upstream toolchain). Correct form shipped in `cascade_reduce_*_i32`.
