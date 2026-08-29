---
bug_id: BUG-002
status: open
severity: high
scope: engine/npu/generators/mm_kernel_reference.cc (aie2p backend)
title: aie2p backend miscompiles the float silu loop — core faults/hangs
---

# BUG-002: aie2p backend miscompiles the float silu loop

## Symptom

The fused GU→SiLU step hangs/faults the core whenever the *float* silu
(`silu_quant_i8_fused`, a `silu_lut(float)` loop) runs for a cg>0. Silicon
bisect is unambiguous:

- `h2s = 127` set directly (no silu kernel) → **WORKS** (c1b=2048).
- with `silu_quant_i8_fused` → **HANGS**.

## Reproduction

Run the single-launch fused decode with `silu_quant_i8_fused` (float LUT
silu). The core never completes the GU phase.

## Root cause

The aie2p backend mis-compiles the float `silu_lut` loop (issue #1836 / #1844
family). This is a toolchain bug, not a design bug. The prior "2-cg GU hangs"
and "gs-tile acquire" theories were red herrings.

## Fix / workaround (in repo)

`silu_quant_i8_fused_q22` — a **pure-int32 fixed-point silu** using the
`silu_sigmoid_q22` LUT (gate clamped to [-4,4], `silu = (g·sig)>>22`, `h =
sat8(silu·u)`). CPU-verified h2=127 = float reference. This avoids the float
miscompile but is a **workaround**, not a fix to the backend. Track upstream
#1836.

## Status

OPEN (upstream toolchain). Workaround `silu_quant_i8_fused_q22` shipped.
