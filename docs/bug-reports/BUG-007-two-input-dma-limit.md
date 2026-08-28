---
bug_id: BUG-007
status: open
severity: medium
scope: AIE2P NPU hardware (limits)
title: AIE2P core tile has only TWO input DMA channels (A+B_gu+B_d=3 exceeds)
---

# BUG-007: AIE2P 2-input-DMA limit on the fused worker

## Symptom

The fused single-launch worker reads three input streams per core:
`x(A)` (GU), `B_gu` (GU), `B_d` (D). Placement fails:

```
tile (0, 2) requires 3 input/0 output DMA channels, but only 2 input/2 output available
```

Confirmed under every allocation scheme (`basic-sequential`, default,
`--dynamic-objFifos` / not).

## Root cause

An AIE2P core tile exposes `DMA_MM2S_0` and `DMA_MM2S_1` only — 2 input DMA
channels. The fused pipeline needs 3 (unless some stream is core-local).

## Workaround attempted (does NOT resolve the launch)

Pack `A` and `B_gu` into ONE merged `[A|B]` channel → 2 channels. But that
merged channel deadlocks (BUG-006). So the workaround is invalidated.

## Status

OPEN (hardware limit). The single launch needs a 2-channel dataflow in which
both the GU and the D work — currently impossible given BUG-006 + the
one-cascade-pass limit (BUG-004). **p1/p2 two-launch (h2 via DDR, ≤2 channels
per launch) remains the production-correct design.**
