---
bug_id: BUG-006
status: open
severity: high
scope: aie.iron ObjectFifo (merged-element stream)
title: iron ObjectFifo merged [A|B] input element deadlocks at count>1
---

# BUG-006: merged [A|B] input FIFO deadlocks at count>1

## Symptom

A single combined `[A|B]` input FIFO (element = A-tile 8×64 followed by
B_gu-tile 64×128, 8704 int8 bytes) consumed by ONE core:

- **count=1 → works** (c1 written correctly),
- **count≥4 → DEADLOCKS** (c1 untouched / C2 stays 0x5A).

Tested at every FIFO depth (1, 2, 3) and both element shapes
(`(8704,)` 1-D and `(64,136)` 2-D) — the deadlock is count/size independent of
shape and depth.

## Reproduction

Single-core probe: acquire the merged element N times (N≥4), run a trivial
copy kernel, release. Works at N=1, hangs at N≥4. By contrast the D's per-core
2-D `(64,N_D)` B_d FIFO pipelines fine at count=4.

## Root cause

The aie.iron ObjectFifo does **not pipeline a merged single-element stream** at
count>1 — the acquire/release handshake stalls. This is an iron-FIFO/dataflow
limitation specific to the merged element (the working p1 pattern uses
separate 2-D A + B_gu FIFOs at depth=BATCH+1, which DO pipeline).

## Why this blocks the single launch

The 2-input-DMA limit (BUG-007) forces packing A + B_gu into ONE merged
channel. That merged channel deadlocks. So the single-launch fused design has
no working 2-channel dataflow today.

## Status

OPEN. Required to unblock the single launch: either (a) a 2-channel dataflow
where the D's B_d is multiplexed over a GU channel, or (b) an iron FIFO
primitive that pipelines a merged element. Both are real engineering.
