---
bug_id: BUG-011
status: open
severity: high
scope: engine/npu (requirement/acceptance)
title: "zero h2 DMA copy + performance improvement" single-launch premise is NOT met
---

# BUG-011: the zero-DMA / performance premise is not delivered

## What was promised vs delivered

The premise (advanced by this work, issue #1775): cascade-reduce kernels give
a **single-launch fused GU→SiLU→D with ZERO h2 DDR round-trip** and hence a
**performance improvement** over the p1/p2 two-launch.

**Current reality**:

| Claim | Status |
|-------|--------|
| Zero h2 DDR round-trip (h2 core-local) | ⚠️ architecture designed, but **no working end-to-end kernel** |
| Performance improvement | ❌ **no evidence** — no working single launch to benchmark |
| Single-launch fused decode | ❌ **NOT working** (BUG-006 + BUG-007 + BUG-004) |

## Why it fails

1. AIE2P gives only **2 input DMA channels** (BUG-007); the fused design needs
   3 (A, B_gu, B_d).
2. Packing A+B_gu into one merged channel **deadlocks at count>1** (BUG-006).
3. The cascade is **single-pass-per-launch** (BUG-004), so the D partial must
   fit L1 (BUG-009), bounding N_D and blocking large-N decode.

## What DOES stand

- `BUG-005` corrected D cascade: **silicon-verified exact at N_D=128**.
- `BUG-002` q22 silu workaround; zero-DMA **h2 core-locality** is designed.

## Accept/reject

**Accept** the bug-fix (BUG-005) and the d2 issues (002/003/004). **Reject**
the single-launch zero-DMA + performance premise until BUG-006/BUG-007 are
resolved. **Do not benchmark or ship** the single-launch design; the
production-correct path remains **p1/p2 two-launch (h2 via DDR)**.

## Recommended follow-up

- Resource BUG-006/BUG-007 as a dedicated effort (2-channel dataflow or a
  pipelined merged FIFO).
- Measure the actual win only after a working single launch exists; the
  h2-DDR round-trip may be small relative to the fused compute throughput, so
  the "performance improvement" claim should be re-validated on hardware, not
  assumed.
