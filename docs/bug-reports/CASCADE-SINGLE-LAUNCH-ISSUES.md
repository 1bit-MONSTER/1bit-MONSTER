# Fused GU→SiLU→D Single-Launch Cascade — Issue Log

**Scope**: `engine/npu` fused GU(H2-RELAY) decode, issue #1775 / #1769, AIE2P + aie.iron API.

## Accountability note (2026-08-28)

The **premise that "cascade-reduce kernels give a zero-h2-DMA-copy single launch
with a performance win" is NOT met.** The cascade *mechanism* works and the
**core correctness bug is fixed and silicon-verified** (see BUG-005), but the
**end-to-end single-launch fused kernel does not run** — it is blocked by a
hard aie2p/iron FIFO deadlock (BUG-006) against the 2-input-DMA limit
(BUG-007). The production-correct path remains the proven p1/p2 **two-launch**
(h2 via DDR). No performance win exists until BUG-006/BUG-007 are resolved;
do not release or benchmark the single-launch design.

## Index

| ID | Title | Sev | Status |
|----|-------|-----|--------|
| [BUG-002](BUG-002-aie2p-float-silu-miscompile.md) | aie2p backend miscompiles the float silu loop → core faults/hangs | HIGH | open (upstream) |
| [BUG-003](BUG-003-aie2p-v4-mmul-cascade-miscompile.md) | aie2p backend miscompiles the V[4] register-array cascade mmul form | HIGH | open (upstream) |
| [BUG-004](BUG-004-cascade-single-pass-limit.md) | AIE2P cascade may be called only ONCE per launch (2nd call stalls at get_scd) | HIGH | open (upstream) |
| [BUG-005](BUG-005-kn-cross-distribution-flaw.md) | D GEMM distributed K AND N across cores but cascade only sums K (same column) | HIGH | **FIXED + silicon-verified** |
| [BUG-006](BUG-006-merged-fifo-count-deadlock.md) | iron ObjectFifo merged [A\|B] element deadlocks at count>1 (works count=1) | HIGH | open |
| [BUG-007](BUG-007-two-input-dma-limit.md) | AIE2P core has only 2 input DMA channels (A+B_gu+B_d=3 exceeds) | MED | workaround-invalidated by BUG-006 |
| [BUG-008](BUG-008-xrt-five-buffer-limit.md) | XRT MLIR_AIE kernel exposes only 5 data buffers (8 per-core AB fifos over) | MED | worked around |
| [BUG-009](BUG-009-core-l1-64k.md) | 64 KB core L1 overflows with wide B_d + c2scr + combined A\|B | MED | worked around |
| [BUG-010](BUG-010-aiesim-chess-blocked.md) | `aiecc --aiesim` needs chess, which rejects the peano cascade object (`.LBB0_2`) | LOW | open (sim blocked) |
| [BUG-011](BUG-011-premise-not-met.md) | the zero-h2-DMA + performance improvement premise is NOT met (no working single launch) | HIGH | open |

## What IS validated (the win that stands)
- **BUG-005 corrected D cascade**: at `N_D=128` (no_gu, h2=1, B_d=1) the
  corrected single-pass cascade writes `C2=2048` everywhere — the exact value.
- q22 fixed-point silu (BUG-002 workaround) + zero-h2-DMA h2 core-locality.

## Files touched
`engine/npu/generators/mm_kernel_reference.cc`,
`engine/npu/generators/n1_core_fused_gu_silu_d_iron.py`,
`engine/npu/generators/build_iron_cascade.sh`,
`engine/npu/tests/test_cascade_reduce.cpp`,
`research/ws01-npu-attention/FUSED-H2-RELAY-DESIGN.md`.
