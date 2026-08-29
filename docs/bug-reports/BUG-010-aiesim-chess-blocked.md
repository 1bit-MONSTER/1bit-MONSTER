---
bug_id: BUG-010
status: open
severity: low
scope: engine/npu/tests/aiesim (simulator tooling)
title: aiecc --aiesim requires chess/xbridge, which rejects the peano cascade object
---

# BUG-010: aie2p simulator blocked for the cascade design

## Symptom

Trying to run the single-launch design through the AIE simulator:

```
aiecc --aiesim
```

fails because `--aiesim` requires the **chess** toolchain, and chess rejects
the PEANO-built cascade object:

```
chess-linker: duplicate local symbol `.LBB0_2`
```

Chess also lacks the aie2p `get_scd`/`put_mcd` cascade intrinsics.

## Root cause

The aie2p cascade kernels are peano-only (the intrinsics live in
llvm-aie's `aie2p_streams.h`, not the Vitis `adf/stream/me` headers). The
aiesim path needs chess, which is incompatible.

## Impact

No cycle-accurate aie2p simulation of the fused cascade — all bring-up is
on-silicon (strixhalo, `/dev/accel0`), which is slow and hard to inspect.

## Workaround prospects

Split the peano kernel objects (avoid the duplicate `.LBB0_2` local wall), or
generate the sim package manually. Not done.

## Status

OPEN (sim blocked). On-silicon bring-up only.
