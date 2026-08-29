---
bug_id: BUG-008
status: worked-around
severity: medium
scope: XRT / aie.iron Runtime (buffer count)
title: XRT MLIR_AIE kernel exposes only FIVE data buffers (groups 3-7)
---

# BUG-008: XRT 5-data-buffer limit

## Symptom

The design originally declared 10 sequence buffers (8 per-core AB + C2 + B_d).
Creating them overflows the XRT kernel's argument groups:

```
vector::_M_range_check: __n (>=) this->size()  (group_id(8+) out of range)
```

`xrt::kernel.group_id(i)` exposes only indices 0..7 (groups 1=insts, 3..7
= the five data buffers; group 0/2 are invalid sentinels `131071`).

## Root cause

The `MLIR_AIE` kernel on the NPU supports at most ~5 data buffers.

## Workaround (in repo)

Merge the 8 per-core AB streams into ONE buffer laid out `[core][ki][cg]`,
each core's fill tapping its own region. Sequence = `(AB, C2, B_d)` = 3
groups. In the test harness, `kern.group_id(3/4/5)` = AB/C2/B_d.

## Status

WORKED-AROUND. Records that per-core multi-buffer designs must fit 5 buffers.
