---
bug_id: BUG-009
status: worked-around
severity: medium
scope: AIE2P core L1 memory (limits)
title: 64 KB core L1 overflows with wide B_d + c2scr + combined A|B
---

# BUG-009: 64 KB core L1 limit

## Symptom

`aiecc` basic-sequential allocation fails on the core tile:

```
'aie.tile' op allocated buffers exceeded available memory
Bd7_cons_buff_0    : 65536 bytes   (wide B_d element)
C2_tail_buff_0     : 32768 bytes
```

An `(8×N_D)` c2scr + `(64×N_D)` wide B_d element + `(8×K)` h2 legacy buffer
together exceed the 64 KB core L1.

## Workarounds (in repo)

1. **Shrink h2buf**: each core only holds its OWN 4 chunks
   (`(8, n_cg_gu·64)` = 2 KB) instead of the full `(8×K)` 16 KB. The D reads
   its own chunk at local `cg·(n//2)`.
2. **FIFO depth**: combined A|B `depth=1` (depth=3 = 26 KB overflows);
   wide B_d `depth=1`.
3. **N_D bounded**: `(8×N_D)` partial must fit — N_D=128 is the verified
   working size; N_D=256 hangs (also relates to BUG-004/BUG-006).

## Status

WORKED-AROUND. Bound: single-pass D limited to small N_D by L1.
