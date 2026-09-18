# Prism GEMV sweep (Q1_0, gfx1151) — evidence for "which kernel stands"

All rows: `Bonsai-27B-Q1_0.1bp` `blk.0.ffn_gate.weight` (17408x5120, 18 B/128 = 12 MB payload) on
gfx1151. `corr` is against `prism::dequant_flat` + an f64 CPU dot. Timings were taken on a shared box
with peer lanes active, so the GB/s are **relative to each other**, not headline numbers (plan risk R16).

| variant | where | GB/s | corr | note |
|---|---|---|---|---|
| warp-per-row (v3) | `kernels/prism_gemv.hip` | 23 | 1.0 | one warp per row, no x reuse |
| 4 rows/warp (row4) | `kernels/prism_gemv_row4.hip` | 29 | 1.0 | x reused across 4 rows |
| tile4, software fp16 | `kernels/prism_gemv_tile.hip` (pre-h2f) | 80–93 | 1.0 | warp-per-block, lane owns 4 consecutive elements, global float4 x |
| tile8 | sweep | 78 | 1.0 | 8 rows/warp: no gain over 4 (register pressure) |
| per-block LDS x | sweep | 74 | 1.0 | share one 128-wide x block across the workgroup; sync cost |
| 1024-element LDS x super-block | sweep | 68 | 1.0 | fewer syncs, but occupancy loss |
| conflict-free strided LDS | sweep | 42 | 1.0 | lane owns l, l+32, l+64, l+96; more byte reads |
| **tile4 + hardware `__half2float`** | `kernels/prism_gemv_tile.hip` | **133.5** | 1.0 | **the fix that stands** (was 79.9) |

## The correction that matters (supersedes the "93.6 GB/s pattern cap")

A dummy kernel with the *identical* full-byte weight loads plus the fp16 decode but no bit-extraction/FMA
measured 93.6 GB/s and was read as "the pattern ceiling". That reading is wrong: the dummy still used the
**software** fp16 decode (a ~10-op bit loop, 4x per lane per block), so it was ALU-limited, not
pattern-limited. After switching the real kernel to hardware `__half2float` it reaches **133.5 GB/s — 42%
above** that supposed cap. So the current limit is decode/ALU and non-GEMV overhead, **not** the weight
access pattern.

## End-to-end backend after the fix

`tools/bench_hip_1bp <pack>.1bp 32 4`, 32 greedy tokens, quietest window observed (load 5.7, no process
above 300% CPU):

| pack | tok/s | gate |
|---|---|---|
| Bonsai-27B-Q1_0 (3.80 GB) | 24 | ≥42 |
| Ternary-Bonsai-2-27B-PTQ1_0 (5.95 GB) | 19 | ≥27 |
| Ternary-Bonsai-27B-PQ2_0 (7.17 GB) | 19 | ≥22 |

That is up from 16 / 11 / 12 before the fix; PQ2_0 is at 86% of its gate. A triad reading in the same
window is still pending, so these are tagged `strixhalo-unknown`, not `strixhalo-quiet`, in the results of
record.

## Reproduce

```
BENCH=/opt/rocm-therock/bin/hipcc
$BENCH --offload-arch=gfx1151 -O3 -std=c++17 -I include -I src \
    tests/prism/bench_gemv_lds.hip src/onebp_model.cpp -o /tmp/bgemv
/tmp/bgemv <pack>.1bp blk.0.ffn_gate.weight 100
```
tile4 is the production kernel (`kernels/prism_gemv_tile.hip`); the other variants were exploratory and
are not kept as files. `corr` must read `1.000000` for a variant to be eligible.
