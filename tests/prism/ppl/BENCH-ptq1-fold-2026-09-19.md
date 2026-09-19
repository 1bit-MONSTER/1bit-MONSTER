# PTQ1_0 decode-speed benchmark: does the fold-group fix regress speed? — 2026-09-19

Goal `mu8oyo0n`. The fold-group correction changes the PTQ1_0 forward; this measures whether it
materially regresses PTQ1_0 **decode** speed. Pre-fix = `PRISM_LEGACY_FOLD_GROUP=1`; post-fix =
default. Q1_0 and PQ2_0 are controls (the fix is a no-op for them, `has_transform=false`).

## Instrument and window

* `tests/prism/prism_forward_hip.hip` — thin CLI over `include/prism_engine.h`, the real 27B HIP
  forward — built from HEAD `c76daa9e9`. Its timer covers only the generated tokens (decode).
* One window: 5 reps × {default, legacy} × {PTQ1_0, Q1_0, PQ2_0}, **128 generated tokens** per run,
  prompt = first 32 ids of `tests/prism/ppl/slice200.ids.txt`; a `hip_bw_probe` triad brackets it.
* Materiality threshold: **>2% slower = material**.

## Result (tok/s, mean of 5; ms/token derived)

| pack | post-fix (default) | pre-fix (legacy) | Δ tok/s | Δ % | ms/token post / pre |
|---|---:|---:|---:|---:|---:|
| Ternary-Bonsai-2-27B-PTQ1_0 | 22.704 | 22.432 | +0.272 | +1.2% | 44.05 / 44.58 |
| Bonsai-27B-Q1_0 (control) | 31.396 | 31.380 | +0.016 | +0.05% | 31.85 / 31.87 |
| Ternary-Bonsai-27B-PQ2_0 (control) | 20.558 | 20.402 | +0.156 | +0.8% | 48.64 / 49.02 |

Post-fix column tag: `[<pack> | <format> | HIP PrismEngine (default = fold fix) | strixhalo-unknown | 128 | slice200 ids[0:32] | 2026-09-19]`.
Pre-fix column tag: `[<pack> | <format> | HIP PrismEngine (legacy = pre-fix) | strixhalo-unknown | 128 | slice200 ids[0:32] | 2026-09-19]`.

PTQ1_0 pre-fix had one `21.54` tok/s outlier; excluding it the pre-fix mean is **22.655** → post-fix
**+0.2%**; by median (pre-fix 22.59) **+0.5%**. Either way the post-fix is not slower.

## Verdict

**The fold-group fix does NOT materially regress PTQ1_0 decode speed.** Post-fix PTQ1_0 is
equal-to-slightly-faster than pre-fix (+0.2% to +1.2% depending on outlier handling), far inside the
2% materiality bar. The controls are identical within noise (+0.05% Q1_0, +0.8% PQ2_0), as expected.
So the fix that closes the +14.9% PPL gap costs no decode throughput.

## Window classification (quiet vs relative-only)

Triad (`hip_bw_probe`, 128/256/512/1024 MB per array): **start 203.3–219.7 GB/s**,
**end 197.7–209.8 GB/s**. The end 1024 MB triad is **197.7 GB/s (< 200)**, so the window is **not
clean-quiet** and every row above is tagged **`strixhalo-unknown`**: only a RELATIVE
pre-fix-vs-post-fix claim is made, and **no absolute throughput claim is asserted**. Triad rows:
`[n/a | probe | HIP hip_bw_probe triad | strixhalo-unknown | - | - | 2026-09-19]`.

## Provenance

```bash
# build from HEAD c76daa9e9
/opt/rocm-therock/bin/hipcc --offload-arch=gfx1151 -O3 -std=c++17 -I include -I src \
  tests/prism/prism_forward_hip.hip kernels/prism_hadamard_fwht.hip kernels/prism_gemv.hip \
  kernels/prism_gemv_row4.hip kernels/prism_gemv_tile.hip kernels/prism_gemv_dp4a.hip \
  kernels/prism_gdn.hip kernels/prism_attn.hip kernels/prism_ops.hip src/onebp_model.cpp -o /tmp/pfhip
/opt/rocm-therock/bin/hipcc --offload-arch=gfx1151 -O3 -std=c++17 tests/hip_bw_probe.cu -o /tmp/hip_bw_probe
# run (single window)
/tmp/pfhip <pack.1bp> <first 32 ids of slice200.ids.txt> --predict 128           # post-fix
PRISM_LEGACY_FOLD_GROUP=1 /tmp/pfhip <pack.1bp> <first 32 ids> --predict 128     # pre-fix
```

Raw log: `tests/prism/ppl/bench_full.log`. Runner: `tests/prism/ppl/run_bench.sh`.
