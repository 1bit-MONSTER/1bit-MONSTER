# P3 — HIP kernels and the full device forward (2026-09-18)

Companion to `docs/plans/prism-bonsai-27b-custom-build.md` §P3. Every number is tagged:
**[gate]** = correctness, timing-immune; **[busy-box]** = wall-clock with other lanes active
(load 1–20), relative only.

## Kernels (all in `kernels/`, wired into `CMakeLists.txt`)

| File | What | Gate |
|---|---|---|
| `prism_hadamard_fwht.hip` | signed, B∈{512..4096}, fp16/bf16/f32, forward+inverse | 19/19 device parity |
| `prism_gemv.hip` | warp-per-row GEMV, all three layouts | corr 1.000000000 |
| `prism_gemv_row4.hip` | 4 rows/warp (x reuse) | corr 1.0; +12–31% over warp-per-row |
| `prism_gemv_tile.hip` | warp-per-block, lane owns 4 consecutive elements, finite float4 x; PTQ1_0 via a 256×5 LDS trit table | corr 1.0; **Q1_0 23→80, PQ2_0 35→93, PTQ1_0 25→93 GB/s** |
| `prism_gdn.hip` | conv1d+silu+rolling state, gated-delta recurrence + gated RMSNorm, ssmo-out perm | 3/3 |
| `prism_attn.hip` | q/k RMSNorm + partial RoPE, flash gated-GQA decode | 4/4 |
| `prism_ops.hip` | l2norm, plain RMSNorm, g/β, residual, silu×up, dense f32 GEMV, q/gate split | 4/4 |

## Full device forward (`tests/prism/prism_forward_hip.hip`)

All 64 layers (48 GDN + 16 full-attn), per-layer conv/rec/KV state, then
output_norm → fwht → lm_head → argmax. `--predict N` generates greedily;
`--max-pos N` sizes the caches.

**[gate]** Matches the fork's own *generated* tokens **5/5 on all three packs**:
PTQ1_0 220/314/279/369/11751; Q1_0 and PQ2_0 2614/314/279/369/11751.

**[busy-box]** 16-token greedy decode: Q1_0 15.05, PTQ1_0 10.15, PQ2_0 12.20 tok/s.
The lane targets (≥42 / ≥27 / ≥22) require a quiet box and are **not claimed**.

Two tile-GEMV decompositions were tested and rejected: 8 rows/warp (78.0 vs 77.7 GB/s)
and per-block LDS x staging shared across the workgroup (74.3 vs 82.3); 4 rows/warp +
global float4 x is the best of the tested set. The gap to the ~201 GB/s triad ceiling is
not explained by x redundancy alone.

## Corrections the independent oracle caught

- `tests/prism/prism_layer0.cpp` and `dump_prism_layer0.py` omitted the folded **ssmo-out
  head permutation** and used the HF **(1+w)** norm: layer_out_l2 502 vs the
  fork-validated 9.93. Both fixed; C++ vs numpy now agree at 2.6e-05.
- The tile GEMV's first cut had the Q1_0 sign-byte offset wrong (`(l>>1)*2+2` instead of
  `2+(l>>1)`); the CPU oracle caught it (corr 0.0385 → 1.000000000).
- The fused full-attn `attn_q` output is per-head `[q | gate]`, not `[all q | all gate]`.

## Fail-closed (R15)

Both 1BP loaders expose `has_prism_transform()`; the generic CPU path and the HIP backend
refuse a folded pack explicitly, and `tests/prism/test_prism_failclosed.cpp` asserts the
per-pack flag (folded PTQ1_0 = 1, unfolded Q1_0/PQ2_0 = 0) inside `run_prism_tests.sh`.
