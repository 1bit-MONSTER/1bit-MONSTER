# Zaya single-launch cascade — blind D read-map (RM) probe, silicon-verified

Worktree: ~/wt/zaya-m1   |   Artifact: `final_cascade_fused_zaya_nd2048.xclbin` (N_D=2048 ROWS=4, 243968 B)
Probe: `NPU_CASCADE_RM` mode added to `src/zaya_decode.cpp` E2 hook (env-gated).
Run: split path (`NPU_FUSED_SPLIT=1`) + `NPU_CASCADE_TEST=1 NPU_CASCADE_RM=1 NPU_CASCADE_RM_H=<h> NPU_CASCADE_RM_N=12`,
     `NPU_XCLBIN_DIR=.../xclbins`, XRT `LD_LIBRARY_PATH=/usr/local/xrt-runlist/lib`, `~/models/zaya1-8b-fresh.q4nx`.

## Method (blind, non-circular — no assumed Zaya read formula)
- `cas.packB_gu_into` with REAL layer-1 expert (e=7) gate/up weights (uniform per-tensor scale).
- B_d OVERWRITTEN with a row-identifying ramp: `bd[kk][nn] = (kk % 99) + 1` (constant across nn for a given row kk).
- h2 input (A-tile) = ONE-HOT at index H0 (`hh[H0]=1`, rest 0). `cas.go(hh, ...)`.
- Read raw C2 (`cas.bC2`); dump group leaders `C2[8q]` (cols were found constant in blocks of 8) + C2[0..15].

## Representative result (H0 = 0..11, group leaders C2[0],C2[8],...,C2[56]; each constant x8)
```
H0=0  groups: -651737 261462 34882 287096 161759 -531209 -76119 597479
H0=1  groups:  465563 2162449 -785397 -155213 -1384332 278925 -21661 127821
H0=2  groups:   32832 389264 -34014 -473862 -1249117 -308096 116676 601496
H0=3  groups:  206936 311991 768810 -323061 150839 -286655 -89888 -49956
H0=4  groups: -812204 -411821 -139689 342186 232904 -1277997 321085 102239
H0=5  groups:  346065 -565006 -776472 328591 451389 735917 992510 -390750
H0=6  groups:-1157413 686517 -1557686 587266 -688077 -336241 -1062962 -808773
H0=7  groups:  147748 12612 957389 -1110717 -972718 -600733 -222632 -355327
H0=8  groups:  333159 -234286 324117 -69390 -129540 715931 276271 -238975
H0=9  groups: -326470 -237862 69425 -761191 -1644813 -790736 163865 -306570
H0=10 groups: 1108754 162273 -1307108 -250004 546409 -464229 -602177 -1489216
H0=11 groups: 497397 -24123 -475533 -5734 52734 1083687 126091 -74692
```
(Each group leader is held identically by the following 7 columns: C2[8q..8q+7] all equal.)

## Interpretation so far
- With B_d constant-per-row, an IDEAL D contraction would give `C2[nn] = Σ_k h2[k]·(k%99+1)` — the
  SAME value for every nn (B_d row kk is column-independent). The NPU instead returns C2 constant in
  BLOCKS OF 8 (C2[8q..8q+7] equal) with DIFFERENT values per 8-block. ⇒ the kernel's D read of B_d
  partitions output columns into groups of 8, each group reading a DIFFERENT k-slice set. This is the
  per-core "own h2 k-slice (ki = cg*8 + col)" cascade structure, NOT a uniform row-major read.
- The blocks-of-8 value varies with H0, so the GU one-hot spreads (dense h2) and the D read for each
  8-block picks up a distinct combination — the read map is therefore NOT the host packer's
  `bd[kk*N_D + nn]` uniform assumption.

## Next (derivation, multi-day)
1. Mirror Zaya GU→silu on host (`h2s[n][cg*64+j] = sat8(silu_q22(C1[2j],C1[2j+1]))`) for the one-hot H0
   inputs → get exact h2.
2. For each 8-block leader, solve the k-set S(block) s.t. `Σ_{k∈S} h2[k]·(k%99+1) == C2[block]` → recover
   the kernel's B_d row index per output-column block.
3. Diff the derived B_d index map against `packB_d_into`'s `bd[kk*N_D+nn]` (= w3[nn*IM+kk]) to locate the
   index mismatch (#2078).
4. Fix generator/host index; rebuild `final_cascade_fused_zaya_nd2048.xclbin`; re-run the real-weight probe
   to close `corr ≥ 0.999` vs the decode reference.

## Context refs
- #2078 (open, the cascade numerics bug), #2080 (whole-layer ELF endgame = the real 8× fix).
- Generator: `engine/npu/generators/n1_core_fused_gu_silu_d_iron.py` (D core_fn: `ki = cg*n_aie_cols+col`,
  `a8s[kstep,c_] = h2b[ks, cg*64 + kstep*8 + c_]`).
- Host packer: `engine/fusion/zero_copy/npu_cascade_kernel.h` (`packB_d_into`: `bd[kk*N_D+nn] = w3[nn*IM+kk]`).
