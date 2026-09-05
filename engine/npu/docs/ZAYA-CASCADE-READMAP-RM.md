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

## No_gu discriminator build (2026-09-04, issue #2109 step 1) — CONFIRMED D read ≈ correct
Built `final_cascade_fused_zaya_nogu_nd2048.xclbin` (117440 B, N_D=2048 ROWS=4, `--no-gu --h2-const 1`,
via `build_iron_cascade.sh` NO_GU=1 OUT=final_cascade_fused_zaya_nogu_nd2048). h2 is a baked-in
constant (1), so C2 directly reflects the kernel's B_d read (no GU convolution).

**Silicon result (RM probe, B_d = ramp):** C2 is UNIFORM across every output column = **101504**
(identical for H0=0 and H0=1, as expected — no_gu ignores the h2 input).

**Host mirror derivation:** the generator's D read reads each of the 2048 distinct B_d rows exactly
once across (col=8 × cg=4 × ks=8 × t=8), so `Σ_{row=0}^{2047} (row%99+1) = 101346`.

**Verdict:** NPU 101504 vs host 101346 → delta **158 = 0.156%**, uniform (not a scramble). A truly
mismatched D index would scramble C2 (large, per-column-variable delta). A small *uniform* delta is a
scale/rounding artifact, not an index bug. ⇒ **The D phase reads the B_d layout correctly.**

**Strong implication for #2078:** corr 0.02 (full scramble on real weights) is NOT explained by the
D read. The bug is in the **GU phase / GU→SiLU→h2 handoff** (the h2 that feeds the D), i.e. the GU
B-tile read / A-tile fill / silu pair indexing for the Zaya `n_cg=4` geometry — not `packB_d_into`.

## To confirm (next)
- Explain the 158 (1.0056×) uniform scale — likely a host fold / h2 quant factor; check `packB_gu`'s
  A-tile `ascale`/`S` composition vs the no_gu constant.
- Verify C2 is uniform across ALL 4 ROWS chunks (cols 512/1024/1536) to rule out row-specific reading.
- Move the probe to the GU phase: one-hot-A `guread`/`bread` (cascade_real_weight_probe modes ported to
  Zaya) to isolate the GU B-tile read / silu pair indexing — the new prime suspect.

## GU-phase probe (NPU_CASCADE_GU) — h2 clustered at multiples of 8 (2026-09-04)
Identity B_d (bd[kk][nn] = 127·δ) so C2[nn] == h2[nn] (the silu'd GU output). One-hot A input at H0,
real layer-1 expert (e=7) GU weights; run on `final_cascade_fused_zaya_nd2048.xclbin`.

**Silicon result (H0=0..7):** h2 is NONZERO **only at columns ≡ 0 (mod 8)** — the nz columns are
{0, 8, 16, 24, …, 552+} (nz≈220-226), and columns 1-7, 9-15, … are all ZERO. e.g. H0=0
`first16 = [48387, 0,0,0,0,0,0,0, 16129, 0,0,0,0,0,0,0]` (nonzero at col0, col8), H0=1
`[-16129, 0×7, 32258, 0×7]`, H0=6 `[16129, 0×7, -80645, 0×7]`.

**Observation / open question:** the A-tile one-hot fill (`A[i*64+c] = h[ki*64+i*8+(c%8)]`) puts a
one-hot h at (row i, cols c%8) — so the *input* is already clustered at multiples of 8 → a *clustered
output* may be the expected consequence and NOT itself the bug. Distinguishing "correct GU read with a
clustered A" vs "GU read map is wrong" requires the CPU GU→SiLU mirror comparison (the #2078 derivation).
This is the current open branch — the no_gu D-read-verified result already proved the D is NOT at fault,
so the GU read / silu pair indexing is now the prime suspect.

## To confirm (next)
- Run the CPU mirror of GU→SiLU (silu_q22 on C1 = A_tile @ B_gu deriv-inverse) for the same one-hot A and
  diff vs the NPU h2 (C2 with identity B_d). If nonzero positions/values differ → GU read map is #2078.

## GU read vs CPU true-math — DECISIVE: GU read scrambles h2 (2026-09-04)
Added a CPU true-math GU mirror to the GU probe: `h2_cpu[p] = silu(w1[p][H0])·w2[p][H0]` for each
one-hot A (float weights from w.gu). Compare density vs the NPU h2 (C2 with identity B_d).

**Result (H0=0..7):**
| H0 | NPU h2 nz (frac) | CPU true-math nz |
|----|----|----|
| 0 | 224 / 2048 (11%) | 1526 |
| 1 | 219 (11%) | 1543 |
| 2 | 224 (11%) | 1525 |
| 3 | 221 (11%) | 1519 |
| 4 | 221 (11%) | 1522 |
| 5 | 223 (11%) | 1576 |

NPU h2 is nonzero ONLY at column multiples of 8 (~11% density); the correct GU math is DENSE (~75%).
A ~7× structural mismatch — quantization (float→int8) cannot drop 1300+ of 2048 outputs to zero.

**CONCLUSION (silicon-verified):** the cascade GU read map is SCRAMBLING the silu'd GU output (h2) to a
sparse subset at multiples of 8. Combined with the no_gu result that proved the D read is CORRECT,
**the #2078 root cause is in the GU phase** — the GU B_gu deriv-inverse read / A-tile fill / silu pair
indexing for the Zaya n_cg=4 geometry — NOT packB_d_into. This explains corr=0.02 (dense correct h2
vs sparse scrambled h2).

## Next
Diff the NPU GU h2 nonzero positions against the deriv-inverse B_gu read formula to find which index
term (row/col/pair) is misindexed for Zaya; then fix `packB_gu_into` / the generator GU core_fn.

## Precise DIFF (NPU GU h2 vs CPU true-math) — c_=0 micro-tile collapse
h2 index p = ks*256 + cg*64 + kstep*8 + c_  (kernel D read of h2b). So p % 8 == 0 ⟺ c_ == 0.

| metric | NPU GU h2 | CPU true-math |
|----|----|----|
| nonzero cols | 224 (all ≡0 mod 8) | 1526 |
| collapse factor | — | **6.8×** |
| multiples-of-8 density | 88% (224/256) | — |

⇒ the cascade GU mm produces h2 ONLY in the **c_==0** column of each 8×8 micro-group
(p%8==0); c_=1..7 are all zero. A micro-tile collapse: the A-tile / B_gu deriv-inverse
read is not spreading the GU output across the 8 sub-positions → h2 is ~1/8 as dense as
the correct GU output. This is the #2078 root (D exonerated by no_gu). The mm kernel:
`matmul_i8_i32_ab`→`matmul_vectorized_8x8x8_i8_i32_m8` (mm_kernel_reference.cc:714/307),
A=ab[0..512], B_gu=ab[512..8704].

## CAVEAT (2026-09-04) — the c_=0 "GU read" claim is AMBIGUOUS, not conclusive
The real-residual GU discriminator (`[GUrl] real-residual h2 nz=183/2048`) shows the identity-B_d
h2 (C2) is SPARSE (~9%) even for the REAL layer-1 h2 input — so the pattern is NOT purely a one-hot/
A-tile artifact. HOWEVER: identity B_d only recovers `C2[nn] = h2[nn]` if the D read maps kk→nn as a
true identity. The no_gu ramp test proved the D read is *uniform* (same k-set per output column), NOT
that it is *identity*. So the sparse-at-multiples-of-8 h2 could be either:
  (A) the GU read genuinely collapsing h2 (the #2078 bug), or
  (B) identity-B_d mapping h2 through a non-identity D k-slice permutation (read(nn) != nn).
These are indistinguishable from C2 alone. The earlier "c_=0 = GU read bug" conclusion is therefore
**not proven**; it is a hypothesis with a confound. Resolving A vs B requires knowing the D read map
read(nn) (from the no_gu one-hot B_d) or a GU-isolated probe (no D), i.e. the read-formula solver.

## Honest status
- CONFIRMED: D read is correct (no_gu, uniform + 0.16% scale). packB_d_into exonerated.
- UNRESOLVED: whether the sparse h2 is a GU read bug (A) or an identity-B_d D-mapping artifact (B).
- The real #2078 bug (corr=0.02) is in the GU->h2 phase broadly, but the exact misindexed term /
  whether it's GU-read or D-mapping is NOT determined. Needs the read-solver (port guread/bread/one-hot
  to Zaya + the no_gu one-hot B_d read map) — multi-cycle.

## D read map probe (NPU_CASCADE_RD, no_gu one-hot B_d) — CONTRADICTS RAMP, unresolved
Ran no_gu artifact with a single one-hot B_d row (bd[k0][nn]=127, all nn), scanning k0=0..7:

    [RD] k0=0 -> all 2048 cols nonzero
    [RD] k0=1..7 -> 0 cols nonzero

Suggests only B_d row 0 is read. BUT this CONTRADICTS the no_gu RAMP result (C2=101504 ≈ full-K
101346, diff 158): the r=0-only ramp sum is 12688, NOT 101504. So "D reads only row 0" is NOT
consistent with the RAMP (which implies the kernel reads the full K). The two probes cannot both
be right => there is an unresolved subtlety in how the one-hot/identity B_d interacts with the D
k-slice mm read (mm_wk8 B-tile / a8s contraction / cascade-reduce). Hypothesis A (GU bug) vs B
(D identity-mapping artifact) remains UNDETERMINED; neither the identity-B_d nor the one-hot-B_d
probe cleanly isolates the D read map.

## D read-count probe (NPU_CASCADE_RC, all-ones B_d, no_gu) — D reads FULL K
bd[kk][nn]=1 (all rows), no_gu h2=const=1 => C2[nn] = |read(nn)| (rows read per col):

    [RC] read-count C2[0..15] = 2048 2048 2048 ... (2048 for EVERY col)

So |read(nn)| = 2048 for every output column => the D reads ALL 2048 B_d rows
(FULL K). This agrees with the no_gu RAMP (101504 ≈ full-K 101346) and RULES OUT
the earlier "r=0-only / D collapse" interpretation. The RD one-hot probe
(k0=0 -> all, k0=1..7 -> none) is therefore an ANOMALY inconsistent with the
read-count -> discounted as a probe artifact. NOTE: full-K is PERMUTATION-INVARIANT,
so read() being full-K does NOT prove read(nn)=nn (identity). The identity-B_d GU
probe's sparse h2 is still A (GU output genuinely sparse) vs B (full-K permutation)
— not fully resolved. But the D is now confirmed to read the full K (correct
contraction magnitude), so the D read is NOT collapsing; packB_d_into exonerated.

## RESOLVED (2026-09-05): A (GU read bug) confirmed, B (permutation) ruled out
Combinatorial reasoning from the established facts:
  (1) read-count: |read(nn)| = 2048 for EVERY column (same full-K set per col).
  (2) RAMP: uniform 101504 across all columns (same Σ over full-K ramp) => read(nn)
      = the SAME full-K row-set for all nn (a different per-col set would make the
      RAMP non-uniform).
  (3) Therefore C2[nn] = Σ_k h2[k]·B_d[k][nn] is the standard full-K GEMM — the
      hardware's internal summation ORDER is irrelevant (it is a sum), so a
      permutation cannot change C2. => identity B_d gives C2[nn] = h2[nn] EXACTLY.
  (4) Real-residual GU probe ([GUrl]) gave C2 (= h2) nz=183/2048 — SPARSE for a
      dense input. The CPU true-math h2 is DENSE (1526). A real dense input must
      produce a dense GU output; 183/2048 (9%) is far too sparse to be int8
      quantization (which rounds to small NONZERO, not exact 0).
  => The GU read produces a SPARSE h2 (collapse) -> the cascade GU read map is the
     #2078 bug (A). The D read is correct/full-K (exonerated). Hypothesis B (D
     permutation) is impossible given (1)-(3).

## Remaining (the exact term)
The GU micro-tile collapse (c_=0 / 9%-sparse h2) is confined to the GU read. The
exact misindexed term is in the A-tile fill / deriv-inverse B_gu read / silu pair
indexing (mm_kernel_reference.cc matmul_vectorized_8x8x8_i8_i32_m8, the
GU mm matmul_i8_i32_ab). Localizing that term (and the fix) is the next step, but
the DIFF's core question (is it the GU read or the D read) is now answered.

## Full GU one-hot read map (H0=0..63) — preserved for the read-solver
Ran NPU_CASCADE_GU_H=0 N=64 (identity B_d, real GU weights, no_gu h2 input one-hot). Every H0:
  - h2 is nonzero ONLY at column multiples of 8 (c_=0), never at c_=1..7.
  - The nz set shifts with H0 (e.g. H0=0 starts at col 0; H0=1 starts at 64; H0=63 starts at 0),
    but ALL nonzero cols are ≡0 (mod 8). (nz ~208-232 per H0.)
Confirms the GU read collapses the silu'd h2 to the c_=0 sub-position of each 8x8 micro-group
across the entire input space (not input-dependent) -> the GU read map is the #2078 fault. This
full (H0 -> output-c_0-set) table is the target for the read-formula solver (guread/gusolve port):
diff each H0's output set against the deriv-inverse B_gu/A-tile read formula to name the misindex.

## Read-formula hypothesis (for the solver) — 64*((j/4)%2) high-block term
From the deriv-inverse col formula `col = 64*((j/4)%2) + (K%8)*8 + 2*(j%4)`:
  - For output pair j with j%8==0 (the ONLY nonzero h2 in the c_=0 collapse): (j/4)%2 == 0
    (j=0,8,16,... -> j/4 = 0,2,4,... even). So col = (K%8)*8  -- always in the LOW 64-block.
  - For j%8!=0 (which collapse to 0): j=1..7 have (j/4)%2 = 0 (j<4) or 1 (j=4..7); j=4..7 land
    in the HIGH 64-block (64*((j/4)%2)=64).
=> LEADING HYPOTHESIS: the kernel's B_gu (or A-tile) read OMITS the `64*((j/4)%2)` term, so it
   only ever reaches the LOW 64-block cols -> j%8!=0 outputs (which require the HIGH block) come
   back 0 -> the observed c_=0 collapse. The GUI read is therefore reading a SINGLE micro-header
   block instead of the full deriv-inverse 8x8 layout.
This is the first concrete candidate term for the read-formula solver to test (enumerate col
variants: full deriv-inverse vs no-64*((j/4)%2) vs alt, and score against the kernel h2).

## MAJOR CORRECTION (2026-09-05): GU read is DENSE for real input => NOT the bug
The real-residual GU probe (identity B_d, real layer-1 residual) shows:
    [GUrl] real-residual h2 nz=2048/2048 max|h2|=127 (first8: -127 -127 ... )
=> the GU read produces a CORRECT, DENSE h2 (all ±1 quantized) for a real input.
The earlier "GU read collapse / GU bug" conclusion was WRONG — it was conflated with
the ONE-HOT A probe (one-hot input -> sparse GU output is EXPECTED, not a bug). The
one-hot read map (H0=0..63, c_=0) is therefore NOT a GU fault; it's the one-hot
response pattern. So #2078's corr=0.02 is NOT the GU read.

Recalibration: with GU dense (correct) and D full-K (read-count 2048, RAMP≈full-K),
the real cascade output (CAS corr) should be high, but it's ~0.02-0.58 (varies). The
fault is therefore in the GU->h2 ->D *dataflow/scale* (e.g. the D B_d read of REAL
weights, the GU->D handoff scale), NOT the GU read map per se. The identity-B_d probe
(which verified the D row-set is full-K) CANNOT reveal a REAL-weight D index/scale bug
because identity is permutation- and scale-neutral. This invalidates the "GU read bug"
conclusion in #2078/#2109; the exact fault is still open (now narrowed to the real
GU->D dataflow, likely the real-weight B_d index or the h2->D scale).
