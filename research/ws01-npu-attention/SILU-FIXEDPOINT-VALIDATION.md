# Fixed-point SiLU validation — fused GU→SiLU→D kernel

**Branch:** `feat/npu-zaya-cca-cpu-port` · **Date:** 2026-08-27 · **Status:** ✅ VALIDATED

Validation of the fixed-point SiLU numerics for the fused GU→SiLU→D kernel
(issue #1759 / FLM-PARITY-PLAN §"Fused GU→SiLU→D design", the on-NPU MoE
path of the Zaya CCA port). The fused kernel computes on the NPU:

```
gate_f = C1[2p]·gs'[2p]     up_f·qn_s = C1[2p+1]·gs'[2p+1]   (gs' host-folded)
h2[p]  = sat8(round(silu(gate_f)·up_f·qn_s))   silu(x) = x·sigmoid_LUT(x)
C2[j]  = Σ_p h2[p]·B_d[p][j]                    (D GEMM, h2 = its int8 A)
```

Since D re-quantizes to int8 anyway, the bar is **not** bit-exactness: after
int8 quantization the kernel h2 must match the float reference to ≤1 LSB for
~all pairs (corr ≥ 0.999, ≥98% within ±1, max |dH2| ≤ 8 — the C++ gate's own
criteria).

**Result: PASS.** The on-core contract is validated on synthetic realistic
data AND on the real Zaya1-8B weights. The fixed-point SiLU is tractable and
numerically sound; nothing blocks folding into `feat/npu-1776-attn-seq`
(which, in fact, already contains the implementation — see §5).

---

## 1. What was validated

Two on-NPU SiLU contracts from `engine/npu/generators/silu_quant.h`
(@ `origin/main` `3ced5bdb`, merged via #1892), emulated **bit-for-bit**:

| Path | Arithmetic | Where |
|---|---|---|
| `silu_quant_i8` / `silu_lut` | float32, 256-entry sigmoid LUT over [-4,4] | CPU-side reference of the fused kernel |
| `silu_pair_q22` | **pure int32** (Q22 fixed point), `silu_sigmoid_q22[256]` LUT | the on-NPU arithmetic (aie2p-safe) |

plus the **host fold math** (`write_silu_pad_meta` in `gu_i4_pack.h`):
per-64-pair-tile `Q = 22 − clamp(15+ceil(log2 min|S'|))`, `fold = round(S'·2^Q)`,
`boundG = (2^31−1)/|fold|`, `boundU = 4·boundG+3`, `shG/shU = Q−11/Q−7`.

Compared against three references:
1. **float-LUT** — the contract's own reference (what the C++ gate gates).
2. **TRUE float** — exact `exp` sigmoid, float64 (the strictest bar; the
   branch's `silu_fixedpoint_explore.py` target).
3. **full float FFN** — dequantized GU→SiLU→D vs full-precision float FFN
   (the end-to-end bar the D GEMM output must meet).

## 2. Method & artifacts

- `silu_fixedpoint_validate.py` (this repo, new) — Python emulation of both
  contracts + host fold math. **Bit-exactness verified against the real C++
  header** via a cross-check driver (`/tmp` build of `silu_quant.h` +
  `gu_i4_pack.h` @ `3ced5bdb`): 0/2000 pair mismatches on fold params
  (Q/shG/shU/fold/bounds) and h2, on both raw and qn_s-folded inputs.
- C++ gates built from main and run locally (x86, no NPU needed):
  `test_i4_silu_q22.cpp` (Q22 contract gate) and `test_i4_grouped_fused.cpp`
  (full FFN gate), on `zaya1-8b.q4nx` (5.58 GB, local copy).
- Distributions: the C++ gate's realistic synthetic envelope
  (g ~ N(0,1.2) clamp ±8, u ~ ±10^U(−0.5,2.8), S' log-uniform [1e-5.5, 1e-1.5],
  c1 = g/S'), adversarial corners (the old v50/v51 overflow class), and a
  wide-gate stress (N(0,2.5) — 2× the measured gate width).

## 3. Numeric results

### 3a. Python — realistic synthetic envelope (N=60000), raw contract (no qn_s)

| comparison | corr | within ±1 LSB | worst \|dH2\| | zero(ref≠0→kern=0) |
|---|---:|---:|---:|---:|
| Q22-int32 vs TRUE float | 0.999967 | 99.32% | 24 | 79 |
| float-LUT vs TRUE float | 0.999982 | 99.91% | 25 | 12 |
| **Q22-int32 vs float-LUT** | **0.999971** | **99.09%** | **6** | 76 |

The `Q22 vs float-LUT` row is the C++ gate's contract bar → **PASS**.
The two "vs TRUE float" rows are informational: without qn_s normalization
the synthetic u reaches ~630, amplifying the sigmoid-LUT's absolute range
error (≤ ~2e-3·|g|·|u|) into several raw units. The real kernel always
folds qn_s first (§3c). Adversarial corners (N=2000): Q22-vs-floatLUT
corr=0.999963, 98.10% within ±1, worst 4 → **PASS**.

### 3b. C++ gates (authoritative, built from main @ 3ced5bdb)

`test_i4_silu_q22.cpp` — Q22 contract vs float-LUT reference:

| set | corr | within ±1 | worst \|dH2\| | result |
|---|---:|---:|---:|---|
| synthetic-v59 (60000) | 0.999970 | 99.08% | 5 | PASS |
| adversarial-u600 (2000) | 0.999962 | 98.20% | 4 | PASS |
| **real-zaya L1/E0** (zaya1-8b.q4nx) | 0.999862 | 99.90% | 2 | PASS |
| **real-zaya L1/E5** | 0.999856 | 100.00% | 1 | PASS |
| **real-zaya L3/E0** | 0.999832 | 100.00% | 1 | PASS |
| **real-zaya L5/E7** | 0.999738 | 100.00% | 1 | PASS |

The real-data runs are the operative validation: **on the actual Zaya
weights the fixed-point Q22 SiLU matches the float reference to ≥99.9%
within 1 int8 LSB, worst deviation 2 LSB, corr ≥ 0.9997.** (Real-data
envelope measured by the gate: H=2048, n_ff=2048, 16 experts, ag≈4.2,
per-token max|h2f|=1.79 → qn_s≈70.8.)

`test_i4_grouped_fused.cpp` — full GU→SiLU→D FFN vs full-precision float FFN:

| variant (layer/expert) | FFN corr | GU corr | h2 corr | result |
|---|---:|---:|---:|---|
| A int8 per-section L1/E0 (two-launch path) | 0.999562 | 0.999929 | 0.999852 | PASS |
| **D int4 on-chip dequant L1/E0 (fused path)** | **0.999640** | **0.999962** | **0.999926** | PASS |
| E packer roundtrip L1/E0 | 0.999643 | 0.999960 | 0.999905 | PASS |
| D int4 on-chip dequant L3/E0 | 0.999705 | 0.999962 | 0.999925 | PASS |
| D int4 on-chip dequant L1/E5 | 0.999577 | 0.999962 | 0.999905 | PASS |

Every variant passes the ≥0.999 FFN gate on real weights; the fused
on-chip-dequant path (RAW Q4NX + Q22 silu, half the GU bytes) matches or
beats the int8 two-launch path. This confirms the header's claim
("corr 0.9993–0.9996 vs float on zaya1-8b.q4nx").

### 3c. Python ⟷ C++ cross-validation

The Python emulation reproduces the C++ gate's numbers nearly exactly
(independent PRNGs):

| set | C++ | Python |
|---|---:|---:|
| synthetic Q22-vs-floatLUT | corr 0.999970, 99.08%, worst 5 | corr 0.999971, 99.09%, worst 6 |
| corners Q22-vs-floatLUT | corr 0.999962, 98.20%, worst 4 | corr 0.999963, 98.10%, worst 4 |

### 3d. Branch's float-space exploration (`silu_fixedpoint_explore.py`)

Run to verify the committed claims (they hold): a 256-entry sigmoid LUT is
99.999–100% within 1 LSB for the swept gate ranges; the cubic
`0.5+0.25x−x³/48` hits 99.997% at the small-gate scale. Note the explore
script used XLUT=8 with a truncated index, so its LUT numbers are a
*concept* check, not the merged contract — §3a/b are the contract-accurate
versions (XLUT=4, `silu_roundf` index, int32 Q22).

## 4. Findings

1. **VALIDATED — the fixed-point SiLU is numerically sound for the fused
   kernel.** Q22-vs-float corr ≥ 0.9997 on real weights, ≥99.9% within 1
   int8 LSB, worst |dH2| = 2, FFN corr ≥ 0.9996 end-to-end. The fused
   kernel's numerics are de-risked; the milestone is CPU-gated and green.

2. **XLUT=4 clamp fits the measured gate range with margin.** Real gate_f
   spans [-3.4, 3.4] → 0% beyond the ±4 clamp. Wide-gate stress (N(0,2.5),
   10.5% beyond ±4) degrades vs-TRUE corr to ~0.999 but leaves the
   Q22-vs-floatLUT arithmetic exact (worst 4) — the fixed-point math itself
   never loses precision; only the LUT's finite range matters, and the
   measured range fits. (The branch's explore sweep reached gate_f ±9.4 only
   at an unrealistic S; the real model's scales keep gates in [-3.4,3.4].)

3. **NEW FINDING — latent contract hazard: `boundU` int32 overflow when
   `foldG < 8`.** `boundU = 4·boundG+3` wraps **negative** for `boundG ≥ 2^29`
   ⟺ `|fold| ≤ 7` ⟺ `|S'|·2^Q < 8`. In that regime the |c1u| clamp
   misbehaves and h2 flips to ±127 (corr collapse to 0.35 — demonstrated in
   the Python hazard probe, §3a folded row). **Reachability:** needs a tile
   column with `|S'|·2^Q < 8`, i.e. a per-token qn_s small enough to push the
   folded up-scale tail below ~2^-16. On the real model: qn_s = 70.8 *folds
   up* (max|h2f| = 1.79), su' ~ O(1), real folds ≥ ~164 → **~3 orders of
   margin**; the passing real-data gate (worst |dH2| = 2) empirically
   confirms no overflow on the actual weights. **No kernel change needed**;
   recommend documenting the invariant (`fold ≥ 8` ⟺ per-token
   `max|h2f| < 127·ag·scol_u·2^Q/8`) in `silu_quant.h`. If ever hardened,
   the fix is a cheap int32 saturate of `boundU` at 2^31−1 (not required,
   and not to be done speculatively — rule: no unverified kernel edits).

4. **The old failure classes are gone.** Zero-pairs (ref≠0 → kernel=0):
   76/60000 synthetic, 2/2000 corners, 6/2048 real — the "host h2=12 → NPU 0"
   v50/v51 wrap is fixed; small-gate/huge-up corners hold worst |dH2| = 4.

## 5. Recommendation — fold into `feat/npu-1776-attn-seq`

**YES — fold it, and note the fold has effectively already happened.**
`origin/feat/npu-1776-attn-seq` already contains this branch's research
files (`silu_fixedpoint_explore.py`, `ZAYA-CCA-CPU-PORT.md`, `cpu_ref.cpp`)
and the full fused GU→SiLU→D implementation + CPU gates (`silu_quant.h`,
`gu_i4_pack.h`, `test_i4_silu_q22.cpp`, `test_i4_grouped_fused.cpp`,
`test_fused_silu.cpp`); `main` merged that via **#1892**. The validation in
this report independently confirms those gates pass: the Q22 contract was
re-implemented bit-exact in Python and cross-checked against the C++ header,
and both gates were built and run on the real `zaya1-8b.q4nx` weights.

What this branch still adds (research/documentation only, no kernel
changes): this report + `silu_fixedpoint_validate.py` (the bit-exact
emulation, reproducible without the C++ gate) + the `boundU`-overflow
finding. Recommend folding these into the fold target as documentation.

## 6. What still needs NPU verification (strixhalo)

The SiLU contract is CPU-gated; the on-NPU round trip was already verified
in main's history (the #1776/#1892 milestone). For a fresh hardware pass on
strixhalo (RyzenAI NPU5, Vitis 2026.1 AIE tools):

```bash
# per-branch clone on strixhalo (repo syncs from origin)
git clone --branch feat/npu-1776-attn-seq https://github.com/1bit-MONSTER/1bit-MONSTER ~/1bit-1776
export AIETOOLS=/home/bcloud/Xilinx/2026.1/Vitis/aietools
export XILINXD_LICENSE_FILE=/home/bcloud/.Xilinx/Xilinx.lic
export LM_LICENSE_FILE=/home/bcloud/.Xilinx/Xilinx.lic
export PATH=$AIETOOLS/bin:$PATH
cd ~/1bit-1776

# 1. CPU contract gates (fast, no hardware):
g++ -std=c++20 -O2 -I engine/npu/generators -I engine/npu/src \
    engine/npu/src/test_i4_silu_q22.cpp -o /tmp/t && /tmp/t zaya1-8b.q4nx 1 0
# 2. fused xclbin build (needs mlir-aie aiecc, see the script's REQUIRES):
engine/npu/generators/build_zaya_fused.sh
# 3. on-NPU h2 / token-parity check against the float path:
#    (the fused engine path in engine/npu/src/npu_engine_fused.hip + the
#    existing corr/token tests; xrt-smi examine first for NPU state)
```

Expected: all gates PASS (this report's numbers), fused xclbin builds, and
decode tokens match the two-launch int8 path (FFN corr ≥ 0.999 ⇒ argmax
parity, per the FLM-PARITY-PLAN bar of corr ~0.999 / token parity, not
bit-exact).
