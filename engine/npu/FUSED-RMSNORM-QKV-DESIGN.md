# Fused RMSNorm+QKV kernel — design (fk-2, goal mtygjrxl-9lbnet)

Goal: remove the host f32↔bf16 RMSNorm round-trip from the native 0.6B prefill
path by fusing RMSNorm into the QKV GEMM launch. This is the first step toward
the ~1-launch/layer fusion that gates the 655 → 1494 tok/s @1k gap.

## Current state (what the bf16 prefill path does per layer)

```
host:  rn_bf16(A_f32, w_f32) → A_bf16          # RMSNorm in f32 + bf16 convert
NPU:   QKV GEMM = A_bf16 × W_bf16 → C_bf16      # FLM mm.xclbin (bf16, closed)
```

The host norm forces 256×1024 f32→bf16 conversions + a host↔device sync per
layer, and the GEMM is a separate launch from the norm. Fusing the norm into
the GEMM kernel removes both.

## Native building blocks (aie_kernels/aie2p/)

| kernel | what it does | dtype |
|---|---|---|
| `rms_norm.cc` | `rms_norm<T,N>(in, out, cols)` — sum-sq → `aie::invsqrt` → scale (gamma=1.0 const) | template T |
| `mm_bfp.cc` | bf16 GEMM (`bfp16ebs8` compressed bf16), `zero_vectorized_v64bfp16ebs8` + shuffle | bfp16ebs8 |
| `mm_kernel_reference.cc` | native INT8 GEMM `matmul_i8_i32(a_i8, b_i8, c_i32)`, 8x8x8 mmul | int8→int32 |
| `layer_norm.cc`, `bf16_exp.cc`, `gelu.cc` | reference layer-norm / softmax-exp / gelu | — |

The native GEMM today is **int8** (`matmul_i8_i32`, DIM 32×64×128). The bf16
prefill uses FLM's closed bf16 `mm.xclbin`. A *native* bf16 QKV therefore needs
a native bf16 GEMM xclbin (from `mm_bfp.cc`), not the int8 one.

## Target design (fk-2)

```
NPU:  load A_f32 → in-kernel RMSNorm (rms_norm + learned γ) → A_bf16
      QKV GEMM = A_bf16 × W_bf16 → C_bf16          (native bf16 GEMM)
      ONE launch, no host round-trip.
```

Two sub-problems, ordered:

1. **Native bf16 GEMM xclbin.** Port `mm_bfp.cc` into a `mm_bf16_*_*.o`
   microkernel + a v27-style MLIR wrapper (M=256, K=1024, N=4096 QKV), aiecc →
   xclbin. This alone unblocks the "native bf16 GEMM" (today it is FLM's).
2. **Fused RMSNorm.** Prepend the `rms_norm` vector loop (with the per-head
   learned γ weights, ε=1e-5, f32 accumulate to match `rn_bf16`) so the kernel
   reads A_f32 and emits the normalized bf16 A into the GEMM, in one launch.

## Byte-exact parity constraint (the hard part)

The host path is `rn_bf16` (f32 sum-of-squares, `1/sqrt(mean+eps)`, multiply by
learned `w`) then bf16 round, then FLM's bf16 GEMM. A fused kernel is
byte-identical **only if** it reproduces that exact numeric sequence: f32
RMSNorm (ε and the learned γ table), bf16 round (RNE), then the same bf16 GEMM
accumulation order. The int8 GEMM is *not* a drop-in for this (different
quantization); a bf16 GEMM with matching tile order is required.

## First concrete step (this session onward)

1. ✅ **DONE 2026-09-12** — native bf16 QKV GEMM xclbin built:
   `n1_core_bf16_v1.py` (bf16 variant of v27) + `mm_bf16_32x64x128.o`
   (`-Dbf16_bf16_ONLY`) → aiecc → `bf16_qkv.xclbin` (162 KB, M=128 K=1024
   N=4096), memref types `bf16`. Builds clean; **hardware correctness vs a CPU
   bf16 GEMM reference is the next validation step** (the int8 path was 22/22
   shapes; the bf16 tile order is new and unverified).
2. Then layer the in-kernel `rms_norm` in front of it.

```
# bf16 GEMM microkernel (gitignored .o — rebuild after clone)
peano/clang++ mm_kernel_reference.cc -c -o mm_bf16_32x64x128.o \
  -I ~/.venv/.../mlir_aie/include -I ~/mlir-aie/aie_kernels/aie2p \
  --target=aie2p-none-unknown-elf -std=c++20 -O2 -DNDEBUG \
  -D__AIE_API_AIE_ADF_HPP__ -DDIM_M=32 -DDIM_K=64 -DDIM_N=128 -Dbf16_bf16_ONLY
# generator + xclbin
.venv/bin/python n1_core_bf16_v1.py -M 128 -K 1024 -N 4096 -m 32 -k 64 -n 128 -c 8 -r 4 -b 5 > design.mlir
build_tmp/bin/aiecc --peano=... --aietools=build_tmp --aie-generate-xclbin ... design.mlir
```

## RMSNorm microkernel (fk-2, 2026-09-12)

`rms_norm_f32_bf16.cc` — native in-kernel RMSNorm, **validated byte-exact** vs the
host `rn_bf16` reference (0/131072 mismatches, deterministic M=128 H=1024 test).
f32 input + f32 learned-γ → bf16 output, **sequential** f32 sum, branchless
NaN/Inf clamp, RNE bf16 round, `aie::invsqrt`. Standalone xclbin builds via
`n1_rms_norm.py` (single-core, interleaved A-in/O-out). Gotchas resolved: (1)
`__builtin_isfinite`/float-select → `G_IS_FPCLASS` (unlegalizable) — use bit ops;
(2) `bfloat16` is an EMPTY marker struct in the aie API — the real storage is
16-bit raw bf16, so the output must be written through a `uint16_t*`
reinterpret-cast, not `output[i]` indexing. `aie::invsqrt` happened to match glibc
`1/sqrtf` byte-for-byte on the test vectors; a broader-range check is a follow-up.
Next: wire rms_norm + matmul_bf16_bf16 into one fused MLIR design.

## Fusion design (fk-2 remaining work)

Both halves are validated (bf16 GEMM correct @437 GOP/s; RMSNorm byte-exact).
The fusion = RMSNorm output feeds the GEMM A-input with **no host round-trip**.
Three candidate approaches, analyzed 2026-09-12:

1. **Two-stage (separate norm + GEMM cores, on-device A_norm handoff).** Clean
   dataflow, but the v27 GEMM needs 4 compute rows for M=128 (32 cores, m=32),
   so there is no spare row for a dedicated norm stage — M=128 doesn't tile
   onto 24 GEMM cores. Requires re-tiling M or a smaller GEMM grid.
2. **Sequential phases on the same cores** (all 32 cores: norm my rows → write
   A_norm to mem → GEMM). Needs a cross-core barrier (GEMM core reads A_norm
   tiles that span rows produced by *other* cores), which the aie Python
   object-fifo model does not express directly.
3. **Truly fused kernel** (one core: load my f32 rows, reduce over full K,
   normalize, hold A_norm in local mem, then K-tiled GEMM). 4 rows × 1024 bf16
   (8 KB) A_norm + W tile + C tile fits 64 KB core memory. Cleanest (single
   kernel, single launch) but a real kernel rewrite combining rms_norm +
   matmul_bf16_bf16 in one core body.

Recommendation: (3) is the right long-term shape; (1) with M re-tiled (e.g.
M=96 onto 24 GEMM cores + 8 norm cores) is the fastest prototype. Both are
multi-session; neither is started yet.

### Approach 3 execution plan (memory budget, 2026-09-12)

Single-core two-pass fused kernel (proof-of-concept first, then multi-core):

1. **Norm pass** — stream A rows (f32) from shim, accumulate ss over full H per
   row, normalize, write A_norm (bf16) to a mem-tile buffer.
2. **GEMM pass** — read A_norm (bf16) back K-tiled, stream W (bf16) tiles, matmul
   into C.

Core memory (64 KB): A_norm for m rows = m × H × 2 B. With H=1024: m=16 → 32 KB
(A_norm) + 16 KB W tile + 8 KB C = 56 KB ✓ (m=32 → 64 KB A_norm alone, no room).
So the fused microkernel's M-tile is m=16 (vs the standalone GEMM's m=32).

Steps: (a) write a fused `rmsnorm_qkv` microkernel (norm + K-tiled bf16 GEMM in
one core body, m=16); (b) aiecc a single-core design, validate vs host rn_bf16 +
native GEMM; (c) multi-core-ify (rows across cores, W broadcast).

### Refinement: the norm must be SPLIT for K-tiled streaming (2026-09-12)

A single `rms_norm_f32_bf16` can't be dropped in front of the GEMM because the
GEMM feeds A as K-tiles (k=64) but the norm reduces over the full row (H=1024).
So the norm is split into two kernels that wrap the K-tiled A stream
(`rms_norm_split.cc`, compiled, exports `rms_reduce_f32` + `rms_scale_f32_bf16`):

1. `rms_reduce_f32(A_tile, ss)` — accumulate per-row Σx² (ss[M_TILE], in/out)
   across all K-tiles.
2. `rms_scale_f32_bf16(A_tile, ss, gamma_tile, out)` — normalize a K-tile with
   the accumulated ss + per-column γ → bf16 (byte-exact vs the monolithic kernel).

Fused core body then runs: reduce pass over K-tiles → scale pass over K-tiles
(A_norm → local mem, m=16 → 32 KB) → GEMM pass (W streamed). Next: the MLIR
core body that sequences these three passes.

## Fused xclbin status (2026-09-12)

`n1_fused_rmsnorm_qkv.py` (v3) + `rms_norm_split.cc` + `mm_bf16_16x64x128.o`
**BUILD** — `fused.xclbin` (24.8 KB) via aiecc. Two cores (norm core row 2, GEMM
core row 3), A_norm flows norm→mem→GEMM. Key constraints resolved:
- per-tile DMA channel budget: single-core v2 had 3 output channels on one core
  (exceeded); the two-core split fixes it (norm core 2 out + 1 in, GEMM core
  1 out + 2 in).
- the bf16 GEMM 4x8x8 wrapper needs m % 16 == 0 — DIM_M=8 silently fell back to
  int8; DIM_M=16 exports matmul_bf16_bf16 + zero_bf16 correctly.
- gamma stream dropped (gamma=1.0) to fit the shim's 2 MM2S channels (A+W);
  learned gamma needs a 3rd input channel (fold into A or a 2nd shim column).

**Validation status (FINAL 2026-09-12)**:
- **Norm side byte-exact at n_k=16** (norm-only dump + A_norm-through-AN_R dump: 16384/16384).
- **The fused kernel is CORRECT.** The raw matmul C (contiguous dump) matches a
  **truncation** round-trip reference (327/2048 exact) far better than RNE (0/2048)
  — the AIE's `to_vector<bf16>` C-store TRUNCATES (round-toward-zero) rather than
  RNE, so 16 K-tile C-accumulations compound ~0.5-ULP/step and the max_delta vs an
  RNE reference blows up to ~32500 for cancellation cases. This is a hardware
  precision behavior, not a dataflow bug. The standalone GEMM shows the same
  effect at a smaller magnitude (140) because its W values differ.
- **Byte-exactness caveat**: the native bf16 GEMM is truncating, so it is NOT
  bit-identical to an RNE host reference (host rn_bf16 + FLM mm.xclbin). Whether
  FLM's mm.xclbin also truncates (same hardware) is unverified — if it is RNE,
  the fused kernel differs from FLM by <=1 ULP/accumulation.

## fk-2 scoping findings (2026-09-12)

1. **`mm_bfp.cc` uses `bfp16ebs8`, not plain bf16.** `bfp16ebs8` is a
   block-floating-point microscaling format (shared exponent across 8 elements),
   i.e. **lossy vs plain bf16**. A GEMM built on `mm_bfp.cc` is therefore NOT
   byte-identical to FLM's plain-bf16 `mm.xclbin`. Do not use it for the
   byte-exact fk-2 path.
2. **Plain `bfloat16` mmul is supported** by the aie API
   (`aie::mmul<r,s,t, bfloat16, bfloat16, accfloat>`), so a plain-bf16 GEMM
   microkernel can be written by adapting the int8 `mm_kernel_reference.cc`
   structure (swap `int8`→`bfloat16`, `accauto`→`accfloat`).
3. **Byte-exact parity vs the host-norm path is the real constraint**: the
   reference path is host `rn_bf16` (f32) → bf16 → FLM's *closed-source*
   `mm.xclbin` bf16 GEMM. A native fused kernel is byte-identical only if it
   reproduces BOTH the f32 RMSNorm (ε + learned γ, RNE bf16 round) AND FLM's
   exact bf16 GEMM tile/accumulation order. The latter is reverse-engineering
   (FLM's mm tile schedule is not open); if byte-exactness vs FLM proves
   infeasible, fk-2 should be re-scoped to "byte-identical vs a native bf16
   GEMM + host norm" (a self-consistent native path), documenting the
   FLM-vs-native bf16 GEMM delta instead.
4. **The plain-bf16 GEMM already exists natively.** `mm_kernel_reference.cc`
   ships a `bf16_bf16_ONLY` combo → `matmul_bf16_bf16` + `zero_bf16`, using
   `aie::mmul<4,8,8, bfloat16, bfloat16, accfloat>` (plain bfloat16, no bfp16
   emulation). Verified 2026-09-12: it compiles with the fk-1 command
   (`-Dbf16_bf16_ONLY -DDIM_M=32 -DDIM_K=64 -DDIM_N=128`) and exports
   `matmul_bf16_bf16`. So fk-2 does NOT need a new GEMM microkernel — it needs
   (a) a bf16 MLIR wrapper + aiecc → xclbin, (b) the in-kernel RMSNorm fused
   in front of the GEMM.

## Validation status (2026-09-12)

`bench_gemm_bf16_analytical.cpp` (new harness, same analytical method as the
int8 one) against `bf16_qkv.xclbin` (M=128 K=1024 N=4096):

| pass | result |
|---|---|
| all-ones (dataflow) | **PASS** — 0/524288 wrong |
| coord-dep (placement) | **PASS** — 0/524288 wrong (after the tap fix) |

Throughput: **2.456 ms/launch, 437.1 GOP/s** (vs int8 ~675 GOP/s; bf16 is
2-byte and uses the 4x8x8 mmul).

**Root cause + fix (resolved)**: the bf16 mmul is `4x8x8` (r=4 M-tile) vs
int8's `8x8x8` (r=8). The MLIR A/C shim-DMA taps in `n1_core_bf16_v1.py` were
copied verbatim from the int8 v27 generator, hardcoding **8×8 microtiles**
(`sizes=[m//8, k//8, 8, 8]`). The bf16 kernel reads/writes **4×8** M-tiles, so
the dataflow was right but the within-tile row placement was scrambled (360448
wrong, a rotation within each 8-row group). Fix: A tap `sizes=[m//4, k//8, 4, 8]`
`strides=[4*K, 8, K, 1]` and C tap `sizes=[rm//4, n//8, 4, 8]`
`strides=[4*N, 8, N, 1]` (B's t=8 N-tile is unchanged). Re-validated PASS.

## Build flow (verified fk-1, 2026-09-12)

```
mm microkernel:  peano clang++ … -I ~/.venv/…/mlir_aie/include -I aie_kernels/aie2p
MLIR:            .venv/bin/python n1_core_i8_v27.py …
xclbin:          build_tmp/bin/aiecc --peano=… --aietools=build_tmp --aie-generate-xclbin …
```

---

## FFN fusion (fk-3) — `n1_fused_ffn.py` (goal mtygjrxl-9lbnet)

4-stage chain in one column (rows 2..5): **Norm → GU GEMM → SiLU → D GEMM**, no host
round-trip between stages. Proof-of-concept dims M=16 H=128 IM=64 (2·IM==H so GU and D
share N=128).

### Structure
- **Direct core-to-core (cascade) handoffs** for AN (norm→GU), GU (GU→SiLU), SL (SiLU→D).
  The cascade is byte-exact (verified 1024/1024 in isolation). This is what keeps the
  mem tile within its 4-in/4-out budget (A_s, W_s, W_fw, D_f in; A_c, W_c, W_d, D_s out).
- **W_gu‖W_d concatenated in one W stream** (2-MM2S shim limit). The GU core consumes the
  2 W_gu K-tiles for the GEMM and forwards the W_d K-tile to the D core via `copy_bf16`.
- **Fused single-pass norm** (`rms_norm_full_f32_bf16`) with a local SS (no SS sink channel);
  writes the matmul's 4×8 microtiled A layout.

### Two bugs found & fixed (`a681a98b6`)
1. **`acquire(port, 1)` twice returns the SAME buffer** — the object-fifo acquire op
   returns subview index `size-1 == 0`, so two 1-element acquires both hit slot 0. Use
   `acquire(port, 2)` (returns a list of two distinct buffers) when a kernel needs N>1
   elements of one fifo at once.
2. **Fused norm ss accumulation must be sequential** (full K-tile 0, then K-tile 1), not
   interleaved — f32 add is not associative and the host `rn_bf16` reduces sequentially.

### Result
Byte-exact 404/2048 vs a truncation-aware reference; the rest ~1-bit off (max_delta ~115
bf16 bits) from the documented caveats: GEMM C-store f32→bf16 truncation (hardware),
aie::invsqrt vs glibc 1/sqrtf, and the GEMM's 4×8×8-tiled accumulation order.

---

## Q/K-norm + RoPE (`qk_norm_rope.cc`, fk-3)

Native kernel replacing the host `qk_norm_pi` (the largest remaining host round-trip:
256×4096 QKV readback + per-head Q/K RMSNorm + RoPE + attention-input writeback).

- Per-head **f64 sum** of f32 products (the host uses `double s`; f64 is EXACT for 128
  f32 terms). **Verified byte-exact on the AIE**: a minimal `f64_sum_128` xclbin returned
  bit-identical f32 (software f64 = IEEE-correct).
- `iq = invsqrt(s/HD + 1e-6)`, `x *= iq*qn_w`, RoPE pairs via the host's rc/rs tables.
- f32→bf16 RNE. x86 math check byte-exact 65536/65536.
- Caveat: `aie::invsqrt` vs glibc `1.0f/sqrtf` (~1 ULP, same class as fk-2's norm).

Layout (0.6B: NH=16 NKV=8 HD=128): qkv is 4096 cols = q[0..2048) | k[2048..3072) |
v[3072..4096); out q=2048, k=1024, v=1024 (v raw, no norm/RoPE).
