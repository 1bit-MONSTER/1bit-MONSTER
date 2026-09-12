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
