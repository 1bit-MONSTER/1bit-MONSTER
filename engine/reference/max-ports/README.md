# MAX Reference Ports (gfx1151 / Strix Halo)

Reference HIP implementations extracted from MAX 26.5.0 production kernels
(disassembled AMDGPU ELFs from real compiled models) and verified on this box.
**Reference only — not wired into the build.** For the complete dossier
(layouts, traps, probe datasets), see the LLM wiki (`gpu-programming` category).

## The kernels

| File | Recipe | Verification |
|---|---|---|
| `gemv_dpp.cu` | Decode GEMV: b128 bf16 loads, cvt+fma, 4-step DPP butterfly + cross-row shuffle, bf16 RNE output | PASS 0/128, deterministic |
| `rmsnorm_dpp.cu` | RMSNorm: warp tiling, DPP butterfly, `v_rsq_f32` fast path | PASS 0/8192; rsq error 7.4e-08 vs IEEE |
| `gemm_wmma.cu` | bf16 WMMA GEMM: 2x2 accumulator tiles per warp, 32x32 per warp | PASS 0/4096, dense random data |
| `gemm_wmma_lds.cu` | LDS-staged version: Y transposed in LDS, `ds_load_b128` operands, double-buffer ready | PASS 0/4096; 512³: 20.5us vs 22.9us direct |
| `gemm_fp8_wmma.cu` | **Fused** fp8(dequant)+bf16 WMMA GEMM — one kernel (MAX does two passes); per-16-k-slice scales | PASS 0/4096 |

## The WMMA layout (solved — do not re-derive)

`v_wmma_f32_16x16x16_bf16`: **C[l][m] = sum_k A_lane_m[k] * B_lane_l[k]** —
the A operand of lane m feeds output column m; the B operand of lane l feeds
output row l; element-wise k pairing over the 16-wide slice.
D registers: lane `(l + 16*half)` holds `C[l][2j + half]` (lanes 0-15 = even
output cols, lanes 16-31 = odd; both halves read the same operand rows).
**A operand = the RIGHT matrix COLUMN (strided by N). B operand = the LEFT
matrix ROW (consecutive).** Getting A/B roles or row/column wrong silently
produces row-sums (A=ones) or shifted output (identity tests pass either way —
the symmetric identity masks it).

## DPP notes

- DPP rows are **16 lanes** (wave32 = two rows); the 4-step butterfly
  (`quad_perm:[1,0,3,2] -> [2,3,0,1] -> row_half_mirror -> row_ror:8`) reduces
  *within* each 16-lane row only. The cross-row merge is a required 5th step:
  `total += __shfl_xor(total, 16);`
- Inline-asm form that works (single register, matches MAX's `v_add_f32_dpp v1, v1, v1`):
  ```c
  asm volatile("v_add_f32_dpp %0, %0, %0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf bound_ctrl:1" : "+v"(v));
  ```
  Do **not** use `%0, %0, %1` with a separate input constraint — the compiler
  splits registers and reads a stale value.

## Platform facts

- No fp8 WMMA on gfx1151 (`v_wmma_..._fp8` invalid; fp8 WMMA is RDNA4/gfx12).
  fp8 = bandwidth savings only; math is bf16 WMMA. This is why the fused fp8
  kernel dequantizes fp8->bf16 in-kernel.
- `v_rsq_f32` (fast) error ~1 ULP — take the fast path; invisible after bf16 output.
- rocwmma.hpp targets CDNA/MFMA (MI-series), NOT RDNA3.5 — don't use it.

## Build & run (each kernel is self-contained with a CPU reference)

```
export LD_LIBRARY_PATH=/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_devel/lib
/opt/rocm-therock/bin/hipcc --offload-arch=gfx1151 gemm_wmma.cu -o gemm_wmma && ./gemm_wmma
```

## Perf status

- GEMM LDS staging: DONE (`gemm_wmma_lds.cu`) — Y columns transposed into LDS,
  operands read as `ds_load_b128`. Next step when prefill perf matters: the
  double-buffer + graded `s_waitcnt lgkmcnt` overlap (MAX's pattern).
- Fused fp8: per-block (per-16-k-slice) scales: DONE, folded into D per slice.
- GEMV: bf16 output store with f32->bf16 RNE: DONE (`bf16(total)`).
- Remaining: FlashAttention reference port (recipe in the port guide);
  double-buffering the GEMM k-loop.
