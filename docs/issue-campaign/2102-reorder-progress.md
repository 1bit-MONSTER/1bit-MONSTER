# #2102 RE progress — reorder_cpy disassembly (2026-09-09)

Two distinct `reorder_cpy` symbols in `amd-oss/fastflowlm/src/lib/xrt/libqwen3_npu.so`:

| symbol | addr | purpose |
|---|---|---|
| global `reorder_cpy(uchar*, buffer<uchar>&, int, int)` | `0x3dbd0` | builds an **offset table** (row-tiling reorder) |
| `Dequant::reorder_cpy(uchar*, buffer<uchar>&, quant_block_t, int,int,int,int)` | `0x5d0d0` (+ cold clone `0x249ce`) | the **within-tile** reorder (the `8*(ir/8)+2*tc+(tr%2)` permutation the issue names) |

## global reorder_cpy @ 0x3dbd0 (offset-table / row-tiling)

Args: `rdi=dst (u64* table), rsi=&src buffer, edx=arg2 (dim), ecx=arg3`.

- `0x18(%rsi)` = source data pointer (the raw tiles).
- Computes `B = (arg2 + 255) >> 8` (≈ arg2/256), then vectorizes an offset table fill: `dst[i] = base + i * B * 5120` for `i = 0..15`, `5120 = 0x1400` = NPU block size.
- This is the **row-tiling** reorder documented in `docs/research/fastflowlm-analysis/Q4NX_FORMAT.md` (`_q4nx_reorder`: `[rows,cols]` → `[chunks,32,cols]`), NOT the within-tile nibble permutation. It confirms the mm.xclbin weight BO is laid out in 5120-byte blocks with a per-block stride of `B*5120`.

## Dequant::reorder_cpy @ 0x5d0d0 (within-tile — the #2102 target)

Not yet fully disassembled. This is the function that produces the `8*(ir/8)+2*tc+(tr%2)` interleave the issue cites (the formula is a HYPOTHESIS from `reorder_cpy`, not yet proven against silicon).

## Recommendation (matches peer diagnosis)

Disassembly of the AVX-512 offset-table loop is not the fastest route — the within-tile permutation is best derived **empirically in one NPU run**:

1. Build a synthetic B tile where each quant group encodes a distinct, invertible value pattern (dequant formula is known: `W = (q − zp) · scale`, bf16 scale/zero at group-major `g*32+lr`, unsigned nibbles — `npu-infer/NPU_GEMM_FIX.md` round-27).
2. Run `tools/mm_gemm_oracle.cpp` with identity A; C[i][n] = dequant(B[i][n]) → read C, invert the position map → exact permutation.
3. Validate against the disassembled Dequant::reorder_cpy (or skip disassembly entirely).

## Next concrete step

Add the "distinct per-position value + invert C" probe to `tools/mm_gemm_oracle.cpp` (it already writes `/tmp/mm_c.bin`), and drive it with a known-shape tile. This needs the NPU (`/dev/accel/accel0`) + a `<xclbin>.bin` from `tools/gen_mm_insts`.
