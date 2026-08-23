# TileFuse → 1bit.MONSTER Q4NX plan (2026-08-08)

Paper: arXiv 2606.11357 (full text: docs/research/papers/tilefuse-xdna2-mixed-precision-kernels.md).
TileFuse = fused mixed-precision (W4A16/W8A16) GEMM/GEMV kernels on AMD XDNA2,
compiled via IRON/MLIR-AIE, running off-the-shelf quantized weights with no
NPU-specific requant. This is the direct fix for our Q4NX wall (official NPU2
weights required; community GGUF → q4nx conversions degenerate).

## TileFuse recipe (what to replicate)

1. **Offline weight pre-tiling** (CPU, once per model):
   - Pack each weight tile: contiguous INT4/INT8 codes → BF16 scales → INT8
     zero-points (zp duplicated so tile payloads are 128 B divisible for DMA).
   - **Interleaved column-major tile layout**: tiles assigned to the same AIE
     column become memory-contiguous → BD streams without large strides →
     GEMM dims up to 32K (our 74B MLP layers need this).
2. **Fused microkernel** (per compute core):
   - Unpack nibbles (mask/shift → INT8 lanes) → widen → BF16 → dequant with
     per-group scale/zp → store BF16 in local buffer → reuse across activation
     tiles (dequant once per weight tile, not per consume) → AIE 8×8 BF16
     matmul primitives.
3. **GEMV dataflow** (decode, batch=1):
   - Shim core → memory core distributes 4 bundled weight tiles → all 4×8
     compute rows (baseline uses 1 row = 8/32 cores).
   - Inner loop: 64×8 weight block × shared 1×8 activation fragment → 64
     outputs/iteration (vs baseline 16).
4. **Hybrid operator split** (matches our design):
   - NPU: quantized linear layers (GEMM/GEMV). iGPU: attention, softmax,
     norms, residual (FlashAttention-2, libtorch/HIP). Sequential operator
     dispatch, not concurrent (our GPU+NPU hybrid backend already does this).
5. **Numbers to beat**: GEMM +121.6% / GEMV +281% vs fp32; >2× perf+energy vs
   iGPU; −64.6% energy end-to-end; GPT-OSS-20B at 12.6 TPS / 0.54 s TTFT on
   Strix/Halo (AMD's own article).

## Mapping onto our code

| TileFuse piece | Our counterpart | Action |
|---|---|---|
| Pre-tiling + metadata packing | `tools/gguf_to_onebp.cpp` + q4nx manifest (d/m packed per tile) | Extend converter to emit TileFuse-style interleaved column-major layout + packed scales/zp (W4A16/W8A16, group 128). The q4nx I8/Q4_1 pack is ~this already — reuse the pack, add interleave. |
| Fused GEMM/GEMV kernels | FLM's `dequant.xclbin` / `generate_dequant_q4_1_seq` (open in third_party/FastFlowLM src/include/modules/dequant.hpp) | Build our own fused kernel via Vitis 2026.1 + aiebu (installed). Consume the pre-tiled layout directly. |
| GEMV full-array dataflow | `backend_npu_flm.cpp` per-request/serve path | Replace FLM's GEMV with our kernel for decode. |
| Hybrid split | `tests/backends/backend_npu.cpp` + GPU backends (attention on iGPU) | Keep; wire our kernel into the same dispatch. |
| Weights source | Community GGUFs (vanilla arch) | Pre-tiled layout accepts AWQ/Q4_0/Q4_1/Q8_0 directly → the Q4NX "everything" promise becomes true. |

## Milestones

1. **M1 ✅ DONE (2026-08-08)**: `docs/research/tilefuse_prep.py` — standalone
   pre-tiler: 128x64 tiles, int4 codes (adjacent-column nibbles), bf16 scales,
   int8 code-domain zps (`(q-zp)*scale`), interleaved column-major (8 AIE
   slots round-robin). Round-trip PASS on synthetic (err 0.0087 <= ceiling
   0.0101) and real llama-3.2-1B q_proj (err 0.0288 <= 0.0383). Also settled:
   the q4nx pack is NOT salvageable (nibble-truncated Q8_0 codes, corr
   0.01-0.09 vs GGUF) — the pre-tiler is a new output, not a reuse.
   Next: fold into gguf_to_onebp as `--tilefuse` output mode.
2. **M2 (kernel)**: fused GEMV for one linear layer on XDNA2 via Vitis
   (aiebu/IRON flow); compare vs FLM's dequant path on the same layer.
3. **M3 (end-to-end)**: llama-3.2-1B, community Q8_0 GGUF → pre-tiled →
   our kernel → coherent output (the test that currently degenerates).
4. **M4 (perf)**: 64.6% energy / 2× iGPU targets; TTFT/TPS vs FLM baseline.

## Open questions
- FLM's q4nx pack already interleaves tiles per AIE column? (manifest shapes
  [ntiles, 5120] suggest tile-major — check against Figure 2c layout.)
- Kernel dispatch overhead for short GEMVs (paper's limitation §4.5) — batch
  multi-layer launches to amortize.
