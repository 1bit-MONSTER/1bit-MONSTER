#!/bin/bash
# Build the fused GU→SiLU→D xclbin (issue #1759) — ONE launch per Zaya MoE
# layer instead of two (GU then D). On-core fixed-point SiLU (256-entry LUT,
# see silu_quant.h) eliminates the GU→CPU→D round trip and halves the 40
# decode launches/token; the plan's ~6.2 → ~7.5 tok/s milestone.
#
# Shape: A = residual [1×2048], GU [2048×4096 interleaved], D [2048×2048].
# Single core row (r=1), M=8 1x4 vectorized mmul (bit-identical to M=16/128).
#
# REQUIRES the mlir-aie toolchain (aiecc) + an NPU2 device for verification —
# this machine only has the CPU-side contract validation
# (engine/npu/tests/test_fused_silu.cpp).
#
# Usage: engine/npu/generators/build_zaya_fused.sh
set -euo pipefail

P=/home/bcloud/mlir-aie/.venv/lib/python3.14/site-packages/llvm-aie
M=/home/bcloud/mlir-aie/.venv/lib/python3.14/site-packages/mlir_aie
PYTHON=/home/bcloud/mlir-aie/.venv/bin/python3
AIECC=/home/bcloud/mlir-aie/build_tmp/bin/aiecc
AIETOOLS=/home/bcloud/mlir-aie/build_tmp
export PATH=/home/bcloud/Xilinx/2026.1/2026.1/Vitis/bin:/opt/xilinx/xrt/bin:$PATH
export PYTHONPATH=/home/bcloud/mlir-aie/install_tmp/python:/home/bcloud/mlir-aie/.venv/lib/python3.14/site-packages
export LD_LIBRARY_PATH=/home/bcloud/mlir-aie/install_tmp/python/aie/_mlir_libs

GENERATOR_DIR="$(cd "$(dirname "$0")" && pwd)"
XCLBIN_DIR="$GENERATOR_DIR/../xclbins"
mkdir -p "$XCLBIN_DIR"

# 1. Compile the DIM_M=8 kernel (1x4 mmul + the fused silu_quant_i8_fused
#    entry from silu_quant.h — the on-core SiLU+quant step).
$P/bin/clang++ --target=aie2p-none-unknown-elf --std=c++20 -O2 \
    -DDIM_M=8 -DDIM_K=64 -DDIM_N=128 -Di8_i32_ONLY -DM8_VECTORIZED \
    -isystem $P/include/c++/v1 \
    -I /home/bcloud/Xilinx/2025.2/Vitis/aietools/include \
    -I $M/include/aie_kernels/aie2p \
    -c "$GENERATOR_DIR/mm_kernel_reference.cc" -o /tmp/mm_8x64x128_fused.o

# The MLIR references the kernel object by the fixed name mm_32x64x128.o.
cp /tmp/mm_8x64x128_fused.o /tmp/mm_32x64x128.o

echo "═══ fused GU→SiLU→D  M=8 K=2048 N_GU=4096 N_D=2048 ═══"
design="/tmp/design_fused_gu_silu_d.mlir"
xclbin="$XCLBIN_DIR/final_i8_MOE_FUSED_zaya.xclbin"
insts="$XCLBIN_DIR/insts_i8_MOE_FUSED_zaya.txt"
$PYTHON "$GENERATOR_DIR/n1_core_fused_gu_silu_d.py" -M 8 -K 2048 \
    -N_GU 4096 -N_D 2048 -m 8 -k 64 -n 128 -c 8 -b 2 2>/dev/null > "$design"
cd /tmp
$AIECC --peano="$P" --aietools="$AIETOOLS" \
    --alloc-scheme=basic-sequential --no-xchesscc --no-xbridge \
    --aie-generate-xclbin --no-compile-host --unified --dynamic-objFifos \
    --aie-generate-npu-insts \
    --xclbin-name="$xclbin" --npu-insts-name="$insts" \
    "$design" 2>&1 | tail -3
cd "$GENERATOR_DIR"
ls -la "$xclbin" "$insts"

cat <<'EOF'
═══ verification checklist (strixhalo) ═══
1. aiecc build: watch for (a) the produce-only C1 fifo lowering, (b) shim
   S2MM channel pressure (2 outbound streams per column), (c) the gs 512B tap.
2. NPU decode with NPU_FUSED=1 (zaya_decode.cpp fused mode):
   - corr of the layer-1 MoE probe vs the CPU float reference ≥ ~0.999
     (contract validated on x86: 0.9993–0.9996, argmax parity — the fused
     int8 path must match the two-launch NPU path, not float exactly).
   - token parity vs the two-launch M=16 path.
3. perf: 40 → 20 launches/token; expect ~6.2 → ~7.5 tok/s if the ~0.85 ms
   per-launch fixed overhead is the binding cost.
Fallbacks if the C1 produce-only fifo misbehaves:
   - Design J: write C1 to bo4 via the v27 C path and read it back for the
     SiLU phase (adds ~16 KB/token DDR traffic; needs a 4th outbound stream
     budget check).
EOF
