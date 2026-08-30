#!/bin/bash
# build_qwen3_0_6b_m32.sh — build M=32 FULL-32-CORE-GRID decode xclbins for
# Qwen3-0.6B FFN (GU: K=1024,N=6144; D: K=3072,N=1024) via n1_core_i8_v27.py
# with -r 4 (all 4 AIE core rows = 32 of 32 compute tiles).
#
# Rationale: the m8 xclbins (-r 1) use only 8 of the 32 compute tiles and the
# launch is B-DMA-bound, so adding rows costs almost nothing.  M=32 with m=8
# tiles spreads the 32 rows across 4 core rows x 8 columns (the full grid):
# one B (weight) DMA per launch now serves 32 rows instead of 8.
# Measured 2026-08-30 on Strix Halo (bench_gemm_analytical, 100 iters, both
# correctness passes 0 err):
#   GU 1024x6144: 1.934 ms/8 rows -> 2.258 ms/32 rows (241.8 -> 70.6 us/row, 3.4x)
#   D  3072x1024: 0.944 ms/8 rows -> 0.954 ms/32 rows (118.0 -> 29.8 us/row, 4.0x)
# These are the multi-sequence-decode xclbins: batch 32 sequences per layer so
# every sequence rides the same B DMA.  Single-sequence decode should keep the
# m1/m8 xclbins (M=32 costs ~21% more wall time than m8 for one row).
#
# Usage: engine/npu/generators/build_qwen3_0_6b_m32.sh
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

# PID-unique workdir (issue #1777): a fixed /tmp path could be clobbered by a
# co-tenant process between generation and aiecc.
workdir="/tmp/qwen3_m32_build.$$"
mkdir -p "$workdir"
trap 'rm -rf "$workdir"' EXIT

# 1. Compile the DIM_M=8 vectorized microkernel (1x4 mmul expansion).
$P/bin/clang++ --target=aie2p-none-unknown-elf --std=c++20 -O2 \
    -DDIM_M=8 -DDIM_K=64 -DDIM_N=128 -Di8_i32_ONLY -DM8_VECTORIZED \
    -isystem $P/include/c++/v1 \
    -I /home/bcloud/Xilinx/2025.2/Vitis/aietools/include \
    -I $M/include/aie_kernels/aie2p \
    -c "$GENERATOR_DIR/mm_kernel_reference.cc" -o "$workdir/mm_8x64x128.o"

# The MLIR references the kernel object by the fixed name mm_32x64x128.o.
cp "$workdir/mm_8x64x128.o" "$workdir/mm_32x64x128.o"

build_one() {
    local proj="$1" K="$2" N="$3"
    local design="$workdir/design_${proj}_m32.mlir"
    local xclbin="$XCLBIN_DIR/final_i8_${proj}_qwen3_0_6b_m32.xclbin"
    local insts="$XCLBIN_DIR/insts_i8_${proj}_qwen3_0_6b_m32.txt"
    echo "═══ ${proj} M=32 K=${K} N=${N} (r=4 c=8, full 32-core grid) ═══"
    $PYTHON "$GENERATOR_DIR/n1_core_i8_v27.py" -M 32 -K "$K" -N "$N" \
        -m 8 -k 64 -n 128 -c 8 -r 4 -b 5 2>/dev/null > "$design"
    [ -s "$design" ] || { echo "ERROR: ${proj}: design generation produced an empty file" >&2; exit 1; }
    cd "$workdir"
    $AIECC --peano="$P" --aietools="$AIETOOLS" \
        --alloc-scheme=basic-sequential --no-xchesscc --no-xbridge \
        --aie-generate-xclbin --no-compile-host --unified --dynamic-objFifos \
        --aie-generate-npu-insts \
        --xclbin-name="$xclbin" --npu-insts-name="$insts" \
        "$design" 2>&1 | tail -1
    cd "$GENERATOR_DIR"
    ls -la "$xclbin" "$insts"
}

build_one GU 1024 6144
build_one D 3072 1024
echo "OK: qwen3_0_6b M=32 full-grid xclbins built"
