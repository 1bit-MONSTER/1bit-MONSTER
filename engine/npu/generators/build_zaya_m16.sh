#!/bin/bash
# Build Zaya M=16 decode xclbins — GU (K=2048, N=4096) + D (K=2048, N=2048).
#
# Decode is M=1, but the M=128-baked xclbins run a fixed 128-row stream per
# launch (AIE2P-FACTS.md §3b). M=16 is the smallest vectorized mmul shape
# (mmul needs m % 16 == 0), so this halves the compute vs the 4-slice M=128
# design (16 rows vs 128) and ~1.4x's decode (measured 4.3 -> 6.1 tok/s).
#
# Usage: engine/npu/generators/build_zaya_m16.sh
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

# PID-unique workdir (issue #1777): a fixed /tmp path for the design OR the
# kernel .o could be clobbered by a co-tenant process between generation and
# aiecc, and the build would silently consume the stale file. $$ = PID.
workdir="/tmp/zaya_m16_build.$$"
mkdir -p "$workdir"
trap 'rm -rf "$workdir"' EXIT

# 1. Compile the DIM_M=16 microkernel (vectorized mmul, single core-row slice)
#    INTO the workdir.
$P/bin/clang++ --target=aie2p-none-unknown-elf --std=c++20 -O2 \
    -DDIM_M=16 -DDIM_K=64 -DDIM_N=128 -Di8_i32_ONLY \
    -isystem $P/include/c++/v1 \
    -I /home/bcloud/Xilinx/2025.2/Vitis/aietools/include \
    -I $M/include/aie_kernels/aie2p \
    -c "$GENERATOR_DIR/mm_kernel_reference.cc" -o "$workdir/mm_16x64x128.o"

# The MLIR references the kernel object by the fixed name mm_32x64x128.o.
cp "$workdir/mm_16x64x128.o" "$workdir/mm_32x64x128.o"

build_one() {
    local proj="$1" K="$2" N="$3"
    local design="$workdir/design_${proj}_m16.mlir"
    local xclbin="$XCLBIN_DIR/final_i8_MOE_${proj}_zaya_m16.xclbin"
    local insts="$XCLBIN_DIR/insts_i8_MOE_${proj}_zaya_m16.txt"
    echo "═══ ${proj} M=16 K=${K} N=${N} ═══"
    $PYTHON "$GENERATOR_DIR/n1_core_i8_v27.py" -M 16 -K "$K" -N "$N" \
        -m 16 -k 64 -n 128 -c 8 -r 1 -b 5 2>/dev/null > "$design"
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

build_one GU 2048 4096
build_one D 2048 2048
