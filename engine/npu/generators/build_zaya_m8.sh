#!/bin/bash
# Build Zaya M=8 decode xclbins — GU (K=2048, N=4096) + D (K=2048, N=2048).
#
# The vectorized mmul needs m % 16 == 0 (2x M expansion), so the M=16 build
# runs 16 rows for a 1-row decode. M=8 uses a single-mmul-row (1x4) kernel —
# the same 8x8x8 mmul accumulation, so it is bit-identical to M=16/M=128.
# Measured: M=8 == M=16 within noise (moe ~126ms) — compute is not the
# bottleneck; the ~2.9ms/launch fixed overhead + weight DMA is.
#
# Usage: engine/npu/generators/build_zaya_m8.sh
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

# 1. Compile the DIM_M=8 kernel (1x4 mmul expansion; M8_VECTORIZED bypasses the
#    DIM_M < 16 scalar alias).
$P/bin/clang++ --target=aie2p-none-unknown-elf --std=c++20 -O2 \
    -DDIM_M=8 -DDIM_K=64 -DDIM_N=128 -Di8_i32_ONLY -DM8_VECTORIZED \
    -isystem $P/include/c++/v1 \
    -I /home/bcloud/Xilinx/2025.2/Vitis/aietools/include \
    -I $M/include/aie_kernels/aie2p \
    -c "$GENERATOR_DIR/mm_kernel_reference.cc" -o /tmp/mm_8x64x128.o

# The MLIR references the kernel object by the fixed name mm_32x64x128.o.
cp /tmp/mm_8x64x128.o /tmp/mm_32x64x128.o

build_one() {
    local proj="$1" K="$2" N="$3"
    # PID-unique design path (issue #1777): a fixed /tmp path could be
    # clobbered by a co-tenant process between generation and aiecc, and the
    # build would silently consume the stale file.
    local design="/tmp/design_${proj}_m8.$$.mlir"
    local xclbin="$XCLBIN_DIR/final_i8_MOE_${proj}_zaya_m8.xclbin"
    local insts="$XCLBIN_DIR/insts_i8_MOE_${proj}_zaya_m8.txt"
    echo "═══ ${proj} M=8 K=${K} N=${N} ═══"
    $PYTHON "$GENERATOR_DIR/n1_core_i8_v27.py" -M 8 -K "$K" -N "$N" \
        -m 8 -k 64 -n 128 -c 8 -r 1 -b 5 2>/dev/null > "$design"
    [ -s "$design" ] || { echo "ERROR: ${proj}: design generation produced an empty file" >&2; exit 1; }
    cd /tmp
    $AIECC --peano="$P" --aietools="$AIETOOLS" \
        --alloc-scheme=basic-sequential --no-xchesscc --no-xbridge \
        --aie-generate-xclbin --no-compile-host --unified --dynamic-objFifos \
        --aie-generate-npu-insts \
        --xclbin-name="$xclbin" --npu-insts-name="$insts" \
        "$design" 2>&1 | tail -1
    cd "$GENERATOR_DIR"
    ls -la "$xclbin" "$insts"
    rm -f "$design"; rm -rf "$design.prj"
}

build_one GU 2048 4096
build_one D 2048 2048
