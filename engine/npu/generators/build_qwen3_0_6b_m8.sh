#!/bin/bash
# build_qwen3_0_6b_m8.sh — build M=8 (vectorized mmul) decode xclbins for
# Qwen3-0.6B FFN (GU: K=1024,N=6144; D: K=3072,N=1024).
#
# The m1 xclbins (n1_core_i8_m1.py) are COMPUTE-bound: the scalar matmul
# consumes ~1 GB/s of B while the DMA delivers ~1.4-2.3 GB/s, so r.wait is
# ~4.3 ms for the 6.3 MB GU B.  The vectorized 8x8x8 mmul (M8_VECTORIZED,
# same recipe as build_zaya_m8.sh) is 8x faster per byte — the launch drops
# to the DMA rate (~2.7 ms).  Runs 8 rows per launch (decode uses row 0);
# am=8 also amortizes the B DMA across 8 sequences (measured 28.1x for the
# M=128 family: am=28 == am=1 wall time).
#
# Usage: engine/npu/generators/build_qwen3_0_6b_m8.sh
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

workdir="/tmp/qwen3_m8_build.$$"
mkdir -p "$workdir"
trap 'rm -rf "$workdir"' EXIT

# 1. Compile the DIM_M=8 vectorized microkernel (1x4 mmul expansion).
$P/bin/clang++ --target=aie2p-none-unknown-elf --std=c++20 -O2 \
    -DDIM_M=8 -DDIM_K=64 -DDIM_N=128 -Di8_i32_ONLY -DM8_VECTORIZED \
    -isystem $P/include/c++/v1 \
    -I /home/bcloud/Xilinx/2025.2/Vitis/aietools/include \
    -I $M/include/aie_kernels/aie2p \
    -c "$GENERATOR_DIR/mm_kernel_reference.cc" -o "$workdir/mm_8x64x128.o"

cp "$workdir/mm_8x64x128.o" "$workdir/mm_32x64x128.o"

build_one() {
    local proj="$1" K="$2" N="$3"
    local design="$workdir/design_${proj}_m8.mlir"
    local xclbin="$XCLBIN_DIR/final_i8_${proj}_qwen3_0_6b_m8.xclbin"
    local insts="$XCLBIN_DIR/insts_i8_${proj}_qwen3_0_6b_m8.txt"
    echo "═══ ${proj} M=8 K=${K} N=${N} ═══"
    $PYTHON "$GENERATOR_DIR/n1_core_i8_v27.py" -M 8 -K "$K" -N "$N" \
        -m 8 -k 64 -n 128 -c 8 -r 1 -b 5 2>/dev/null > "$design"
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
echo "OK: qwen3_0_6b M=8 xclbins built"
