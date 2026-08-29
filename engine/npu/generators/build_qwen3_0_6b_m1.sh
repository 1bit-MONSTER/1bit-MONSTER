#!/bin/bash
# build_qwen3_0_6b_m1.sh — build TRUE M=1 decode xclbins for Qwen3-0.6B FFN
# (GU: K=1024,N=6144; D: K=3072,N=1024) via n1_core_i8_m1.py.
#
# The shipped final_i8_{GU,D}_qwen3_0_6b xclbins bake a fixed M=128 AIE tile
# stream — every decode launch runs a 128-row stream for 1 row of data
# (~3.4-4.9 ms kernel wait per launch, AIE2P-FACTS.md §3b).  The m1 generator
# emits a single-core-row design with m=1 tiles (linear 1-row A/C DMA taps,
# same 8x8-microtile B transform), so the stream is a true 1-row decode:
# expected ~50 us/launch.  The scalar matmul_i8_i32 alias (DIM_M=1 < 16) is
# exact integer arithmetic, bit-identical to the vectorized M=16/128 kernels.
#
# Usage: engine/npu/generators/build_qwen3_0_6b_m1.sh
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
workdir="/tmp/qwen3_m1_build.$$"
mkdir -p "$workdir"
trap 'rm -rf "$workdir"' EXIT

# 1. Compile the DIM_M=1 scalar microkernel (1x64x128 tile) into the workdir.
$P/bin/clang++ --target=aie2p-none-unknown-elf --std=c++20 -O2 \
    -DDIM_M=1 -DDIM_K=64 -DDIM_N=128 -Di8_i32_ONLY \
    -isystem $P/include/c++/v1 \
    -I /home/bcloud/Xilinx/2025.2/Vitis/aietools/include \
    -I $M/include/aie_kernels/aie2p \
    -c "$GENERATOR_DIR/mm_kernel_reference.cc" -o "$workdir/mm_1x64x128.o"

# The MLIR references the kernel object by the fixed name mm_32x64x128.o.
cp "$workdir/mm_1x64x128.o" "$workdir/mm_32x64x128.o"

build_one() {
    local proj="$1" K="$2" N="$3"
    local design="$workdir/design_${proj}_m1.mlir"
    local xclbin="$XCLBIN_DIR/final_i8_${proj}_qwen3_0_6b_m1.xclbin"
    local insts="$XCLBIN_DIR/insts_i8_${proj}_qwen3_0_6b_m1.txt"
    echo "═══ ${proj} M=1 K=${K} N=${N} ═══"
    $PYTHON "$GENERATOR_DIR/n1_core_i8_m1.py" -K "$K" -N "$N" \
        -k 64 -n 128 -c 8 -b 5 2>/dev/null > "$design"
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
echo "OK: qwen3_0_6b M=1 xclbins built"
