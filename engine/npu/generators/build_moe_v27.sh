#!/bin/bash
# Build MOE_GU/MOE_D/MOE_SGU/MOE_SD xclbins with the v27 multi-row flow.
set -euo pipefail
PYTHON=/home/bcloud/mlir-aie/.venv/bin/python3
AIECC=/home/bcloud/mlir-aie/build_tmp/bin/aiecc
PEANO=/home/bcloud/mlir-aie/.venv/lib/python3.14/site-packages/llvm-aie
AIETOOLS=/home/bcloud/mlir-aie/build_tmp
export PATH=/home/bcloud/Xilinx/2026.1/2026.1/Vitis/bin:/opt/xilinx/xrt/bin:$PATH
KERNEL_O="$(cd "$(dirname "$0")" && pwd)/mm_32x64x128.o"
export PYTHONPATH=/home/bcloud/mlir-aie/install_tmp/python:/home/bcloud/mlir-aie/.venv/lib/python3.14/site-packages
export LD_LIBRARY_PATH=/home/bcloud/mlir-aie/install_tmp/python/aie/_mlir_libs
XCLBIN_DIR="$(cd "$(dirname "$0")/../xclbins" && pwd)"
GENERATOR_DIR="$(cd "$(dirname "$0")" && pwd)"
TAG=qwen3.6-moe_35b

# PID-unique workdir (issue #1777): a fixed /tmp path for the design OR the
# kernel .o could be clobbered by a co-tenant process between generation and
# aiecc, and the build would silently consume the stale file. $$ = PID.
workdir="/tmp/zaya_moe_v27_build.$$"
mkdir -p "$workdir"
trap 'rm -rf "$workdir"' EXIT

build_one() {
    local proj="$1" K="$2" N="$3" cols="$4"
    local design="$workdir/design_${proj}_moe.mlir"
    local xclbin="$XCLBIN_DIR/final_i8_${proj}_${TAG}.xclbin"
    echo "═══ ${proj} K=${K} N=${N} cols=${cols} ═══"
    $PYTHON "$GENERATOR_DIR/n1_core_i8_v27.py" -M 128 -K "$K" -N "$N" \
        -m 32 -k 64 -n 128 -c "$cols" -r 4 -b 5 2>/dev/null > "$design"
    if [ ! -s "$design" ]; then
        echo "ERROR: ${proj}: design generation produced an empty file" >&2
        rm -rf "$workdir"
        exit 1
    fi
    cp "$KERNEL_O" "$workdir/mm_32x64x128.o"
    cd "$workdir"
    $AIECC --peano="$PEANO" --aietools="$AIETOOLS" \
        --alloc-scheme=basic-sequential --no-xchesscc --no-xbridge \
        --aie-generate-xclbin --no-compile-host --unified --dynamic-objFifos \
        --aie-generate-npu-insts \
        --xclbin-name="$xclbin" \
        --npu-insts-name="$XCLBIN_DIR/insts_i8_${proj}_${TAG}.txt" \
        "$design" 2>&1 | tail -1
    cd "$GENERATOR_DIR"
    ls -la "$xclbin" "$XCLBIN_DIR/insts_i8_${proj}_${TAG}.txt"
}

build_one MOE_GU 2048 8192 8
build_one MOE_D  8192 2048 8
build_one MOE_SGU 2048 1024 4
build_one MOE_SD 512 2048 8
