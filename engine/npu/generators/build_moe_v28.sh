#!/bin/bash
# Build v28 fused MoE xclbins: MOE_GUSGU (routed GU + shared GU in one
# launch, N=8192+1024) and MOE_DSD (routed D + shared D in one launch,
# K=4096+512, N=2*H). Same single-GEMM kernel as v27 — the fusion is
# pure concat along N (GUSGU) and K (DSD), so no new kernel code.
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

build_one() {
    local proj="$1" K="$2" N="$3" cols="$4"
    # PID-unique design path (issue #1777): a fixed /tmp path could be
    # clobbered by a co-tenant process between generation and aiecc, and the
    # build would silently consume the stale file.
    local design="/tmp/design_${proj}_moe.$$.mlir"
    local xclbin="$XCLBIN_DIR/final_i8_${proj}_${TAG}.xclbin"
    echo "═══ ${proj} K=${K} N=${N} cols=${cols} ═══"
    $PYTHON "$GENERATOR_DIR/n1_core_i8_v27.py" -M 128 -K "$K" -N "$N" \
        -m 32 -k 64 -n 128 -c "$cols" -r 4 -b 5 2>/dev/null > "$design"
    [ -s "$design" ] || { echo "ERROR: ${proj}: design generation produced an empty file" >&2; exit 1; }
    cp "$KERNEL_O" /tmp/mm_32x64x128.o
    cd /tmp
    $AIECC --peano="$PEANO" --aietools="$AIETOOLS" \
        --alloc-scheme=basic-sequential --no-xchesscc --no-xbridge \
        --aie-generate-xclbin --no-compile-host --unified --dynamic-objFifos \
        --aie-generate-npu-insts \
        --xclbin-name="$xclbin" \
        --npu-insts-name="$XCLBIN_DIR/insts_i8_${proj}_${TAG}.txt" \
        "$design" 2>&1 | tail -1
    cd "$GENERATOR_DIR"
    # backward-compat symlink for the a3b tag variant
    ln -sf "final_i8_${proj}_${TAG}.xclbin" "$XCLBIN_DIR/final_i8_${proj}_qwen3_6_35b_a3b.xclbin"
    ln -sf "insts_i8_${proj}_${TAG}.txt"   "$XCLBIN_DIR/insts_i8_${proj}_qwen3_6_35b_a3b.txt"
    ls -la "$xclbin" "$XCLBIN_DIR/insts_i8_${proj}_${TAG}.txt"
    rm -f "$design"; rm -rf "$design.prj"
}

build_one MOE_GUSGU 2048 9216 8
build_one MOE_DSD   4608 4096 8
