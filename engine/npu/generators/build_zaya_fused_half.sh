#!/bin/bash
# build_zaya_fused_half.sh — 4-col half-array FUSED GU→SiLU→D xclbins.
# Single-context full-FFN kernels at col offsets 0 and 4, so a decode
# process = ONE NPU context on its half → two decode streams fit the
# measured 2-context-per-partition co-schedule (issue #2128).
#
# Outputs (into engine/npu/xclbins):
#   final_i8_MOE_FUSED_zaya_h0.xclbin / insts_..._h0.txt  (cols 0-3)
#   final_i8_MOE_FUSED_zaya_h1.xclbin / insts_..._h1.txt  (cols 4-7)
#
# Strixhalo toolchain paths (see build_zaya_m16_half.sh). Run from generators/.
set -euo pipefail

P=/home/bcloud/mlir-aie/.venv/lib/python3.14/site-packages/llvm-aie
M=/home/bcloud/mlir-aie/.venv/lib/python3.14/site-packages/mlir_aie
PYTHON=/home/bcloud/mlir-aie/.venv/bin/python3
AIECC="$M/bin/aiecc"
AIETOOLS=/home/bcloud/Xilinx/2026.1/Vitis/aietools
export PATH=/home/bcloud/Xilinx/2026.1/Vitis/bin:/opt/xilinx/xrt/bin:$PATH
export PYTHONPATH=/home/bcloud/mlir-aie/.venv/lib/python3.14/site-packages
export LD_LIBRARY_PATH=$AIETOOLS/lib:/home/bcloud/mlir-aie/.venv/lib/python3.14/site-packages/mlir_aie/_mlir_libs

GENERATOR_DIR="$(cd "$(dirname "$0")" && pwd)"
XCLBIN_DIR="$GENERATOR_DIR/../xclbins"
mkdir -p "$XCLBIN_DIR"

workdir="/tmp/zaya_fusedh_build.$$"
mkdir -p "$workdir"
trap 'rm -rf "$workdir"' EXIT

# DIM_M=8 fused microkernel (1x4 mmul + on-core silu_quant), one object reused
# for both halves.
$P/bin/clang++ --target=aie2p-none-unknown-elf --std=c++20 -O2 \
    -DDIM_M=8 -DDIM_K=64 -DDIM_N=128 -Di8_i32_ONLY -DM8_VECTORIZED \
    -DI4_SCALAR_C1 -DI4_SCALAR_C1_ACK_1864 \
    -isystem $P/include/c++/v1 \
    -I $AIETOOLS/include \
    -I $M/include/aie_kernels/aie2p \
    -c "$GENERATOR_DIR/mm_kernel_reference.cc" -o "$workdir/mm_8x64x128_fused.o"
cp "$workdir/mm_8x64x128_fused.o" "$workdir/mm_32x64x128.o"

build_one() {
    local half="$1" off="$2"
    local design="$workdir/design_fused_${half}.mlir"
    local xclbin="$XCLBIN_DIR/final_i8_MOE_FUSED_zaya_${half}.xclbin"
    local insts="$XCLBIN_DIR/insts_i8_MOE_FUSED_zaya_${half}.txt"
    echo "═══ FUSED GU→SiLU→D M=8 K=2048 N_GU=4096 N_D=2048 cols=4 offset=${off} (${half}) ═══"
    $PYTHON "$GENERATOR_DIR/n1_core_fused_gu_silu_d.py" -M 8 -K 2048 \
        -N_GU 4096 -N_D 2048 -m 8 -k 64 -n 128 -c 4 -b 2 -x "$off" 2>/dev/null > "$design"
    [ -s "$design" ] || { echo "ERROR: ${half}: empty design" >&2; exit 1; }
    ( cd "$workdir"
      $AIECC --peano="$P" --aietools="$AIETOOLS" \
        --alloc-scheme=basic-sequential --no-xchesscc --no-xbridge \
        --aie-generate-xclbin --no-compile-host --unified --dynamic-objFifos \
        --aie-generate-npu-insts \
        --xclbin-name="$xclbin" --npu-insts-name="$insts" \
        "$design" 2>&1 | tail -1 )
    ls -la "$xclbin" "$insts"
}

build_one h0 0
build_one h1 4
echo "DONE"
