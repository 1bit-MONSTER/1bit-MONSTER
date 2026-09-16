#!/bin/bash
# build_zaya_fused_quarter.sh — 2-col QUARTER-array FUSED GU→SiLU→D xclbins.
# Four single-context full-FFN kernels at col offsets 0/2/4/6, so a decode
# process = ONE NPU context on its 2-col quarter → up to four decode streams
# per 8-col partition (tests whether the runqueue co-schedules 4 disjoint
# quarter-arrays, or only the measured 2 halves).
#
# Outputs (into engine/npu/xclbins):
#   final_i8_MOE_FUSED_zaya_q0.xclbin / insts_..._q0.txt  (cols 0-1)
#   final_i8_MOE_FUSED_zaya_q1.xclbin / insts_..._q1.txt  (cols 2-3)
#   final_i8_MOE_FUSED_zaya_q2.xclbin / insts_..._q2.txt  (cols 4-5)
#   final_i8_MOE_FUSED_zaya_q3.xclbin / insts_..._q3.txt  (cols 6-7)
#
# Toolchain = build_zaya_fused_half.sh (venv mlir_aie + llvm-aie Peano,
# Xilinx 2026.1 aietools). Run from generators/.
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

workdir="/tmp/zaya_fusedq_build.$$"
mkdir -p "$workdir"
trap 'rm -rf "$workdir"' EXIT

# DIM_M=8 fused microkernel (1x4 mmul + on-core silu_quant), one object reused
# for all four quarters.
$P/bin/clang++ --target=aie2p-none-unknown-elf --std=c++20 -O2 \
    -DDIM_M=8 -DDIM_K=64 -DDIM_N=128 -Di8_i32_ONLY -DM8_VECTORIZED \
    -DI4_SCALAR_C1 -DI4_SCALAR_C1_ACK_1864 \
    -isystem $P/include/c++/v1 \
    -I $AIETOOLS/include \
    -I $M/include/aie_kernels/aie2p \
    -c "$GENERATOR_DIR/mm_kernel_reference.cc" -o "$workdir/mm_8x64x128_fused.o"
cp "$workdir/mm_8x64x128_fused.o" "$workdir/mm_32x64x128.o"

build_one() {
    local q="$1" off="$2"
    local design="$workdir/design_fused_${q}.mlir"
    local xclbin="$XCLBIN_DIR/final_i8_MOE_FUSED_zaya_${q}.xclbin"
    local insts="$XCLBIN_DIR/insts_i8_MOE_FUSED_zaya_${q}.txt"
    echo "═══ FUSED GU→SiLU→D M=8 K=2048 N_GU=4096 N_D=2048 cols=2 offset=${off} (${q}) ═══"
    $PYTHON "$GENERATOR_DIR/n1_core_fused_gu_silu_d.py" -M 8 -K 2048 \
        -N_GU 4096 -N_D 2048 -m 8 -k 64 -n 128 -c 2 -b 2 -x "$off" 2>/dev/null > "$design"
    [ -s "$design" ] || { echo "ERROR: ${q}: empty design" >&2; exit 1; }
    echo "    compute tiles: $(grep -oE '[a-zA-Z_0-9]+ = aie.tile\([0-9]+, 2\)' "$design" | grep -oE 'tile\([0-9]+' | grep -oE '[0-9]+' | sort -n | uniq | tr '\n' ' ' | head -c 40)"
    ( cd "$workdir"
      $AIECC --peano="$P" --aietools="$AIETOOLS" \
        --alloc-scheme=basic-sequential --no-xchesscc --no-xbridge \
        --aie-generate-xclbin --no-compile-host --unified --dynamic-objFifos \
        --aie-generate-npu-insts \
        --xclbin-name="$xclbin" --npu-insts-name="$insts" \
        "$design" 2>&1 | tail -1 )
    ls -la "$xclbin" "$insts"
}

build_one q0 0
build_one q1 2
build_one q2 4
build_one q3 6
echo "DONE"
