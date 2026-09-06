#!/bin/bash
# build_zaya_m16_half.sh — 4-col half-array Zaya M=16 GU/D xclbins (option A:
# column-sliced kernels so two contexts own disjoint array halves and the
# runqueue can co-schedule them — issue #2128 full-array mutual exclusion).
#
# Builds 4 xclbins + insts into engine/npu/xclbins:
#   GU/D at col offset 0 (cols 0-3)  -> *_h0 (half 0)
#   GU/D at col offset 4 (cols 4-7)  -> *_h1 (half 1)
#
# Strixhalo toolchain (paths verified 2026-09-06):
#   venv mlir_aie + llvm-aie(Peano) wheels, aiecc in the wheel bin dir,
#   Xilinx 2026.1 Vitis aietools.  Run from engine/npu/generators.
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

workdir="/tmp/zaya_m16h_build.$$"
mkdir -p "$workdir"
trap 'rm -rf "$workdir"' EXIT

# 1. Compile the DIM_M=16 microkernel once (same object for both halves).
KERNOBJ="$workdir/mm_16x64x128.o"
$P/bin/clang++ --target=aie2p-none-unknown-elf --std=c++20 -O2 \
    -DDIM_M=16 -DDIM_K=64 -DDIM_N=128 -Di8_i32_ONLY \
    -isystem $P/include/c++/v1 \
    -I $AIETOOLS/include \
    -I $M/include/aie_kernels/aie2p \
    -c "$GENERATOR_DIR/mm_kernel_reference.cc" -o "$KERNOBJ"

cp "$KERNOBJ" "$workdir/mm_32x64x128.o"

build_one() {
    local proj="$1" K="$2" N="$3" half="$4" off="$5"
    local design="$workdir/design_${proj}_${half}.mlir"
    local xclbin="$XCLBIN_DIR/final_i8_MOE_${proj}_zaya_m16_${half}.xclbin"
    local insts="$XCLBIN_DIR/insts_i8_MOE_${proj}_zaya_m16_${half}.txt"
    echo "═══ ${proj} M=16 K=${K} N=${N} cols=4 offset=${off} (${half}) ═══"
    $PYTHON "$GENERATOR_DIR/n1_core_i8_v27.py" -M 16 -K "$K" -N "$N" \
        -m 16 -k 64 -n 128 -c 4 -r 1 -b 5 -x "$off" 2>/dev/null > "$design"
    [ -s "$design" ] || { echo "ERROR: ${proj}/${half}: empty design" >&2; exit 1; }
    # sanity: confirm the grid landed on the intended columns
    echo "    grid cols: $(grep -oE 'tile\(([0-9]+), ?[0-9]+\)' "$design" | grep -oE '[0-9]+' | sort -n | uniq | tr '\n' ' ' | head -c 60)"
    ( cd "$workdir"
      $AIECC --peano="$P" --aietools="$AIETOOLS" \
        --alloc-scheme=basic-sequential --no-xchesscc --no-xbridge \
        --aie-generate-xclbin --no-compile-host --unified --dynamic-objFifos \
        --aie-generate-npu-insts \
        --xclbin-name="$xclbin" --npu-insts-name="$insts" \
        "$design" 2>&1 | tail -1 )
    ls -la "$xclbin" "$insts"
}

build_one GU 2048 4096 h0 0
build_one GU 2048 4096 h1 4
build_one D  2048 2048 h0 0
build_one D  2048 2048 h1 4
echo "DONE"
