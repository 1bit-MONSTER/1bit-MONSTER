#!/bin/bash
# Build the split two-launch fused decode xclbins (issue #1775 fix):
#   p1 = GU→SiLU→h2 (one launch), p2 = D from h2 (second launch).
# Host-side h2_bo sync between the launches provides the write→read
# visibility barrier the single-launch design lacked.
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

$P/bin/clang++ --target=aie2p-none-unknown-elf --std=c++20 -O2 \
    -DDIM_M=8 -DDIM_K=64 -DDIM_N=128 -Di8_i32_ONLY -DM8_VECTORIZED \
    -isystem $P/include/c++/v1 \
    -I /home/bcloud/Xilinx/2025.2/Vitis/aietools/include \
    -I $M/include/aie_kernels/aie2p \
    -c "$GENERATOR_DIR/mm_kernel_reference.cc" -o /tmp/mm_8x64x128_fused.o
cp /tmp/mm_8x64x128_fused.o /tmp/mm_32x64x128.o

build_one() {  # $1=gen $2=xclbin $3=insts
    local design="/tmp/design_split.$$.mlir"
    trap 'rm -f "$design"; rm -rf "$design.prj"' EXIT
    $PYTHON "$GENERATOR_DIR/$1" -M 8 -K 2048 -N_GU 4096 -N_D 2048 \
        -m 8 -k 64 -n 128 -c 8 -b 2 2>/dev/null > "$design"
    [ -s "$design" ] || { echo "ERROR: $1 empty design" >&2; exit 1; }
    echo "═══ $1 ═══"
    $AIECC --peano="$P" --aietools="$AIETOOLS" \
        --alloc-scheme=basic-sequential --no-xchesscc --no-xbridge \
        --aie-generate-xclbin --no-compile-host --unified --dynamic-objFifos \
        --aie-generate-npu-insts \
        --xclbin-name="$XCLBIN_DIR/$2" --npu-insts-name="$XCLBIN_DIR/$3" \
        "$design" 2>&1 | tail -2
    rm -f "$design"; rm -rf "$design.prj"
}

build_one n1_core_fused_gu_silu_d_p1.py final_i8_MOE_GUSILU_zaya.xclbin insts_i8_MOE_GUSILU_zaya.txt
build_one n1_core_fused_gu_silu_d_p2.py final_i8_MOE_D_zaya_m8h2.xclbin insts_i8_MOE_D_zaya_m8h2.txt
echo "SPLIT BUILD DONE"
