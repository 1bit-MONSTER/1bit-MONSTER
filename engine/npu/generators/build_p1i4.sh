#!/bin/bash
# Build the int4-GU fused GUSILU xclbin (issue #1769, ws09 kernel round).
set -euo pipefail
P=/home/bcloud/mlir-aie/.venv/lib/python3.14/site-packages/llvm-aie
M=/home/bcloud/mlir-aie/.venv/lib/python3.14/site-packages/mlir_aie
PYTHON=/home/bcloud/mlir-aie/.venv/bin/python3
G=$(cd "$(dirname "$0")" && pwd)
W=/tmp/p1i4_build.$$
mkdir -p "$W"; trap 'rm -rf "$W"' EXIT

# kernel: matmul + silu + unpack + dequant in one object
$P/bin/clang++ --target=aie2p-none-unknown-elf --std=c++20 -O2 \
    -DDIM_M=8 -DDIM_K=64 -DDIM_N=128 -Di8_i32_ONLY -DM8_VECTORIZED \
    -isystem $P/include/c++/v1 \
    -I /home/bcloud/Xilinx/2025.2/Vitis/aietools/include \
    -I $M/include/aie_kernels/aie2p \
    -c "$G/mm_kernel_reference.cc" -o "$W/mm.o" 2>/dev/null
$P/bin/clang++ --target=aie2p-none-unknown-elf --std=c++20 -O2 \
    -DDIM_M=8 -DDIM_K=64 -DDIM_N=128 -Di8_i32_ONLY -DM8_VECTORIZED \
    -isystem $P/include/c++/v1 \
    -I /home/bcloud/Xilinx/2025.2/Vitis/aietools/include \
    -I $M/include/aie_kernels/aie2p \
    -c "$G/attn_kernel_reference.cc" -o "$W/silu.o" 2>/dev/null
$P/bin/clang++ --target=aie2p-none-unknown-elf --std=c++20 -O2 \
    -isystem $P/include/c++/v1 \
    -I /home/bcloud/Xilinx/2025.2/Vitis/aietools/include \
    -I $M/include/aie_kernels/aie2p \
    -I "$G" \
    -c "$G/i4_dequant_kernel.cc" -o "$W/dequant.o" 2>/dev/null
$P/bin/ld.lld -r "$W/mm.o" "$W/silu.o" "$W/dequant.o" -o "$W/mm_32x64x128.o"

$PYTHON "$G/n1_core_fused_gu_silu_d_p1_i4.py" -M 8 -K 2048 -N_GU 4096 -N_D 2048 \
    -m 8 -k 64 -n 128 -c 8 -b 2 > "$W/design.mlir" 2>/dev/null
export PATH=/home/bcloud/Xilinx/2026.1/2026.1/Vitis/bin:/opt/xilinx/xrt/bin:$PATH
export PYTHONPATH=/home/bcloud/mlir-aie/install_tmp/python:/home/bcloud/mlir-aie/.venv/lib/python3.14/site-packages
export LD_LIBRARY_PATH=/home/bcloud/mlir-aie/install_tmp/python/aie/_mlir_libs
cd "$W"   # aiecc resolves link_with objects relative to the CWD (stale generators/mm_32x64x128.o bug)
/home/bcloud/mlir-aie/build_tmp/bin/aiecc --peano="$P" --aietools="$M" \
    --alloc-scheme=basic-sequential --no-xchesscc --no-xbridge \
    --aie-generate-xclbin --no-compile-host --unified --dynamic-objFifos \
    --aie-generate-npu-insts \
    --xclbin-name="$G/../xclbins/final_i8_MOE_GUSILU_i4_zaya.xclbin" \
    --npu-insts-name="$G/../xclbins/insts_i8_MOE_GUSILU_i4_zaya.txt" \
    "$W/design.mlir"
