#!/bin/bash
# Build the GQA flash-attention xclbin (issue #1776).
#
# STATUS (2026-08-23): the design BUILDS (aiecc) and the kernel contract is
# verified on x86 (test_attn.cpp), but the hardware run hangs in the multi-
# phase core (QK^T → softmax → PV with the A2 round-trip through DDR). The
# no-QK^T variant runs in 0.4 ms, isolating the hang to the QK^T A/B fifo
# phase. Next debugging step: the A/B fifo sync or the A2O writeback timing.
#
# Usage: bash build_attn.sh
set -euo pipefail
P=/home/bcloud/mlir-aie/.venv/lib/python3.14/site-packages/llvm-aie
M=/home/bcloud/mlir-aie/.venv/lib/python3.14/site-packages/mlir_aie
PYTHON=/home/bcloud/mlir-aie/.venv/bin/python3
G=$(cd "$(dirname "$0")" && pwd)
W=/tmp/attn_build.$$
mkdir -p "$W"; trap 'rm -rf "$W"' EXIT

# 1. kernel: matmul (mm_kernel_reference.cc) + attn_softmax_i8 in one object
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
    -c "$G/attn_kernel_reference.cc" -o "$W/softmax.o" 2>/dev/null
$P/bin/ld.lld -r "$W/mm.o" "$W/softmax.o" -o "$W/attn_kernel.o"

# 2. design
$PYTHON "$G/n1_core_attn.py" -M 8 -K 128 -N 256 -m 8 -k 64 -n 128 -c 8 -b 2 \
    > "$W/design.mlir" 2>/dev/null
cp "$W/attn_kernel.o" "$W/attn_kernel.o"  # link_with resolves from CWD
cd "$W" && cp attn_kernel.o attn_kernel.o
export PATH=/home/bcloud/Xilinx/2026.1/2026.1/Vitis/bin:/opt/xilinx/xrt/bin:$PATH
export PYTHONPATH=/home/bcloud/mlir-aie/install_tmp/python:/home/bcloud/mlir-aie/.venv/lib/python3.14/site-packages
export LD_LIBRARY_PATH=/home/bcloud/mlir-aie/install_tmp/python/aie/_mlir_libs
/home/bcloud/mlir-aie/build_tmp/bin/aiecc --peano="$P" --aietools="$M" \
    --alloc-scheme=basic-sequential --no-xchesscc --no-xbridge \
    --aie-generate-xclbin --no-compile-host --unified --dynamic-objFifos \
    --aie-generate-npu-insts \
    --xclbin-name="$G/../xclbins/attn.xclbin" \
    --npu-insts-name="$G/../xclbins/attn_insts.txt" \
    "$W/design.mlir"
