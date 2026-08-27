#!/bin/bash
# Build the GQA flash-attention xclbin (issue #1776).
#
# STATUS (2026-08-24): the full multi-phase design is VERIFIED on strixhalo.
# QK^T (c1a/c1b = 73984 exactly for the 0x11/0x22 pattern), softmax
# (A2 = 127 for t < seq=200, 0 for t >= seq — causal mask + rows 1-7 zero),
# and PV (C2 = 127·Σ_{t<200}(t%7+1) = 100838) all match the x86 contract
# (test_attn.cpp — PASS). Hardware blockers found & fixed this round:
#   - the i4 B-path used std::roundf, which the Peano libc++ cannot resolve
#     ("reference to unresolved using declaration") — the attention kernel
#     object failed to compile; fixed by using silu_roundf (no-libm, same
#     round-half-away-from-zero semantics) in mm_kernel_reference.cc;
#   - stale prebuilt kernel .o files (mm_32x64x128.o) from earlier probe
#     sessions double-write their C result into the adjacent buffer,
#     corrupting C1b (3 QK^T dots instead of 2) → wrong softmax max → all-zero
#     A2. Always rebuild the kernel from source; do not reuse old .o files.
# Run: bash build_attn.sh
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
$PYTHON "$G/n1_core_attn.py" -M 8 -K 128 -N "${NPU_ATTN_N:-512}" -m 8 -k 64 -n 128 -c 8 -b 2 \
    > "$W/design.mlir" 2>/dev/null
cd "$W"  # link_with resolves attn_kernel.o from CWD
export PATH=/home/bcloud/Xilinx/2026.1/Vitis/bin:/opt/xilinx/xrt/bin:$PATH
export PYTHONPATH=/home/bcloud/mlir-aie/install_tmp/python:/home/bcloud/mlir-aie/.venv/lib/python3.14/site-packages
export LD_LIBRARY_PATH=/home/bcloud/mlir-aie/install_tmp/python/aie/_mlir_libs
/home/bcloud/mlir-aie/build_tmp/bin/aiecc --peano="$P" --aietools="$M" \
    --alloc-scheme=basic-sequential --no-xchesscc --no-xbridge \
    --aie-generate-xclbin --no-compile-host --unified --dynamic-objFifos \
    --aie-generate-npu-insts \
    --xclbin-name="$G/../xclbins/attn.xclbin" \
    --npu-insts-name="$G/../xclbins/attn_insts.txt" \
    "$W/design.mlir"
