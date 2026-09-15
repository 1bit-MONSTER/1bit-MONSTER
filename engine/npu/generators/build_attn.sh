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
#   NPU_ATTN_N=1024       MAX_SEQ (default 512)
#   NPU_ATTN_K=256        head dim (default 128); K != 128 needs the PV N-split,
#                         which divides K into K/n tiles of n=128
#   NPU_ATTN_COLS=8       AIE columns == q heads per pass (default 8)
#   NPU_ATTN_HEADS=<nh>   total q heads, when nh > cols (multi-pass head blocks;
#                         must be a multiple of --cols)
#   NPU_ATTN_NKV=<nkv>    kv heads (default 2 -> gqa 4)
#   NPU_ATTN_XCLBIN=/path NPU_ATTN_INSTS=/path   override the outputs (default
#                         writes the shipped engine/npu/xclbins/attn{.xclbin,_insts.txt})
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
# Outputs default to the shipped artifacts; override them (e.g. to build a
# candidate without touching the tracked attn.xclbin / attn_insts.txt).
XCLBIN_OUT="${NPU_ATTN_XCLBIN:-$G/../xclbins/attn.xclbin}"
INSTS_OUT="${NPU_ATTN_INSTS:-$G/../xclbins/attn_insts.txt}"
HEADS_ARG=""
[ -n "${NPU_ATTN_HEADS:-}" ] && HEADS_ARG="-H ${NPU_ATTN_HEADS}"
NKV_ARG=""
[ -n "${NPU_ATTN_NKV:-}" ] && NKV_ARG="--nkv ${NPU_ATTN_NKV}"
$PYTHON "$G/n1_core_attn.py" -M 8 -K "${NPU_ATTN_K:-128}" -N "${NPU_ATTN_N:-512}" \
    -m 8 -k 64 -n 128 -c "${NPU_ATTN_COLS:-8}" $NKV_ARG -b 2 $HEADS_ARG \
    > "$W/design.mlir" 2>/dev/null
cd "$W"  # link_with resolves attn_kernel.o from CWD
export PATH=/home/bcloud/Xilinx/2026.1/Vitis/bin:/opt/xilinx/xrt/bin:$PATH
# The python bindings and the aiecc binary MUST come from the same install. The
# bindings here are install_tmp's (the emitted MLIR is in that revision's dialect
# syntax) while aiecc was being taken from build_tmp -- a tree rebuilt later, whose
# parser rejects it:
#     build_tmp aiecc + install_tmp bindings -> "design.mlir:NNN:44: error: expected ')'"
#     install_tmp aiecc + install_tmp bindings -> compiles, and reproduces the
#                                                 shipped attn_insts.txt byte-for-byte
# (AttnNL: the mismatch is a stale path in this script, NOT the local mlir-aie WIP
# patch -- that was my misdiagnosis. Do not "fix" it by reverting the patch.)
export PYTHONPATH=/home/bcloud/mlir-aie/install_tmp/python:/home/bcloud/mlir-aie/.venv/lib/python3.14/site-packages
export LD_LIBRARY_PATH=/home/bcloud/mlir-aie/install_tmp/python/aie/_mlir_libs
/home/bcloud/mlir-aie/install_tmp/bin/aiecc --peano="$P" --aietools="$M" \
    --alloc-scheme=basic-sequential --no-xchesscc --no-xbridge \
    --aie-generate-xclbin --no-compile-host --unified --dynamic-objFifos \
    --aie-generate-npu-insts \
    --xclbin-name="$XCLBIN_OUT" \
    --npu-insts-name="$INSTS_OUT" \
    "$W/design.mlir"
echo "built: $XCLBIN_OUT"
echo "insts: $INSTS_OUT ($(sha256sum "$INSTS_OUT" | cut -c1-16))"
