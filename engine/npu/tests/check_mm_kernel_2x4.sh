#!/bin/bash
# check_mm_kernel_2x4.sh — rebuild the 2x4 n-expansion GEMM kernel (mm_kernel_reference.cc)
# and verify it on the NPU with the analytical correctness harness.
#
# Fails (non-zero exit) if the kernel computes wrong results — wrong B-tile
# selection, wrong C-store offsets, or a broken k-reduction all surface as
# mismatches in bench_gemm_analytical (pass 0 = dataflow/k-reduction,
# pass 1 = output-tile placement).
#
# Usage: ./check_mm_kernel_2x4.sh   (needs the Strix Halo NPU host + XRT)
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
GEN="$REPO/engine/npu/generators"
TESTS="$REPO/engine/npu/tests"
CLANG=~/mlir-aie/.venv/lib/python3.14/site-packages/llvm-aie/bin/clang++
PYTHON=~/mlir-aie/.venv/bin/python3
AIECC=~/mlir-aie/build_tmp/bin/aiecc
PEANO=~/mlir-aie/.venv/lib/python3.14/site-packages/llvm-aie
AIETOOLS=~/mlir-aie/build_tmp
# Issue #1913 guard: AIETOOLS=~/mlir-aie/build_tmp is only valid because this
# script defaults to --no-xchesscc. Set USE_XCHESSCC=1 to enable the chess arm
# (--xchesscc --xbridge) — then --aietools MUST be the Vitis aietools root, or
# aiecc silently skips chess-llvm-link and fails later with a confusing
# 'main_input.chesslinked.ll' missing error. The guard fails loudly instead.
# shellcheck source=../generators/check_chess_aietools.sh
# shellcheck disable=SC1091
source "$GEN/check_chess_aietools.sh"
USE_XCHESSCC="${USE_XCHESSCC:-0}"
XCHESS_ARGS=("--no-xchesscc" "--no-xbridge")
if [ "$USE_XCHESSCC" = "1" ]; then
    XCHESS_ARGS=("--xchesscc" "--xbridge")
fi
export PATH=/home/bcloud/Xilinx/2026.1/2026.1/Vitis/bin:/opt/xilinx/xrt/bin:$PATH
export PYTHONPATH=~/mlir-aie/install_tmp/python:~/mlir-aie/.venv/lib/python3.14/site-packages
export LD_LIBRARY_PATH=~/mlir-aie/install_tmp/python/aie/_mlir_libs

W=$(mktemp -d /tmp/mm2x4_check.XXXXXX)
trap 'rm -rf "$W"' EXIT

echo "== 1/4 compile kernel (i8_i32, 32x64x128) =="
"$CLANG" "$GEN/mm_kernel_reference.cc" -c -o "$W/mm_32x64x128.o" \
  -I ~/mlir-aie/.venv/lib/python3.14/site-packages/mlir_aie/include \
  -I ~/mlir-aie/aie_kernels/aie2p \
  -std=c++20 -O2 -DNDEBUG -D__AIE_API_AIE_ADF_HPP__ \
  --target=aie2p-none-unknown-elf \
  -DDIM_M=32 -DDIM_K=64 -DDIM_N=128 -Di8_i32_ONLY

echo "== 2/4 generate design + build xclbin (QKV 128x2048x8192) =="
cd "$W"
if ! check_chess_aietools "$AIETOOLS" "$([ "$USE_XCHESSCC" = "1" ] && echo true || echo false)" \
    "/home/bcloud/Xilinx/2026.1/2026.1/Vitis/bin:/opt/xilinx/xrt/bin"; then
    exit 1
fi
"$PYTHON" "$GEN/n1_core_i8_v27.py" -M 128 -K 2048 -N 8192 -m 32 -k 64 -n 128 -c 8 -r 4 -b 5 \
  2>/dev/null > design.mlir
"$AIECC" --peano="$PEANO" --aietools="$AIETOOLS" \
  --alloc-scheme=basic-sequential "${XCHESS_ARGS[@]}" \
  --aie-generate-xclbin --no-compile-host --unified --dynamic-objFifos \
  --aie-generate-npu-insts \
  --xclbin-name="$W/final.xclbin" --npu-insts-name="$W/insts.txt" \
  design.mlir >/dev/null 2>&1

echo "== 3/4 build harness =="
g++ -std=gnu++17 -O2 "$TESTS/bench_gemm_analytical.cpp" \
  -I/opt/xilinx/xrt/include -L/opt/xilinx/xrt/lib \
  -Wl,-rpath,/opt/xilinx/xrt/lib -lxrt_coreutil -lxrt_core -o "$W/bench"

echo "== 4/4 hardware check =="
OUT=$(LD_LIBRARY_PATH=/opt/xilinx/xrt/lib "$W/bench" \
  "$W/final.xclbin" "$W/insts.txt" 128 2048 8192 20)
echo "$OUT"
if ! echo "$OUT" | grep -q "^PASS$"; then
  echo "FAIL: kernel produced wrong results" >&2
  exit 1
fi
echo "OK: 2x4 kernel correct"
