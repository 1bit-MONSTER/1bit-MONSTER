#!/bin/bash
# Build the ZAYA int4 fused GU->SiLU->D xclbin with the VECTORIZED aie2p
# backend. Two fixes bundle here:
#   1. Toolchain: LLVM-AIE rebuilt with commit a36c62b9d (ACC1024 fp-acc spills
#      to cmh, not stack) so the vectorized mmul path is selectable (NO
#      I4_SCALAR_C1); without it the compiler forces the scalar fallback.
#   2. Kernel:  mm_kernel_reference.cc dequant restructured from
#      `int8_t Bb[4][64]` + nested jt-loop (the aie2p "register-array V[4] +
#      nested inner-loop" miscompile, #1872 — B'' arrived scrambled at the
#      mmul) into four named aie::vector<int8,64> registers fed straight to the
#      mmul. This makes the vectorized mmul C1 numeric-correct.
# Verified on strixhalo: corr=0.999336 (matches scalar int4), GU 35.3→9.68 ms,
# 0.6→3.1-3.3 tok/s (5x faster than scalar). Emits a NEW xclbin so the
# production final_i8_MOE_GUSILU_i4_zaya.xclbin is untouched.
set -euo pipefail
LLVM=/home/bcloud/llvm-aie-src/install_aie
MLIAIE=/home/bcloud/mlir-aie/build_tmp
G="$(cd "$(dirname "$0")" && pwd)"
W=/tmp/fixed_zaya_i4.$$
mkdir -p "$W"; trap 'rm -rf "$W"' EXIT
CLANG="$LLVM/bin/clang++"
MI=$MLIAIE/install/include  # may not exist; use Vitis include below
AIE_INC=/home/bcloud/Xilinx/2026.1/Vitis/aietools/include
PYTHON=/home/bcloud/mlir-aie/.venv/bin/python3
AIECC=/home/bcloud/mlir-aie/build_tmp/bin/aiecc
# The aie2p kernel includes live in the mlir_aie python package
M_INC=/home/bcloud/mlir-aie/.venv/lib/python3.14/site-packages/mlir_aie/include
# libc++ headers for the aie2p target (from the llvm-aie install)
CPP_INC=/home/bcloud/llvm-aie-src/install_aie/include/c++/v1

echo "═══ compile vectorized mm_kernel_reference.cc with FIXED clang ═══"
"$CLANG" --target=aie2p-none-unknown-elf --std=c++20 -O2 \
  -DDIM_M=8 -DDIM_K=64 -DDIM_N=128 -Di8_i32_ONLY -DM8_VECTORIZED \
  -isystem "$CPP_INC" -I "$M_INC" -I "$M_INC/aie_kernels/aie2p" -I "$AIE_INC" \
  -c "$G/mm_kernel_reference.cc" -o "$W/mm.o" 2>&1 | tail -5 || { echo "KERNEL COMPILE FAILED"; exit 1; }
"$CLANG" --target=aie2p-none-unknown-elf --std=c++20 -O2 \
  -DDIM_M=8 -DDIM_K=64 -DDIM_N=128 -Di8_i32_ONLY -DM8_VECTORIZED \
  -isystem "$CPP_INC" -I "$M_INC" -I "$M_INC/aie_kernels/aie2p" -I "$AIE_INC" \
  -c "$G/attn_kernel_reference.cc" -o "$W/silu.o" 2>&1 | tail -5 || { echo "SILU COMPILE FAILED"; exit 1; }
"$CLANG" --target=aie2p-none-unknown-elf --std=c++20 -O2 \
  -isystem "$CPP_INC" -I "$M_INC" -I "$M_INC" -I "$M_INC/aie_kernels/aie2p" -I "$AIE_INC" -I "$G" \
  -c "$G/i4_dequant_kernel.cc" -o "$W/dequant.o" 2>&1 | tail -5 || { echo "DEQUANT COMPILE FAILED"; exit 1; }
"$LLVM/bin/ld.lld" -r "$W/mm.o" "$W/silu.o" "$W/dequant.o" -o "$W/mm_32x64x128.o"
for sym in matmul_i8_i32_i4 silu_quant_i8_fused_i4 unpack_i4_b zero_i32 zero_c1 copy_c1; do
  if ! "$LLVM/bin/llvm-nm" "$W/mm_32x64x128.o" 2>/dev/null | grep -qE " T $sym\$"; then
    echo "ERROR: missing symbol '$sym'" >&2; exit 1
  fi
done
echo "kernel object OK"

echo "═══ generate zaya int4 design ═══"
"$PYTHON" "$G/n1_core_fused_gu_silu_d_p1_i4.py" -M 8 -K 2048 -N_GU 4096 -N_D 2048 \
  -m 8 -k 64 -n 128 -c 8 -b 2 > "$W/design.mlir" 2>/dev/null
[ -s "$W/design.mlir" ] || { echo "design empty"; exit 1; }

echo "═══ aiecc → xclbin (vectorized int4 for Zaya) ═══"
export PATH="/home/bcloud/Xilinx/2026.1/2026.1/Vitis/bin:/opt/xilinx/xrt/bin:$PATH"
export PYTHONPATH="/home/bcloud/mlir-aie/install_tmp/python"
export LD_LIBRARY_PATH="/home/bcloud/mlir-aie/install_tmp/python/aie/_mlir_libs"
XCLBIN="$G/../xclbins/final_i8_MOE_GUSILU_i4_zaya_VECFIX.xclbin"
INSTS="$G/../xclbins/insts_final_i8_MOE_GUSILU_i4_zaya_VECFIX.txt"
cd "$W"
"$AIECC" --peano="$LLVM" --aietools="$MLIAIE" \
  --alloc-scheme=basic-sequential --no-xchesscc --no-xbridge \
  --aie-generate-xclbin --no-compile-host --unified --dynamic-objFifos \
  --aie-generate-npu-insts \
  --xclbin-name="$XCLBIN" --npu-insts-name="$INSTS" \
  "$W/design.mlir" 2>&1 | tail -8
ls -la "$XCLBIN" "$INSTS"
echo "DONE: $XCLBIN"
