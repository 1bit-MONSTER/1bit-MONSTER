#!/usr/bin/env bash
# build_fk2.sh — reproducible build for the fk-2 fused RMSNorm+QKV xclbin.
# C output is now f32 (f32 K-tile accumulator) — consumers convert to bf16.
#
# Reconstructed 2026-09-12 (dsh agent) from the crashed agent's session history;
# the original .o files (mm_bf16_16x64x128.o) lived in ephemeral /tmp and were
# lost. HOST-SIDE build only; running the xclbin needs the NPU.
#
# Layout: A is (M+1) x H f32 — rows 0..M-1 = activations, row M = learned gamma.
#         W is H x N bf16, C is M x N bf16.  The norm is K-tiled: k=64, n_k=H/k.
#         rms_norm_split.cc folds gamma from A row M_TILE, so M_TILE must == M.
#
# Usage: bash build_fk2.sh [M] [H] [N] [k] [outdir]
#   defaults: M=16 H=1024 N=128 k=64  (real Qwen3-0.6B hidden size)
set -euo pipefail

M="${1:-16}"; H="${2:-1024}"; N="${3:-128}"; K="${4:-64}"
OUT="${5:-$HOME/npu-build/fk2_m${M}_H${H}_N${N}_k${K}}"

G="$(cd "$(dirname "$0")" && pwd)"
MLIR=/home/bcloud/mlir-aie
P="$MLIR/.venv/lib/python3.14/site-packages/llvm-aie"
PY="$MLIR/.venv/bin/python"
AIECC="$MLIR/build_tmp/bin/aiecc"
AIETOOLS="$MLIR/build_tmp"
VITIS=/home/bcloud/Xilinx/2026.1/Vitis
export PATH="$VITIS/bin:/opt/xilinx/xrt/bin:$PATH"
export PYTHONPATH="$AIETOOLS/python:$MLIR/.venv/lib/python3.14/site-packages"
export LD_LIBRARY_PATH="$AIETOOLS/python/aie/_mlir_libs"

CLANG="$P/bin/clang++"
CFLAGS=(--target=aie2p-none-unknown-elf --std=c++20 -O2 -DNDEBUG -D__AIE_API_AIE_ADF_HPP__
        -isystem "$P/include/c++/v1" -I "$VITIS/aietools/include"
        -I "$MLIR/aie_kernels/aie2p"
        -I "$MLIR/.venv/lib/python3.14/site-packages/mlir_aie/include")

rm -rf "$OUT"; mkdir -p "$OUT"; cd "$OUT"

echo "== generator: n1_fused_rmsnorm_qkv.py -m $M -H $H -N $N -k $K"
"$PY" "$G/n1_fused_rmsnorm_qkv.py" -m "$M" -H "$H" -N "$N" -k "$K" >design.mlir 2>gen.err \
  || { echo "== GENERATOR FAILED"; tail -20 gen.err; exit 1; }
[ -s design.mlir ] || { echo "== EMPTY design.mlir"; tail -20 gen.err; exit 1; }
echo "   design.mlir: $(wc -l <design.mlir) lines"

echo "== kernels (norm M_TILE=$M K_TILE=$K H=$H; gemm ${M}x${K}x${N})"
if ! "$CLANG" "${CFLAGS[@]}" -DM_TILE="$M" -DK_TILE="$K" -DH="$H" \
      -c "$G/rms_norm_split.cc" -o rms_split.o >>cc.log 2>&1; then
  echo "== CC FAILED rms_norm_split.o"; tail -15 cc.log; exit 1
fi
echo "   cc rms_split.o"
# f32 C accumulator. The bf16-C variant (mm_bf16_16x64x128.o) rounds the
# accumulator on EVERY K-tile call (bf16 has 8 mantissa bits), which is wrong
# for n_k>1. matmul_bf16_f32 keeps C in f32; acc_zero clears it per output tile.
if ! "$CLANG" "${CFLAGS[@]}" -DDIM_M="$M" -DDIM_K="$K" -DDIM_N="$N" -Dbf16_f32_ONLY \
      -c "$MLIR/aie_kernels/aie2p/mm.cc" -o mm_bf16_f32.o >>cc.log 2>&1; then
  echo "== CC FAILED mm_bf16_f32.o"; tail -15 cc.log; exit 1
fi
echo "   cc mm_bf16_f32.o (f32 K-tile accumulator)"
if ! "$CLANG" "${CFLAGS[@]}" -DDIM_M="$M" -DDIM_N="$N" \
      -c "$G/mm_acc.cc" -o mm_acc.o >>cc.log 2>&1; then
  echo "== CC FAILED mm_acc.o"; tail -15 cc.log; exit 1
fi
echo "   cc mm_acc.o"

echo "== aiecc"
if ! "$AIECC" --peano="$P" --aietools="$AIETOOLS" \
      --alloc-scheme=basic-sequential --no-xchesscc --no-xbridge --dynamic-objFifos \
      --aie-generate-xclbin --no-compile-host --unified --aie-generate-npu-insts \
      design.mlir -o fk2.xclbin >aiecc.log 2>&1; then
  echo "== AIECC FAILED"; tail -25 aiecc.log; exit 1
fi
[ -f main.xclbin ] && cp -f main.xclbin fk2.xclbin
[ -f main_seq.bin ] && cp -f main_seq.bin fk2_insts.txt
[ -f fk2.xclbin ] || { echo "== NO XCLBIN"; tail -20 aiecc.log; exit 1; }
echo "== OK: $OUT/fk2.xclbin ($(stat -c%s fk2.xclbin) B), fk2_insts.txt ($(stat -c%s fk2_insts.txt 2>/dev/null || echo 0) B)"
