#!/usr/bin/env bash
# build_nt_gemm.sh — reproducible build for the N-tiled PLAIN bf16 GEMM xclbin
# (fk-3 scale-up): the O-proj (K=NH*HD=2048, N=H=1024) and D (K=IM=3072,
# N=H=1024) stages. Same core-local-A mechanism as build_fk2nt.sh, but no norm
# core (A is already bf16).
#
# Usage: bash build_nt_gemm.sh [M] [K] [N] [k] [NT] [WDEPTH] [outdir]
#   defaults: M=8 K=2048 N=1024 k=64 NT=64 WDEPTH=2  (the O-proj stage)
#   WDEPTH=1 for K=3072 (D): the extra W buffer would not fit with a 48 KB A.
#   env: STACK (default 4096)
set -euo pipefail

M="${1:-8}"; K="${2:-2048}"; N="${3:-1024}"; KT="${4:-64}"; NT="${5:-64}"; WDEPTH="${6:-2}"
STACK="${STACK:-4096}"
OUT="${7:-$HOME/npu-build/ntgemm_M${M}_K${K}_N${N}_k${KT}_NT${NT}_w${WDEPTH}}"

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
[ -x "$CLANG" ] || { echo "FATAL: no clang++ at $CLANG"; exit 1; }
CFLAGS=(--target=aie2p-none-unknown-elf --std=c++20 -O2 -DNDEBUG -D__AIE_API_AIE_ADF_HPP__
        -isystem "$P/include/c++/v1" -I "$VITIS/aietools/include"
        -I "$MLIR/aie_kernels/aie2p"
        -I "$MLIR/.venv/lib/python3.14/site-packages/mlir_aie/include")

rm -rf "$OUT"; mkdir -p "$OUT"; cd "$OUT"
N_K=$(( K / KT ))

echo "== generator: n1_nt_gemm.py -m $M -K $K -N $N -k $KT -NT $NT -wdepth $WDEPTH -stack $STACK"
"$PY" "$G/n1_nt_gemm.py" -m "$M" -K "$K" -N "$N" -k "$KT" -NT "$NT" \
      -wdepth "$WDEPTH" -stack "$STACK" >design.mlir 2>gen.err \
  || { echo "== GENERATOR FAILED"; tail -20 gen.err; exit 1; }
[ -s design.mlir ] || { echo "== EMPTY design.mlir"; tail -20 gen.err; exit 1; }
echo "   design.mlir: $(wc -l <design.mlir) lines"

echo "== kernels (gemm ${M}x${KT}x${NT}, N_K=$N_K)"
if ! "$CLANG" "${CFLAGS[@]}" -DDIM_M="$M" -DDIM_N="$NT" \
      -c "$G/mm_acc.cc" -o mm_acc.o >>cc.log 2>&1; then
  echo "== CC FAILED mm_acc.o"; tail -15 cc.log; exit 1
fi
echo "   cc mm_acc.o"
if ! "$CLANG" "${CFLAGS[@]}" -DDIM_M="$M" -DDIM_K="$KT" -DDIM_N="$NT" -DN_K="$N_K" -Dbf16_f32_ONLY \
      -c "$G/nq_nt.cc" -o nq_nt.o >>cc.log 2>&1; then
  echo "== CC FAILED nq_nt.o"; tail -15 cc.log; exit 1
fi
echo "   cc nq_nt.o (mm.cc + core-local A, $((N_K * M * KT * 2 / 1024)) KB)"

echo "== aiecc"
if ! "$AIECC" --peano="$P" --aietools="$AIETOOLS" \
      --alloc-scheme=basic-sequential --no-xchesscc --no-xbridge --dynamic-objFifos \
      --aie-generate-xclbin --no-compile-host --unified --aie-generate-npu-insts \
      design.mlir -o ntgemm.xclbin >aiecc.log 2>&1; then
  echo "== AIECC FAILED"; tail -25 aiecc.log; exit 1
fi
[ -f main.xclbin ] && cp -f main.xclbin ntgemm.xclbin
[ -f main_seq.bin ] && cp -f main_seq.bin ntgemm_insts.txt
[ -f ntgemm.xclbin ] || { echo "== NO XCLBIN"; tail -20 aiecc.log; exit 1; }
echo "== OK: $OUT/ntgemm.xclbin ($(stat -c%s ntgemm.xclbin) B), ntgemm_insts.txt ($(stat -c%s ntgemm_insts.txt 2>/dev/null || echo 0) B)"
