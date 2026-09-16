#!/usr/bin/env bash
# build_fk2nt.sh — reproducible build for the N-TILED fused RMSNorm+QKV xclbin
# (fk-3 scale-up). fk-2 is single-N-tile (N=128); this design holds the whole
# (M x H) A_norm in the GEMM core's local memory so the real 0.6B QKV width
# (N=4096) can be produced with the A_norm handoff streamed exactly once
# (see nq_nt.cc for why the re-stream path is avoided).
#
# C output is f32 (f32 K-tile accumulator), same as fk-2.
#
# Usage: bash build_fk2nt.sh [M] [H] [N] [k] [NT] [outdir]
#   defaults: M=16 H=1024 N=4096 k=64 NT=128  (Qwen3-0.6B QKV)
set -euo pipefail

M="${1:-16}"; H="${2:-1024}"; N="${3:-4096}"; K="${4:-64}"; NT="${5:-128}"
OUT="${6:-$HOME/npu-build/fk2nt_m${M}_H${H}_N${N}_k${K}_NT${NT}}"

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

echo "== generator: n1_fused_norm_qkv_nt.py -m $M -H $H -N $N -k $K -NT $NT"
"$PY" "$G/n1_fused_norm_qkv_nt.py" -m "$M" -H "$H" -N "$N" -k "$K" -NT "$NT" \
    >design.mlir 2>gen.err \
  || { echo "== GENERATOR FAILED"; tail -20 gen.err; exit 1; }
[ -s design.mlir ] || { echo "== EMPTY design.mlir"; tail -20 gen.err; exit 1; }
echo "   design.mlir: $(wc -l <design.mlir) lines"

N_K=$(( H / K ))
echo "== kernels (norm M_TILE=$M K_TILE=$K H=$H; gemm ${M}x${K}x${NT}, N_K=$N_K)"
if ! "$CLANG" "${CFLAGS[@]}" -DM_TILE="$M" -DK_TILE="$K" -DH="$H" \
      -c "$G/rms_norm_split.cc" -o rms_split.o >>cc.log 2>&1; then
  echo "== CC FAILED rms_split.o"; tail -15 cc.log; exit 1
fi
echo "   cc rms_split.o"
if ! "$CLANG" "${CFLAGS[@]}" -DDIM_M="$M" -DDIM_N="$NT" \
      -c "$G/mm_acc.cc" -o mm_acc.o >>cc.log 2>&1; then
  echo "== CC FAILED mm_acc.o"; tail -15 cc.log; exit 1
fi
echo "   cc mm_acc.o"
if ! "$CLANG" "${CFLAGS[@]}" -DDIM_M="$M" -DDIM_K="$K" -DDIM_N="$NT" -DN_K="$N_K" -Dbf16_f32_ONLY \
      -c "$G/nq_nt.cc" -o nq_nt.o >>cc.log 2>&1; then
  echo "== CC FAILED nq_nt.o"; tail -15 cc.log; exit 1
fi
echo "   cc nq_nt.o (core-local A_norm, $((N_K * M * K * 2 / 1024)) KB)"

echo "== aiecc"
if ! "$AIECC" --peano="$P" --aietools="$AIETOOLS" \
      --alloc-scheme=basic-sequential --no-xchesscc --no-xbridge --dynamic-objFifos \
      --aie-generate-xclbin --no-compile-host --unified --aie-generate-npu-insts \
      design.mlir -o fk2nt.xclbin >aiecc.log 2>&1; then
  echo "== AIECC FAILED"; tail -25 aiecc.log; exit 1
fi
[ -f main.xclbin ] && cp -f main.xclbin fk2nt.xclbin
[ -f main_seq.bin ] && cp -f main_seq.bin fk2nt_insts.txt
[ -f fk2nt.xclbin ] || { echo "== NO XCLBIN"; tail -20 aiecc.log; exit 1; }
echo "== OK: $OUT/fk2nt.xclbin ($(stat -c%s fk2nt.xclbin) B), fk2nt_insts.txt ($(stat -c%s fk2nt_insts.txt 2>/dev/null || echo 0) B)"
