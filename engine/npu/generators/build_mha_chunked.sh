#!/usr/bin/env bash
# build_mha_chunked.sh — reproducible build for the chunked (flash-attention)
# native bf16 MHA generator n1_mha_chunked.py.
#
# Reconstructed 2026-09-12 (dsh agent) from the crashed agent's session history.
# HOST-SIDE only; running the xclbin needs the NPU.
#
# Usage: bash build_mha_chunked.sh <N_chunk> <C_chunks> [depth] [outdir]
#   e.g. bash build_mha_chunked.sh 128 2            # depth defaults to 2
#        bash build_mha_chunked.sh 64 4 4 ~/npu-build/mha_n64_c4_d4
# Outputs: design.mlir, mha.xclbin, mha_insts.txt, logs.
set -euo pipefail

N="${1:?usage: build_mha_chunked.sh <N_chunk> <C_chunks> [depth] [outdir]}"
C="${2:?usage: build_mha_chunked.sh <N_chunk> <C_chunks> [depth] [outdir]}"
DEPTH="${3:-2}"
OUT="${4:-$HOME/npu-build/mha_n${N}_c${C}_d${DEPTH}}"
HD=128
M=16

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

echo "== generator: n1_mha_chunked.py -M $M -N $N -C $C -HD $HD -D $DEPTH ${VDIRECT:+--v-direct }${VSD:+--vs-depth $VSD}"
"$PY" "$G/n1_mha_chunked.py" -M "$M" -N "$N" -C "$C" -HD "$HD" -D "$DEPTH" \
      ${VDIRECT:+--v-direct} ${VSD:+--vs-depth "$VSD"} ${ASYNCDMA:+--async-dma} >design.mlir 2>gen.err \
  || { echo "== GENERATOR FAILED"; tail -20 gen.err; exit 1; }
[ -s design.mlir ] || { echo "== EMPTY design.mlir"; tail -20 gen.err; exit 1; }
echo "   design.mlir: $(wc -l <design.mlir) lines"

build_cc() {  # build_cc <out.o> <src.cc> [defines...]
  local obj="$1" src="$2"; shift 2
  if ! "$CLANG" "${CFLAGS[@]}" "$@" -c "$G/$src" -o "$obj" >>cc.log 2>&1; then
    echo "== CC FAILED: $src -> $obj"; tail -15 cc.log; exit 1
  fi
  echo "   cc $obj"
}
build_upstream_mm() {  # build_upstream_mm <out.o> [defines...]
  local obj="$1"; shift
  if ! "$CLANG" "${CFLAGS[@]}" "$@" -c "$MLIR/aie_kernels/aie2p/mm.cc" -o "$obj" >>cc.log 2>&1; then
    echo "== CC FAILED: mm.cc -> $obj"; tail -15 cc.log; exit 1
  fi
  echo "   cc $obj"
}

echo "== kernels (QK^T M=$M K=$HD N=$N; PV M=$M K=$N N=$HD)"
build_upstream_mm mm_bf16_f32.o -DDIM_M=$M -DDIM_K=$N -DDIM_N=$HD -Dbf16_f32_ONLY
build_cc mm_qk_concat.o mm_qk_concat.cc -DDIM_M=$M -DDIM_K=$HD -DDIM_N=$N -Dbf16_bf16_ONLY
build_cc zero_qk.o      zero_bf16.cc     -DDIM_M=$M -DDIM_N=$N
build_cc softmax_online.o softmax_online.cc -DM_TILE=$M -DN_KEYS=$N
build_cc combine_attn.o combine_attn.cc   -DM_TILE=$M -DHD=$HD

echo "== aiecc (OBJFIFO='${OBJFIFO:-<default>}')"
if ! "$AIECC" --peano="$P" --aietools="$AIETOOLS" \
      --alloc-scheme=basic-sequential --no-xchesscc --no-xbridge \
      ${OBJFIFO:-} \
      --aie-generate-xclbin --no-compile-host --unified --aie-generate-npu-insts \
      design.mlir -o mha.xclbin >aiecc.log 2>&1; then
  echo "== AIECC FAILED"; tail -25 aiecc.log; exit 1
fi
[ -f main.xclbin ] && cp -f main.xclbin mha.xclbin
[ -f main_seq.bin ] && cp -f main_seq.bin mha_insts.txt
[ -f mha.xclbin ] || { echo "== NO XCLBIN"; tail -20 aiecc.log; exit 1; }
echo "== OK: $OUT/mha.xclbin ($(stat -c%s mha.xclbin) B), mha_insts.txt ($(stat -c%s mha_insts.txt 2>/dev/null || echo 0) B)"
