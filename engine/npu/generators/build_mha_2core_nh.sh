#!/usr/bin/env bash
# build_mha_2core_nh.sh — build the 2-cores-per-head fused chunked attention
# (n1_mha_2core_nh.py: qk_softmax.cc + pv_combine.cc).
#
# Usage: bash build_mha_2core_nh.sh <N_chunk> <C_chunks> <NH> [depth] [outdir]
# Outputs: design.mlir, mha2.xclbin, mha2_insts.txt, logs.
set -euo pipefail

N="${1:?usage: build_mha_2core_nh.sh <N_chunk> <C_chunks> <NH> [depth] [outdir]}"
C="${2:?}"
NH="${3:-2}"; DEPTH="${4:-2}"; PERCOL="${PERCOL:-1}"; OUT="${5:-$HOME/npu-build/mha2_nh${NH}_p${PERCOL}_n${N}_c${C}_d${DEPTH}}"
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

echo "== generator: n1_mha_2core_nh.py -M $M -N $N -C $C -HD $HD -D $DEPTH -NH $NH"
"$PY" "$G/n1_mha_2core_nh.py" -M "$M" -N "$N" -C "$C" -HD "$HD" -D "$DEPTH" -NH "$NH" -P "$PERCOL" \
      >design.mlir 2>gen.err \
  || { echo "== GENERATOR FAILED"; tail -20 gen.err; exit 1; }
[ -s design.mlir ] || { echo "== EMPTY design.mlir"; tail -20 gen.err; exit 1; }
echo "   design.mlir: $(wc -l <design.mlir) lines"

build_cc() { local obj="$1" src="$2"; shift 2
  if ! "$CLANG" "${CFLAGS[@]}" "$@" -c "$G/$src" -o "$obj" >>cc.log 2>&1; then
    echo "== CC FAILED: $src -> $obj"; tail -15 cc.log; exit 1; fi
  echo "   cc $obj"; }

echo "== fused kernels (qk_softmax QK^T ${M}x${HD}x${N}; pv_combine ${M}x${N}x${HD})"
build_cc qk_softmax.o qk_softmax.cc -DDIM_M=$M -DDIM_K=$HD -DDIM_N=$N -DM_TILE=$M -DN_KEYS=$N -Dbf16_bf16_ONLY
build_cc pv_combine.o pv_combine.cc -DDIM_M=$M -DDIM_K=$N -DDIM_N=$HD -DM_TILE=$M -DHD=$HD -Dbf16_f32_ONLY

echo "== aiecc"
if ! "$AIECC" --peano="$P" --aietools="$AIETOOLS" \
      --alloc-scheme=basic-sequential --no-xchesscc --no-xbridge \
      --aie-generate-xclbin --no-compile-host --unified --aie-generate-npu-insts \
      design.mlir -o mha2.xclbin >aiecc.log 2>&1; then
  echo "== AIECC FAILED"; tail -25 aiecc.log; exit 1
fi
[ -f main.xclbin ] && cp -f main.xclbin mha2.xclbin
[ -f main_seq.bin ] && cp -f main_seq.bin mha2_insts.txt
[ -f mha2.xclbin ] || { echo "== NO XCLBIN"; tail -20 aiecc.log; exit 1; }
echo "== OK: $OUT/mha2.xclbin ($(stat -c%s mha2.xclbin) B), mha2_insts.txt ($(stat -c%s mha2_insts.txt 2>/dev/null || echo 0) B)"
