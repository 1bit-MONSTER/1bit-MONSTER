#!/usr/bin/env bash
# build_mha_1core_nh.sh — build the 1-core-per-head fused chunked attention
# (n1_mha_1core_nh.py + attn1.cc). NH=16 then costs 16 compute tiles instead of
# 32, which is what leaves room for the layer's linear stages.
#
# Usage: bash build_mha_1core_nh.sh <N_chunk> <C_chunks> <NH> [percol] [outdir]
set -euo pipefail

N="${1:?usage: build_mha_1core_nh.sh <N_chunk> <C_chunks> <NH> [percol] [outdir]}"
C="${2:?}"
NH="${3:-16}"; PERCOL="${4:-2}"; PASSES="${PASSES:-1}"; OUT="${5:-$HOME/npu-build/mha1_nh${NH}_p${PERCOL}_g${PASSES}_n${N}_c${C}}"
HD=128
M="${MGEN:-16}"

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

echo "== generator: n1_mha_1core_nh.py -M $M -N $N -C $C -HD $HD -NH $NH -P $PERCOL"
"$PY" "$G/n1_mha_1core_nh.py" -M "$M" -N "$N" -C "$C" -HD "$HD" -NH "$NH" -P "$PERCOL" --passes "$PASSES" \
      >design.mlir 2>gen.err \
  || { echo "== GENERATOR FAILED"; tail -20 gen.err; exit 1; }
[ -s design.mlir ] || { echo "== EMPTY design.mlir"; tail -20 gen.err; exit 1; }
echo "   design.mlir: $(wc -l <design.mlir) lines"

build_cc() { local obj="$1" src="$2"; shift 2
  if ! "$CLANG" "${CFLAGS[@]}" "$@" -c "$G/$src" -o "$obj" >>cc.log 2>&1; then
    echo "== CC FAILED: $src -> $obj"; tail -15 cc.log; exit 1; fi
  echo "   cc $obj"; }

echo "== kernels (QK^T ${M}x${HD}x${N} bf16-C; PV ${M}x${N}x${HD} f32-C)"
# attn1.cc includes mm.cc (bf16->bf16, DIM = M/HD/N) for the QK^T.
build_cc attn1.o attn1.cc -DM_TILE=$M -DHD=$HD -DN_KEYS=$N -DDIM_M=$M -DDIM_K=$HD -DDIM_N=$N -Dbf16_bf16_ONLY

echo "== aiecc"
if ! "$AIECC" --peano="$P" --aietools="$AIETOOLS" \
      --alloc-scheme=basic-sequential --no-xchesscc --no-xbridge \
      --aie-generate-xclbin --no-compile-host --unified --aie-generate-npu-insts \
      design.mlir -o mha1.xclbin >aiecc.log 2>&1; then
  echo "== AIECC FAILED"; tail -25 aiecc.log; exit 1
fi
[ -f main.xclbin ] && cp -f main.xclbin mha1.xclbin
[ -f main_seq.bin ] && cp -f main_seq.bin mha1_insts.txt
[ -f mha1.xclbin ] || { echo "== NO XCLBIN"; tail -20 aiecc.log; exit 1; }
echo "== OK: $OUT/mha1.xclbin ($(stat -c%s mha1.xclbin) B), mha1_insts.txt ($(stat -c%s mha1_insts.txt 2>/dev/null || echo 0) B)"
