#!/usr/bin/env bash
# build_fk3_attn_oproj.sh — first real composition: the attention (cols 0..ncol-1,
# 1 core/head, multi-pass) PLUS the O-proj (the next free column) in ONE xclbin,
# with the attention output round-tripping through a host BO.
#
# Usage: bash build_fk3_attn_oproj.sh [M] [N] [C] [NH] [PERCOL] [PASSES] [NO] [outdir]
#   defaults: M=8 N=64 C=16 NH=16 PERCOL=2 PASSES=2 NO=1024  (the 0.6B attention
#   + O-proj at the composition's M=8; 1024 keys)
set -euo pipefail

M="${1:-8}"; N="${2:-64}"; C="${3:-16}"; NH="${4:-16}"; PERCOL="${5:-2}"; PASSES="${6:-2}"; NO="${7:-1024}"
OUT="${8:-$HOME/npu-build/fk3_ao_m${M}_n${N}_c${C}_nh${NH}_p${PERCOL}_g${PASSES}}"
HD=128
KO=64

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
N_K=$(( NH * HD / KO ))

echo "== generator: n1_fk3_attn_oproj.py -M $M -N $N -C $C -HD $HD -NH $NH -P $PERCOL --passes $PASSES -NO $NO"
"$PY" "$G/n1_fk3_attn_oproj.py" -M "$M" -N "$N" -C "$C" -HD "$HD" -NH "$NH" \
      -P "$PERCOL" --passes "$PASSES" -NO "$NO" -kO "$KO" >design.mlir 2>gen.err \
  || { echo "== GENERATOR FAILED"; tail -20 gen.err; exit 1; }
[ -s design.mlir ] || { echo "== EMPTY design.mlir"; tail -20 gen.err; exit 1; }
echo "   design.mlir: $(wc -l <design.mlir) lines"

build_cc() { local obj="$1" src="$2"; shift 2
  if ! "$CLANG" "${CFLAGS[@]}" "$@" -c "$G/$src" -o "$obj" >>cc.log 2>&1; then
    echo "== CC FAILED: $src -> $obj"; tail -15 cc.log; exit 1; fi
  echo "   cc $obj"; }

echo "== kernels (attn1 M=$M HD=$HD N=$N ; oproj M=$M K=$KO N=64 n_k=$N_K)"
build_cc attn1.o attn1.cc -DM_TILE=$M -DHD=$HD -DN_KEYS=$N -DDIM_M=$M -DDIM_K=$HD -DDIM_N=$N -Dbf16_bf16_ONLY
build_cc nq_nt.o nq_nt.cc -DDIM_M=$M -DDIM_K=$KO -DDIM_N=64 -DN_K=$N_K -Dbf16_f32_ONLY
build_cc mm_acc.o mm_acc.cc -DDIM_M=$M -DDIM_N=64

echo "== aiecc"
if ! "$AIECC" --peano="$P" --aietools="$AIETOOLS" \
      --alloc-scheme=basic-sequential --no-xchesscc --no-xbridge \
      --aie-generate-xclbin --no-compile-host --unified --aie-generate-npu-insts \
      design.mlir -o fk3_ao.xclbin >aiecc.log 2>&1; then
  echo "== AIECC FAILED"; tail -25 aiecc.log; exit 1
fi
[ -f main.xclbin ] && cp -f main.xclbin fk3_ao.xclbin
[ -f main_seq.bin ] && cp -f main_seq.bin fk3_ao_insts.txt
[ -f fk3_ao.xclbin ] || { echo "== NO XCLBIN"; tail -20 aiecc.log; exit 1; }
echo "== OK: $OUT/fk3_ao.xclbin ($(stat -c%s fk3_ao.xclbin) B), fk3_ao_insts.txt ($(stat -c%s fk3_ao_insts.txt 2>/dev/null || echo 0) B)"
