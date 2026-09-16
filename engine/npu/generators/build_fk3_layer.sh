#!/usr/bin/env bash
# build_fk3_layer.sh — the fk-3 ATTENTION BLOCK in ONE launch:
#   col 5 norm (RMSNorm -> A_norm to DDR) | col 6 GEMM (re-read -> QKV bf16)
#   cols 0..3 attention (Q/K^T/V gathered straight out of QKV) | col 4 O-proj
#
# Usage: bash build_fk3_layer.sh [M] [H] [NH] [HD] [NO] [PERCOL] [PASSES] [K] [NT] [KO] [outdir]
set -euo pipefail

M="${1:-16}"; H="${2:-1024}"; NH="${3:-16}"; HD="${4:-128}"; NO="${5:-1024}"
PERCOL="${6:-2}"; PASSES="${7:-2}"; K="${8:-64}"; NT="${9:-64}"; KO="${10:-64}"
OUT="${11:-$HOME/npu-build/fk3_layer_m${M}_k${K}_nt${NT}}"
N="$M"                                   # single chunk: keys == query tokens

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

echo "== generator -M $M -H $H -NH $NH -HD $HD -NO $NO -P $PERCOL --passes $PASSES -k $K -NT $NT -kO $KO"
"$PY" "$G/n1_fk3_layer.py" -M "$M" -H "$H" -NH "$NH" -HD "$HD" -NO "$NO" -MA ${MA:-16} -NC ${NC:-16} -NDEP ${NDEP:-2} ${NOQKV:+-NOQKV} -N2 ${N2:-6144} \
      -NI ${NI:-3072} -ND ${ND:-1024} \
      -P "$PERCOL" --passes "$PASSES" -k "$K" -NT "$NT" -kO "$KO" >design.mlir 2>gen.err \
  || { echo "== GENERATOR FAILED"; tail -20 gen.err; exit 1; }
[ -s design.mlir ] || { echo "== EMPTY design.mlir"; tail -20 gen.err; exit 1; }
echo "   design.mlir: $(wc -l <design.mlir) lines"

build_cc() { local obj="$1" src="$2"; shift 2
  if ! "$CLANG" "${CFLAGS[@]}" "$@" -c "$G/$src" -o "$obj" >>cc.log 2>&1; then
    echo "== CC FAILED: $src -> $obj"; tail -15 cc.log; exit 1; fi
  echo "   cc $obj"; }

# One nq_nt.o serves BOTH GEMMs: DIM_K = K = KO = 64 and DIM_N = NT = 64 for the
# QKV and the O-proj alike, so the two share the matmul/acc helpers. N_K only
# sizes the core-local g_an (unused now that both stages re-read), so keep it 1.
build_cc rms_split.o rms_norm_split.cc -DM_TILE=$M -DK_TILE=$K -DH=$H
build_cc nq_nt.o      nq_nt.cc        -DDIM_M=$M -DDIM_K=$K -DDIM_N=$NT -Dbf16_f32_ONLY
build_cc silu_split.o silu_split.cc   -DM_TILE=$M -DIM_TILE=$NT
NQB=$(( M / ${MA:-16} ))     # query blocks per pass
NCH=$(( M / ${NC:-16} ))     # key chunks per query block
build_cc attn1.o      attn1.cc        -DM_TILE=${MA:-16} -DHD=$HD -DN_KEYS=${NC:-16} -DN_QB=${NQB:-1} -DN_CH=${NCH:-1} -DDIM_M=${MA:-16} -DDIM_K=$HD -DDIM_N=${NC:-16} -Dbf16_bf16_ONLY -DK_ROW_MAJOR ${ATTN_DBG:+-DATTN_DUMP_SOFTMAX}

echo "== aiecc"
if ! "$AIECC" --peano="$P" --aietools="$AIETOOLS" \
      --alloc-scheme=basic-sequential --no-xchesscc --no-xbridge \
      --aie-generate-xclbin --aie-generate-npu-insts \
      --xclbin-name=fk3_layer.xclbin --npu-insts-name=fk3_layer_insts.txt \
      design.mlir >aiecc.log 2>&1; then
  echo "== AIECC FAILED"; grep -E "error|Error" aiecc.log | head -8; exit 1; fi
echo "== OK: $OUT/fk3_layer.xclbin ($(stat -c%s fk3_layer.xclbin) B), insts ($(stat -c%s fk3_layer_insts.txt) B)"
