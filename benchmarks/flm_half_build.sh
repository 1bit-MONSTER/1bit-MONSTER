#!/bin/bash
# build_half.sh — build one engine bf16 GEMM kernel as a half-array (4 columns) at a column offset,
# using the -x/--col-offset knob ported from upstream (n1_core_fused_gu_silu_d.py, issue #2128).
# The device stays npu2: the tiles simply sit at 4..7 for the second half, which is how upstream's
# two-stream halves are built.  usage: build_half.sh <PROJ> <K> <N> <cols> <offset> [outdir]
set -euo pipefail
PROJ=$1; K=$2; N=$3; COLS=${4:-4}; OFF=${5:-0}; OUT=${6:-$HOME/xclbins-h${OFF}}
G="$HOME/1bit-MONSTER-goal/engine/npu/generators"
P=/home/bcloud/mlir-aie/.venv/bin/python3
AICC=/home/bcloud/mlir-aie/install_tmp/bin/aiecc
export PEANO_INSTALL_DIR=/home/bcloud/mlir-aie/.venv/lib/python3.14/site-packages/llvm-aie
export MLIR_AIE_DIR=/home/bcloud/mlir-aie
mkdir -p "$OUT"
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
cp "$G/mm_bf16_32x64x128.o" "$W/"
(
  cd "$W"
  PYTHONPATH=/home/bcloud/mlir-aie/install_tmp/python:$PEANO_INSTALL_DIR \
    "$P" "$G/n1_core_bf16_v1.py" -M 128 -K "$K" -N "$N" -m 32 -k 64 -n 128 -c "$COLS" -r 4 -b 5 -x "$OFF" \
    > design.mlir 2>gen.err || { echo "  generator failed"; tail -3 gen.err; exit 1; }
  LD_LIBRARY_PATH=/home/bcloud/mlir-aie/install_tmp/python/aie/_mlir_libs \
    "$AICC" --peano="$PEANO_INSTALL_DIR" --aietools="$MLIR_AIE_DIR" \
      --alloc-scheme=basic-sequential --no-xchesscc --no-xbridge \
      --aie-generate-xclbin --no-compile-host --unified --dynamic-objFifos \
      --aie-generate-npu-insts \
      --xclbin-name="$OUT/final_bf16_${PROJ}_K${K}_N${N}.xclbin" \
      --npu-insts-name="$OUT/insts_bf16_${PROJ}_K${K}_N${N}.txt" \
      design.mlir > aiecc.log 2>&1 || { echo "  aiecc FAILED"; tail -4 aiecc.log; exit 1; }
)
cols=$(grep -oE "aie\.tile\([0-9]+, [0-9]+\)" "$W/design.mlir" | sed -E 's/aie\.tile\(([0-9]+),.*/\1/' | sort -un | tr '\n' ',')
echo "  ${PROJ} K=$K N=$N cols=$COLS offset=$OFF -> columns {${cols%,}}  $(stat -c%s "$OUT/final_bf16_${PROJ}_K${K}_N${N}.xclbin") B"
