#!/bin/bash
# build4col_dev.sh — build one engine bf16 kernel with the *narrow device* in the MLIR, so the
# xclbin declares a 4-wide partition (the stock generator hardcodes aie.device(npu2), which is why
# a -c 4 design still stamped column_width=8). usage: build4col_dev.sh <PROJ> <K> <N> <cols>
set -euo pipefail
PROJ=${1:-QKV}; K=${2:-1024}; N=${3:-4096}; COLS=${4:-4}
G="$HOME/1bit-MONSTER-goal/engine/npu/generators"
OUT=${OUT:-$HOME/xclbins-4col}
mkdir -p "$OUT"
P=/home/bcloud/mlir-aie/.venv/bin/python3
AICC=/home/bcloud/mlir-aie/install_tmp/bin/aiecc
export PEANO_INSTALL_DIR=/home/bcloud/mlir-aie/.venv/lib/python3.14/site-packages/llvm-aie
export MLIR_AIE_DIR=/home/bcloud/mlir-aie
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
cp "$G/mm_bf16_32x64x128.o" "$W/"
(
  cd "$W"
  PYTHONPATH=/home/bcloud/mlir-aie/install_tmp/python:$PEANO_INSTALL_DIR \
    "$P" "$G/n1_core_bf16_v1.py" -M 128 -K "$K" -N "$N" -m 32 -k 64 -n 128 -c "$COLS" -r 4 -b 5 \
    > design.mlir 2>gen.err
  # narrow device: the design uses columns 0..COLS-1, so say so
  if [ "$COLS" != 8 ]; then
      sed -i "s/aie\.device(npu2)/aie.device(npu2_${COLS}col)/" design.mlir
  fi
  grep -oE "aie\.device\([a-z0-9_]+\)" design.mlir | head -1 | sed 's/^/  device: /'
  LD_LIBRARY_PATH=/home/bcloud/mlir-aie/install_tmp/python/aie/_mlir_libs \
    "$AICC" --peano="$PEANO_INSTALL_DIR" --aietools="$MLIR_AIE_DIR" \
      --alloc-scheme=basic-sequential --no-xchesscc --no-xbridge \
      --aie-generate-xclbin --no-compile-host --unified --dynamic-objFifos \
      --aie-generate-npu-insts \
      --xclbin-name="$OUT/final_bf16_${PROJ}_K${K}_N${N}.xclbin" \
      --npu-insts-name="$OUT/insts_bf16_${PROJ}_K${K}_N${N}.txt" \
      design.mlir > aiecc.log 2>&1 || { echo "  aiecc FAILED"; tail -5 aiecc.log; exit 1; }
)
f="$OUT/final_bf16_${PROJ}_K${K}_N${N}.xclbin"
xclbinutil --dump-section AIE_PARTITION:JSON:/tmp/apd.json --input "$f" >/dev/null 2>&1
python3 -c "
import json,os
p=json.load(open('/tmp/apd.json'))['aie_partition']['partition']
print('  %s  %dB  column_width=%s start=%s' % (os.path.basename('$f'), os.path.getsize('$f'), p.get('column_width'), p.get('start_columns')))
"
