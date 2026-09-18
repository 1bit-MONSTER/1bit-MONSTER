#!/bin/bash
# build_attn4.sh — build the attention design at N columns, directly replicating generators/build_attn.sh
# (the wrapper's sed copy produced nothing). usage: build_attn4.sh <cols> <outbox>
set -euo pipefail
COLS="${1:-4}"
OUT="${2:-/tmp/attn4build}"
P=/home/bcloud/mlir-aie/.venv/lib/python3.14/site-packages/llvm-aie
M=/home/bcloud/mlir-aie/.venv/lib/python3.14/site-packages/mlir_aie
PYTHON=/home/bcloud/mlir-aie/.venv/bin/python3
G=/home/bcloud/1bit-MONSTER-goal/engine/npu/generators
N=${NPU_ATTN_N:-512}
W="$OUT"; mkdir -p "$W"
echo "== attn build: cols=$COLS N=$N out=$W =="
$P/bin/clang++ --target=aie2p-none-unknown-elf --std=c++20 -O2 \
    -DDIM_M=8 -DDIM_K=64 -DDIM_N=128 -Di8_i32_ONLY -DM8_VECTORIZED \
    -isystem $P/include/c++/v1 -I /home/bcloud/Xilinx/2025.2/Vitis/aietools/include \
    -I $M/include/aie_kernels/aie2p -c "$G/mm_kernel_reference.cc" -o "$W/mm.o"
$P/bin/clang++ --target=aie2p-none-unknown-elf --std=c++20 -O2 \
    -DDIM_M=8 -DDIM_K=64 -DDIM_N=128 -Di8_i32_ONLY -DM8_VECTORIZED \
    -isystem $P/include/c++/v1 -I /home/bcloud/Xilinx/2025.2/Vitis/aietools/include \
    -I $M/include/aie_kernels/aie2p -c "$G/attn_kernel_reference.cc" -o "$W/softmax.o"
$P/bin/ld.lld -r "$W/mm.o" "$W/softmax.o" -o "$W/attn_kernel.o"
$PYTHON "$G/n1_core_attn.py" -M 8 -K 128 -N "$N" -m 8 -k 64 -n 128 -c "$COLS" -b 2 > "$W/design.mlir"
cd "$W"
export PATH=/home/bcloud/Xilinx/2026.1/Vitis/bin:/opt/xilinx/xrt/bin:$PATH
export PYTHONPATH=/home/bcloud/mlir-aie/install_tmp/python:/home/bcloud/mlir-aie/.venv/lib/python3.14/site-packages
export LD_LIBRARY_PATH=/home/bcloud/mlir-aie/install_tmp/python/aie/_mlir_libs
/home/bcloud/mlir-aie/install_tmp/bin/aiecc --peano="$P" --aietools="$M" \
    --alloc-scheme=basic-sequential --no-xchesscc --no-xbridge \
    --aie-generate-xclbin --no-compile-host --unified --dynamic-objFifos \
    --aie-generate-npu-insts \
    --xclbin-name="$W/attn${COLS}.xclbin" \
    --npu-insts-name="$W/attn${COLS}_insts.txt" \
    "$W/design.mlir"
echo "== built =="; ls -l "$W/attn${COLS}.xclbin" "$W/attn${COLS}_insts.txt" 2>/dev/null | awk '{print "  "$5, $9}'
xclbinutil --dump-section AIE_PARTITION:JSON:"$W/ap.json" --input "$W/attn${COLS}.xclbin" >/dev/null 2>&1 || true
python3 - "$W/ap.json" <<'PY'
import json,sys
try:
    p=json.load(open(sys.argv[1]))["aie_partition"]["partition"]
    print("  partition: column_width=%s start_columns=%s" % (p.get("column_width"), p.get("start_columns")))
except Exception as e:
    print("  partition parse failed:", e)
PY
