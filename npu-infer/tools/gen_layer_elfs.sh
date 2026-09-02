#!/bin/bash
# gen_layer_elfs.sh — generate the per-context layer ELFs the runtime_layers
# decode path needs (issue #2006). The decode dies at the first ctx without
# an ELF; generate up to MAX_CTX (default 1024).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL="${1:-/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2}"
OUT="${2:-$HERE/captures/txn-elfs}"
MAX_CTX="${3:-1024}"
cd "$HERE/tools"
g++ -O2 -std=c++17 -include climits gen_layer_elfs.cpp -o gen_layer_elfs \
  -I/home/bcloud/amd-oss/fastflowlm/src/include \
  -I/home/bcloud/amd-oss/fastflowlm/src/include/npu_utils \
  -I/usr/include/aiebu \
  -L/home/bcloud/amd-oss/fastflowlm/src/lib/xrt -lqwen3_npu -lgemm -lmha -lq4_npu_eXpress \
  -L/usr/local/lib -laiebu -lxrt_coreutil -lxrt_core \
  -Wl,-rpath,/home/bcloud/amd-oss/fastflowlm/src/lib/xrt
mkdir -p "$OUT"
./gen_layer_elfs "$MODEL" "$OUT" 1 "$MAX_CTX"
echo "layer ELFs 1..$MAX_CTX -> $OUT"
