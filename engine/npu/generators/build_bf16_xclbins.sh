#!/bin/bash
# Build bf16 xclbins under the name the ENGINE asks for:
#   final_bf16_<PROJ>_K<K>_N<N>.xclbin
#
# src/npu_engine_universal.cpp:1302 constructs exactly that name
#   xd+"/final_bf16_"+t+"_K"+K+"_N"+N+".xclbin"
# but no script in the tree emitted it: build_new_xclbins.sh emits
# final_i8_${proj}_${model_tag}.xclbin via n1_core_i8_v26.py -- a different name
# AND a different generator. The bf16 generator (n1_core_bf16_v1.py) existed only
# with the manual invocation documented in FUSED-RMSNORM-QKV-DESIGN.md:74.
# This is that producer. See
# benchmarks/RESULTS-attention-c2-regression-2026-09-15.md.
#
# Usage: bash build_bf16_xclbins.sh [PROJ:K:N[:cols] ...]
#        (default: Nanbeige QKV K=2560 N=3584 cols=8)
set -euo pipefail

export PEANO_INSTALL_DIR=/home/bcloud/mlir-aie/.venv/lib/python3.14/site-packages/llvm-aie
export AIETOOLS_DIR=/home/bcloud/mlir-aie/npu2_40_toolchain
export MLIR_AIE_DIR=/home/bcloud/mlir-aie
export PATH="$AIETOOLS_DIR/bin:$PEANO_INSTALL_DIR/bin:$PATH"

G="$(cd "$(dirname "$0")" && pwd)"
XCLBIN_DIR="$G/../xclbins"
KERNEL_OBJ="$G/mm_bf16_32x64x128.o"
PY=/home/bcloud/mlir-aie/.venv/bin/python3
AICC=/home/bcloud/mlir-aie/install_tmp/bin/aiecc

[ -f "$KERNEL_OBJ" ] || { echo "ERROR: $KERNEL_OBJ missing (build the bf16 kernel object first)"; exit 1; }

build() {
    local proj="$1" K="$2" N="$3" cols="${4:-8}"
    local name="final_bf16_${proj}_K${K}_N${N}.xclbin"
    local insts="insts_bf16_${proj}_K${K}_N${N}.txt"
    echo ""
    echo "=== bf16 ${proj} K=${K} N=${N} cols=${cols} -> ${name} ==="
    local W
    W="$(mktemp -d)"
    # aiecc resolves link_with against CWD, so stage the kernel object there
    cp "$KERNEL_OBJ" "$W/"
    ( cd "$W" && PYTHONPATH=/home/bcloud/mlir-aie/install_tmp/python:$PEANO_INSTALL_DIR \
        "$PY" "$G/n1_core_bf16_v1.py" \
        -M 128 -K "$K" -N "$N" -m 32 -k 64 -n 128 -c "$cols" -r 4 -b 5 \
        > design.mlir 2>"$W/gen.err" ) || { echo "  ❌ generator failed"; tail -5 "$W/gen.err"; rm -rf "$W"; return 1; }

    # aiecc's bindings and binary must come from the same install (see build_attn.sh).
    # link_with resolves attn_kernel.o / mm_*.o from CWD, so run inside $W.
    ( cd "$W" && \
      LD_LIBRARY_PATH=/home/bcloud/mlir-aie/install_tmp/python/aie/_mlir_libs \
      "$AICC" --peano="$PEANO_INSTALL_DIR" --aietools="$MLIR_AIE_DIR" \
          --alloc-scheme=basic-sequential --no-xchesscc --no-xbridge \
          --aie-generate-xclbin --no-compile-host --unified --dynamic-objFifos \
          --aie-generate-npu-insts \
          --xclbin-name="$XCLBIN_DIR/$name" \
          --npu-insts-name="$XCLBIN_DIR/$insts" \
          "$W/design.mlir" ) || { echo "  ❌ aiecc failed"; rm -rf "$W"; return 1; }
    rm -rf "$W"
    echo "  ✅ $name ($(stat -c%s "$XCLBIN_DIR/$name" 2>/dev/null || echo 0) B)"
    echo "     $insts"
}

if [ $# -gt 0 ]; then
    for spec in "$@"; do IFS=':' read -r p k n c <<< "$spec"; build "$p" "$k" "$n" "${c:-8}"; done
else
    build QKV 2560 3584 8
fi
