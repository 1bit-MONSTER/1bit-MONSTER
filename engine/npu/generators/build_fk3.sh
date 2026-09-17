#!/usr/bin/env bash
# build_fk3.sh — reproducible build for the fk-3 fused-layer PoC generators.
#
# Reconstructed 2026-09-12 (dsh agent) from the crashed agent's session history
# after the original /tmp build dirs were lost. Builds HOST-SIDE only (aiecc /
# peano); running the resulting xclbin needs the NPU.
#
# Usage:
#   bash build_fk3.sh [gen] [outdir]
#     gen   = fk3 | fk3_full | fk3_qkv   (default: fk3_full)
#     outdir defaults to ~/npu-build/<gen>
#
# Outputs (in outdir): design.mlir, <gen>.xclbin, <gen>_insts.txt, logs.
set -euo pipefail

G="$(cd "$(dirname "$0")" && pwd)"
GEN="${1:-fk3_full}"
OUT="${2:-$HOME/npu-build/$GEN}"

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
[ -x "$CLANG" ] || { echo "FATAL: no clang++ at $CLANG"; exit 1; }
[ -f "$MLIR/aie_kernels/aie2p/mm.cc" ] || { echo "FATAL: missing $MLIR/aie_kernels/aie2p/mm.cc"; exit 1; }
CFLAGS=(--target=aie2p-none-unknown-elf --std=c++20 -O2 -DNDEBUG -D__AIE_API_AIE_ADF_HPP__
        -isystem "$P/include/c++/v1" -I "$VITIS/aietools/include"
        -I "$MLIR/aie_kernels/aie2p"
        -I "$MLIR/.venv/lib/python3.14/site-packages/mlir_aie/include")

rm -rf "$OUT"; mkdir -p "$OUT"; cd "$OUT"

echo "== generator: $GEN"
case "$GEN" in
  fk3)      "$PY" "$G/n1_fk3.py"      -M 16 -N 64 -HD 64 -ON 64        >design.mlir 2>gen.err ;;
  fk3_full) "$PY" "$G/n1_fk3_full.py" -M 16 -N 64 -HD 64 -H 64 -IM 64  >design.mlir 2>gen.err ;;
  fk3_qkv)  "$PY" "$G/n1_fk3_qkv.py"                                   >design.mlir 2>gen.err ;;
  *) echo "unknown gen '$GEN'"; exit 2 ;;
esac
if [ ! -s design.mlir ]; then echo "== GENERATOR FAILED"; tail -20 gen.err; exit 1; fi
echo "   design.mlir: $(wc -l <design.mlir) lines"

build_cc() {  # build_cc <src.cc> [defines...]
  local src="$1"; shift
  local obj="${src%.cc}.o"
  if ! "$CLANG" "${CFLAGS[@]}" "$@" -c "$G/$src" -o "$obj" >>cc.log 2>&1; then
    echo "== CC FAILED: $src"; tail -15 cc.log; exit 1
  fi
  echo "   cc $obj"
}
build_cc_named() {  # build_cc_named <out.o> <src.cc> [defines...]  (link_with name != source name)
  local obj="$1" src="$2"; shift 2
  if ! "$CLANG" "${CFLAGS[@]}" "$@" -c "$G/$src" -o "$obj" >>cc.log 2>&1; then
    echo "== CC FAILED: $src -> $obj"; tail -15 cc.log; exit 1
  fi
  echo "   cc $obj"
}
build_upstream_mm() {  # the reference mm.cc -> named object
  local obj="$1"; shift
  if ! "$CLANG" "${CFLAGS[@]}" "$@" -c "$MLIR/aie_kernels/aie2p/mm.cc" -o "$obj" >>cc.log 2>&1; then
    echo "== CC FAILED: mm.cc -> $obj"; tail -15 cc.log; exit 1
  fi
  echo "   cc $obj"
}

echo "== kernels"
build_upstream_mm mm_bf16_16x64x64.o -DDIM_M=16 -DDIM_K=64 -DDIM_N=64 -Dbf16_bf16_ONLY
build_cc_named zero_qk.o zero_bf16.cc -DDIM_M=16 -DDIM_N=64
build_cc softmax_bf16.cc  -DM_TILE=16 -DN_KEYS=64
build_cc rescale_bf16.cc  -DM_TILE=16 -DHD=64
build_cc copy_64x64.cc

case "$GEN" in
  fk3)
    build_cc mm_qk_concat.cc -DDIM_M=16 -DDIM_K=64 -DDIM_N=64 -Dbf16_bf16_ONLY
    build_cc mm_oproj.cc     -DDIM_M=16 -DDIM_K=64 -DDIM_N=64 -Dbf16_bf16_ONLY
    ;;
  fk3_full)
    build_cc mm_qk_concat.cc -DDIM_M=16 -DDIM_K=64 -DDIM_N=64 -Dbf16_bf16_ONLY
    build_cc mm_oproj.cc     -DDIM_M=16 -DDIM_K=64 -DDIM_N=64 -Dbf16_bf16_ONLY
    build_cc mm_ffn.cc       -DDIM_M=16 -DDIM_K=64 -DDIM_N=64 -Dbf16_bf16_ONLY
    build_cc silu_split.cc   -DM_TILE=16 -DIM_TILE=64
    ;;
  fk3_qkv)
    build_cc mm_qk_concat.cc -DDIM_M=16 -DDIM_K=64 -DDIM_N=64 -Dbf16_bf16_ONLY
    build_cc mm_oproj.cc     -DDIM_M=16 -DDIM_K=64 -DDIM_N=64 -Dbf16_bf16_ONLY
    build_cc mm_ffn.cc       -DDIM_M=16 -DDIM_K=64 -DDIM_N=64 -Dbf16_bf16_ONLY
    build_cc silu_split.cc   -DM_TILE=16 -DIM_TILE=64
    build_cc qkv_gemm.cc     -DDIM_M=16 -DDIM_K=64 -DDIM_N=64 -Dbf16_bf16_ONLY -DN_PAD=64
    ;;
esac

echo "== aiecc"
if ! "$AIECC" --peano="$P" --aietools="$AIETOOLS" \
      --alloc-scheme=basic-sequential --no-xchesscc --no-xbridge \
      --aie-generate-xclbin --no-compile-host --unified --aie-generate-npu-insts \
      design.mlir -o "$GEN.xclbin" >aiecc.log 2>&1; then
  echo "== AIECC FAILED"; tail -25 aiecc.log; exit 1
fi
# aiecc names its artifacts 'main.*' regardless of -o in this flow
[ -f main.xclbin ] && cp -f main.xclbin "$GEN.xclbin"
[ -f main_seq.bin ] && cp -f main_seq.bin "${GEN}_insts.txt"
[ -f "$GEN.xclbin" ] || { echo "== NO XCLBIN PRODUCED"; tail -20 aiecc.log; exit 1; }
echo "== OK: $OUT/$GEN.xclbin ($(stat -c%s "$GEN.xclbin") B), ${GEN}_insts.txt ($(stat -c%s "${GEN}_insts.txt" 2>/dev/null || echo 0) B)"
