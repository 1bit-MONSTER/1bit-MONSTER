#!/usr/bin/env bash
# build_fused_norm_gemm_rr.sh — fused RMSNorm+GEMM with the SHIM RE-READ structure
# (norm col 0 -> A_norm to DDR -> GEMM col 1 re-reads it per N-tile). Removes the
# core-local-A M cap, so this runs at the engine's prefill M.
#
# Usage: bash build_fused_norm_gemm_rr.sh [M] [H] [N] [k] [NT] [wdepth] [outdir]
#   defaults: M=128 H=1024 N=4096 k=64 NT=32 wdepth=1   (QKV at prefill M)
set -euo pipefail

M="${1:-128}"; H="${2:-1024}"; N="${3:-4096}"; K="${4:-64}"; NT="${5:-32}"; WD="${6:-1}"
NSTACK="${NSTACK:-4096}"; GSTACK="${GSTACK:-4096}"
OUT="${7:-$HOME/npu-build/normgemm_rr_m${M}_H${H}_N${N}_k${K}_NT${NT}}"

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
N_K=$(( H / K ))

echo "== generator: n1_fused_norm_gemm_rr.py -m $M -H $H -N $N -k $K -NT $NT -wdepth $WD"
"$PY" "$G/n1_fused_norm_gemm_rr.py" -m "$M" -H "$H" -N "$N" -k "$K" -NT "$NT" \
      -wdepth "$WD" -stack "$NSTACK" -gstack "$GSTACK" ${BF16OUT:+-bf16out} >design.mlir 2>gen.err \
  || { echo "== GENERATOR FAILED"; tail -20 gen.err; exit 1; }
[ -s design.mlir ] || { echo "== EMPTY design.mlir"; tail -20 gen.err; exit 1; }
echo "   design.mlir: $(wc -l <design.mlir) lines"

build_cc() { local obj="$1" src="$2"; shift 2
  if ! "$CLANG" "${CFLAGS[@]}" "$@" -c "$G/$src" -o "$obj" >>cc.log 2>&1; then
    echo "== CC FAILED: $src -> $obj"; tail -15 cc.log; exit 1; fi
  echo "   cc $obj"; }

echo "== kernels (norm M_TILE=$M K_TILE=$K H=$H; gemm ${M}x${K}x${NT}, n_k=$N_K)"
build_cc rms_split.o rms_norm_split.cc -DM_TILE=$M -DK_TILE=$K -DH=$H
build_cc nq_nt.o nq_nt.cc -DDIM_M=$M -DDIM_K=$K -DDIM_N=$NT -DN_K=$N_K -Dbf16_f32_ONLY
build_cc mm_acc.o mm_acc.cc -DDIM_M=$M -DDIM_N=$NT

echo "== aiecc"
if ! "$AIECC" --peano="$P" --aietools="$AIETOOLS" \
      --alloc-scheme=basic-sequential --no-xchesscc --no-xbridge --dynamic-objFifos \
      --aie-generate-xclbin --no-compile-host --unified --aie-generate-npu-insts \
      design.mlir -o normgemm_rr.xclbin >aiecc.log 2>&1; then
  echo "== AIECC FAILED"; tail -25 aiecc.log; exit 1
fi
[ -f main.xclbin ] && cp -f main.xclbin normgemm_rr.xclbin
[ -f main_seq.bin ] && cp -f main_seq.bin normgemm_rr_insts.txt
[ -f normgemm_rr.xclbin ] || { echo "== NO XCLBIN"; tail -20 aiecc.log; exit 1; }
echo "== OK: $OUT/normgemm_rr.xclbin ($(stat -c%s normgemm_rr.xclbin) B), normgemm_rr_insts.txt ($(stat -c%s normgemm_rr_insts.txt 2>/dev/null || echo 0) B)"
