#!/bin/bash
# build_q34b_split_g.sh — build the split-G artifacts Qwen3-4B needs (issue #2329)
#
# Why this file exists: the shape selector sends Qwen3-4B down the split-GU path
# (`cfg.gu_split = (IM*2 > 14336)`, IM=9728), and the artifact that path requires
# — `final_i8_G_K2560_N9728.xclbin` — was not in the repo or anywhere on this box,
# so `npu_engine_qwen3_4b` could not initialize at all (`FAIL G`) on any route.
# These three shapes were built with this script and are committed:
#
#   final_i8_G_K2560_N9728.xclbin   final_i8_U_K2560_N9728.xclbin
#   final_i8_D_K9728_N2560.xclbin   (+ the tag-named qwen3_4b copies + insts)
#
# TWO THINGS DIFFER FROM run_build.sh, both measured on 2026-09-14 rather than
# assumed (see issue #2262 for the wider "the flow cannot reproduce the committed
# artifacts" problem):
#
#   1. AIECC ARM. run_build.sh pins /home/bcloud/mlir-aie/build_tmp/bin/aiecc
#      (AOMP-23.0-60), which cannot parse what n1_core_i8_v27.py emits:
#          loc("design.mlir":1059:45): error: expected ')'
#          Error parsing MLIR file
#      That is not shape-specific — it fails identically on a committed shape
#      (K=2560 N=9216 cols=8), which is the control that proved it. The
#      install_tmp arm (/home/bcloud/mlir-aie/install_tmp/bin/aiecc, LLVM 23.0.0)
#      parses the same design and compiles it.
#
#   2. COLS. n1_core_i8_v27.py asserts (N//n) % n_aie_cols == 0 with n=128, so
#      N=9728 needs cols that divide 76. The K=2560 neighbours use cols=8 (for
#      N=9216: 72 % 8 == 0), which is invalid here; 4 is the largest usable value
#      with rows=4 under the generator's other asserts.
#
# Usage:  bash engine/npu/generators/build_q34b_split_g.sh
# Needs:  the mlir-aie venv + install_tmp build; mm_32x64x128.o beside this script
#         (it is gitignored; see run_build.sh's header for the rebuild command).
set -uo pipefail

GENDIR="$(cd "$(dirname "$0")" && pwd)"
# Output directory: the committed artifacts by default. Pass one to rebuild into a temp dir
# instead — Testing/xclbin_rebuild_selfcheck.sh does exactly that, so a reproducibility check
# never writes into the tree it is checking.
XCLBIN_DIR="${1:-$GENDIR/../xclbins}"
mkdir -p "$XCLBIN_DIR"
PYTHON=/home/bcloud/mlir-aie/.venv/bin/python3
AIECC=/home/bcloud/mlir-aie/install_tmp/bin/aiecc
PEANO=/home/bcloud/mlir-aie/.venv/lib/python3.14/site-packages/llvm-aie
AIETOOLS=/home/bcloud/mlir-aie/build_tmp
export PYTHONPATH=/home/bcloud/mlir-aie/install_tmp/python:/home/bcloud/mlir-aie/.venv/lib/python3.14/site-packages
export LD_LIBRARY_PATH=/home/bcloud/mlir-aie/install_tmp/python/aie/_mlir_libs

SHAPES=("G:2560:9728:4" "U:2560:9728:4" "D:9728:2560:4")
TAG=qwen3_4b
ok=0; fail=0

for entry in "${SHAPES[@]}"; do
    IFS=':' read -r proj K N cols <<< "$entry"
    W="/tmp/q34b_${proj}_$$"
    mkdir -p "$W"
    echo "═══ ${proj} K=${K} N=${N} cols=${cols} ═══"
    "$PYTHON" "$GENDIR/n1_core_i8_v27.py" -M 128 -K "$K" -N "$N" \
        -m 32 -k 64 -n 128 -c "$cols" -r 4 -b 5 2>/dev/null > "$W/design.mlir"
    if [ ! -s "$W/design.mlir" ]; then
        echo "  ✗ design generation produced nothing (check (N//128) % ${cols} == 0)"
        fail=$((fail+1)); rm -rf "$W"; continue
    fi
    cp "$GENDIR/mm_32x64x128.o" "$W/" || { echo "  ✗ mm_32x64x128.o missing (see run_build.sh header)"; fail=$((fail+1)); rm -rf "$W"; continue; }
    ( cd "$W" && "$AIECC" --peano="$PEANO" --aietools="$AIETOOLS" \
        --alloc-scheme=basic-sequential --no-xchesscc --no-xbridge \
        --aie-generate-xclbin --no-compile-host --unified --dynamic-objFifos \
        --aie-generate-npu-insts \
        --xclbin-name="$XCLBIN_DIR/final_i8_${proj}_${TAG}.xclbin" \
        --npu-insts-name="$XCLBIN_DIR/insts_i8_${proj}_${TAG}.txt" \
        design.mlir > "$W/aiecc.log" 2>&1 )
    rc=$?
    if [ $rc -eq 0 ] && [ -f "$XCLBIN_DIR/final_i8_${proj}_${TAG}.xclbin" ]; then
        # dimension-keyed copies: what the engine looks up first (#1481)
        cp -f "$XCLBIN_DIR/final_i8_${proj}_${TAG}.xclbin" "$XCLBIN_DIR/final_i8_${proj}_K${K}_N${N}.xclbin"
        cp -f "$XCLBIN_DIR/insts_i8_${proj}_${TAG}.txt" "$XCLBIN_DIR/insts_i8_${proj}_K${K}_N${N}.txt"
        echo "  ✓ final_i8_${proj}_K${K}_N${N}.xclbin ($(numfmt --to=iec "$(stat -c%s "$XCLBIN_DIR/final_i8_${proj}_K${K}_N${N}.xclbin")"))"
        ok=$((ok+1))
    else
        echo "  ✗ FAILED (rc=$rc)"; tail -3 "$W/aiecc.log" | sed 's/^/    /'
        fail=$((fail+1))
    fi
    rm -rf "$W"
done

echo "═══ ${ok} built, ${fail} failed ═══"
[ "$XCLBIN_DIR" = "$GENDIR/../xclbins" ] || { echo "(built into $XCLBIN_DIR — not the committed set)"; exit $(( fail > 0 )); }
echo "Then regenerate the manifest so the new artifacts have provenance:"
echo "  python3 engine/npu/tests/check_xclbin_provenance.py --write-manifest \\"
echo "      --toolchain 'aiecc (mlir-aie install_tmp, LLVM 23.0.0) + peano, npu2'"
