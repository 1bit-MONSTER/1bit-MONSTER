#!/bin/bash
# build_zaya_fused_cols.sh — column-sliced FUSED GU→SiLU→D xclbins.
#
# Why this file exists
# -------------------
# The two-stream decode path (tools/two_stream_decode.sh) needs 4-column
# half-array kernels at col offsets 0 and 4. The generator that produced them
# (engine/npu/generators/build_zaya_fused_half.sh) landed in ffb6af90b, which
# is NOT an ancestor of main, and the file was later lost from this box — only
# the full-array build_zaya_fused.sh survived. So the halves could not be
# regenerated at all, and the surviving h0/h1 pair in ~/npu-verify is from the
# Aug-25 lineage: running it reads [MoE L1 single dbg] corr ≈ -0.0016, the
# #2163/#2172 host↔xclbin ABI mismatch, which the launcher's gate refuses.
#
# This script is build_zaya_fused.sh with the column geometry parameterized.
# It reuses that script's toolchain block and aiecc invocation verbatim, and
# relies on the -c/--cols and -x/--col-offset knobs that ffb6af90b added to
# n1_core_fused_gu_silu_d.py (re-applied in this tree).
#
# The kernels MUST be built from the same revision as the npu_engine binary:
# that pairing is what makes the MoE-L1 correlation read ~0.998 (full) / ~0.87
# (half) instead of ~-0.0015.
#
# Usage:
#   engine/npu/generators/build_zaya_fused_cols.sh              # h0 + h1 halves
#   engine/npu/generators/build_zaya_fused_cols.sh 4 0 h0       # cols offset tag
#   engine/npu/generators/build_zaya_fused_cols.sh 8 0 full     # full array
#
# Outputs (into engine/npu/xclbins):
#   final_i8_MOE_FUSED_zaya_<tag>.xclbin
#   insts_i8_MOE_FUSED_zaya_<tag>.txt
set -euo pipefail

P=/home/bcloud/mlir-aie/.venv/lib/python3.14/site-packages/llvm-aie
M=/home/bcloud/mlir-aie/.venv/lib/python3.14/site-packages/mlir_aie
PYTHON=/home/bcloud/mlir-aie/.venv/bin/python3
AIECC=/home/bcloud/mlir-aie/install_tmp/bin/aiecc
AIETOOLS=/home/bcloud/mlir-aie/install_tmp
export PATH=/home/bcloud/Xilinx/2026.1/Vitis/bin:/opt/xilinx/xrt/bin:$PATH
export PYTHONPATH=/home/bcloud/mlir-aie/install_tmp/python:/home/bcloud/mlir-aie/.venv/lib/python3.14/site-packages
export LD_LIBRARY_PATH=/home/bcloud/mlir-aie/install_tmp/python/aie/_mlir_libs

GENERATOR_DIR="$(cd "$(dirname "$0")" && pwd)"
XCLBIN_DIR="$GENERATOR_DIR/../xclbins"
mkdir -p "$XCLBIN_DIR"

# PID-unique workdir (issue #1777): no two builds share /tmp paths, and the
# trap removes the design + kernel object, so a co-tenant build cannot
# overwrite one between generation and aiecc (the 08-22 incident, corr 0.374).
workdir="/tmp/zaya_fused_cols_build.$$"
mkdir -p "$workdir"
trap 'rm -rf "$workdir"' EXIT

# DIM_M=8 fused microkernel (1x4 mmul + the on-core silu_quant_i8_fused entry
# from silu_quant.h). I4_SCALAR_C1 is the production default (issue #1874; the
# mmul C1 store was miscompiled for non-uniform B).
$P/bin/clang++ --target=aie2p-none-unknown-elf --std=c++20 -O2 \
    -DDIM_M=8 -DDIM_K=64 -DDIM_N=128 -Di8_i32_ONLY -DM8_VECTORIZED \
    -DI4_SCALAR_C1 -DI4_SCALAR_C1_ACK_1864 \
    -isystem $P/include/c++/v1 \
    -I $M/include \
    -I /home/bcloud/Xilinx/2025.2/Vitis/aietools/include \
    -I $M/include/aie_kernels/aie2p \
    -c "$GENERATOR_DIR/mm_kernel_reference.cc" -o "$workdir/mm_8x64x128_fused.o"
# The MLIR references the kernel object by this fixed name.
cp "$workdir/mm_8x64x128_fused.o" "$workdir/mm_32x64x128.o"

build_one() {  # $1=cols  $2=col-offset  $3=tag
    local cols="$1" off="$2" tag="$3"
    local design="$workdir/design_fused_${tag}.mlir"
    local ref="$workdir/design_fused_${tag}.ref.mlir"
    local xclbin="$XCLBIN_DIR/final_i8_MOE_FUSED_zaya_${tag}.xclbin"
    local insts="$XCLBIN_DIR/insts_i8_MOE_FUSED_zaya_${tag}.txt"

    gen() {
        $PYTHON "$GENERATOR_DIR/n1_core_fused_gu_silu_d.py" -M 8 -K 2048 \
            -N_GU 4096 -N_D 2048 -m 8 -k 64 -n 128 -c "$cols" -b 2 -x "$off" \
            2>/dev/null > "$1"
    }

    echo "═══ FUSED GU→SiLU→D M=8 K=2048 N_GU=4096 N_D=2048 cols=${cols} offset=${off} (${tag}) ═══"
    gen "$design"
    [ -s "$design" ] || { echo "ERROR: ${tag}: design generation produced an empty file" >&2; exit 1; }

    # Determinism check (as build_zaya_fused.sh): a fresh regeneration must be
    # byte-identical, and the design must carry the fused-kernel markers.
    gen "$ref"
    if ! cmp -s "$design" "$ref"; then
        echo "ERROR: ${tag}: design differs from a fresh regeneration — stale/tampered or" >&2
        echo "       nondeterministic generator. Refusing to build (issue #1777)." >&2
        exit 1
    fi
    for marker in "silu_quant_i8_fused" "mm_32x64x128.o"; do
        grep -q "$marker" "$design" || {
            echo "ERROR: ${tag}: design lacks marker '$marker' — not the fused design?" >&2
            exit 1; }
    done
    local used
    used=$(grep -oE 'tile\([0-9]+, [0-9]+\)' "$design" | grep -oE 'tile\([0-9]+' | \
           grep -oE '[0-9]+$' | sort -n | uniq | tr '\n' ' ')
    echo "    AIE columns used: ${used}"
    case "$used" in
      "0 1 2 3 "|"4 5 6 7 "|"0 1 2 3 4 5 6 7 ") : ;;
      *) echo "ERROR: ${tag}: unexpected column set [${used}] for cols=${cols} off=${off}" >&2
         exit 1 ;;
    esac

    ( cd "$workdir"
      $AIECC --peano="$P" --aietools="$AIETOOLS" \
        --alloc-scheme=basic-sequential --no-xchesscc --no-xbridge \
        --aie-generate-xclbin --no-compile-host --unified --dynamic-objFifos \
        --aie-generate-npu-insts \
        --xclbin-name="$xclbin" --npu-insts-name="$insts" \
        "$design" 2>&1 | tail -1 )
    ls -la "$xclbin" "$insts"
}

if [ "$#" -ge 3 ]; then
    build_one "$1" "$2" "$3"
elif [ "$#" -eq 0 ]; then
    build_one 4 0 h0      # cols 0-3
    build_one 4 4 h1      # cols 4-7
else
    echo "usage: $0 [cols offset tag]" >&2
    echo "   no args  -> h0 (4 cols, offset 0) + h1 (4 cols, offset 4)" >&2
    exit 2
fi

cat <<'EOF'

═══ next ═══
Run the gated driver against these halves (same binary revision):
  XCLBIN_DIR=<this tree>/engine/npu/xclbins \
  BIN=<this tree>/engine/npu/build/npu_engine_zr1 \
  tools/two_stream_decode.sh --two
Expect: MoE-L1 corr ~0.87 (half-array), gate MIN_CORR=0.80, exit 0.
A corr of ~-0.0015 means the halves and the binary are NOT a matched pair.
EOF
