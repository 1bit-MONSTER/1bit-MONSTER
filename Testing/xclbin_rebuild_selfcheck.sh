#!/usr/bin/env bash
# xclbin_rebuild_selfcheck.sh — the committed split-G artifacts must be reproducible from the tree.
#
# WHY (issue #2262)
# -----------------
# The committed xclbin set is the source of truth because the MLIR/chess toolchain could not
# reliably rebuild it, so PROVENANCE.json records what shipped. That record could never be checked,
# because a rebuild of the same shape produces a *different xclbin*: measured on this box, the
# committed `final_i8_G_K2560_N9728.xclbin` is 61,706 B and a rebuild is 69,389 B, ~72% of bytes
# differing — toolchain drift, not nondeterminism (two rebuilds differ in exactly 68 bytes: the
# embedded XclBinUUID in its three copies and the TimeStamp in its two, 16+16+31+2+3).
#
# What *is* reproducible, and therefore worth asserting, is the part the engine actually loads:
# every rebuilt `insts_i8_*_qwen3_4b.txt` is byte-identical to the committed one, and identical
# across rebuilds. This check pins that, so the next toolchain or generator change that alters the
# instruction stream is caught here instead of silently invalidating the committed set.
#
# It rebuilds into a temp directory (the recipe takes one), so the tree it checks is never written
# to; the only repo write is the gitignored microkernel object the documented build produces, and
# that is removed again if this run created it.
#
# Exit: 0 ok, 1 a rebuilt instruction stream differs (or the floors fail), 2 cannot judge
#       (toolchain absent — "could not run" must not read as "clean").

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# Toolchain roots: overridable so this can be pointed at another install, and so the
# "toolchain absent" path is testable.
MLIR_AIE_ROOT="${MLIR_AIE_ROOT:-/home/bcloud/mlir-aie}"
AIETOOLS_INCLUDE="${AIETOOLS_INCLUDE:-/home/bcloud/Xilinx/2025.2/Vitis/aietools/include}"
PYTHON="$MLIR_AIE_ROOT/.venv/bin/python3"
AIECC="$MLIR_AIE_ROOT/install_tmp/bin/aiecc"
PEANO="$MLIR_AIE_ROOT/.venv/lib/python3.14/site-packages/llvm-aie"
KERNELS="$MLIR_AIE_ROOT/install_tmp/include/aie_kernels/aie2p"
CLANG="$PEANO/bin/clang++"

# The three shapes the committed recipe builds, and the committed stream that must come back.
SHAPES=(G U D)
TAG=qwen3_4b
MIN_COMPARED=3

for f in "$PYTHON" "$AIECC" "$CLANG" "$CLANG"; do
    if [ ! -x "$f" ]; then
        echo "ENVIRONMENT: $f is missing — no toolchain to rebuild with" >&2
        echo "  set MLIR_AIE_ROOT to the mlir-aie install (AIECC/PEANO/PYTHON are derived from it)" >&2
        exit 2
    fi
done
[ -d "$KERNELS" ] || { echo "ENVIRONMENT: $KERNELS missing (AIETOOLS_INCLUDE=$AIETOOLS_INCLUDE)" >&2; exit 2; }

# Floor: the committed streams must be there and non-empty, or "identical" means nothing.
for proj in "${SHAPES[@]}"; do
    f="engine/npu/xclbins/insts_i8_${proj}_${TAG}.txt"
    if [ ! -s "$f" ]; then
        echo "ENVIRONMENT: committed $f is missing or empty — nothing to compare against" >&2
        exit 2
    fi
done

W="$(mktemp -d)"; trap 'rm -rf "$W"' EXIT
KO="engine/npu/generators/mm_32x64x128.o"
made_ko=0
if [ ! -f "$KO" ]; then
    echo "== building the gitignored microkernel object (run_build.sh:10-19) =="
    "$CLANG" --target=aie2p-none-unknown-elf --std=c++20 -O2 \
        -DDIM_M=32 -DDIM_K=64 -DDIM_N=128 -Di8_i32_ONLY \
        -isystem "$PEANO/include/c++/v1" \
        -I "$AIETOOLS_INCLUDE" -I "$KERNELS" \
        -c engine/npu/generators/mm_kernel_reference.cc -o "$KO" >"$W/cc.log" 2>&1 \
        || { echo "FAIL: the microkernel object did not build:" >&2; tail -5 "$W/cc.log" >&2; exit 2; }
    made_ko=1
fi

echo "== rebuilding ${#SHAPES[@]} committed shape(s) into $W =="
out="$(bash engine/npu/generators/build_q34b_split_g.sh "$W" 2>&1)"; rc=$?
[ "$made_ko" -eq 1 ] && rm -f "$KO"
printf '%s\n' "$out" | grep -E 'built,|✓|✗' | sed 's/^/  /'

compared=0; bad=0
for proj in "${SHAPES[@]}"; do
    committed="engine/npu/xclbins/insts_i8_${proj}_${TAG}.txt"
    rebuilt="$W/insts_i8_${proj}_${TAG}.txt"
    if [ ! -s "$rebuilt" ]; then
        echo "  ✗ $proj: the rebuild produced no instruction stream"; bad=1; continue
    fi
    compared=$((compared+1))
    if cmp -s "$committed" "$rebuilt"; then
        printf '  ✓ %-2s %s (%s B, identical)\n' "$proj" "$(basename "$committed")" "$(stat -c%s "$committed")"
    else
        printf '  ✗ %-2s %s DIFFERS from the rebuild:\n' "$proj" "$(basename "$committed")"
        diff <(xxd "$committed") <(xxd "$rebuilt") | head -6 | sed 's/^/      /'
        bad=1
    fi
done

echo
if [ "$compared" -lt "$MIN_COMPARED" ]; then
    echo "FAIL: only $compared of $MIN_COMPARED shape(s) produced a stream — the rebuild did not run" >&2
    [ "$rc" -ne 0 ] && echo "      recipe exit was $rc" >&2
    exit 1
fi
if [ "$bad" -ne 0 ]; then
    echo "FAIL: a rebuilt instruction stream differs from the committed one. Either the generator or" >&2
    echo "      the toolchain changed; the committed xclbin set no longer describes what builds here." >&2
    exit 1
fi
echo "OK: $compared/$MIN_COMPARED committed instruction streams reproduce byte-for-byte"
echo "    (the xclbin payloads are expected to differ — the embedded UUID/timestamp, and the"
echo "     toolchain drift #2262 records; the instructions the engine loads are what is pinned here)"
