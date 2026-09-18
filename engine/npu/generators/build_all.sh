#!/bin/bash
# Sequential build of all new xclbins with isolated workdirs
#
# The toolchain roots are derived from this script's location and from the mlir-aie
# tree, never from a hardcoded absolute path: the previous KERNEL/XDIR/GEN trio
# pointed at /home/bcloud/1bit-monster/... (lowercase), which does not exist on any
# of our machines, and the script never mkdir'd it, so every write failed against a
# nonexistent directory. Same "declared vs effective path" class as issue #1913.
#
# PEANO-only by construction: this script passes --no-xchesscc explicitly.
#
# The toolchain root below must be the SAME one the bindings come from. This file
# used to take aiecc from build_tmp while exporting PYTHONPATH=install_tmp/python,
# so the generator emitted one dialect form and aiecc parsed another, and every
# build died on its first artifact with a bare
#
#     loc("design.mlir":171:45): error: expected ')'
#
# which reads like a generator bug. Measured 2026-09-17 on one shape, same design
# and same command in both arms, varying only the root:
#     build_tmp   (AIECC+AIETOOLS=build_tmp, PYTHONPATH=install_tmp) -> rc=1, 0 bytes
#     install_tmp (both)                                            -> rc=0, 27738 bytes
# install_tmp is also what engine/npu/build_xclbins.sh:19 documents as the
# known-good toolchain setup. The old comment here claimed build_tmp was "the
# legitimate --aietools value"; that is not true for this pairing.
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MLIR_AIE="${MLIR_AIE:-$HOME/mlir-aie}"

# "xclbins\" dir = the repo's tracked set (this file lives in engine/npu/generators)
XDIR="${XDIR:-$SCRIPT_DIR/../xclbins}"
mkdir -p "$XDIR"

export PYTHON="${PYTHON:-$MLIR_AIE/.venv/bin/python3}"
export AIECC="${AIECC:-$MLIR_AIE/install_tmp/bin/aiecc}"
export PEANO="${PEANO:-$MLIR_AIE/.venv/lib/python3.14/site-packages/llvm-aie}"
export AIETOOLS="${AIETOOLS:-$MLIR_AIE/install_tmp}"
export KERNEL="${KERNEL:-$SCRIPT_DIR/mm_32x64x128.o}"
export GEN="${GEN:-$SCRIPT_DIR}"
export PYTHONPATH="${PYTHONPATH:-$MLIR_AIE/install_tmp/python:$MLIR_AIE/.venv/lib/python3.14/site-packages}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-$MLIR_AIE/install_tmp/python/aie/_mlir_libs}"

# Guard the pairing rather than trusting the two variables to agree later.
if [ ! -x "$AIECC" ]; then
    echo "build_all.sh: aiecc not executable at $AIECC" >&2; exit 1
fi
if [ "$(dirname "$(dirname "$AIECC")")" != "$AIETOOLS" ]; then
    echo "build_all.sh: AIECC ($AIECC) and AIETOOLS ($AIETOOLS) are different roots." >&2
    echo "  The generating bindings and the parsing aiecc must come from the same one." >&2
    exit 1
fi

build_one() {
    local tag="$1" proj="$2" K="$3" N="$4" cols="$5"
    local workdir="/tmp/b_${proj}_${tag}"
    rm -rf "$workdir"
    mkdir -p "$workdir"
    
    $PYTHON "$GEN/n1_core_i8_v26.py" -M 128 -K "$K" -N "$N" -m 32 -k 64 -n 128 -c "$cols" -b 5 \
        2>/dev/null > "$workdir/design.mlir"
    cp "$KERNEL" "$workdir/"
    
    cd "$workdir"
    # Build into the workdir and verify before writing through to the tracked set.
    # Writing straight to $XDIR and checking `[ -f ]` afterwards could not tell a
    # fresh xclbin from a stale one left by an earlier run, so a failed aiecc could
    # report ✅; and a failure here must not damage a committed artifact.
    $AIECC --peano="$PEANO" --aietools="$AIETOOLS" \
        --alloc-scheme=basic-sequential --no-xchesscc --no-xbridge \
        --aie-generate-xclbin --no-compile-host --unified --dynamic-objFifos \
        --aie-generate-npu-insts \
        --xclbin-name="$workdir/out.xclbin" \
        --npu-insts-name="$workdir/insts.txt" \
        design.mlir > "$workdir/aiecc.log" 2>&1
    local rc=$?

    local ok=0
    if [ "$rc" -eq 0 ] && [ -f "$workdir/out.xclbin" ] \
       && [ "$(head -c8 "$workdir/out.xclbin" 2>/dev/null | tr -d '\0')" = "xclbin2" ]; then
        cp -f "$workdir/out.xclbin" "$XDIR/final_i8_${proj}_${tag}.xclbin"
        local sz; sz=$(stat -c%s "$XDIR/final_i8_${proj}_${tag}.xclbin" 2>/dev/null)
        echo "  ✅ $proj $tag ($(numfmt --to=iec "$sz"))"
        ok=1
    else
        # Say WHY. aiecc's output used to go to /dev/null, so a build that failed on
        # its first artifact printed only "❌" and the actual error was discarded.
        echo "  ❌ $proj $tag (aiecc rc=$rc)"
        sed 's/^/       /' "$workdir/aiecc.log" | tail -6
    fi
    
    cd "$GEN"
    rm -rf "$workdir"
    sleep 2
    return $(( 1 - ok ))
}

ok=0
fail=0

# BUILD_ALL_FILTER limits the run to entries containing the given substring, so one
# shape can be rebuilt — and this script verified — without running all 23. Mirrors
# SHAPES_FILTER in run_build.sh and MODELS_FILTER in build_new_xclbins.sh. Example:
#   BUILD_ALL_FILTER="moe_35b:O:" ./build_all.sh
BUILD_ALL_FILTER="${BUILD_ALL_FILTER:-}"

for entry in \
    "qwen3.6-moe_35b:O:4096:2048:8" \
    "qwen3.6-moe_35b:U:2048:512:4" \
    "qwen3.6-moe_35b:D:512:2048:4" \
    "qwen3.5_4b:QKV:2560:6144:8" \
    "qwen3.5_4b:O:4096:2560:4" \
    "qwen3.5_4b:G:2560:9216:8" \
    "qwen3.5_4b:U:2560:9216:8" \
    "qwen3.5_4b:D:9216:2560:4" \
    "gemma4_e4b:QKV:2560:6144:8" \
    "gemma4_e4b:O:4096:2560:4" \
    "gemma4_e4b:G:2560:12288:8" \
    "gemma4_e4b:U:2560:12288:8" \
    "gemma4_e4b:D:12288:2560:4" \
    "phi4-mini_4b:QKV:3072:5120:8" \
    "phi4-mini_4b:O:3072:3072:4" \
    "phi4-mini_4b:G:3072:8192:8" \
    "phi4-mini_4b:U:3072:8192:8" \
    "phi4-mini_4b:D:8192:3072:4" \
    "nanbeige4.1_3b:QKV:2560:3840:8" \
    "nanbeige4.1_3b:O:2560:2560:4" \
    "nanbeige4.1_3b:G:2560:8192:8" \
    "nanbeige4.1_3b:U:2560:8192:8" \
    "nanbeige4.1_3b:D:8192:2560:4"; do
    
    IFS=':' read -r tag proj K N cols <<< "$entry"
    if [ -n "$BUILD_ALL_FILTER" ] && [[ "$entry" != *"$BUILD_ALL_FILTER"* ]]; then
        continue
    fi
    if build_one "$tag" "$proj" "$K" "$N" "$cols"; then
        ((ok++))
    else
        ((fail++))
    fi
done

echo ""
echo "=== Done: $ok OK, $fail FAILED ==="
find "$XDIR" -name "final_i8_*.xclbin" -not -path "*backup*" 2>/dev/null | wc -l
echo "total xclbins"
