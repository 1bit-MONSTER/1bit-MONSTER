#!/bin/bash
# run_build.sh — Build all 25 new xclbins
set -euo pipefail

PYTHON=/home/bcloud/mlir-aie/.venv/bin/python3
AIECC=/home/bcloud/mlir-aie/install_tmp/bin/aiecc
PEANO=/home/bcloud/mlir-aie/.venv/lib/python3.14/site-packages/llvm-aie
AIETOOLS=/home/bcloud/mlir-aie/install_tmp
# Same directory as this script. The previous expression was
#   "$(cd "$(dirname "$0")/../.." && pwd)/engine/npu/generators/mm_32x64x128.o"
# which appends the engine/npu/generators suffix a SECOND time: from this directory
# `dirname "$0"/../..` is <repo>/engine, so the result was
# <repo>/engine/engine/npu/generators/... - a path that never exists. The cp below
# swallowed the failure with `|| true`, aiecc then could not find the kernel and died
# with "could not copy .../mm_32x64x128.o ... No such file or directory". Verified
# 2026-09-17: the old form resolves to the doubled path and src_exists=NO.
KERNEL_O="$(cd "$(dirname "$0")" && pwd)/mm_32x64x128.o"
# The microkernel .o is gitignored and was missing from clones. Rebuild it with
# the peano clang (no xchesscc needed) — verified 2026-08-15:
#   P=/home/bcloud/mlir-aie/.venv/lib/python3.14/site-packages/llvm-aie
#   $P/bin/clang++ --target=aie2p-none-unknown-elf --std=c++20 -O2 \
#       -DDIM_M=32 -DDIM_K=64 -DDIM_N=128 -Di8_i32_ONLY \
#       -isystem $P/include/c++/v1 \
#       -I /home/bcloud/Xilinx/2025.2/Vitis/aietools/include \
#       -I <mlir_aie>/include/aie_kernels/aie2p \
#       -c <repo>/engine/npu/generators/mm_kernel_reference.cc -o mm_32x64x128.o
# (aie_clang++'s wrapper is broken — hardcoded /aietools paths.)

export PATH=/home/bcloud/Xilinx/2026.1/2026.1/Vitis/bin:/opt/xilinx/xrt/bin:$PATH

export PYTHONPATH=/home/bcloud/mlir-aie/install_tmp/python:/home/bcloud/mlir-aie/.venv/lib/python3.14/site-packages
export LD_LIBRARY_PATH=/home/bcloud/mlir-aie/install_tmp/python/aie/_mlir_libs

# The bindings that GENERATE and the aiecc that PARSES must come from the same
# mlir-aie root. A mismatch fails with a bare
#     loc("design.mlir":1059:45): error: expected ')'
# which reads like a generator bug but is a dialect-version split: install_tmp's
# bindings emit the pre-#3306 positional form and install_tmp/bin/aiecc parses it,
# while build_tmp/bin/aiecc only parses the post-#3306 operand form that install/
# emits. engine/npu/generators/check_aie_dialect.sh documents this and
# engine/npu/build_xclbins.sh:19 names the known-good root. This script hardcoded
# build_tmp/bin/aiecc WITH install_tmp's PYTHONPATH - a mismatched pair - so every
# build failed. Verified 2026-09-17 with the repo's own checker.
_AIETOOLS_BIN="$(dirname "$AIECC")"
if [ ! -x "$AIECC" ]; then
    echo "  ❌ aiecc not executable at $AIECC" >&2
    exit 1
fi
if [ "$(dirname "$_AIETOOLS_BIN")" != "$AIETOOLS" ]; then
    echo "  ❌ AIECC ($AIECC) and AIETOOLS ($AIETOOLS) are different roots." >&2
    echo "     The generating bindings and the parsing aiecc must match - see comment above." >&2
    exit 1
fi
_AIE_DIR=$("$PYTHON" -c 'import aie,os;print(os.path.dirname(aie.__file__))' 2>/dev/null || true)
case "${_AIE_DIR:-}" in
    "$AIETOOLS"/*) ;;
    *)
        echo "  ❌ aie bindings resolve to '${_AIE_DIR:-<unresolvable>}', expected under $AIETOOLS" >&2
        echo "     Generator and parser must come from the same mlir-aie root - see comment above." >&2
        exit 1
        ;;
esac

GENERATOR_DIR="$(cd "$(dirname "$0")" && pwd)"
XCLBIN_DIR="$GENERATOR_DIR/../xclbins"
mkdir -p "$XCLBIN_DIR"

SHAPES=(
    "qwen3_6_35b_a3b:QKV:2048:8192:8"
    "qwen3_6_35b_a3b:O:4096:2048:8"
    "qwen3_6_35b_a3b:G:2048:512:4"
    "qwen3_6_35b_a3b:U:2048:512:4"
    "qwen3_6_35b_a3b:D:512:2048:4"
    # Fused gate+up for the 35B (engine looks for final_i8_GU_K2048_N1024.xclbin,
    # which shipped nowhere -- it was the FAIL GU blocker). N = 2*IM_EXP = 2*512 = 1024;
    # cols divides N/128 = 8.  See ledger §78.
    "qwen3.6-moe_35b:GU:2048:1024:8"
    "qwen3_5_4b:QKV:2560:6144:8"
    "qwen3_5_4b:O:4096:2560:4"
    "qwen3_5_4b:G:2560:9216:8"
    "qwen3_5_4b:U:2560:9216:8"
    "qwen3_5_4b:D:9216:2560:4"
    "gemma4_e4b:QKV:2560:6144:8"
    "gemma4_e4b:O:4096:2560:4"
    "gemma4_e4b:G:2560:12288:8"
    "gemma4_e4b:U:2560:12288:8"
    "gemma4_e4b:D:12288:2560:4"
    "phi4_mini_4b:QKV:3072:5120:8"
    "phi4_mini_4b:O:3072:3072:4"
    "phi4_mini_4b:G:3072:8192:8"
    "phi4_mini_4b:U:3072:8192:8"
    "phi4_mini_4b:D:8192:3072:4"
    "nanbeige4_1_3b:QKV:2560:3840:6"   # N//n=30 not %8; 6 cols divides 30
    "nanbeige4_1_3b:O:2560:2560:4"
    "nanbeige4_1_3b:G:2560:8192:8"
    "nanbeige4_1_3b:U:2560:8192:8"
    "nanbeige4_1_3b:D:8192:2560:4"
)

build_one() {
    local tag="$1" proj="$2" K="$3" N="$4" cols="$5"
    # PID-unique workdir (issue #1777): a fixed /tmp path for the design OR
    # the kernel .o could be clobbered by a co-tenant process between
    # generation and aiecc, and the build would silently consume the stale
    # file. $$ = PID. aiecc needs the kernel .o in its CWD, so the design and
    # the kernel are staged together here.
    local workdir="/tmp/build_${proj}_${tag}.$$"
    mkdir -p "$workdir"
    local design="$workdir/design.mlir"
    local xclbin="$XCLBIN_DIR/final_i8_${proj}_${tag}.xclbin"
    local insts_dir="$XCLBIN_DIR"   # engine reads insts_i8_<op>_<tag>.txt from the xclbin dir
    mkdir -p "$insts_dir"
    
    echo ""
    echo "══════ Building ${tag} ${proj} K=${K} N=${N} cols=${cols} ══════"
    
    # Generate clean MLIR (stderr to /dev/null, stdout to file).
    # v27 spreads the tile grid over all 4 AIE core rows; v26 used only row 2,
    # i.e. 8 of the 32 compute tiles.  Both emit the same xclbin interface, but
    # an xclbin and its instruction stream encode the same topology and must be
    # regenerated as a pair — never mix a v27 xclbin with v26 instructions.
    $PYTHON "$GENERATOR_DIR/n1_core_i8_v27.py" \
        -M 128 -K "$K" -N "$N" -m 32 -k 64 -n 128 -c "$cols" -r 4 -b 5 \
        2>/dev/null > "$design"
    if [ ! -s "$design" ]; then
        echo "  ❌ ${proj} ${tag}: design generation produced an empty file" >&2
        rm -rf "$workdir"
        return 1
    fi
    
    # No `|| true`: a missing kernel .o is fatal (it is gitignored, so a fresh clone
    # legitimately lacks it - see the rebuild recipe above). Swallowing this made
    # aiecc fail later with a confusing "could not copy ... No such file or directory".
    if [ ! -f "$KERNEL_O" ]; then
        echo "  ❌ FAILED: kernel object not found at $KERNEL_O" >&2
        echo "     it is gitignored; rebuild it with the peano clang recipe in the header." >&2
        rm -rf "$workdir"
        return 1
    fi
    cp "$KERNEL_O" "$workdir/mm_32x64x128.o"

    # Pick an output path, then delete it BEFORE invoking aiecc. Deleting first is the
    # load-bearing part of the gate: aiecc exits 0 even when it prints "Error parsing
    # MLIR file" and writes nothing, so `set -e` never fires and any staleness check
    # passes against the PREVIOUS build's artifact. Measured 2026-09-17: with the
    # magic/size check but without the delete, four shapes reported ✅ while their
    # xclbins were byte-identical to the 09-16 22:26 set and aiecc had failed on all.
    #
    # The 35B's tag-keyed path is a tracked ALIAS SYMLINK (final_i8_QKV_<tag> ->
    # final_i8_QKV_qwen3.6-moe_35b). Writing to the tag path would materialise it and
    # trip check_xclbin_provenance.py's "tracked as a symlink ... not a symlink" rule.
    # So retarget aiecc at the link's TARGET and leave the link itself alone; the
    # success path's `cp -f` then copies the canonical artifact to the tag path, which
    # over a symlink writes through to the same file already built.
    # Resolve the real file to build. For an alias, that is the link's target; the
    # link at the tag path is left untouched so the repo's aliasing invariant holds.
    build_path="$xclbin"
    _link=""
    if [ -L "$xclbin" ]; then
        _rt=$(readlink -f "$xclbin" 2>/dev/null || true)
        if [ -n "$_rt" ]; then
            build_path="$_rt"
            _link=$(readlink "$xclbin" 2>/dev/null || true)
            echo "  (alias: building $_link, link left in place)"
        fi
    fi
    # Build into the workdir, NOT over the target. The previous `rm -f "$build_path"` was
    # there so the post-build check could not be satisfied by a stale file — but when
    # $xclbin is an alias, build_path is the link's TARGET, which is a TRACKED artifact.
    # Deleting it before aiecc meant a failed build left it deleted and the alias dangling,
    # with nothing to restore it. A fresh path inside the PID-unique workdir gives the same
    # freshness guarantee while leaving the repo untouched until the build has succeeded.
    _build_out="$workdir/$(basename "$build_path")"
    _insts_out="$workdir/insts_i8_${proj}_${tag}.txt"

    cd "$workdir"
    $AIECC --peano="$PEANO" --aietools="$AIETOOLS" \
        --alloc-scheme=basic-sequential --no-xchesscc --no-xbridge \
        --aie-generate-xclbin --no-compile-host --unified --dynamic-objFifos \
        --aie-generate-npu-insts \
        --xclbin-name="$_build_out" \
        --npu-insts-name="$_insts_out" \
        "$design" 2>&1 | tail -1
    cd "$GENERATOR_DIR"
    
    if [ -f "$_build_out" ] && [ "$(head -c8 "$_build_out" 2>/dev/null)" = "xclbin2" ] \
       && [ "$(stat -Lc%s "$_build_out" 2>/dev/null || echo 0)" -gt 4096 ]; then
        local size; size=$(stat -Lc%s "$_build_out" 2>/dev/null)
        # Nothing in the repo has been touched up to this point; only a verified build
        # writes through, so a failure cannot damage a committed artifact.
        cp -f "$_build_out" "$build_path"
        cp -f "$_insts_out" "$insts_dir/insts_i8_${proj}_${tag}.txt"
        # Dimension-keyed copies: the engine falls back to
        # final_i8_<op>_K<K>_N<N>.xclbin when no tag-keyed file exists (#1481),
        # so any model with identical GEMM shapes loads without a rebuild.
        cp -f "$build_path" "$XCLBIN_DIR/final_i8_${proj}_K${K}_N${N}.xclbin"
        cp -f "$insts_dir/insts_i8_${proj}_${tag}.txt" "$XCLBIN_DIR/insts_i8_${proj}_K${K}_N${N}.txt"
        rm -rf "$workdir"
        echo "  ✅ $(basename "$xclbin") ($(numfmt --to=iec "$size")) + dim-keyed copies"
        return 0
    else
        # Say WHICH precondition failed. Plain `[ -f ]` was not enough: a stale xclbin
        # left by an earlier run satisfies it, so a failed aiecc still reported ✅ and
        # the existing artifact was then re-copied over the dim-keyed names as if fresh.
        if [ ! -e "$_build_out" ]; then
            echo "  ❌ FAILED: aiecc produced no xclbin (see the ${proj}_${tag} error above)"
        elif [ "$(head -c8 "$_build_out" 2>/dev/null)" != "xclbin2" ]; then
            echo "  ❌ FAILED: $xclbin is not an AXLF container (bad magic) - not copying it forward"
        else
            echo "  ❌ FAILED: $_build_out is $(stat -Lc%s "$_build_out" 2>/dev/null) B, below the 4096 B floor"
        fi
        echo "  (the committed artifact, if any, was not touched)"
        rm -rf "$workdir"
        return 1
    fi
}

ok=0
fail=0
# SHAPES_FILTER limits the run to entries containing the given substring, so a
# single shape can be rebuilt without regenerating all of them (each entry is a
# full MLIR design + aiecc pass).
SHAPES_FILTER="${SHAPES_FILTER:-}"
for entry in "${SHAPES[@]}"; do
    if [ -n "$SHAPES_FILTER" ] && [[ "$entry" != *"$SHAPES_FILTER"* ]]; then
        continue
    fi
    IFS=':' read -r tag proj K N cols <<< "$entry"
    if build_one "$tag" "$proj" "$K" "$N" "$cols"; then
        ok=$((ok+1))
    else
        fail=$((fail+1))
    fi
done

echo ""
echo "══════ RESULTS: ${ok} OK, ${fail} FAILED ══════"
echo "Xclbins in: $XCLBIN_DIR"
find "$XCLBIN_DIR" -name "final_i8_*.xclbin" -not -path "*backup*" 2>/dev/null | wc -l
echo "total xclbins"
