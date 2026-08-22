#!/bin/bash
# run_build.sh — Build all 25 new xclbins
set -euo pipefail

PYTHON=/home/bcloud/mlir-aie/.venv/bin/python3
AIECC=/home/bcloud/mlir-aie/build_tmp/bin/aiecc
PEANO=/home/bcloud/mlir-aie/.venv/lib/python3.14/site-packages/llvm-aie
AIETOOLS=/home/bcloud/mlir-aie/build_tmp
KERNEL_O="$(cd "$(dirname "$0")/../.." && pwd)/engine/npu/generators/mm_32x64x128.o"
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

GENERATOR_DIR="$(cd "$(dirname "$0")" && pwd)"
XCLBIN_DIR="$GENERATOR_DIR/../xclbins"
mkdir -p "$XCLBIN_DIR"

SHAPES=(
    "qwen3_6_35b_a3b:QKV:2048:8192:8"
    "qwen3_6_35b_a3b:O:4096:2048:8"
    "qwen3_6_35b_a3b:G:2048:512:4"
    "qwen3_6_35b_a3b:U:2048:512:4"
    "qwen3_6_35b_a3b:D:512:2048:4"
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
    # PID-unique design path (issue #1777): a fixed /tmp path could be
    # clobbered by a co-tenant process between generation and aiecc, and the
    # build would silently consume the stale file.
    local design="/tmp/design_${proj}_${tag}.$$.mlir"
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
    [ -s "$design" ] || { echo "  ❌ ${proj} ${tag}: design generation produced an empty file"; return 1; }
    
    # aiecc needs kernel .o in CWD and runs from the design directory
    local workdir; workdir=$(dirname "$design")
    cp "$KERNEL_O" "$workdir/mm_32x64x128.o" 2>/dev/null || true
    
    cd "$workdir"
    $AIECC --peano="$PEANO" --aietools="$AIETOOLS" \
        --alloc-scheme=basic-sequential --no-xchesscc --no-xbridge \
        --aie-generate-xclbin --no-compile-host --unified --dynamic-objFifos \
        --aie-generate-npu-insts \
        --xclbin-name="$xclbin" \
        --npu-insts-name="$insts_dir/insts_i8_${proj}_${tag}.txt" \
        "$design" 2>&1 | tail -1
    cd "$GENERATOR_DIR"
    rm -f "$design"; rm -rf "$design.prj"
    
    if [ -f "$xclbin" ]; then
        local size; size=$(stat -c%s "$xclbin" 2>/dev/null)
        # Dimension-keyed copies: the engine falls back to
        # final_i8_<op>_K<K>_N<N>.xclbin when no tag-keyed file exists (#1481),
        # so any model with identical GEMM shapes loads without a rebuild.
        cp -f "$xclbin" "$XCLBIN_DIR/final_i8_${proj}_K${K}_N${N}.xclbin"
        cp -f "$insts_dir/insts_i8_${proj}_${tag}.txt" "$XCLBIN_DIR/insts_i8_${proj}_K${K}_N${N}.txt"
        echo "  ✅ $(basename "$xclbin") ($(numfmt --to=iec "$size")) + dim-keyed copies"
        return 0
    else
        echo "  ❌ FAILED"
        return 1
    fi
}

ok=0
fail=0
for entry in "${SHAPES[@]}"; do
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
