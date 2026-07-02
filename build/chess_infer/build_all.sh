#!/usr/bin/env bash
# Build Chess-compiled inference xclbins for Qwen3-0.6B
# Uses config1 generator + Chess kernel (m=128, k=64, n=128)
# License: Xilinx.lic required (installed at ~/torch2aie/licenses/)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TORCH2AIE_ROOT="/home/bcloud/torch2aie"
CONFIG1_DIR="$TORCH2AIE_ROOT/examples/gemm_asymmetric_tile_buffering/config1"
KERNEL_SRC="$CONFIG1_DIR/mm_bfp_mixed.cc"
GENERATOR="$CONFIG1_DIR/n32_core_placed.py"
BUILD_DIR="$SCRIPT_DIR"

source "$TORCH2AIE_ROOT/scripts/env.sh"

# Verify license
if [ ! -f "$XILINXD_LICENSE_FILE" ]; then
    echo "ERROR: License not found at $XILINXD_LICENSE_FILE" >&2
    exit 1
fi
echo "LICENSE: $XILINXD_LICENSE_FILE"
echo "AIETOOLS: $AIETOOLS_DIR"
echo "MLIR_AIE: $MLIR_AIE_DIR"

# Tile sizes (config1 defaults, Chess kernel hardcoded)
m=128 k=64 n=128

# Qwen3-0.6B projection shapes
# M=1024 (padded decode batch, 4 row groups × 256 rows each)
declare -A SHAPES=(
    ["qkv"]="1024 1024 4096"   # Q:2048 + K:1024 + V:1024
    ["o"]="1024 2048 1024"     # O: attention output projection
    ["gu"]="1024 1024 6144"    # G:3072 + U:3072 (gate+up fused)
    ["d"]="1024 3072 1024"     # D: down projection
)

# Build kernel once (same .o for all shapes)
KERNEL_O="$BUILD_DIR/mm_${m}x${k}x${n}.o"
echo ""
echo "=== Building Chess kernel: mm_${m}x${k}x${n}.o ==="
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"
xchesscc_wrapper aie2p -c \
    -I "$AIETOOLS_DIR/include" \
    -I "$MLIR_AIE_DIR/include" \
    -DDIM_M=$m -DDIM_K=$k -DDIM_N=$n \
    -I"$MLIR_AIE_DIR/include/aie_kernels" \
    "$KERNEL_SRC" \
    -o "$KERNEL_O"
echo "Kernel built: $(ls -lh $KERNEL_O | awk '{print $5}')"

# Build xclbin for each projection
for proj in qkv o gu d; do
    read M K N <<< "${SHAPES[$proj]}"
    echo ""
    echo "============================================================"
    echo "=== Building $proj: M=$M K=$K N=$N (m=$m k=$k n=$n) ==="
    echo "============================================================"

    PROJ_DIR="$BUILD_DIR/$proj"
    mkdir -p "$PROJ_DIR"
    cd "$PROJ_DIR"

    # Copy kernel .o
    cp "$KERNEL_O" "mm_${m}x${k}x${n}.o"

    # Generate MLIR
    echo "--- Generating MLIR ---"
    $TORCH2AIE_PYTHON "$GENERATOR" -M $M -K $K -N $N -m $m -k $k -n $n \
        > "design_${M}x${K}x${N}.mlir"
    echo "MLIR generated: $(wc -l < design_${M}x${K}x${N}.mlir) lines"

    # Build xclbin
    echo "--- Building xclbin with aiecc (Chess kernel) ---"
    aiecc --aietools="$AIETOOLS_DIR" \
        --alloc-scheme=basic-sequential \
        --aie-generate-xclbin --no-compile-host \
        --xclbin-name="qwen3_${proj}_chess.xclbin" \
        --unified --dynamic-objFifos \
        --aie-generate-npu-insts \
        --npu-insts-name="qwen3_${proj}_chess.insts" \
        "design_${M}x${K}x${N}.mlir"

    XCLBIN_SIZE=$(ls -lh "qwen3_${proj}_chess.xclbin" | awk '{print $5}')
    echo "=== $proj DONE: $XCLBIN_SIZE ==="
done

echo ""
echo "============================================================"
echo "=== ALL XCLBINS BUILT ==="
echo "============================================================"
for proj in qkv o gu d; do
    xclbin="$BUILD_DIR/$proj/qwen3_${proj}_chess.xclbin"
    insts="$BUILD_DIR/$proj/qwen3_${proj}_chess.insts"
    if [ -f "$xclbin" ]; then
        size=$(ls -lh "$xclbin" | awk '{print $5}')
        echo "  $proj: $xclbin ($size)"
    else
        echo "  $proj: FAILED"
    fi
done
