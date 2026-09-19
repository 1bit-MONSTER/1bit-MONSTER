#!/bin/bash
# Run GEMM bench with TheRock rocBLAS vs system rocBLAS
set -euo pipefail

SCRIPT_DIR=$(dirname "$(readlink -f "$0")")
THEROCK=$HOME/therock/build
OUT=$HOME/Desktop/strixhalo-output/gemm-bench-results.txt

# NOTE: HSA_OVERRIDE_GFX_VERSION=11.5.1 used to be set here. That forces the
# runtime to REPORT gfx1151 — on a gfx1201 card it would select the gfx1151 device
# build, which is precisely the defect this work removed (rocBLAS then aborts with
# an empty Tensile list). Do not reintroduce it; build for the real arch below.
export HSA_ENABLE_SDMA=0
export HIP_VISIBLE_DEVICES=0

echo "=== Compiling bench_gemm ==="
# Build for THIS machine's arch (was hardcoded gfx1151).
ARCH="$(bash "$SCRIPT_DIR/../scripts/detect-gfx-targets.sh" --cmake 2>/dev/null || echo gfx1151)"
echo "    arch: $ARCH"
hipcc -O3 --offload-arch="$ARCH" \
    -I/opt/rocm/include \
    -L/opt/rocm/lib -lrocblas -lamdhip64 \
    "$SCRIPT_DIR/bench_gemm.cpp" -o "$SCRIPT_DIR/bench_gemm"

echo ""
echo "=== System rocBLAS ===" | tee $OUT
echo "Date: $(date)" | tee -a $OUT
echo "" | tee -a $OUT
"$SCRIPT_DIR/bench_gemm" 2>&1 | tee -a $OUT

echo "" | tee -a $OUT
echo "=== TheRock rocBLAS (native Tensile for this arch) ===" | tee -a $OUT
echo "" | tee -a $OUT
export LD_LIBRARY_PATH=$THEROCK/math-libs/BLAS/rocBLAS/dist/lib:$THEROCK/math-libs/BLAS/hipBLASLt/dist/lib:$THEROCK/core/clr/dist/lib:/opt/rocm/lib
export ROCBLAS_TENSILE_LIBPATH=$THEROCK/math-libs/BLAS/rocBLAS/dist/lib/rocblas/library
"$SCRIPT_DIR/bench_gemm" 2>&1 | tee -a $OUT

echo "" | tee -a $OUT
echo "=== Complete ===" | tee -a $OUT
