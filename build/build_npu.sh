#!/bin/bash
# Build NPU engine binaries with AVX-512 + fast-math optimization
# This reproduces the 174ms/tok speed of the pre-built binary
set -euo pipefail

SRC="$(cd "$(dirname "$0")/../src" && pwd)"
BUILD="$(cd "$(dirname "$0")" && pwd)"
OPT="-O3 -mavx2 -mfma -mavx512f -ffast-math -march=native -funroll-loops -fno-math-errno -fvect-cost-model=unlimited"

echo "=== Building NPU engine binaries (AVX-512 optimized) ==="

# Step 1: dequant library
echo "  dequant_q4nx.o ..."
gcc -std=c11 $OPT -c "$SRC/dequant_q4nx.c" -o "$BUILD/dequant_q4nx.o"

# Step 2: npu_engine_i8 (original, proven engine)
echo "  npu_engine_i8 ..."
c++ -std=gnu++17 $OPT -g -I"$(dirname "$SRC")/include" \
    -o "$BUILD/npu_engine_i8" "$SRC/npu_engine_i8.cpp" "$BUILD/dequant_q4nx.o" \
    -lxrt_coreutil

# Step 3: npu_engine_fused_i8 (fused pipeline, same performance)
echo "  npu_engine_fused_i8 ..."
c++ -std=gnu++17 $OPT -g -I"$(dirname "$SRC")/include" \
    -o "$BUILD/npu_engine_fused_i8" "$SRC/npu_engine_fused_i8.cpp" "$BUILD/dequant_q4nx.o" \
    -lxrt_coreutil

echo ""
echo "=== Done ==="
echo "  npu_engine_i8       — original INT8 engine"
echo "  npu_engine_fused_i8 — fused INT8 engine (no attention xclbins)"
echo ""
echo "Run:"
echo "  sudo LD_LIBRARY_PATH=/home/bcloud/torch2aie/toolchain/xrt/lib64:\\"
echo "    /home/bcloud/torch2aie/toolchain/mlir_aie.libs:\\"
echo "    /home/bcloud/torch2aie/toolchain/sysroot/usr/lib64 \\"
echo "    ./npu_engine_i8"
