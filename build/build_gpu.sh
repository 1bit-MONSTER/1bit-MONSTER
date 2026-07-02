#!/bin/bash
# Build script for GPU Inference Engine

set -e

cd "$(dirname "$0")"

echo "=== Building GPU Inference Engine ==="

# Check dependencies
if ! command -v hipcc &> /dev/null; then
    echo "ERROR: hipcc not found"
    exit 1
fi

if [ ! -f "dequant_q4nx.o" ]; then
    echo "ERROR: dequant_q4nx.o not found in build directory"
    exit 1
fi

# Compiler flags
CXX=hipcc
CXXFLAGS="-std=c++17 -O2 -g"
INCLUDES="-I../include -I/home/bcloud/torch2aie/examples -I/opt/rocm/include"
LIBS="-L/opt/rocm/lib -lhipblas -lm"

# Source and output
SRC="../src/gpu_engine.cpp"
OBJ="dequant_q4nx.o"
OUTPUT="gpu_engine"

echo "Compiling..."
$CXX $CXXFLAGS $INCLUDES $SRC $OBJ $LIBS -o $OUTPUT

if [ -f "$OUTPUT" ]; then
    echo "Build successful: $OUTPUT"
    ls -lh $OUTPUT

    # Verify it's a valid executable
    file $OUTPUT
    echo ""

    # Optional: check for required symbols
    echo "Checking symbols..."
    nm $OUTPUT 2>/dev/null | grep -c "hipblas" || true
    echo "hipblas symbols found"

    echo ""
    echo "=== Build complete ==="
    echo "Run with: ./$OUTPUT"
else
    echo "Build FAILED"
    exit 1
fi
