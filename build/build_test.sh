#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
g++ -std=c++23 -O2 -g -o test_128 ../tools/test_128.cpp \
  -I/home/bcloud/torch2aie/toolchain/xrt/include \
  -I/home/bcloud/torch2aie/examples \
  -I/home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering \
  -L/home/bcloud/torch2aie/toolchain/xrt/lib64 \
  -L/home/bcloud/torch2aie/toolchain/mlir_aie.libs \
  -lxrt_coreutil -luuid -ldl
