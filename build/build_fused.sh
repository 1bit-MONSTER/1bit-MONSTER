#!/bin/bash
set -e
cd /home/bcloud/npu-sandbox/npu-infer/build
g++ -std=c++23 -O3 -march=native -flto -g -o npu_infer_fused \
  ../src/npu_engine_fused.cpp \
  dequant_q4nx.o \
  -I../include \
  -I/home/bcloud/torch2aie/toolchain/xrt/include \
  -I/home/bcloud/torch2aie/examples \
  -I/home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering \
  -L/home/bcloud/torch2aie/toolchain/xrt/lib64 \
  -L/home/bcloud/torch2aie/toolchain/mlir_aie.libs \
  -Wl,-rpath,/home/bcloud/torch2aie/toolchain/xrt/lib64 \
  -Wl,-rpath,/home/bcloud/torch2aie/toolchain/mlir_aie.libs \
  -lxrt_coreutil -luuid -ldl -lm
