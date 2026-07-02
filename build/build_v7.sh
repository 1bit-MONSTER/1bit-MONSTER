#!/bin/bash
cd /home/bcloud/npu-sandbox/npu-infer
g++ -std=c++23 -O3 -o build/npu_engine_v7 src/npu_engine_v7.cpp build/dequant_q4nx.o \
  -Iinclude -I/home/bcloud/torch2aie/toolchain/xrt/include \
  -I/home/bcloud/torch2aie/examples \
  -I/home/bcloud/torch2aie/examples/gemm_asymmetric_tile_buffering \
  -L/home/bcloud/torch2aie/toolchain/xrt/lib64 \
  -L/home/bcloud/torch2aie/toolchain/mlir_aie.libs \
  -lxrt_coreutil -luuid -lm
