#!/bin/bash
cd /home/bcloud/npu-sandbox/npu-infer/build
LD_LIBRARY_PATH=/home/bcloud/torch2aie/toolchain/xrt/lib64:/home/bcloud/torch2aie/toolchain/mlir_aie.libs:/home/bcloud/torch2aie/toolchain/sysroot/usr/lib64 \
  ./npu_infer_fused
