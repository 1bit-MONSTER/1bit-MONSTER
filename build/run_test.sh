#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
LD_LIBRARY_PATH=/home/bcloud/torch2aie/toolchain/xrt/lib64:/home/bcloud/torch2aie/toolchain/mlir_aie.libs:/home/bcloud/torch2aie/toolchain/sysroot/usr/lib64 \
  ./test_128
