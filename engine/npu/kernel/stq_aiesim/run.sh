#!/usr/bin/env bash
# run.sh — build + run the STQ kernel in x86sim, check against golden.
# Needs: Vitis 2026.1 aietools on PATH, XILINXD_LICENSE_FILE set.
# Usage: ./run.sh [probe <k>]   (probe mode: A-delta debug, needs FINGERPRINT=1 for col tagging)
set -euo pipefail
cd "$(dirname "$0")"
V=${XILINX_VITIS:-/home/bcloud/Xilinx/2026.1/2026.1/Vitis}
export PATH=$V/aietools/bin:$V/bin:$PATH
export LD_LIBRARY_PATH=$V/aietools/lib/lnx64.o:$V/lib/lnx64.o:${LD_LIBRARY_PATH:-}
g++ -O2 -o gen_data gen_data.cpp
./gen_data "$@"
aiecompiler --target=x86sim \
  --platform=$V/base_platforms/xilinx_vek280_base_202610_1/xilinx_vek280_base_202610_1.xpfm \
  --include=. --include=/home/bcloud/mlir-aie/aie_kernels stq_main.cpp
rm -rf x86simulator_output
x86simulator --pkg-dir=Work
if [ "${1:-}" = "" ]; then
  cp x86simulator_output/outC.txt outC.txt
  ./gen_data check
else
  echo "probe output: x86simulator_output/outC.txt"
fi
