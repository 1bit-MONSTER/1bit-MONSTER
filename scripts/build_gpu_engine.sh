#!/usr/bin/env bash
# build_gpu_engine.sh — build the GPU+NPU fused engine (npu_engine_fused).
# Verified 2026-08-15 on Strix Halo (Radeon 8060S gfx1151 + XDNA2 NPU):
# the engine was unwired from CMake and unbuildable; this recipe is the
# exact proven sequence. Requires /opt/rocm-therock + /opt/xilinx/xrt.
#
# Run: bash scripts/build_gpu_engine.sh   (output: /tmp/npu_engine_fused)
#   ./npu_engine_fused model.q4nx N  with NPU_XCLBIN_DIR=engine/npu/xclbins
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH=/opt/rocm-therock/bin:$PATH
HIP_INC=/opt/rocm-therock/include
S=engine/npu/src

hipcc -c -std=c++17 -O2 -march=native -DMODEL_qwen3_0_6b \
    $S/npu_engine_fused.hip -I include -I $S -I $S/../../npu-infer/include -I $HIP_INC -o /tmp/fused.o
g++ -std=c++17 -O2 -march=native -c $S/dequant_q4nx.cpp -I include -I $S -o /tmp/dequant.o
hipcc -c -std=c++17 -O2 -march=native -I include -I $HIP_INC \
    src/kv_cache_attn.hip -o /tmp/kvattn.o
hipcc /tmp/fused.o /tmp/dequant.o /tmp/kvattn.o \
    -L /opt/rocm-therock/lib -lamdhip64 \
    -L /opt/xilinx/xrt/lib -lxrt_coreutil -lxrt_core -laiebu -luuid -lpthread \
    -o /tmp/npu_engine_fused
echo "built /tmp/npu_engine_fused"
