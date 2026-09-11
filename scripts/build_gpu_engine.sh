#!/usr/bin/env bash
# build_gpu_engine.sh — build the GPU+NPU fused engine (npu_engine_fused).
# Verified 2026-08-15 on Strix Halo (Radeon 8060S gfx1151 + XDNA2 NPU), and
# re-verified 2026-09-11 with the corrected include/link paths below: build rc=0,
# and the resulting binary generated 8 tokens on the NPU at ~580 ms/tok.
#
# Requires: hipcc on PATH (the aie2p clang from the ROCm SDK) plus the XRT runtime.
# The ROCm SDK headers come from the *devel* include tree, via -isystem. Both halves
# are load-bearing and neither is obvious:
#   * /opt/rocm-therock/include is EMPTY — the real headers live in
#     …/_rocm_sdk_devel/include (the path is derived and checked below);
#   * without it, hipcc resolves <hip/hip_runtime.h> to /usr/include/hip, i.e. the
#     distro libamdhip64-dev header set, which is a DIFFERENT set (its amd_hip_bf16.h
#     etc. diverge). A plain -I does NOT win that lookup; -isystem does.
# The former -L flags were inert: /opt/rocm-therock/lib contains only python3.14 and
# /opt/xilinx/xrt/lib does not exist at all. hipcc injects the correct -L itself, so
# no hand-written -L is needed to link.
#
# Run: OUT=/tmp/npu_engine_fused bash scripts/build_gpu_engine.sh
#   ./npu_engine_fused model.q4nx N  with NPU_XCLBIN_DIR=engine/npu/xclbins
# NOTE: the link output used to be hardcoded, so a rebuild silently overwrote the
# previous binary — which corrupted an A/B comparison once (both files turned out to
# be the same build). Set OUT per run, or copy the binary aside immediately.
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH=/opt/rocm-therock/bin:$PATH
HIP_INC=$(ls -d /opt/rocm-therock/lib/python3*/site-packages/_rocm_sdk_devel/include 2>/dev/null | head -n1 || true)
if [ -z "${HIP_INC:-}" ] || [ ! -d "$HIP_INC" ]; then
    echo "ERROR: ROCm SDK devel include tree not found." >&2
    echo "       Expected /opt/rocm-therock/lib/python3*/site-packages/_rocm_sdk_devel/include" >&2
    exit 1
fi
OUT="${OUT:-/tmp/npu_engine_fused}"
S=engine/npu/src

hipcc -c -std=c++17 -O2 -march=native -DMODEL_qwen3_0_6b \
    $S/npu_engine_fused.hip -I include -I $S -I $S/../../npu-infer/include -isystem "$HIP_INC" -o /tmp/fused.o
g++ -std=c++17 -O2 -march=native -c $S/dequant_q4nx.cpp -I include -I $S -o /tmp/dequant.o
hipcc -c -std=c++17 -O2 -march=native -I include -isystem "$HIP_INC" \
    src/kv_cache_attn.hip -o /tmp/kvattn.o
hipcc /tmp/fused.o /tmp/dequant.o /tmp/kvattn.o \
    -lamdhip64 -lxrt_coreutil -lxrt_core -laiebu -luuid -lpthread \
    -o "$OUT"
echo "built $OUT"
