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

# Private object dir, and make OUT's own directory if it is missing.
#
# The three objects were written to FIXED /tmp/fused.o, /tmp/dequant.o and
# /tmp/kvattn.o and left there, so every run of this recipe shares one set of
# intermediates. I ran two concurrently to see what that costs: both SUCCEEDED and
# the binaries were the same size, because the script hard-codes
# -DMODEL_qwen3_0_6b and both built the same tree, so the shared objects happened
# to be interchangeable. It is not harmless across two WORKTREES, where the
# sources differ and one run can link the other's object - this repo keeps ~48 of
# them. It also means the recipe cannot build at all when /tmp is full, which has
# happened here (~/.dsh/scratch/mesh/ALERT-tmp-full-2026-09-16.txt; the
# coordination note says "never build or park artifacts in /tmp"), and that a
# successful build leaves objects behind. A private dir honours TMPDIR, so a run
# can be kept off /tmp entirely, and the trap removes it.
#
# OUT was used without ever creating its directory - demonstrated, not assumed.
# With OUT=/tmp/gputest/deep/new/engine the old recipe exits 1 with
#     error: cannot open output file ...: No such file or directory
#     error: 'ld.lld' failed
# i.e. the error names the linker, not the missing directory. Same class as
# #2439, which fixed exactly this for engine/npu/build_npu.sh.
OBJDIR="$(mktemp -d "${TMPDIR:-/tmp}/build_gpu_engine.XXXXXX")" || exit 1
trap 'rm -rf "$OBJDIR"' EXIT
mkdir -p "$(dirname "$OUT")" || exit 1

hipcc -c -std=c++17 -O2 -march=native -DMODEL_qwen3_0_6b \
    $S/npu_engine_fused.hip -I include -I $S -I $S/../../npu-infer/include -isystem "$HIP_INC" -o "$OBJDIR/fused.o"
g++ -std=c++17 -O2 -march=native -c $S/dequant_q4nx.cpp -I include -I $S -o "$OBJDIR/dequant.o"
hipcc -c -std=c++17 -O2 -march=native -I include -isystem "$HIP_INC" \
    src/kv_cache_attn.hip -o "$OBJDIR/kvattn.o"
hipcc "$OBJDIR/fused.o" "$OBJDIR/dequant.o" "$OBJDIR/kvattn.o" \
    -lamdhip64 -lxrt_coreutil -lxrt_core -laiebu -luuid -lpthread \
    -o "$OUT"
echo "built $OUT"
