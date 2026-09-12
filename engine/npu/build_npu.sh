#!/bin/bash
# build_npu.sh — Build all model variants of the universal NPU engine
set -euo pipefail

SRCDIR="$(cd "$(dirname "$0")" && pwd)"
BUILDDIR="$SRCDIR/build"
REPO_ROOT="$(cd "$SRCDIR/../.." && pwd)"
SRC="$SRCDIR/src/npu_engine_universal.cpp"
DEQUANT="$SRCDIR/src/dequant_q4nx.cpp"
DEQUANT_O="$BUILDDIR/dequant_q4nx.o"
INSTR_GEN="$SRCDIR/src/gemm_npu_instructions.cpp"
INSTR_GEN_O="$BUILDDIR/gemm_npu_instructions.o"
# Zaya decode path (zaya_decode_main — CCA attention on CPU, MoE FFN on NPU,
# NPU_FUSED=1 fused GU->SiLU->D mode). npu_engine_universal dispatches to it
# when the model header contains "zaya", so it must be linked into every
# variant. Requires the generators/ include dir (silu_quant.h) + -mavx2
# (zaya_moe_cpu.h uses AVX2 for the host amax pass).
ZAYA_DECODE="$SRCDIR/src/zaya_decode.cpp"
ZAYA_DECODE_O="$BUILDDIR/zaya_decode.o"
# Single-launch whole-layer per-ctx ELF path (#2080/#2150): RuntimeLayerEngine
# bridge + npu-infer model loader + runtime_layer. model.c is C; the bridge and
# runtime_layer.cpp are C++. Isolated TU: npu-infer's ModelConfig (common.h) must
# NOT reach npu_engine_universal.cpp (name clash with engine's model_config.h).
RUNLIST_BRIDGE="$SRCDIR/src/npu_runlist_bridge.cpp"
RUNLIST_BRIDGE_O="$BUILDDIR/npu_runlist_bridge.o"
# FLM bf16 GEMM bridge (dequant.xclbin + mm.xclbin via libgemm/libdequant) — the
# prefill mm path. Built as a SEPARATE TU with the FLM headers (its Bf16Mm needs
# FLM's lm_config/modules/npu_utils_xrt, which must NOT reach the main engine TU).
FLM_ROOT="${FLM_ROOT:-/home/bcloud/.local/flm-v0946}"
# v0.9.46 drop-in (task-4 MoE unblock): the v0.9.46 headers + the official v0.9.46
# .deb libs (md5 39a6c36a) — the v1.0.x libs NaNs the MoE GDN.
FLM_INC="${FLM_INC:-/home/bcloud/.local/flm-v0946/include}"
FLM_LIB="${FLM_LIB:-/home/bcloud/.local/flm-v0946/lib/xrt}"
BF16MM_BRIDGE="$SRCDIR/src/npu_engine_bf16_mm_bridge.cpp"
BF16MM_BRIDGE_O="$BUILDDIR/npu_engine_bf16_mm_bridge.o"
# FLM prefill bridge (libqwen3_npu::prefill — the prefill/TTFT measurement path)
FLM_PREFILL_BRIDGE="$SRCDIR/src/flm_prefill_bridge.cpp"
FLM_PREFILL_BRIDGE_O="$BUILDDIR/flm_prefill_bridge.o"
RUNLIST_RT="$REPO_ROOT/npu-infer/src/runtime_layer.cpp"
RUNLIST_RT_O="$BUILDDIR/npu_runlist_runtime.o"
NPU_MODEL_C="$REPO_ROOT/npu-infer/src/model.c"
NPU_MODEL_O="$BUILDDIR/npu_model.o"
NPU_INFER_INC="$REPO_ROOT/npu-infer/include"

# XRT headers at /usr/include, libs at system default path
XRT_INC="/usr/include"

# One-time: compile dequantizer
if [ ! -f "$DEQUANT_O" ] || [ "$DEQUANT" -nt "$DEQUANT_O" ]; then
    echo "gcc -c -O3 -o $DEQUANT_O $DEQUANT"
    gcc -c -O3 -o "$DEQUANT_O" "$DEQUANT"
fi

# One-time: compile NPU instruction generator
if [ ! -f "$INSTR_GEN_O" ] || [ "$INSTR_GEN" -nt "$INSTR_GEN_O" ]; then
    echo "g++ -c -std=c++26 -O3 -o $INSTR_GEN_O $INSTR_GEN"
    g++ -c -std=c++26 -O3 -fopenmp -I"$SRCDIR"/src -I"$SRCDIR"/include -I$XRT_INC \
        -o "$INSTR_GEN_O" "$INSTR_GEN"
fi

# One-time: compile the Zaya decode path
if [ ! -f "$ZAYA_DECODE_O" ] || [ "$ZAYA_DECODE" -nt "$ZAYA_DECODE_O" ]; then
    echo "g++ -c -std=c++26 -O3 -mavx2 -o $ZAYA_DECODE_O $ZAYA_DECODE"
    g++ -c -std=c++26 -O3 -mavx2 -fopenmp -DONEBP_SUPPORT \
        -I"$SRCDIR"/src -I"$SRCDIR"/include -I"$SRCDIR"/generators \
        -I"$REPO_ROOT"/include -I$XRT_INC \
        -o "$ZAYA_DECODE_O" "$ZAYA_DECODE"
fi

# One-time: compile the runlist whole-layer stack (model.c + runtime_layer + bridge)
if [ ! -f "$NPU_MODEL_O" ] || [ "$NPU_MODEL_C" -nt "$NPU_MODEL_O" ]; then
    echo "gcc -c -O3 -std=c11 -o $NPU_MODEL_O $NPU_MODEL_C"
    gcc -c -O3 -std=c11 -I"$NPU_INFER_INC" -o "$NPU_MODEL_O" "$NPU_MODEL_C"
fi
if [ ! -f "$RUNLIST_RT_O" ] || [ "$RUNLIST_RT" -nt "$RUNLIST_RT_O" ]; then
    echo "g++ -c -std=c++17 -O3 -o $RUNLIST_RT_O $RUNLIST_RT"
    g++ -c -std=c++17 -O3 -I"$NPU_INFER_INC" -I"$XRT_INC" -o "$RUNLIST_RT_O" "$RUNLIST_RT"
fi
if [ ! -f "$RUNLIST_BRIDGE_O" ] || [ "$RUNLIST_BRIDGE" -nt "$RUNLIST_BRIDGE_O" ]; then
    echo "g++ -c -std=c++17 -O3 -o $RUNLIST_BRIDGE_O $RUNLIST_BRIDGE"
    g++ -c -std=c++17 -O3 -I"$NPU_INFER_INC" -I"$XRT_INC" -o "$RUNLIST_BRIDGE_O" "$RUNLIST_BRIDGE"
fi
# bf16 mm bridge (FLM headers + libgemm/libdequant at link time)
if [ ! -f "$BF16MM_BRIDGE_O" ] || [ "$BF16MM_BRIDGE" -nt "$BF16MM_BRIDGE_O" ] || [ "$SRCDIR/src/npu_engine_bf16_mm.h" -nt "$BF16MM_BRIDGE_O" ]; then
    echo "g++ -c -std=c++17 -O2 -o $BF16MM_BRIDGE_O $BF16MM_BRIDGE"
    g++ -c -std=c++17 -O2 -I"$SRCDIR/src" -I"$FLM_INC" -I"$FLM_INC/npu_utils" -I"$XRT_INC" -o "$BF16MM_BRIDGE_O" "$BF16MM_BRIDGE"
fi
# flm prefill bridge (libqwen3_npu)
if [ ! -f "$FLM_PREFILL_BRIDGE_O" ] || [ "$FLM_PREFILL_BRIDGE" -nt "$FLM_PREFILL_BRIDGE_O" ]; then
    echo "g++ -c -std=c++17 -O2 -mavx2 -o $FLM_PREFILL_BRIDGE_O $FLM_PREFILL_BRIDGE"
    g++ -c -std=c++17 -O2 -mavx2 -include climits -I"$FLM_INC" -I"$FLM_INC/npu_utils" -I"$XRT_INC" -o "$FLM_PREFILL_BRIDGE_O" "$FLM_PREFILL_BRIDGE"
fi

# Models to build
MODELS=(
    "qwen3_0_6b"
    "qwen3_8b"
    "qwen3_vl_4b"
    "llama"
    "gemma4_e2b"
    "qwen3_6_moe_35b"
    "qwen3_5_4b"
    "gemma4_e4b"
    "phi4_mini_4b"
    "nanbeige4_1_3b"
    "zr1"
    "deepseek_v4_flash"
    "qwen3_1_7b"
    "qwen3_4b"
    "qwen3_14b"
    "gemma3_1b"
    "gemma3_4b"
    "smollm2_135m"
)

CXX="${CXX:-g++}"
# Runlist-capable XRT 2.26.0 (exports xrt::runlist, #2150) at a dedicated
# prefix. The system XRT 2.21.75 declares xrt::runlist but does not export it
# (link fails), so prefer the scoped stack when present; fall back to system
# XRT (which still builds the split path, just without runlist batching).
XRT_RUNLIST_LIB="${XRT_RUNLIST_LIB:-/usr/local/xrt-runlist/lib}"
if [ -f "$XRT_RUNLIST_LIB/libxrt_coreutil.so.2" ]; then
    XRT_LIBS=(-L"$XRT_RUNLIST_LIB" -l:libxrt_coreutil.so.2 -l:libxrt_core.so.2 -Wl,-rpath,"$XRT_RUNLIST_LIB")
else
    XRT_LIBS=(-lxrt_coreutil -lxrt_core)
fi
# XRT uses shared libs (must come AFTER source on command line)
LIBS=("${XRT_LIBS[@]}" -laiebu -luuid -lm -ldl -L"$FLM_LIB" -lgemm -ldequant -lqwen3_npu -lqwen3_6_moe_npu -lq4_npu_eXpress -lmha -llm_head -lllama_npu -lgemma4e_npu -lphi4_npu -lnanbeige_npu -llfm2_npu -Wl,-rpath,"$FLM_LIB")
CXXFLAGS=(-std=c++26 -O3 -mavx2 -fopenmp -DONEBP_SUPPORT -I"$SRCDIR/src" -I"$SRCDIR/include" -I"$SRCDIR/generators" -I"$REPO_ROOT/include" -I"$XRT_INC")
ENGINE_OBJS=("$DEQUANT_O" "$INSTR_GEN_O" "$ZAYA_DECODE_O" "$NPU_MODEL_O" "$RUNLIST_RT_O" "$RUNLIST_BRIDGE_O" "$BF16MM_BRIDGE_O" "$FLM_PREFILL_BRIDGE_O")

echo "=== Building NPU engine variants ==="
mkdir -p "$BUILDDIR"

for model in "${MODELS[@]}"; do
    binary="$BUILDDIR/npu_engine_$model"
    echo ""
    echo "--- $model -> $binary ---"
    $CXX "-DMODEL_$model" "${CXXFLAGS[@]}" -o "$binary" "$SRC" "${ENGINE_OBJS[@]}" "${LIBS[@]}"
    ls -lh "$binary"
done

# Also build a default (qwen3_0_6b) as npu_engine for backward compat
echo ""
echo "--- default (qwen3_0_6b) -> $BUILDDIR/npu_engine ---"
$CXX -DMODEL_qwen3_0_6b "${CXXFLAGS[@]}" -o "$BUILDDIR/npu_engine" "$SRC" "${ENGINE_OBJS[@]}" "${LIBS[@]}"
ls -lh "$BUILDDIR/npu_engine"

echo ""
echo "=== All builds complete ==="
ls -lh "$BUILDDIR"/npu_engine*
