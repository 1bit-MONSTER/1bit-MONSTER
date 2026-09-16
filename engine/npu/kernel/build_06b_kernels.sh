#!/usr/bin/env bash
set -euo pipefail
# build_06b_kernels.sh — Compile Qwen3-0.6B NPU AIE kernels
# Uses AMD AIE compiler toolchain (aiecc/xchesscc) from torch2aie.
#
# Prerequisites:
#   TORCH2AIE_ROOT=/home/bcloud/torch2aie  (or set env)
#   source $TORCH2AIE_ROOT/setup_env.sh    (sets PATH, LD_LIBRARY_PATH)
set -euo pipefail

TORCH2AIE_ROOT="${TORCH2AIE_ROOT:-${HOME}/torch2aie}"
SRCDIR="$(cd "$(dirname "$0")" && pwd)"
BUILDDIR="${SRCDIR}/../build"
KERNELDIR="${SRCDIR}"
OUTDIR="${BUILDDIR}/qwen3_0_6b_kernels"
mkdir -p "$OUTDIR"

# Toolchain paths
# The chess arm needs an aietools ROOT (the Vitis one), not mlir-aie's build_tmp:
# with build_tmp aiecc silently skips chess-llvm-link and dies later with a
# confusing 'main_input.chesslinked.ll' missing error (issue #1913). The old
# default here (${TORCH2AIE_ROOT}/toolchain/aietools) does not exist on strixhalo,
# and torch2aie is not installed on ryzen at all, so resolve it explicitly and
# fail at the call site instead of inside the compiler.
# shellcheck source=../generators/check_chess_aietools.sh
# shellcheck disable=SC1091
source "${SRCDIR}/../generators/check_chess_aietools.sh"
if [ -z "${AIETOOLS:-}" ]; then
    AIETOOLS="$(find_chess_aietools_root)" || {
        echo "ERROR (#1913): no Vitis aietools root with chess-llvm-link found under ${HOME}/Xilinx*" >&2
        echo "  Set AIETOOLS=<Vitis aietools root> (e.g. ~/Xilinx/2026.1/Vitis/aietools) and retry." >&2
        exit 1
    }
fi
check_chess_aietools "$AIETOOLS" true || exit 1
MLIR_AIE="${MLIR_AIE:-$HOME/mlir-aie}"
# xchesscc_wrapper is an mlir-aie tool, NOT part of the Vitis aietools root (that
# ships only bin/xchesscc + bin/xchessmk). Resolve it from the same mlir-aie tree
# the other kernel scripts use (bench_compiler_ab.sh's default), then the tree's
# install/bin, then PATH - and fail loudly rather than letting a stale default
# reach the compiler.
if [ -z "${XCHESSCC:-}" ]; then
    for candidate in "${MLIR_AIE}/tools/chess-clang/xchesscc_wrapper" \
                     "${MLIR_AIE}/install/bin/xchesscc_wrapper"; do
        if [ -x "$candidate" ]; then
            XCHESSCC="$candidate"
            break
        fi
    done
fi
if [ -z "${XCHESSCC:-}" ] || [ ! -x "$XCHESSCC" ]; then
    XCHESSCC="$(command -v xchesscc_wrapper || true)"
fi
if [ -z "$XCHESSCC" ] || [ ! -x "$XCHESSCC" ]; then
    echo "ERROR: xchesscc_wrapper not found (looked in ${MLIR_AIE}/tools/chess-clang/," >&2
    echo "  ${MLIR_AIE}/install/bin/ and PATH)." >&2
    echo "  Set MLIR_AIE=<mlir-aie tree> (e.g. ~/mlir-aie) or XCHESSCC=<wrapper path>." >&2
    exit 1
fi
# The Vitis launcher must win the xchesscc lookup: aiecc's getAietoolsDir()
# derives its root from `which xchesscc`, so the raw symlink must not precede it.
export PATH="${AIETOOLS}/bin:${PATH}"

# Include paths for AIE kernel compilation
INCLUDES=(
  -I"${KERNELDIR}"
  -I"${AIETOOLS}/include"
  -I"${MLIR_AIE}/include"
  -I"${MLIR_AIE}/include/aie_kernels"
  -I"${MLIR_AIE}/include/aie_kernels/aie2p"
)

echo "=== Building Qwen3-0.6B NPU AIE kernels ==="
echo "Output: ${OUTDIR}"
echo ""

compile_kernel() {
    local src="$1"
    local out="$2"
    local extra_defs="${3:-}"
    echo "  Compiling: $(basename "$src") -> $(basename "$out")"
    $XCHESSCC aie2p \
        ${extra_defs} \
        "${INCLUDES[@]}" \
        -c "$src" \
        -o "$out"
}

# 1. main16 Q4NX GEMM/dequant kernel (06b variant)
compile_kernel \
    "${KERNELDIR}/qwen3_decode_kernels_06b.cc" \
    "${OUTDIR}/qwen3_decode_kernels_06b.o"

# 2. Edge attention kernel (06b: kHeads=4 per worker)
compile_kernel \
    "${KERNELDIR}/edge_attention.cc" \
    "${OUTDIR}/edge_attention_06b.o" \
    "-DMODEL_QWEN3_0_6B"

# 3. Post-process QKV (shared source, uses qwen3_constants_06b.h)
#    Q=2048bf16 output (NH×HD=16×128) from 1024bf16 input (H)
#    K=V=1024bf16 output (NKV×HD=8×128) from 1024bf16 input (H)
compile_kernel \
    "${KERNELDIR}/postprocess_qkv_06b.cc" \
    "${OUTDIR}/postprocess_qkv_06b.o"

# 4. Full vector station (residual add + RMSNorm, H=1024)
compile_kernel \
    "${KERNELDIR}/full_vector_station_06b.cc" \
    "${OUTDIR}/full_vector_station_06b.o"

# 5. SwiGLU (IM-independent, same for all models)
compile_kernel \
    "${KERNELDIR}/swiglu_06b.cc" \
    "${OUTDIR}/swiglu_06b.o"

echo ""
echo "=== All 5 kernels compiled successfully ==="
ls -lh "${OUTDIR}/"*.o
