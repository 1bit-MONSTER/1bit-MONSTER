#!/bin/bash
# build_rms35b_m1.sh — build the Qwen3.6-35B-A3B RMSNorm M=1 kernel (H=2048).
#
# Addendum 27: this is the ONLY op missing from the 35B whole-layer ELF build.
# Every GEMM the layer needs already has a built, bit-identical M=1 kernel
# (QKV K=2048 N=8192; O K=4096 N=2048; MOE_GUSGU K=2048 N=9216; MOE_DSD K=4608
# N=4096) plus attention and SiLU. fk-3's n1_rms_norm.py supplies the design;
# this script builds it for the 35B hidden size so the whole-layer ELF can be
# authored from OUR OWN kernels rather than the broken lib sequence.
#
# 35B dims (from the engine's own banner): H=2048 NC=40 NH=32 NKV=16 HD=128
# IM=512 rope_theta=1e7.
#
# Usage: engine/npu/generators/build_rms35b_m1.sh [-M 1]
set -euo pipefail

M_ROWS=1
[ "${1:-}" = "-M" ] && M_ROWS="${2:-1}"

P=/home/bcloud/mlir-aie/.venv/lib/python3.14/site-packages/llvm-aie
MA=/home/bcloud/mlir-aie/.venv/lib/python3.14/site-packages/mlir_aie
PYTHON=/home/bcloud/mlir-aie/.venv/bin/python3
AIECC=/home/bcloud/mlir-aie/install/bin/aiecc
AIETOOLS=/home/bcloud/mlir-aie/build_tmp
export PATH=/home/bcloud/Xilinx/2026.1/2026.1/Vitis/bin:/opt/xilinx/xrt/bin:$PATH
export PYTHONPATH=/home/bcloud/mlir-aie/install_tmp/python:/home/bcloud/mlir-aie/.venv/lib/python3.14/site-packages
export LD_LIBRARY_PATH=/home/bcloud/mlir-aie/install_tmp/python/aie/_mlir_libs

GENERATOR_DIR="$(cd "$(dirname "$0")" && pwd)"
XCLBIN_DIR="$GENERATOR_DIR/../xclbins"
H=2048

# PID-unique workdir (issue #1777: a fixed /tmp path can be clobbered by a
# co-tenant between generation and aiecc).
workdir="/tmp/rms35_m1_build.$$"
mkdir -p "$workdir"
trap 'rm -rf "$workdir"' EXIT

design="$workdir/design_rms_${H}_m${M_ROWS}.mlir"
xclbin="$XCLBIN_DIR/final_rms_qwen3_6_35b_a3b_m1.xclbin"
insts="$XCLBIN_DIR/insts_rms_qwen3_6_35b_a3b_m1.txt"

echo "═══ RMSNorm M=${M_ROWS} H=${H} ═══"
$PYTHON "$GENERATOR_DIR/n1_rms_norm.py" -M "$M_ROWS" -H "$H" 2>/dev/null > "$design"
[ -s "$design" ] || { echo "ERROR: design generation produced an empty file" >&2; exit 1; }

# addendum-28 discipline: verify by CALL SITES, not declarations.
decls=$(grep -c 'func.func' "$design" || true)
calls=$(grep -c 'func.call' "$design" || true)
echo "  mlir: ${decls} func.func decl(s), ${calls} func.call site(s)"
[ "$calls" -ge 1 ] || { echo "ERROR: no kernel call sites -- the design does not invoke rms_norm" >&2; exit 1; }

# the extern kernel object must be resolvable from the aiecc working dir
[ -f "$GENERATOR_DIR/rms_norm_f32_bf16.o" ] || { echo "ERROR: rms_norm_f32_bf16.o missing (build it from rms_norm_f32_bf16.cc for aie2p)" >&2; exit 1; }
cp "$GENERATOR_DIR/rms_norm_f32_bf16.o" "$workdir/"

cd "$workdir"
$AIECC --peano="$P" --aietools="$AIETOOLS" \
    --alloc-scheme=basic-sequential --no-xchesscc --no-xbridge \
    --aie-generate-xclbin --no-compile-host --unified --dynamic-objFifos \
    --aie-generate-npu-insts \
    --xclbin-name="$xclbin" --npu-insts-name="$insts" \
    "$design" 2>&1 | tail -3
cd "$GENERATOR_DIR"
ls -la "$xclbin" "$insts"
echo "OK: 35B RMSNorm M=${M_ROWS} H=${H} built"
