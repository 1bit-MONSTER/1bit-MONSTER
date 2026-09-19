#!/bin/bash
# build_moe_ffn_fused.sh — build the engine-side SINGLE-LAUNCH fused MoE FFN
# xclbin for Qwen3.6-35B-A3B (routed GU + shared GU → SiLU → D), using the
# silicon-verified fused GU→SiLU→D cascade generator
# (n1_core_fused_gu_silu_d_iron.py).
#
# Shapes (35B-A3B: H=2048, IM_EXP=512, TOP_K=8, shared=1):
#   GU : K_GU=2048 -> N_GU = routed gate|up (8*1024=8192) + shared (1024) = 9216
#   SiLU on-device: halves to K = 4608
#   D  : K=4608 -> N_D = H = 2048
# This is the FFN half of a whole-layer MoE ELF: attention + router + this
# FFN, all in ONE xclbin partition, so the full layer is ONE runlist submit.
#
# Toolchain: mlir-aie venv (OLD iron Runtime API — .sequence(), which the
# generator targets) + install_tmp aiecc (has --aie-generate-xclbin).
set -euo pipefail
G="$(cd "$(dirname "$0")" && pwd)"
P=/home/bcloud/mlir-aie/.venv/lib/python3.14/site-packages/llvm-aie
M=/home/bcloud/mlir-aie/.venv/lib/python3.14/site-packages/mlir_aie
PYTHON=/home/bcloud/mlir-aie/.venv/bin/python3
AIECC=/home/bcloud/mlir-aie/install_tmp/bin/aiecc
N_D="${N_D:-2048}"
ROWS="${ROWS:-4}"
N_DROW=$((N_D / ROWS))
W=/tmp/moe_ffn_fused.$$ ; mkdir -p "$W"; trap 'rm -rf "$W"' EXIT

echo "== shapes: K_GU=2048 N_GU=9216 K=4608 N_D=$N_D rows=$ROWS N_D_row=$N_DROW =="

# 1. GU kernel object (n=128): matmul_i8_i32_ab (combined A|B) + silu_quant_i8_fused_q22
$P/bin/clang++ --target=aie2p-none-unknown-elf --std=c++20 -O2 \
    -DDIM_M=8 -DDIM_K=64 -DDIM_N=128 -Di8_i32_ONLY -DM8_VECTORIZED \
    -isystem $P/include/c++/v1 \
    -I /home/bcloud/Xilinx/2025.2/Vitis/aietools/include \
    -I $M/include/aie_kernels/aie2p \
    -c "$G/mm_kernel_reference.cc" -o "$W/mm.o" 2>/dev/null
$P/bin/ld.lld -r "$W/mm.o" -o "$W/mm_32x64x128.o"

# 2. WIDE D kernel object (n=N_D_row): matmul_i8_i32_wide_k8 + cascade_reduce_*_i32_wide
$P/bin/clang++ --target=aie2p-none-unknown-elf --std=c++20 -O2 \
    -DDIM_M=8 -DDIM_K=64 -DDIM_N=128 -Di8_i32_ONLY -DM8_VECTORIZED \
    -DWIDE_DIM_N="$N_DROW" \
    -isystem $P/include/c++/v1 \
    -I /home/bcloud/Xilinx/2025.2/Vitis/aietools/include \
    -I $M/include/aie_kernels/aie2p \
    -c "$G/mm_kernel_reference.cc" -o "$W/wide.o" 2>/dev/null
$P/bin/ld.lld -r "$W/wide.o" -o "$W/wide_d.o"

for sym in matmul_i8_i32_ab silu_quant_i8_fused_q22; do
    if ! $P/bin/llvm-nm "$W/mm_32x64x128.o" 2>/dev/null | grep -qE " T $sym\$"; then
        echo "ERROR: missing symbol '$sym' in mm_32x64x128.o" >&2; exit 1
    fi
done
for sym in matmul_i8_i32_wide_k8 cascade_reduce_first_i32_wide cascade_reduce_mid_i32_wide \
           cascade_reduce_last_i32_wide cascade_reduce_last_i32_wide_add; do
    if ! $P/bin/llvm-nm "$W/wide_d.o" 2>/dev/null | grep -qE " T $sym\$"; then
        echo "ERROR: missing symbol '$sym' in wide_d.o" >&2; exit 1
    fi
done
echo "OK: kernel objects built, all symbols present"

# 3. design.mlir (old iron API — the generator calls Runtime() with .sequence())
$PYTHON "$G/n1_core_fused_gu_silu_d_iron.py" -M 8 -K 4608 -N_GU 9216 -N_D "$N_D" \
    -m 8 -k 64 -n 128 -c 8 --rows "$ROWS" -b 2 --K_GU 2048 > "$W/design.mlir" 2>/dev/null
grep -q "cascade_flow" "$W/design.mlir" || { echo "ERROR: no cascade_flow in design" >&2; exit 1; }
echo "OK: design.mlir generated ($(wc -l < "$W/design.mlir") lines)"

# 4. aiecc -> xclbin + insts (install_tmp bindings + install_tmp aiecc)
cd "$W"
export PATH=/home/bcloud/Xilinx/2026.1/Vitis/bin:/opt/xilinx/xrt/bin:$PATH
export PYTHONPATH=/home/bcloud/mlir-aie/install_tmp/python:/home/bcloud/mlir-aie/.venv/lib/python3.14/site-packages
export LD_LIBRARY_PATH=/home/bcloud/mlir-aie/install_tmp/python/aie/_mlir_libs
mkdir -p "$G/../xclbins"
$AIECC --peano="$P" --aietools="$M" \
    --alloc-scheme=basic-sequential --no-xchesscc --no-xbridge \
    --aie-generate-xclbin --no-compile-host --unified --dynamic-objFifos \
    --aie-generate-npu-insts \
    --xclbin-name="$G/../xclbins/final_i8_MOE_FFN_fused_qwen3_6_35b_a3b.xclbin" \
    --npu-insts-name="$G/../xclbins/insts_i8_MOE_FFN_fused_qwen3_6_35b_a3b.txt" \
    "$W/design.mlir" 2>&1 | tail -5
echo "OK: $(ls -la "$G/../xclbins/final_i8_MOE_FFN_fused_qwen3_6_35b_a3b.xclbin" 2>/dev/null | awk '{print $5}') B xclbin"
