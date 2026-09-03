#!/bin/bash
# Build the Qwen3-0.6B int4 fused GU→SiLU→D xclbin with the IRON toolchain.
# vs the default venv (c9c5ecb7) — the newer backend may fix the scalar
# multiply miscompile (issue #1769).
#
# I4_BF16_PAIR=1 (issue #1934 round-11): build the kernel with the additive
# zero-point dequant (B'' = round(q4*a + b), a=s/S_col, b=zp/S_col as a bf16
# pair in the [4096,5120) region) instead of the symmetric-only v66 ratioQ22.
# Emits final_i8_GUSILU_i4_qwen3_0_6b_bf16pair.xclbin so the default build is
# untouched; the zaya path (symmetric zp=0) keeps the verified ratioQ22 kernel.
set -euo pipefail
P=/home/bcloud/iron/lib/python3.14/site-packages/llvm-aie
M=/home/bcloud/iron/lib/python3.14/site-packages/mlir_aie
PYTHON=/home/bcloud/iron/bin/python3
G=$(cd "$(dirname "$0")" && pwd)
W=/tmp/p1i4_iron.$$
mkdir -p "$W"; trap 'rm -rf "$W"' EXIT

MM_FLAGS="-DDIM_M=8 -DDIM_K=64 -DDIM_N=128 -Di8_i32_ONLY -DM8_VECTORIZED"
XCLBIN="$G/../xclbins/final_i8_GUSILU_i4_qwen3_0_6b.xclbin"
INSTS="$G/../xclbins/insts_i8_GUSILU_i4_qwen3_0_6b.txt"
if [ "${I4_BF16_PAIR:-0}" = "1" ]; then
    MM_FLAGS="$MM_FLAGS -DI4_BF16_PAIR -DI4_SCALAR_C1"
    XCLBIN="$G/../xclbins/final_i8_GUSILU_i4_qwen3_0_6b_bf16pair.xclbin"
    INSTS="$G/../xclbins/insts_i8_GUSILU_i4_qwen3_0_6b_bf16pair.txt"
    echo "building bf16-pair variant -> $(basename "$XCLBIN")"
fi

$P/bin/clang++ --target=aie2p-none-unknown-elf --std=c++20 -O2 \
    $MM_FLAGS \
    -isystem $P/include/c++/v1 \
    -I /home/bcloud/Xilinx/2025.2/Vitis/aietools/include \
    -I $M/include/aie_kernels/aie2p \
    -c "$G/mm_kernel_reference.cc" -o "$W/mm.o" 2>/dev/null
$P/bin/clang++ --target=aie2p-none-unknown-elf --std=c++20 -O2 \
    -DDIM_M=8 -DDIM_K=64 -DDIM_N=128 -Di8_i32_ONLY -DM8_VECTORIZED \
    -isystem $P/include/c++/v1 \
    -I /home/bcloud/Xilinx/2025.2/Vitis/aietools/include \
    -I $M/include/aie_kernels/aie2p \
    -c "$G/attn_kernel_reference.cc" -o "$W/silu.o" 2>/dev/null
$P/bin/clang++ --target=aie2p-none-unknown-elf --std=c++20 -O2 \
    -isystem $P/include/c++/v1 \
    -I /home/bcloud/Xilinx/2025.2/Vitis/aietools/include \
    -I $M/include/aie_kernels/aie2p \
    -I "$G" \
    -c "$G/i4_dequant_kernel.cc" -o "$W/dequant.o" 2>/dev/null
$P/bin/ld.lld -r "$W/mm.o" "$W/silu.o" "$W/dequant.o" -o "$W/mm_32x64x128.o"

for sym in matmul_i8_i32_i4 silu_quant_i8_fused_i4 unpack_i4_b zero_i32 zero_c1 copy_c1; do
    if ! $P/bin/llvm-nm "$W/mm_32x64x128.o" 2>/dev/null | grep -qE " T $sym\$"; then
        echo "ERROR: missing symbol '$sym'" >&2
        exit 1
    fi
done

$PYTHON "$G/n1_core_fused_gu_silu_d_p1_i4.py" -M 8 -K 1024 -N_GU 6144 -N_D 1024 \
    -m 8 -k 64 -n 128 -c 8 -b 2 > "$W/design.mlir" 2>/dev/null
export PATH=/home/bcloud/Xilinx/2026.1/2026.1/Vitis/bin:/opt/xilinx/xrt/bin:$PATH
export PYTHONPATH=/home/bcloud/mlir-aie/install_tmp/python:/home/bcloud/iron/lib/python3.14/site-packages
export LD_LIBRARY_PATH=/home/bcloud/mlir-aie/install_tmp/python/aie/_mlir_libs:/home/bcloud/iron/lib/python3.14/site-packages/aie/_mlir_libs
cd "$W"
/home/bcloud/iron/bin/aiecc --peano="$P" --aietools="$M" \
    --alloc-scheme=basic-sequential --no-xchesscc --no-xbridge \
    --aie-generate-xclbin --no-compile-host --unified --dynamic-objFifos \
    --aie-generate-npu-insts \
    --xclbin-name="$XCLBIN" \
    --npu-insts-name="$INSTS" \
    "$W/design.mlir" 2>&1 | tail -1
