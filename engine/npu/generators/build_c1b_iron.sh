#!/bin/bash
# Build the int4 fused GUSILU xclbin with the IRON toolchain (llvm-aie cb664e8c)
# vs the default venv (c9c5ecb7) — the newer backend may fix the scalar
# multiply miscompile (issue #1769).
set -euo pipefail
P=/home/bcloud/iron/lib/python3.14/site-packages/llvm-aie
M=/home/bcloud/iron/lib/python3.14/site-packages/mlir_aie
PYTHON=/home/bcloud/iron/bin/python3
G=$(cd "$(dirname "$0")" && pwd)
W=/tmp/p1i4_iron.$$
mkdir -p "$W"; trap 'rm -rf "$W"' EXIT

$P/bin/clang++ --target=aie2p-none-unknown-elf --std=c++20 -O2 \
    -DDIM_M=8 -DDIM_K=64 -DDIM_N=128 -Di8_i32_ONLY -DM8_VECTORIZED -DNPU_C1_DUMP \
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

for sym in matmul_i8_i32_i4 silu_quant_i8_fused_i4 unpack_i4_b zero_i32 zero_c1; do
    if ! $P/bin/llvm-nm "$W/mm_32x64x128.o" 2>/dev/null | grep -qE " T $sym\$"; then
        echo "ERROR: missing symbol '$sym'" >&2
        exit 1
    fi
done

$PYTHON "$G/n1_core_fused_gu_silu_d_p1_i4.py" -M 8 -K 2048 -N_GU 4096 -N_D 2048 \
    -m 8 -k 64 -n 128 -c 8 -b 2 > "$W/design.mlir" 2>/dev/null
export PATH=/home/bcloud/Xilinx/2026.1/2026.1/Vitis/bin:/opt/xilinx/xrt/bin:$PATH
export PYTHONPATH=/home/bcloud/mlir-aie/install_tmp/python:/home/bcloud/iron/lib/python3.14/site-packages
export LD_LIBRARY_PATH=/home/bcloud/mlir-aie/install_tmp/python/aie/_mlir_libs:/home/bcloud/iron/lib/python3.14/site-packages/aie/_mlir_libs
cd "$W"
/home/bcloud/iron/bin/aiecc --peano="$P" --aietools="$M" \
    --alloc-scheme=basic-sequential --no-xchesscc --no-xbridge \
    --aie-generate-xclbin --no-compile-host --unified --dynamic-objFifos \
    --aie-generate-npu-insts \
    --xclbin-name="$G/../xclbins/final_i8_MOE_GUSILU_i4_zaya.xclbin" \
    --npu-insts-name="$G/../xclbins/insts_i8_MOE_GUSILU_i4_zaya.txt" \
    "$W/design.mlir" 2>&1 | tail -1
