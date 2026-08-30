#!/bin/bash
# NOTE (2026-08-29): Qwen3 shapes with the K_GU split — the generator now
# takes K = D input width = IM = 3072 and --K_GU = GU input width = 1024 (the
# assert "D input width must equal the silu'd GU output" holds: 3072 == 6*64*8).
# Emits a valid design.mlir; the xclbin build + silicon round follow.

# build_iron_cascade.sh — build the SINGLE-LAUNCH fused GU→SiLU→D cascade
# xclbin (issue #1775) with the iron API + peano toolchain.
#
# Design (n1_core_fused_gu_silu_d_iron.py): h2 stays core-local; the D GEMM
# is a SINGLE-PASS CASCADE REDUCE down the AIE2P hardware cascade (col c →
# col c+1). Each core accumulates its OWN K-slice partial with the WIDE-N mm
# (matmul_i8_i32_wide_k8, n=N_D_row), then the 8 partials are summed in ONE
# cascade pass (cascade_reduce_{first,mid,last}_i32_wide); col 7 writes the
# row's C2 chunk. No h2 DDR round-trip → no cross-shim visibility race.
#
# Rows (default 1): with ROWS=1 the single-row design is used and N_D must
# fit L1 as (8×N_D) int32 = N_D*4 KB (32 KB @ N_D=1024; N_D=2048 is 64 KB =
# the whole L1). With ROWS>1 the multi-row memtile-fanout design partitions
# N_D across the NPU2 core rows (N_D_row = N_D/ROWS columns per row; c2scr
# 4*N_D_row KB — 20 KB @ N_D_row=640), so N_D up to 1023*ROWS works
# (N_D=2560 @ ROWS=4 silicon-verified, N_D=3840 verified; the shim BD size
# field is 10-bit so N_D_row ≤ 1023).
set -euo pipefail
G="$(cd "$(dirname "$0")" && pwd)"
P=/home/bcloud/iron/lib/python3.14/site-packages/llvm-aie
M=/home/bcloud/iron/lib/python3.14/site-packages/mlir_aie
PYTHON=/home/bcloud/iron/bin/python3
N_D="${N_D:-1024}"
ROWS="${ROWS:-2}"
N_DROW=$((N_D / ROWS))
W=/tmp/iron_cascade.$$ ; mkdir -p "$W"; trap 'rm -rf "$W"' EXIT

# 1. GU kernel object (n=128): matmul_i8_i32 + q22 silu
$P/bin/clang++ --target=aie2p-none-unknown-elf --std=c++20 -O2 \
    -DDIM_M=8 -DDIM_K=64 -DDIM_N=128 -Di8_i32_ONLY -DM8_VECTORIZED \
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
    -I $M/include/aie_kernels/aie2p -I "$G" \
    -c "$G/i4_dequant_kernel.cc" -o "$W/dequant.o" 2>/dev/null
$P/bin/ld.lld -r "$W/mm.o" "$W/silu.o" "$W/dequant.o" -o "$W/mm_32x64x128.o"

# 2. WIDE D kernel object (n=N_D_row): matmul_i8_i32_wide_k8 + cascade_reduce_*_wide
$P/bin/clang++ --target=aie2p-none-unknown-elf --std=c++20 -O2 \
    -DDIM_M=8 -DDIM_K=64 -DDIM_N=128 -Di8_i32_ONLY -DM8_VECTORIZED \
    -DWIDE_DIM_N="$N_DROW" \
    -isystem $P/include/c++/v1 \
    -I /home/bcloud/Xilinx/2025.2/Vitis/aietools/include \
    -I $M/include/aie_kernels/aie2p \
    -c "$G/mm_kernel_reference.cc" -o "$W/wide.o" 2>/dev/null
$P/bin/ld.lld -r "$W/wide.o" -o "$W/wide_d.o"

for sym in matmul_i8_i32_ab silu_quant_i8_fused_q22 matmul_i8_i32_wide matmul_i8_i32_wide_k8 \
           cascade_reduce_first_i32_wide cascade_reduce_mid_i32_wide \
           cascade_reduce_last_i32_wide cascade_reduce_last_i32_wide_add; do
    obj="$W/mm_32x64x128.o"; [[ "$sym" == *wide* ]] && obj="$W/wide_d.o"
    if ! $P/bin/llvm-nm "$obj" 2>/dev/null | grep -qE " T $sym\$"; then
        echo "ERROR: missing symbol '$sym' in $obj" >&2; exit 1
    fi
done

# 3. design.mlir (iron API)
$PYTHON "$G/n1_core_fused_gu_silu_d_iron.py" -M 8 -K 3072 -N_GU 6144 -N_D "$N_D" \
    -m 8 -k 64 -n 128 -c 8 --rows "$ROWS" -b 2 --K_GU 1024 > "$W/design.mlir" 2>/dev/null
grep -q "cascade_flow" "$W/design.mlir" || { echo "ERROR: no cascade_flow in design" >&2; exit 1; }

# 4. aiecc → xclbin (peano flow; the cascade kernels are peano-only)
export PATH=/home/bcloud/Xilinx/2026.1/2026.1/Vitis/bin:/opt/xilinx/xrt/bin:$PATH
export PYTHONPATH=/home/bcloud/mlir-aie/install_tmp/python:/home/bcloud/iron/lib/python3.14/site-packages
export LD_LIBRARY_PATH=/home/bcloud/mlir-aie/install_tmp/python/aie/_mlir_libs:/home/bcloud/iron/lib/python3.14/site-packages/aie/_mlir_libs
mkdir -p "$G/../xclbins"
cd "$W"
/home/bcloud/iron/bin/aiecc --peano="$P" --aietools="$M" \
    --alloc-scheme=basic-sequential --no-xchesscc --no-xbridge \
    --aie-generate-xclbin --no-compile-host --unified --dynamic-objFifos \
    --aie-generate-npu-insts \
    --xclbin-name="$G/../xclbins/final_cascade_fused_qwen3_0_6b.xclbin" \
    --npu-insts-name="$G/../xclbins/insts_cascade_fused_qwen3_0_6b.txt" \
    "$W/design.mlir" 2>&1 | tail -3
echo "OK: $(ls -la "$G/../xclbins/final_cascade_fused_qwen3_0_6b.xclbin" | awk '{print $5}') B xclbin + insts (N_D=$N_D rows=$ROWS nd_row=$N_DROW)"
