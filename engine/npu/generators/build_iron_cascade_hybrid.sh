#!/bin/bash
# build_iron_cascade_hybrid.sh — kernels compiled with the IRON peano (which has
# the aie_api includes) but the design compiled with the PRODUCTION mlir-aie
# aiecc (build_tmp). Tests whether the build_tmp aiecc resolves the flaky builds.
set -euo pipefail
G="$(cd "$(dirname "$0")" && pwd)"
PYTHON=/home/bcloud/iron/bin/python3
AIECC=/home/bcloud/mlir-aie/build_tmp/bin/aiecc
IP=/home/bcloud/iron/lib/python3.14/site-packages/llvm-aie
IM=/home/bcloud/iron/lib/python3.14/site-packages/mlir_aie
N_D="${N_D:-2048}"
ROWS="${ROWS:-4}"
N_DROW=$((N_D / ROWS))
OUT="${OUT:-final_cascade_fused_zaya_nd2048}"
W=/tmp/iron_cascade_hyb.$$ ; mkdir -p "$W"; trap 'rm -rf "$W"' EXIT

# 1. GU kernel object (iron peano, has the includes)
$IP/bin/clang++ --target=aie2p-none-unknown-elf --std=c++20 -O2 \
    -DDIM_M=8 -DDIM_K=64 -DDIM_N=128 -Di8_i32_ONLY -DM8_VECTORIZED \
    -isystem $IP/include/c++/v1 -isystem $IM/include \
    -I $IM/include/aie_kernels/aie2p \
    -c "$G/mm_kernel_reference.cc" -o "$W/mm.o" 2>/dev/null
$IP/bin/clang++ --target=aie2p-none-unknown-elf --std=c++20 -O2 \
    -DDIM_M=8 -DDIM_K=64 -DDIM_N=128 -Di8_i32_ONLY -DM8_VECTORIZED \
    -isystem $IP/include/c++/v1 -isystem $IM/include \
    -I $IM/include/aie_kernels/aie2p \
    -c "$G/attn_kernel_reference.cc" -o "$W/silu.o" 2>/dev/null
$IP/bin/clang++ --target=aie2p-none-unknown-elf --std=c++20 -O2 \
    -isystem $IP/include/c++/v1 -isystem $IM/include \
    -I $IM/include/aie_kernels/aie2p -I "$G" \
    -c "$G/i4_dequant_kernel.cc" -o "$W/dequant.o" 2>/dev/null
$IP/bin/ld.lld -r "$W/mm.o" "$W/silu.o" "$W/dequant.o" -o "$W/mm_32x64x128.o"
$IP/bin/clang++ --target=aie2p-none-unknown-elf --std=c++20 -O2 \
    -DDIM_M=8 -DDIM_K=64 -DDIM_N=128 -Di8_i32_ONLY -DM8_VECTORIZED \
    -DWIDE_DIM_N="$N_DROW" \
    -isystem $IP/include/c++/v1 -isystem $IM/include \
    -I $IM/include/aie_kernels/aie2p \
    -c "$G/mm_kernel_reference.cc" -o "$W/wide.o" 2>/dev/null
$IP/bin/ld.lld -r "$W/wide.o" -o "$W/wide_d.o"

# 2. design.mlir via the iron python (aie.iron)
export PATH=/home/bcloud/Xilinx/2026.1/2026.1/Vitis/bin:/opt/xilinx/xrt/bin:$PATH
export PYTHONPATH=/home/bcloud/mlir-aie/install_tmp/python:/home/bcloud/iron/lib/python3.14/site-packages
export LD_LIBRARY_PATH=/home/bcloud/mlir-aie/install_tmp/python/aie/_mlir_libs:/home/bcloud/iron/lib/python3.14/site-packages/aie/_mlir_libs
$PYTHON "$G/n1_core_fused_gu_silu_d_iron.py" -M 8 -K 2048 -N_GU 4096 -N_D "$N_D" \
    -m 8 -k 64 -n 128 -c 8 --rows "$ROWS" -b 2 > "$W/design.mlir" 2>/dev/null
grep -q "cascade_flow" "$W/design.mlir" || { echo "ERROR: no cascade_flow" >&2; exit 1; }
mkdir -p "$G/../xclbins"
cd "$W"
$AIECC --peano="$IP" --aietools="$IM" \
    --alloc-scheme=basic-sequential --no-xchesscc --no-xbridge \
    --aie-generate-xclbin --no-compile-host --unified --dynamic-objFifos \
    --aie-generate-npu-insts \
    --xclbin-name="$G/../xclbins/${OUT}.xclbin" \
    --npu-insts-name="$G/../xclbins/insts_${OUT}.txt" \
    "$W/design.mlir" 2>&1 | tail -6
echo "OK: $(ls -la "$G/../xclbins/${OUT}.xclbin" | awk '{print $5}') B"
