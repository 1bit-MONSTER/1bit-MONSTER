#!/bin/bash
# Build the int4-GU fused GUSILU xclbin (issue #1769, ws09 kernel round).
set -euo pipefail
P=/home/bcloud/mlir-aie/.venv/lib/python3.14/site-packages/llvm-aie
M=/home/bcloud/mlir-aie/.venv/lib/python3.14/site-packages/mlir_aie
PYTHON=/home/bcloud/mlir-aie/.venv/bin/python3
G=$(cd "$(dirname "$0")" && pwd)
W=/tmp/p1i4_build.$$
mkdir -p "$W"; trap 'rm -rf "$W"' EXIT

# kernel: matmul + silu + unpack + dequant in one object
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

# Post-build symbol check (issue #1841 follow-up): the merged kernel object
# must contain every entry symbol the generator links. A stale/partial object
# (the old checked-in generators/mm_32x64x128.o contained only zero_i32)
# fails here loudly instead of surfacing as random aiecc "undefined symbol"
# errors that look like kernel bugs.
for sym in matmul_i8_i32_i4 silu_quant_i8_fused_i4 unpack_i4_b zero_i32 zero_c1; do
    if ! $P/bin/llvm-nm "$W/mm_32x64x128.o" 2>/dev/null | grep -qE " T $sym\$"; then
        echo "ERROR: merged kernel object missing symbol '$sym' — stale/partial build?" >&2
        exit 1
    fi
done
# Duplicate-definition check (issue #1845 prevention): the Q22 sigma LUT
# lives ONLY in silu_quant.h; a stray copy in mm_kernel_reference.cc
# redefines it and silently breaks the aiecc build (the old 2>/dev/null hid
# the compile error, surfacing as unrelated "undefined symbol" failures).
if grep -q "silu_sigmoid_q22\[256\]" "$G/mm_kernel_reference.cc"; then
    echo "ERROR: mm_kernel_reference.cc must NOT define silu_sigmoid_q22 (it lives in silu_quant.h — issue #1845)" >&2
    exit 1
fi

$PYTHON "$G/n1_core_fused_gu_silu_d_p1_i4.py" -M 8 -K 2048 -N_GU 4096 -N_D 2048 \
    -m 8 -k 64 -n 128 -c 8 -b 2 > "$W/design.mlir" 2>/dev/null
export PATH=/home/bcloud/Xilinx/2026.1/2026.1/Vitis/bin:/opt/xilinx/xrt/bin:$PATH
export PYTHONPATH=/home/bcloud/mlir-aie/install_tmp/python:/home/bcloud/mlir-aie/.venv/lib/python3.14/site-packages
export LD_LIBRARY_PATH=/home/bcloud/mlir-aie/install_tmp/python/aie/_mlir_libs
cd "$W"   # aiecc resolves link_with objects relative to the CWD (stale generators/mm_32x64x128.o bug)
/home/bcloud/mlir-aie/build_tmp/bin/aiecc --peano="$P" --aietools="$M" \
    --alloc-scheme=basic-sequential --no-xchesscc --no-xbridge \
    --aie-generate-xclbin --no-compile-host --unified --dynamic-objFifos \
    --aie-generate-npu-insts \
    --xclbin-name="$G/../xclbins/final_i8_MOE_GUSILU_i4_zaya.xclbin" \
    --npu-insts-name="$G/../xclbins/insts_i8_MOE_GUSILU_i4_zaya.txt" \
    "$W/design.mlir"

# Hardcoded-kernel-address verification (issue #1842): the fused int4 kernel
# stashes its silu metadata at 0x6000 (Gg_0) and writes h2 to 0x7F000 (wraps
# to the H2 fifo @ 0xF000). The aiecc emits the buffer address map in
# design.mlir.prj/input_with_addresses.mlir; any generator change that moves
# those buffers silently corrupts the silu (the 2026-08-24 incident: the
# stash at 0x76000 missed Gg_0 @ 0x6000 by 458 KB -> h2 all-+127, 0.3 tok/s).
"$PYTHON" - "$W" <<'PYEOF' || { echo "ERROR: kernel address verification FAILED (issue #1842)" >&2; exit 1; }
import glob, os, re, sys
wd = sys.argv[1]
cands = (glob.glob(os.path.join(wd, "**", "input_with_addresses.mlir"), recursive=True)
         + glob.glob(os.path.join(wd, "**", "*address*.mlir"), recursive=True))
if not cands:
    # aiecc version may not emit the map — nothing to verify, but note it.
    print("WARN: input_with_addresses.mlir not found; skipped kernel-address check (#1842)")
    sys.exit(0)
text = open(cands[0], encoding="utf-8", errors="replace").read()
# tolerant parse of the aie.buffer allocations, e.g.
#   %Gg_0 = aie.buffer(%tile_0_2) {address = 24576 : i32, sym_name = "Gg_0"}
addrs = {}   # name -> set(addresses)
for m in re.finditer(r"%(\w+)\s*=\s*aie\.buffer\([^)]*\)\s*\{[^}]*?\baddress\s*=\s*(\d+)", text):
    addrs.setdefault(m.group(1), set()).add(int(m.group(2)))
if not addrs:
    print("WARN: no 'aie.buffer' allocations found in %s; skipped (#1842)" % cands[0])
    sys.exit(0)
def have(name_pat, want):
    for n, s in addrs.items():
        if re.search(name_pat, n) and any(a == want or a & 0xFFFF == want & 0xFFFF for a in s):
            return True
    return False
ok = True
# v59-critical hardcoded addresses (verified against this map on 2026-08-24):
#   Gg_0 @ 0x6000 (the silu metadata stash — the 0x76000 incident missed it
#                  by 458 KB -> h2 all-+127), C1 accumulator @ 0xE000 (the
#                  zero_c1 target — issue #1769 integration).
# The H2 fifo is not an aie.buffer (objectfifos are absent from this map), so
# its 0x7F000 wrap is not verifiable here.
for pat, want, what in (("Gg_0", 0x6000, "silu metadata stash (0x6000)"),
                        ("C1", 0xE000, "C1 accumulator (0xE000)")):
    if not have(pat, want):
        print("ERROR: no buffer %r at %s (%s) in %s" % (pat, hex(want), what, cands[0]))
        for n, s in addrs.items():
            print("   %-24s %s" % (n, ", ".join(hex(a) for a in sorted(s))))
        ok = False
sys.exit(0 if ok else 1)
PYEOF
