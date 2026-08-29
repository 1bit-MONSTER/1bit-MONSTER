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
# Issue #1874: I4_SCALAR_C1 is the PRODUCTION DEFAULT — the aie::mmul C1
# store is miscompiled on this toolchain (scrambled for non-uniform B), and
# the v66 scalar-C1 path is the verified-correct fallback (corr 1.0 via the
# #1897 h2/C2 byte-identity gate). Set I4_USE_MMUL=1 to go back to the mmul
# path (experimental: the C-store scramble makes C1 wrong for data-dependent
# B). Requires I4_SCALAR_C1_ACK_1864 (the scalar RMW pattern is #1864-flagged;
# the ack is verified by the CPU + NPU gates on every toolchain bump).
I4_FLAGS=(-DI4_SCALAR_C1 -DI4_SCALAR_C1_ACK_1864)
if [ "${I4_USE_MMUL:-0}" = "1" ]; then
    # Experimental mmul path: B'' dequant must avoid the Bb memory round-trip
    # (#1872 — computed byte-stores dropped/misplaced). I4_DIRECT_VECTOR_DEQ
    # keeps B'' in registers (aie::mul -> acc32 -> sat8); without it the mmul
    # path falls back to the Bb round-trip (unsafe on this toolchain).
    I4_FLAGS=(-DI4_DIRECT_VECTOR_DEQ)
fi
$P/bin/clang++ --target=aie2p-none-unknown-elf --std=c++20 -O2 \
    -DDIM_M=8 -DDIM_K=64 -DDIM_N=128 -Di8_i32_ONLY -DM8_VECTORIZED \
    "${I4_FLAGS[@]}" \
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
# Issue #1865: zero_c1 is no longer in the required list — the generator no
# longer calls it (matmul_i8_i32_i4 zeroes pC via the delivered arg). If a
# stale generator still references it, the aiecc link fails loudly on the
# undefined symbol, which is the desired failure mode.
for sym in matmul_i8_i32_i4 silu_quant_i8_fused_i4 unpack_i4_b zero_i32; do
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
# .bss lint (issue #1838): the aiecc-generated bare-metal ld.script maps only
# .text/.data — zero-init statics land in .bss, which is DROPPED from the
# kernel ELF, so kernel reads of them return garbage (observed in the #1769
# round). mm_kernel_reference.cc forces every mutable static into .data via
# KERNEL_STATIC; fail loudly if any .bss symbol survives in the merged object
# (a future kernel edit that forgets the attribute regresses silently on NPU).
if $P/bin/llvm-nm "$W/mm_32x64x128.o" 2>/dev/null | grep -E ' [bB] ' >/dev/null; then
    echo "ERROR: kernel object has .bss symbols (issue #1838) — add KERNEL_STATIC:" >&2
    $P/bin/llvm-nm "$W/mm_32x64x128.o" 2>/dev/null | grep -E ' [bB] ' >&2 || true
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

# Issue #1837 guard: the aiecc extern-call lowering has dropped p1/p2 setup
# for 3-arg calls (only p0 delivered) — the fused silu call is exactly a
# 3-arg extern (c1, b4, h2). Disassemble the emitted core ELFs and check
# every call site of silu_quant_i8_fused_i4: the instructions before the
# call must set up at least two distinct argument registers. The current
# tree works around the defect (generator keeps a dummy Gg/gs fifo "to keep
# the aiecc's 3-arg extern call codegen healthy"; the kernel reads metadata
# via the reliable c1-arg), so by default this prints a LOUD WARNING listing
# affected call sites in every build log; set NPU_STRICT_1837=1 to turn the
# same detection into a hard build failure.
if [ -d "$W/design.mlir.prj" ]; then
  NPU_STRICT_1837="${NPU_STRICT_1837:-0}" "$PYTHON" - "$W" "$P/bin/llvm-objdump" <<'PYEOF' || { [ "${NPU_STRICT_1837:-0}" = "1" ] && { echo "ERROR: extern-call arg-setup verification FAILED (issue #1837, NPU_STRICT_1837=1)" >&2; exit 1; }; }
import glob, os, re, subprocess, sys
wd, objdump = sys.argv[1], sys.argv[2]
strict = os.environ.get("NPU_STRICT_1837", "0") == "1"
elvs = glob.glob(os.path.join(wd, "design.mlir.prj", "**", "*.elf"), recursive=True)
if not elvs:
    print("WARN (#1837): no core ELFs under %s/design.mlir.prj — guard skipped" % wd)
    sys.exit(0)
bad = 0
checked = 0
for elf in elvs:
    try:
        out = subprocess.run([objdump, "-d", elf], capture_output=True, text=True, timeout=120).stdout
        syms = subprocess.run([objdump, "-t", elf], capture_output=True, text=True, timeout=120).stdout
    except Exception as e:
        print("WARN (#1837): objdump %s failed: %s" % (elf, e))
        continue
    lines = out.splitlines()
    # FIX (2026-08-28, verified against main_core_*_2.elf): the first version
    # matched ANY line containing "silu_quant_i8_fused_i4" — including the
    # function DEFINITION label ('00000b10 <silu_quant_i8_fused_i4>:') — and
    # then inspected the PREVIOUS function's tail (the window showed 'ret lr'
    # and stores that belong to matmul), producing a false "p1/p2 dropped".
    # The real call is a direct branch: 'jl #0xb10' (AIE2P has no indirect
    # calls here), with p0 (c1) set before and p2 (h2) set in the jl delay
    # slot — verified both ARE delivered. Only match actual jl targets:
    # resolve the silu symbol's address from the symbol table (-t) first.
    sym_addr = None
    for ln in syms.splitlines():
        # '00000b10 g F .text 000003c0 silu_quant_i8_fused_i4'
        m = re.search(r"^([0-9a-f]+)\s+g\s+F\s+\.text\s+[0-9a-f]+\s+silu_quant_i8_fused_i4\s*$", ln.strip())
        if m:
            sym_addr = int(m.group(1), 16)
            break
    if sym_addr is None:
        print("WARN (#1837): silu_quant_i8_fused_i4 symbol not found in %s — guard skipped" % elf)
        continue
    for i, ln in enumerate(lines):
        # a jl to the silu address, e.g. '414: 04 01 00 88 05 00 jl #0xb10'
        if not re.search(r"\bjl\s+#0x%x\b" % sym_addr, ln):
            continue
        # real call site — inspect the window around it for arg-reg writes
        # (AIE2P: p0/p1/p2/p3 pointer args; the jl delay slot may hold the
        # last arg mov — include up to 4 instructions after the jl too).
        win = lines[max(0, i - 10):i + 5]
        writes = set()
        for w in win:
            m = re.search(r"\b([pr][0-3])\b", w)
            if m:
                writes.add(m.group(1))
        if len(writes) < 2:
            print("%s (#1837): call to silu_quant_i8_fused_i4 in %s at line %d" % ("ERROR" if strict else "WARNING", elf, i + 1))
            print("  window: " + " | ".join(x.strip() for x in win))
            print("  distinct arg-reg writes found: %s (< 2 -> arg setup may be dropped)" % sorted(writes))
            bad += 1
        checked += 1
if bad:
    if strict:
        print("FATAL (#1837): %d silu call site(s) lack arg setup — the aiecc" % bad)
        print("  extern lowering dropped args; the silu reads stale regs. Fix upstream,")
        print("  or collapse the extern ABI to a single context-buffer arg.")
    else:
        print("NOTE (#1837): %d silu call site(s) lack arg setup (see above)." % bad)
        print("  This is the KNOWN worked-around aiecc extern-call defect (generator keeps")
        print("  a dummy Gg/gs fifo; kernel reads metadata via the reliable c1-arg). Set")
        print("  NPU_STRICT_1837=1 to make this a hard failure.")
    sys.exit(1 if strict else 0)
print("OK (#1837): %d silu call site(s) checked, arg-reg setup present" % checked)
PYEOF
fi

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
# Issue #1865: the hardcoded-address pins (Gg_0@0x6000 silu stash, C1@0xE000
# zero_c1 target, H2@0x7F000 fifo) are RETIRED — the kernel no longer
# hardcodes any tile-local address: matmul_i8_i32_i4 zeroes pC via the
# delivered pC arg, and silu_quant_i8_fused_i4 writes h2 through the
# delivered h2 arg. Nothing pins 0x6000/0xE000/0x7F000 anymore, so a buffer
# move can no longer silently corrupt the silu. Keep a soft informational
# dump of the address map (still useful for debugging) but do NOT fail on
# specific addresses.
print("INFO (#1865): kernel uses delivered args only — no hardcoded tile addresses to pin.")
for n, s in sorted(addrs.items()):
    print("   %-24s %s" % (n, ", ".join(hex(a) for a in sorted(s))))
sys.exit(0)
PYEOF
