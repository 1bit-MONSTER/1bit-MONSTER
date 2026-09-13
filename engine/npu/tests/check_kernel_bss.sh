#!/usr/bin/env bash
# check_kernel_bss.sh — the fused int4-GU kernel object must contain no .bss symbols.
#
# This is the executable form of the lint inside engine/npu/generators/build_p1i4.sh
# (lines 49-81), extracted so it can run *without* that script's aiecc stage — which cannot
# currently reproduce an xclbin on this box (tracked as #2262) — and so CI can run it as its
# own job (the complement to the xclbin provenance check).
#
# Why .bss is fatal (issue #1838, observed in the #1769 round): the aiecc-generated
# bare-metal ld.script maps only .text/.data, so zero-initialised statics land in .bss,
# which is DROPPED from the kernel ELF — kernel reads of them return garbage on the NPU.
# mm_kernel_reference.cc is supposed to force every mutable static into .data via
# KERNEL_STATIC (__attribute__((section(".data")))).
#
# STATE ON MAIN (2026-09-12): all three assertions hold — #2199 is fixed, by #2282
# (f3825fbb6), which restored KERNEL_STATIC on the two counters that had gone back to .bss:
#     b _ZL9g_i4_call
#     b _ZZ16matmul_i8_i32_i4E4call
# and rebuilt the committed xclbin against the new object (67,306 -> 75,040 B) with a
# regenerated provenance manifest. The pre-regression artifact is not reproducible on this
# box, so the #1897 byte-identity gate was re-established by its documented substitute: the
# fused-path [MoE L1 fused dbg] corr comparison, run side by side on both artifacts with
# byte-identical diagnostics (#2282's PR body).
# The .bss assertion below is therefore a plain regression guard, not a known-open state: if
# it fires, read #2199's closing record before assuming a new cause. One caveat stands — the
# xclbin build flow is still not reproducible here (#2262), so a future .data change cannot
# be gated end-to-end until that lands.
#
# Toolchain roots are DERIVED and a missing one is fatal. The generator this replaces
# declared /home/bcloud/Xilinx/2025.2/Vitis/aietools/include (no such path; the real one is
# ~/Xilinx2025/2025.2/...) and redirected stderr on all three compiles, so on this box it
# failed with exit 1 and no message at all — its own lint was unreachable.
#
# Usage: engine/npu/tests/check_kernel_bss.sh
# Exit:  0 = all assertions hold (RESULT: PASS)
#        1 = an assertion failed (RESULT: FAIL; FAIL_BSS_ONLY when only the .bss
#            assertion trips — the #2199 regression signature)
#        2 = toolchain/environment problem — not a verdict about the kernel
set -euo pipefail

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"   # engine/npu/tests -> repo root
G="$REPO/engine/npu/generators"

derive() { ls -d "$@" 2>/dev/null | head -n1 || true; }
P="$(derive "$HOME"/mlir-aie/.venv/lib/python3*/site-packages/llvm-aie)"
M="$(derive "$HOME"/mlir-aie/.venv/lib/python3*/site-packages/mlir_aie)"
AI="$(derive "$HOME"/Xilinx*/2025.2/Vitis/aietools/include)"
for pair in "llvm-aie(peano):$P" "mlir_aie:$M" "aietools-include:$AI"; do
    name="${pair%%:*}"; val="${pair#*:}"
    if [ -z "$val" ] || [ ! -d "$val" ]; then
        echo "ERROR: toolchain root '$name' not found under \$HOME — cannot judge the kernels." >&2
        exit 2
    fi
done

W="$(mktemp -d)"; trap 'rm -rf "$W"' EXIT

# Free-space precheck. The failure this guards against is NOT an error message: on a full
# filesystem a build/extract fails to write, leaves no usable artifact, and a downstream run
# then produces no output at all — which a counter reads as "agreement". That happened for real
# on this box (another lane's 59 GB /tmp capture filled a 62 GB tmpfs) and it was invisible in
# every log line. So: assert headroom before compiling, and assert the artifact afterwards.
need_kb=$((512 * 1024))   # the three TUs plus the merged object
avail_kb=$(df -Pk "$W" 2>/dev/null | awk 'NR==2{print $4}')
if [ -z "${avail_kb:-}" ] || [ "$avail_kb" -lt "$need_kb" ]; then
    echo "ERROR: only ${avail_kb:-?} KB free on $W (need ${need_kb} KB) — refusing to compile into a full filesystem, because the resulting empty artifact would look like a passing run." >&2
    exit 2
fi
I4=(-DDIM_M=8 -DDIM_K=64 -DDIM_N=128 -Di8_i32_ONLY -DM8_VECTORIZED -DI4_SCALAR_C1 -DI4_SCALAR_C1_ACK_1864)
INCS=(-isystem "$P/include/c++/v1" -I "$AI" -I "$M/include/aie_kernels/aie2p")
cc() { "$P/bin/clang++" --target=aie2p-none-unknown-elf --std=c++20 -O2 "$@"; }

echo "== compiling the three TUs (stderr NOT suppressed: a silent compile failure is how the generator hid this) =="
cc "${I4[@]}" "${INCS[@]}" -c "$G/mm_kernel_reference.cc"    -o "$W/mm.o"
cc "${I4[@]}" "${INCS[@]}" -c "$G/attn_kernel_reference.cc"  -o "$W/silu.o"
cc "${INCS[@]}" -I "$G" -c "$G/i4_dequant_kernel.cc"         -o "$W/dequant.o"
"$P/bin/ld.lld" -r "$W/mm.o" "$W/silu.o" "$W/dequant.o" -o "$W/mm_32x64x128.o"

# The artifact must exist and be non-empty before anything is inferred from it.
if [ ! -s "$W/mm_32x64x128.o" ]; then
    echo "ERROR: merged object is missing or zero-length — the build was truncated (filesystem space?)." >&2
    exit 2
fi

sym_ok=ok; dup_ok=ok; other_fail=0

# 1. every entry symbol the generator links must be a defined text symbol (issue #1841:
#    a stale/partial object otherwise surfaces as random aiecc "undefined symbol" errors).
for sym in matmul_i8_i32_i4 silu_quant_i8_fused_i4 unpack_i4_b zero_i32; do
    if ! "$P/bin/llvm-nm" "$W/mm_32x64x128.o" | grep -qE " T $sym\$"; then
        echo "ERROR: merged kernel object is missing symbol '$sym' — stale or partial build?" >&2
        sym_ok="missing:$sym"; other_fail=1
    fi
done

# 2. the Q22 sigma LUT lives ONLY in silu_quant.h; a stray copy redefines it and silently
#    breaks the aiecc build (issue #1845).
if grep -q "silu_sigmoid_q22\[256\]" "$G/mm_kernel_reference.cc"; then
    echo "ERROR: mm_kernel_reference.cc must NOT define silu_sigmoid_q22 (it lives in silu_quant.h — issue #1845)" >&2
    dup_ok="stray-lut"; other_fail=1
fi

# 3. no zero-initialised statics may survive: .bss is dropped from the kernel ELF (#1838).
bss_count="$("$P/bin/llvm-nm" "$W/mm_32x64x128.o" | grep -cE ' [bB] ' || true)"
if [ "$bss_count" != "0" ]; then
    echo "ERROR: kernel object has $bss_count .bss symbol(s) (issue #1838) — add KERNEL_STATIC (issue #2199):" >&2
    "$P/bin/llvm-nm" "$W/mm_32x64x128.o" | grep -E ' [bB] ' >&2 || true
fi

echo "SUMMARY: symbols=$sym_ok duplicate=$dup_ok bss=$bss_count"
if [ "$other_fail" != "0" ]; then
    echo "RESULT: FAIL"
    exit 1
fi
if [ "$bss_count" != "0" ]; then
    echo "RESULT: FAIL_BSS_ONLY"
    exit 1
fi
echo "RESULT: PASS"
