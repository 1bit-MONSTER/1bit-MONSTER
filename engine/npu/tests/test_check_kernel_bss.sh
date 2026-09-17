#!/usr/bin/env bash
# test_check_kernel_bss.sh — does check_kernel_bss.sh actually FIRE on the regression it exists for?
#
# The gate asserts the fused int4-GU kernel object has no .bss symbols, because
# aiecc's bare-metal ld.script maps only .text/.data, so zero-initialised statics
# land in .bss, which is DROPPED from the kernel ELF and read back as garbage on
# the NPU (#1838). A gate that cannot fail is worse than no gate, so this proves
# it fails -- by mutating the one thing that decides it.
#
# Mutation: mm_kernel_reference.cc defines
#     #define KERNEL_STATIC __attribute__((section(".data")))
# Emptying that attribute sends the two counters it decorates back to .bss. The
# gate must then report FAIL_BSS_ONLY (exit 1) and name them -- the exact
# signature its header records for the #2199 regression.
#
# The repo source is restored by a trap, so an interrupted run cannot leave it
# mutated. Read-only otherwise.
#
# Usage: engine/npu/tests/test_check_kernel_bss.sh
# Exit:  0 = the gate fired on the mutation and passed on the original
#        1 = it did not (a false negative — the gate is not doing its job)
#        2 = environment problem, not a verdict

set -uo pipefail

REPO="$(cd "$(dirname "$0")/../../.." && pwd)"
G="$REPO/engine/npu/generators"
GATE="$REPO/engine/npu/tests/check_kernel_bss.sh"
SRC="$G/mm_kernel_reference.cc"
DEF='#define KERNEL_STATIC __attribute__((section(".data")))'

for f in "$GATE" "$SRC"; do
    if [ ! -f "$f" ]; then
        echo "ERROR: missing $f" >&2
        exit 2
    fi
done
if ! grep -qF "$DEF" "$SRC"; then
    echo "ERROR: the mutation target is not present in $SRC:" >&2
    echo "  expected: $DEF" >&2
    echo "  If the define was reworded, update this test -- do not delete it, or the" >&2
    echo "  gate stops being verified." >&2
    exit 2
fi

BACKUP="$(mktemp)"
cp "$SRC" "$BACKUP"
restore() { cp "$BACKUP" "$SRC"; rm -f "$BACKUP"; }
trap restore EXIT INT TERM

fail=0

echo "== 1/2 baseline: the gate must PASS on the unmutated source =="
if "$GATE" >/tmp/tckb_base.$$ 2>&1; then
    echo "  PASS  gate exit 0 on the original source"
else
    rc=$?
    echo "  FAIL  gate exited $rc on the ORIGINAL source — the tree is already dirty"
    tail -5 /tmp/tckb_base.$$ | sed 's/^/        /'
    fail=1
fi

echo
echo "== 2/2 mutant: EMPTY KERNEL_STATIC -> the counters must fall into .bss =="
# shellcheck disable=SC2016
sed -i "s|^${DEF}\$|#define KERNEL_STATIC|" "$SRC"
if grep -qF "$DEF" "$SRC"; then
    echo "  FAIL  mutation did not apply (sed matched nothing)"
    fail=1
else
    set +e
    "$GATE" >/tmp/tckb_mut.$$ 2>&1
    rc=$?
    set -e
    if [ "$rc" -eq 0 ]; then
        echo "  FAIL  gate PASSED on a source with the .data attribute removed — false negative"
        tail -5 /tmp/tckb_mut.$$ | sed 's/^/        /'
        fail=1
    elif [ "$rc" -ne 1 ]; then
        echo "  FAIL  gate exited $rc, expected 1 (1 = an assertion failed)"
        tail -5 /tmp/tckb_mut.$$ | sed 's/^/        /'
        fail=1
    else
        echo "  PASS  gate exit 1 as required"
        if grep -q "FAIL_BSS_ONLY" /tmp/tckb_mut.$$; then
            echo "  PASS  reported FAIL_BSS_ONLY (the #2199 signature)"
        else
            echo "  FAIL  exit 1 but not FAIL_BSS_ONLY — some other assertion tripped first:"
            grep -E "SUMMARY|RESULT|ERROR" /tmp/tckb_mut.$$ | sed 's/^/        /'
            fail=1
        fi
        # The gate is supposed to NAME the symbols it found.
        for sym in _ZL9g_i4_call _ZZ16matmul_i8_i32_i4E4call; do
            if grep -q "$sym" /tmp/tckb_mut.$$; then
                echo "  PASS  named the .bss symbol $sym"
            else
                echo "  FAIL  did not name $sym among the .bss symbols"
                fail=1
            fi
        done
    fi
fi

rm -f /tmp/tckb_base.$$ /tmp/tckb_mut.$$

echo
if [ "$fail" -eq 0 ]; then
    echo "RESULT: PASS — the gate fires on the regression it exists for"
else
    echo "RESULT: FAIL — see the FAIL lines above"
fi
exit "$fail"
