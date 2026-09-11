#!/bin/bash
# test_check_chess_aietools.sh — verification for the chess pre-flight guard.
#
# A guard that has never been shown to fail is documentation. This drives
# check_chess_aietools.sh through every path that matters, asserting exit status
# AND the text that names the cause, plus the model detector itself in both
# directions (an essentials tree must read plain, a 2025.2/2026.1 tree guarded).
#
# It builds its own throwaway aiecc trees, so it touches nothing shared, and it
# skips (exit 0) when the toolchains it needs are not installed.
#
# Usage: ./test_check_chess_aietools.sh [essentials-dir] [vitis-aietools-dir] [mlir-aie-root]
set -u

GEN="$(cd "$(dirname "$0")/../generators" && pwd)"
GUARD="$GEN/check_chess_aietools.sh"

ESSENTIALS="${1:-${LEGACY_AIETOOLS:-$HOME/Downloads/ryzen_ai-1.3.0/vitis_aie_essentials}}"
VITIS="${2:-${VITIS_AIETOOLS:-$(ls -d "$HOME"/Xilinx/*/Vitis/aietools 2>/dev/null | tail -1)}}"
MLIR_AIE="${3:-${MLIR_AIE:-$HOME/mlir-aie}}"

pass=0; fail=0; skipped=0
expect() { # $1 = label, $2 = expected rc, $3 = required substring, rest = command
    local label="$1" want_rc="$2" want_txt="$3"; shift 3
    local out rc
    out=$("$@" 2>&1); rc=$?
    if [ "$rc" = "$want_rc" ] && { [ -z "$want_txt" ] || printf '%s' "$out" | grep -qF "$want_txt"; }; then
        printf '  PASS  %-58s rc=%s\n' "$label" "$rc"; pass=$((pass+1))
    else
        printf '  FAIL  %-58s rc=%s (wanted %s, text %s)\n' "$label" "$rc" "$want_rc" "$want_txt"
        printf '        %s\n' "$(printf '%s' "$out" | head -2 | tr '\n' ' ')"
        fail=$((fail+1))
    fi
}
skip() { printf '  SKIP  %s\n' "$1"; skipped=$((skipped+1)); }

[ -f "$GUARD" ] || { echo "guard not found at $GUARD" >&2; exit 2; }

# ── private fixtures: two throwaway aiecc trees + a PLAIN wrapper ────────────
WORK=$(mktemp -d /tmp/chess-guard-test.XXXXXX)
trap 'rm -rf "$WORK"' EXIT
mkdir -p "$WORK/plain/bin" "$WORK/plain/aie_runtime_lib/AIE2P" "$WORK/none/bin"
printf '#!/bin/sh\nexit 0\n' > "$WORK/plain/bin/aiecc"; chmod +x "$WORK/plain/bin/aiecc"
printf '#!/bin/sh\nexit 0\n' > "$WORK/none/bin/aiecc";  chmod +x "$WORK/none/bin/aiecc"

INSTALL_AIECC="$MLIR_AIE/install/bin/aiecc"
BUILD_AIECC="$MLIR_AIE/build_tmp/bin/aiecc"
WRAPPER="$MLIR_AIE/install/aie_runtime_lib/AIE2P/chess_intrinsic_wrapper.ll"

# A PLAIN wrapper derived from the installed one, so the fixture needs no /tmp state.
derived_plain_ok=0
if [ -f "$WRAPPER" ]; then
    sed -e 's/void_acquire_guarded___uint___uint/void_acquire____uint___uint/g' \
        -e 's/void_release_guarded___uint___sint/void_release____uint___sint/g' \
        "$WRAPPER" > "$WORK/plain/aie_runtime_lib/AIE2P/chess_intrinsic_wrapper.ll"
    derived_plain_ok=1
fi

echo "== check_chess_aietools.sh guard =="
expect "A peano arm short-circuits (no checks, no toolchain needed)" 0 "" \
    bash "$GUARD" --aietools /nonexistent --use-chess false

if [ -n "$VITIS" ] && [ -x "$INSTALL_AIECC" ] && [ "$derived_plain_ok" = 1 ]; then
    expect "B Vitis chess + install aiecc (guarded/guarded compiles)" 0 "never completes" \
        bash "$GUARD" --aietools "$VITIS" --aiecc "$INSTALL_AIECC"
else
    skip "B (needs Vitis aietools, install aiecc and the wrapper)"
fi

if [ -n "$ESSENTIALS" ] && [ -x "$INSTALL_AIECC" ]; then
    expect "C essentials model (plain) + guarded wrapper = FATAL" 1 "LEGACY_PATCH_WRAPPER=1" \
        bash "$GUARD" --aietools "$ESSENTIALS" --aiecc "$INSTALL_AIECC"
else
    skip "C (needs the vitis_aie_essentials tree)"
fi

if [ -n "$VITIS" ] && [ -x "$BUILD_AIECC" ]; then
    expect "D aiecc tree with no runtime lib is rejected, names the fix" 1 "install/bin/aiecc" \
        bash "$GUARD" --aietools "$VITIS" --aiecc "$BUILD_AIECC"
else
    skip "D (needs mlir-aie build_tmp/bin/aiecc)"
fi

if [ -n "$ESSENTIALS" ]; then
    expect "E essentials + PLAIN wrapper = the working configuration" 0 "pre-flight OK" \
        bash "$GUARD" --aietools "$ESSENTIALS" --aiecc "$WORK/plain/bin/aiecc"
    expect "F no runtime lib at all is rejected, names the fix" 1 "install/bin/aiecc" \
        bash "$GUARD" --aietools "$ESSENTIALS" --aiecc "$WORK/none/bin/aiecc"
    expect "G essentials-style tree missing its aie2ps aliases is FATAL" 1 "ln -s aie2p" \
        bash -c "source '$GUARD'; rm -rf '$WORK/ess'; mkdir -p '$WORK/ess/tps/lnx64/target_aie2p' '$WORK/ess/data'; check_chess_aie2ps_aliases '$WORK/ess'"
    expect "H essentials-style tree WITH aliases passes that check" 0 "" \
        bash -c "source '$GUARD'; check_chess_aie2ps_aliases '$ESSENTIALS'"
else
    skip "E/F/G/H (needs the vitis_aie_essentials tree)"
fi

expect "I wrong --aietools names chess-llvm-link (#1913)" 1 "chess-llvm-link not found" \
    bash "$GUARD" --aietools "$WORK/none" --aiecc "${INSTALL_AIECC:-$WORK/plain/bin/aiecc}"

echo "== model detector (it is itself a check: both answers must be produced) =="
if [ -n "$ESSENTIALS" ]; then
    expect "J essentials aie2p model reads PLAIN" 0 "plain" \
        bash -c "source '$GUARD'; r=\$(chess_model_acquire_form '$ESSENTIALS'); echo \$r; [ \$r = plain ]"
else
    skip "J (needs the vitis_aie_essentials tree)"
fi
if [ -n "$VITIS" ]; then
    expect "K Vitis 2025.2/2026.1 aie2p model reads GUARDED" 0 "guarded" \
        bash -c "source '$GUARD'; r=\$(chess_model_acquire_form '$VITIS'); echo \$r; [ \$r = guarded ]"
else
    skip "K (needs Vitis aietools)"
fi

echo "== $pass passed, $fail failed, $skipped skipped =="
[ "$fail" = 0 ]
