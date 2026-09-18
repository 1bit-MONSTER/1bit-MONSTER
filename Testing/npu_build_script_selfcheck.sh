#!/usr/bin/env bash
# npu_build_script_selfcheck.sh — exercise engine/npu/build_npu.sh without XRT,
# without a 5-minute -O3 build, and without a device.
#
# Why this exists (issue #2440): nothing in CI invokes build_npu.sh, so a fresh
# clone could not build the NPU engine AT ALL — the script used $BUILDDIR 30 lines
# before creating it and died on its first gcc — and no job noticed until someone
# cloned. The real build needs XRT headers and libs, which a hosted runner does not
# have; that is why I first filed this as a decision rather than a patch. It does
# not need them: the compilers are stubs, so what is under test is the script's own
# file and directory handling, which is where the bug was.
#
# It runs the script twice:
#   1. as committed, on a scratch tree that has never been built -> must succeed;
#   2. on a copy with `mkdir -p "$BUILDDIR"` deleted              -> must FAIL.
#
# Case 2 is the point. A check that cannot fail is not a check, and this repo has
# been bitten by exactly that (#2424: "a gate that never ran is not a pass").
#
# Run: bash Testing/npu_build_script_selfcheck.sh
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$REPO/engine/npu/build_npu.sh"
[ -f "$SCRIPT" ] || { echo "FAIL: no $SCRIPT"; exit 1; }

fail=0

scratch() { # scratch <dir> — a tree that has never been built
    local d="$1"
    mkdir -p "$d/engine/npu/src" "$d/engine/npu/include" \
             "$d/engine/npu/generators" "$d/include" "$d/stubs"
    cp "$SCRIPT" "$d/engine/npu/build_npu.sh"
    # The stubs never read them, but the script stats some of these paths.
    local f
    for f in dequant_q4nx.cpp gemm_npu_instructions.cpp zaya_decode.cpp \
             npu_engine_universal.cpp; do
        : > "$d/engine/npu/src/$f"
    done
    # A stub compiler that creates whatever -o names. If its directory does not
    # exist the redirection fails, which is exactly how the real gcc behaves — so
    # the use-before-create bug reproduces faithfully under a stub.
    local c
    for c in gcc g++; do
        cat > "$d/stubs/$c" <<'STUB'
#!/usr/bin/env bash
out=""; prev=""
for a in "$@"; do [ "$prev" = "-o" ] && out="$a"; prev="$a"; done
[ -n "$out" ] || exit 0
: > "$out" || exit 1
exit 0
STUB
        chmod +x "$d/stubs/$c"
    done
}

run_script() { # run_script <dir> — prints combined output; returns the script's rc
    ( cd "$1" && PATH="$1/stubs:$PATH" bash engine/npu/build_npu.sh 2>&1 )
}

T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT

# ---- 1. positive: the committed script, on a tree with no build directory ----
A="$T/a"; scratch "$A"
if [ -d "$A/engine/npu/build" ]; then
    echo "FAIL: scratch tree already has engine/npu/build — the precondition is not set"
    fail=1
elif ! out="$(run_script "$A")"; then
    echo "FAIL: build_npu.sh failed on a scratch tree that has never been built"
    printf '%s\n' "$out" | tail -6 | sed 's/^/    /'
    fail=1
elif [ ! -f "$A/engine/npu/build/npu_engine" ]; then
    echo "FAIL: build_npu.sh exited 0 but produced no engine/npu/build/npu_engine"
    fail=1
else
    echo "ok: build_npu.sh runs on a never-built tree ($(ls "$A/engine/npu/build" | wc -l) outputs)"
fi

# ---- 2. negative control: remove the mkdir and require a failure ----
B="$T/b"; scratch "$B"
if ! grep -q '^mkdir -p "\$BUILDDIR"' "$B/engine/npu/build_npu.sh"; then
    echo "FAIL: no line 'mkdir -p \"\$BUILDDIR\"' to remove — control is meaningless"
    fail=1
else
    sed -i '/^mkdir -p "\$BUILDDIR"/d' "$B/engine/npu/build_npu.sh"
    if out="$(run_script "$B")"; then
        echo "FAIL: with the mkdir removed the script still exited 0 — this check cannot see the bug"
        fail=1
    else
        n=$(printf '%s\n' "$out" | grep -ciE 'no such file' || true)
        echo "ok: control — with the mkdir removed it fails ($n 'No such file' line(s))"
    fi
fi

exit "$fail"
