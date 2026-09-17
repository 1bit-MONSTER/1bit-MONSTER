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
# It also compares the engine's SOURCE SET between the two places that describe
# it: NPU_ENGINE_SOURCES in engine/npu/CMakeLists.txt and the hand-written g++
# line in .github/workflows/bench.yml. Only bench.yml compiles the engine with
# XRT — the required `C++ (cmake configure + build)` job runs without it, so
# CMakeLists adds engine/npu only `if(XRT_FOUND)` and never builds the target —
# which makes bench.yml's copy the only thing that can catch a missing TU. The
# two had already drifted when this was added (#2505).
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

# engine_sources_missing <CMakeLists> <workflow> — prints, one per line, each
# NPU_ENGINE_SOURCES entry the workflow never references (as .cpp or as a linked
# .o). Empty output means the two agree. A source compiled in a separate step and
# linked as an object counts: bench.yml builds dequant_q4nx.cpp in its own step
# and then links build/dequant_q4nx.o, so both spellings are accepted.
engine_sources_missing() {
    local cm="$1" wf="$2" f base
    local srcs
    srcs=$(awk '/set\(NPU_ENGINE_SOURCES/,/^\)/' "$cm" \
           | grep -oE '[A-Za-z0-9_]+\.cpp' | sort -u)
    for f in $srcs; do
        base="${f%.cpp}"
        grep -qE "${base}\.(cpp|o)([^A-Za-z0-9_]|$)" "$wf" || printf '%s\n' "$f"
    done
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

# ---- 3. the engine's source set: CMakeLists.txt vs the job that compiles it ----
CM="$REPO/engine/npu/CMakeLists.txt"
BENCH="$REPO/.github/workflows/bench.yml"
if [ ! -f "$CM" ] || [ ! -f "$BENCH" ]; then
    echo "FAIL: cannot compare the engine's source sets — missing $CM or $BENCH"
    fail=1
else
    n_target=$(awk '/set\(NPU_ENGINE_SOURCES/,/^\)/' "$CM" \
               | grep -cE '[A-Za-z0-9_]+\.cpp' || true)
    missing="$(engine_sources_missing "$CM" "$BENCH")"
    if [ "$n_target" -lt 3 ]; then
        # A parse that finds nothing would pass the comparison below vacuously.
        echo "FAIL: parsed only $n_target source(s) out of NPU_ENGINE_SOURCES — nothing to compare"
        fail=1
    elif [ -n "$missing" ]; then
        echo "FAIL: bench.yml never compiles these NPU_ENGINE_SOURCES:"
        printf '        %s\n' $missing
        echo "      bench.yml is the ONLY job that builds the engine with XRT, so a TU"
        echo "      added to CMakeLists.txt without it here links-fails there alone (#2505)."
        fail=1
    else
        echo "ok: bench.yml references all $n_target NPU_ENGINE_SOURCES"
    fi
    # Negative control: drop one reference from a copy and require the comparison
    # to notice. Without this the check above could be passing on a bad pattern.
    if [ -z "$missing" ]; then
        sed '0,/gemm_npu_instructions\.cpp/s//gemm_npu_instructions_dropped.cpp/' \
            "$BENCH" > "$T/bench.yml"
        ctl="$(engine_sources_missing "$CM" "$T/bench.yml")"
        if [ -z "$ctl" ]; then
            echo "FAIL: control — removing a reference did not make the comparison fail"
            fail=1
        else
            echo "ok: control — with one reference removed it reports:$(printf ' %s' $ctl)"
        fi
    fi
fi

exit "$fail"
