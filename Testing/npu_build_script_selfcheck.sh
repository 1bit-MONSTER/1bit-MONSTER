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
# That comparison has four controls, because a comparison that cannot fail is
# not a comparison: a complete hand list passes, dropping one entry reports the
# missing TU, a hand list sitting BESIDE a `--target` line is still compared
# (the first version skipped both comparisons on the mere presence of `--target`,
# and printed "no source list to drift" over a stale list), and a source named
# only inside a YAML comment does not count as a compile.
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

# strip_comments <file> — the file without YAML comments. A comment is not a
# compile, and this repo has the scar: bench.yml's own line
# `# target already compiles dequant_q4nx.cpp into the binary` satisfied the
# reference test below, so the comparison passed on prose while the .cpp was
# compiled nowhere in that job. `#` starts a comment in YAML when it opens the
# line or follows whitespace.
strip_comments() {
    sed -e 's/^[[:space:]]*#.*$//' -e 's/[[:space:]]#.*$//' "$1"
}

# delegates <workflow> — does the workflow build the engine target itself? Only
# ever used together with "names no source": on its own it must not switch the
# comparisons off (see the header).
delegates() {
    grep -qE -- '--target[= ]+npu_engine_universal' < <(strip_comments "$1")
}

# npu_engine_sources <CMakeLists> — the .cpp names in NPU_ENGINE_SOURCES, with
# comments removed and duplicates collapsed. Comment stripping is not cosmetic on
# this side either: the block carries the note "model.c is C; runtime_layer.cpp +
# bridge are C++", which names a real source in prose. Counting it made n_target 7
# against 6 real sources, so n_ref came out 1 and sources_verdict reported a hand
# list in bench.yml — a FAIL for a workflow that delegates to the CMake target and
# names no source at all. Deduping is what keeps one name from being counted twice.
npu_engine_sources() {
    awk '/set\(NPU_ENGINE_SOURCES/,/^\)/' "$1" \
        | sed -e 's/^[[:space:]]*#.*$//' -e 's/[[:space:]]#.*$//' \
        | grep -oE '[A-Za-z0-9_]+\.cpp' | sort -u
}

# engine_sources_missing <CMakeLists> <workflow> — prints, one per line, each
# NPU_ENGINE_SOURCES entry the workflow never references (as .cpp or as a linked
# .o). Empty output means the two agree. A source compiled in a separate step and
# linked as an object counts: bench.yml builds dequant_q4nx.cpp in its own step
# and then links build/dequant_q4nx.o, so both spellings are accepted.
engine_sources_missing() {
    local cm="$1" wf="$2" f base
    local srcs
    srcs=$(npu_engine_sources "$cm")
    for f in $srcs; do
        base="${f%.cpp}"
        grep -qE "${base}\.(cpp|o)([^A-Za-z0-9_]|$)" < <(strip_comments "$wf") \
            || printf '%s\n' "$f"
    done
}

# resolve_engine_objs <build_npu.sh> — the objects ENGINE_OBJS links, one per
# line, resolved through the script's own variables: build_npu.sh names them
# indirectly (`RUNLIST_BRIDGE_O="$BUILDDIR/npu_runlist_bridge.o"`), so this is two
# steps. Factored out of the comparison below so the control names exactly the
# objects the comparison looks for. The variable class MUST include digits: plain
# `[A-Z_]` matches the tail of `BF16MM_BRIDGE_O` ("MM_BRIDGE_O"), finds no
# assignment for it, and reports an unparsed name instead of the object.
resolve_engine_objs() {
    local script="$1" v o
    awk '/^ENGINE_OBJS=\(/,/\)/' "$script" \
        | grep -oE '\$\{?[A-Z0-9_]+\}?_O' | tr -d '${}' | sed 's/_O$//' \
        | while read -r v; do
            [ -n "$v" ] || continue
            o=$(sed -n "s/^${v}_O=\"\$BUILDDIR\/\([A-Za-z0-9_.]*\)\".*/\1/p" "$script" | head -1)
            [ -n "$o" ] || o="<unparsed ${v}_O>"
            printf '%s\n' "$o"
        done
}

# engine_objs_missing <build_npu.sh> <workflow> — the objects ENGINE_OBJS links
# that the workflow never references. Empty output means the two agree.
engine_objs_missing() {
    local script="$1" wf="$2" o
    resolve_engine_objs "$script" | while read -r o; do
        grep -qE "${o%.o}\.(cpp|c|o)([^A-Za-z0-9_]|$)" < <(strip_comments "$wf") \
            || printf '%s\n' "$o"
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
BUILD_NPU="$REPO/engine/npu/build_npu.sh"

# Two comparisons that serve different builds — either can drift alone:
#   3a  NPU_ENGINE_SOURCES    vs what bench.yml compiles
#   3b  build_npu.sh ENGINE_OBJS vs what bench.yml links
#
# One shape is not a drift: a workflow that builds the target and names no source
# has no copy to compare, and that is said out loud instead of being read as
# "every source is missing". What the skip must NOT key on is the mere presence of
# a --target line, because a hand list beside it is still what compiles (or
# overwrites) the binary — see control (a) in 3c, which reproduces the bypass.
sources_verdict() {
    local cm="$1" wf="$2" wfname n_target missing n_ref
    wfname="$(basename "$wf")"
    n_target=$(npu_engine_sources "$cm" | wc -l)
    if [ "$n_target" -lt 3 ]; then
        printf 'FAIL: parsed only %s source(s) out of NPU_ENGINE_SOURCES — nothing to compare\n' \
               "$n_target"
        return 1
    fi
    missing="$(engine_sources_missing "$cm" "$wf")"
    set -- $missing; n_ref=$(( n_target - $# ))
    if [ -z "$missing" ]; then
        printf 'ok: %s references all %s NPU_ENGINE_SOURCES\n' "$wfname" "$n_target"
        return 0
    fi
    if [ "$n_ref" -eq 0 ] && delegates "$wf"; then
        printf 'ok: %s builds npu_engine_universal through its CMake target and names none of the %s NPU_ENGINE_SOURCES — no copy to drift\n' \
               "$wfname" "$n_target"
        return 0
    fi
    printf 'FAIL: %s never compiles these NPU_ENGINE_SOURCES:\n' "$wfname"
    printf '        %s\n' $missing
    if [ "$n_ref" -gt 0 ] && delegates "$wf"; then
        printf '      It also builds the target, but it names %s source(s) anyway, and a hand list is what compiles (or overwrites) the binary — the --target line does not excuse it (#2505).\n' \
               "$n_ref"
    else
        printf '      %s is the ONLY job that builds the engine with XRT, so a TU added to CMakeLists.txt without it here links-fails there alone (#2505).\n' \
               "$wfname"
    fi
    return 1
}

objs_verdict() {
    local script="$1" wf="$2" wfname n_obj missing n_ref
    wfname="$(basename "$wf")"
    n_obj=$(awk '/^ENGINE_OBJS=\(/,/\)/' "$script" \
            | sed -e 's/^[[:space:]]*#.*$//' -e 's/[[:space:]]#.*$//' \
            | grep -oE '\$\{?[A-Z0-9_]+\}?_O' | sort -u | wc -l)
    if [ "$n_obj" -lt 2 ]; then
        printf 'FAIL: parsed only %s object(s) out of ENGINE_OBJS — nothing to compare\n' "$n_obj"
        return 1
    fi
    missing="$(engine_objs_missing "$script" "$wf")"
    set -- $missing; n_ref=$(( n_obj - $# ))
    if [ -z "$missing" ]; then
        printf 'ok: %s references all %s ENGINE_OBJS objects\n' "$wfname" "$n_obj"
        return 0
    fi
    if [ "$n_ref" -eq 0 ] && delegates "$wf"; then
        printf 'ok: %s builds npu_engine_universal through its CMake target and names none of the %s ENGINE_OBJS objects — no copy to drift\n' \
               "$wfname" "$n_obj"
        return 0
    fi
    printf 'FAIL: %s never compiles these ENGINE_OBJS objects:\n' "$wfname"
    printf '        %s\n' $missing
    printf '      Each is linked into the engine by build_npu.sh, so bench.yml — the only XRT-enabled engine build in CI — link-fails without it (#2505).\n'
    return 1
}

if [ ! -f "$CM" ] || [ ! -f "$BENCH" ]; then
    echo "FAIL: cannot compare the engine's source sets — missing $CM or $BENCH"
    fail=1
else
    v="$(sources_verdict "$CM" "$BENCH")"; rc=$?
    printf '%s\n' "$v"; [ "$rc" = 0 ] || fail=1
fi
if [ -f "$BUILD_NPU" ] && [ -f "$BENCH" ]; then
    v="$(objs_verdict "$BUILD_NPU" "$BENCH")"; rc=$?
    printf '%s\n' "$v"; [ "$rc" = 0 ] || fail=1
fi

# ---- 3c. controls: the ways either comparison could pass without a compile ----
srcs="$(awk '/set\(NPU_ENGINE_SOURCES/,/^\)/' "$CM" 2>/dev/null \
        | grep -oE '[A-Za-z0-9_]+\.cpp' | sort -u)"
set -- $srcs; first_src="$1"

if [ -n "$srcs" ]; then
    # (a) a hand list BESIDE the --target line must still be compared. Before this
    # control existed, adding one g++ line to the delegating workflow made the
    # guard report "no source list to drift" over a stale list.
    if [ -f "$BENCH" ] && delegates "$BENCH"; then
        strip_comments "$BENCH" > "$T/bench_inject.yml"
        printf '        g++ -O2 -o build/npu_engine_universal engine/npu/src/%s\n' \
               "$first_src" >> "$T/bench_inject.yml"
        v="$(sources_verdict "$CM" "$T/bench_inject.yml")"; rc=$?
        if [ "$rc" = 0 ]; then
            echo "FAIL: control — a hand list beside the --target line was not compared:"
            printf '        %s\n' "$v"
            fail=1
        else
            echo "ok: control — a hand list beside the --target line is still compared"
        fi
    fi

    # (b) a name inside a comment is not a compile reference.
    { printf '# no compiler in this file, only prose:\n'
      for s in $srcs; do printf '#   g++ compiles %s here\n' "$s"; done
    } > "$T/bench_comment.yml"
    v="$(sources_verdict "$CM" "$T/bench_comment.yml")"; rc=$?
    if [ "$rc" = 0 ]; then
        echo "FAIL: control — a comment counted as a compile reference:"
        printf '        %s\n' "$v"
        fail=1
    else
        echo "ok: control — a comment naming every source is still 'never compiles'"
    fi

    # (c) a COMPLETE hand list passes, and dropping one entry reports the TU. This
    # is the shape the comparison exists for, tested independently of whatever
    # shape bench.yml is in today.
    { printf 'name: control-complete-hand-list\n'
      for s in $srcs; do printf '        g++ -c engine/npu/src/%s\n' "$s"; done
    } > "$T/bench_full.yml"
    v="$(sources_verdict "$CM" "$T/bench_full.yml")"; rc=$?
    if [ "$rc" != 0 ]; then
        echo "FAIL: control — a complete hand list was reported as drifted:"
        printf '        %s\n' "$v"
        fail=1
    else
        echo "ok: control — a complete hand list passes ($(printf '%s ' $srcs | wc -w) source(s))"
    fi
    sed "0,/${first_src}/s//${first_src%.cpp}_dropped.cpp/" "$T/bench_full.yml" \
        > "$T/bench_drop.yml"
    v="$(sources_verdict "$CM" "$T/bench_drop.yml")"; rc=$?
    if [ "$rc" = 0 ]; then
        echo "FAIL: control — dropping one entry did not make the comparison fail"
        fail=1
    else
        echo "ok: control — dropping one entry reports the missing TU"
    fi

    # (d) the source LIST is read, not its prose.  Two facts at once: a commented
    # name is invisible, and a real entry is still counted — so this control cannot
    # pass merely because the extractor went blind.
    { printf 'set(NPU_ENGINE_SOURCES\n'
      printf '    ${NPU_SRC_DIR}/alpha.cpp\n'
      printf '    # prose: alpha.cpp is C; gamma.cpp is not a real entry\n'
      printf '    ${NPU_SRC_DIR}/delta.cpp\n'
      printf '    ${NPU_SRC_DIR}/epsilon.cpp\n'
      printf ')\n'
    } > "$T/CM_prose.txt"
    got="$(npu_engine_sources "$T/CM_prose.txt" | tr '\n' ' ')"
    if [ "$got" = "alpha.cpp delta.cpp epsilon.cpp " ]; then
        echo "ok: control — a commented source name is not counted in NPU_ENGINE_SOURCES"
    else
        echo "FAIL: control — NPU_ENGINE_SOURCES read prose (or lost an entry): '$got'"
        fail=1
    fi
    { printf 'set(NPU_ENGINE_SOURCES\n'
      printf '    ${NPU_SRC_DIR}/alpha.cpp\n'
      printf '    ${NPU_SRC_DIR}/delta.cpp\n'
      printf '    ${NPU_SRC_DIR}/epsilon.cpp\n'
      printf '    ${NPU_SRC_DIR}/zeta.cpp\n'
      printf ')\n'
    } > "$T/CM_extra.txt"
    n_extra=$(npu_engine_sources "$T/CM_extra.txt" | wc -l)
    if [ "$n_extra" -eq 4 ]; then
        echo "ok: control — a real added NPU_ENGINE_SOURCES entry IS counted (4)"
    else
        echo "FAIL: control — a real added source was not counted (got $n_extra)"
        fail=1
    fi
fi

if [ -f "$BUILD_NPU" ]; then
    engobjs="$(resolve_engine_objs "$BUILD_NPU" | grep -v '^<unparsed' | sort -u)"
    set -- $engobjs; first_obj="$1"
    if [ -n "$engobjs" ]; then
        { printf 'name: control-complete-object-list\n'
          for o in $engobjs; do printf '        g++ -c engine/npu/src/%s\n' "${o%.o}.cpp"; done
        } > "$T/bench_objfull.yml"
        v="$(objs_verdict "$BUILD_NPU" "$T/bench_objfull.yml")"; rc=$?
        if [ "$rc" != 0 ]; then
            echo "FAIL: control — a complete object list was reported as drifted:"
            printf '        %s\n' "$v"
            fail=1
        else
            echo "ok: control — a complete object list passes ($(printf '%s ' $engobjs | wc -w) object(s))"
        fi
        sed "0,/${first_obj%.o}\.cpp/s//${first_obj%.o}_dropped.cpp/" "$T/bench_objfull.yml" \
            > "$T/bench_objdrop.yml"
        v="$(objs_verdict "$BUILD_NPU" "$T/bench_objdrop.yml")"; rc=$?
        if [ "$rc" = 0 ]; then
            echo "FAIL: control — dropping one object reference did not make the comparison fail"
            fail=1
        else
            echo "ok: control — dropping one object reference reports the missing object"
        fi
    fi
fi

exit "$fail"
