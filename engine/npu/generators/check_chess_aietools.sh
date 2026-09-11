#!/bin/bash
# check_chess_aietools.sh — pre-flight guard for the aiecc --xchesscc arm
# (issue #1913, extended for #3690).
#
# When --xchesscc is requested, aiecc derives the chess toolchain path from
# --aietools and looks for chess-llvm-link at:
#   <aietools>/tps/lnx64/target_aie2p/bin/LNa64bin/chess-llvm-link
# If --aietools points at mlir-aie's own build_tmp (the peano arm tolerates
# that — and check_mm_kernel_2x4.sh's AIETOOLS default does exactly that),
# aiecc SILENTLY skips the chess-llvm-link step and fails later with a
# confusing 'main_input.chesslinked.ll' missing error at xchesscc_wrapper.
#
# Four traps are covered. Each one was measured on hardware, and each one fails
# in a way that does NOT name its cause, which is why they are checked here:
#
#  1. chess-llvm-link lookup (#1913) — --aietools must be the aietools ROOT.
#  2. aie_runtime_lib goes with the aiecc BINARY, not with --aietools: aiecc
#     resolves <aiecc-root>/aie_runtime_lib/AIE2P/chess_intrinsic_wrapper.ll, so
#     an aiecc whose tree lacks that file silently skips the link step and then
#     dies with 'xchesscc Failed No such device' (#3690; measured today:
#     build_tmp/AIE2P has no wrapper, install/AIE2P does).
#  3. The acquire form the installed wrapper emits must match what the chess
#     MODEL supports. The 2025.2/2026.1 aie2ps models declare only the guarded
#     (and no_fence) intrinsic; the 2024 vitis_aie_essentials model declares only
#     the plain one. Mismatch = 'unrecognised intrinsic ... model inconsistency'
#     + DSFG errors + a noodle segfault (#3690), not a readable error.
#     Separately: a guarded core does not complete on AIE2P silicon at all, so a
#     guarded build is reported as a runtime warning even when it compiles.
#  4. An essentials-style tree names its targets aie2p while the tools ask for
#     aie2ps; without the two aliases the launcher cannot source its env and
#     chesscc exits with the same opaque 'Failed No such device'.
#
# Usage:
#   source check_chess_aietools.sh
#   check_chess_aietools <aietools-dir> <true|false: use_xchesscc> [extra-path] [aiecc-path]
# Returns 0 (ok) or 1 (fatal) — does not exit on its own so callers can
# add context to the error message.
#
# Standalone (for CI / a manual pre-flight):
#   ./check_chess_aietools.sh [--aietools DIR] [--aiecc PATH] [--use-chess true|false]
set -u

# ── which acquire form does the installed wrapper emit? ──────────────────────
# $1 = aiecc path. Prints: plain | guarded | absent | unknown
chess_wrapper_form() {
    local aiecc="$1" root wrapper
    root=$(dirname "$(dirname "$aiecc")")
    wrapper="${root%/}/aie_runtime_lib/AIE2P/chess_intrinsic_wrapper.ll"
    if [ ! -f "$wrapper" ]; then echo "absent"; return 0; fi
    if grep -q "chessintr_void_acquire____uint___uint" "$wrapper" 2>/dev/null; then
        echo "plain"; return 0
    fi
    if grep -q "chessintr_void_acquire_guarded" "$wrapper" 2>/dev/null; then
        echo "guarded"; return 0
    fi
    echo "unknown"
}

# ── which acquire form does the chess MODEL declare? ─────────────────────────
# $1 = aietools dir. Prints: plain | guarded | unknown
#
# Two details this must get right (both measured the hard way):
#  * scope it to the AIE2P model. An essentials tree ships three models
#    (aie2p, aie_ml, versal_prod) and only aie2p is ours; scanning the whole
#    data/ tree returns whichever file the walk reaches first (aie_ml is guarded,
#    aie2p is plain), i.e. the wrong answer.
#  * use -l with -a (list matching files, treat them as text). Some of these
#    generated headers read as binary to grep, and `grep -o` then prints nothing
#    even though the file matches.
chess_model_acquire_form() {
    local t="$1" d dirs=()
    for d in "$t/data/aie2p" "$t/data/aie2ps" \
             "$t/tps/lnx64/target_aie2p" "$t/tps/lnx64/target_aie2ps"; do
        [ -d "$d" ] && dirs+=("$d")
    done
    if [ ${#dirs[@]} -eq 0 ]; then echo "unknown"; return 0; fi
    if timeout 90 grep -rlsa "chessintr_void_acquire____uint___uint" "${dirs[@]}" \
            >/dev/null 2>&1; then
        echo "plain"; return 0    # 2024 vitis_aie_essentials aie2p model
    fi
    if timeout 90 grep -rlsa -e "chessintr_void_acquire_guarded" \
            -e "chessintr_void_acquire_no_fence" "${dirs[@]}" >/dev/null 2>&1; then
        echo "guarded"; return 0  # 2025.2 / 2026.1 aie2ps models
    fi
    echo "unknown"
}

# ── essentials-style trees need their aie2ps aliases (trap 4) ────────────────
check_chess_aie2ps_aliases() { # $1 = aietools dir
    local t="$1" rc=0
    # Only essentials-style trees ship a real target_aie2p directory; a Vitis
    # aietools root symlinks target_aie2p -> target_aie2ps and needs nothing.
    if [ -d "$t/tps/lnx64/target_aie2p" ] && [ ! -L "$t/tps/lnx64/target_aie2p" ]; then
        if [ ! -e "$t/tps/lnx64/target_aie2ps" ]; then
            echo "ERROR (#3690): $t ships no tps/lnx64/target_aie2ps, but aiecc asks" >&2
            echo "  for target_aie2ps: the launcher cannot source its env and chesscc" >&2
            echo "  exits with 'Failed No such device'.  Fix:" >&2
            echo "    ln -s target_aie2p $t/tps/lnx64/target_aie2ps" >&2
            rc=1
        fi
        if [ ! -e "$t/data/aie2ps" ]; then
            echo "ERROR (#3690): $t ships no data/aie2ps (chesscc needs data/aie2ps/lib)." >&2
            echo "  Fix:" >&2
            echo "    ln -s aie2p $t/data/aie2ps" >&2
            rc=1
        fi
    fi
    return $rc
}

check_chess_aietools() {
    local aietools="$1"
    local use_chess="$2"
    local extra_path="${3:-}"
    local aiecc="${4:-}"

    if [ "$use_chess" != "true" ]; then
        return 0   # peano arm: --aietools may legitimately be mlir-aie build_tmp
    fi

    local chess_link="${aietools%/}/tps/lnx64/target_aie2p/bin/LNa64bin/chess-llvm-link"
    if [ ! -x "$chess_link" ]; then
        echo "ERROR (#1913): --xchesscc requested but chess-llvm-link not found at:" >&2
        echo "  $chess_link" >&2
        echo "  --aietools must be the Vitis aietools ROOT (e.g. ~/Xilinx/<ver>/Vitis/aietools)," >&2
        echo "  not mlir-aie's build_tmp — aiecc silently skips chess-llvm-link there and" >&2
        echo "  fails later with 'main_input.chesslinked.ll' missing." >&2
        return 1
    fi

    # Related setup gotcha (#1913): if an mlir-aie install/bin/xchesscc symlink
    # precedes the Vitis aietools/bin on PATH, aiecc's getAietoolsDir() derives
    # the wrong aietools root from `which xchesscc`. The Vitis launcher must be
    # found first.
    if [ -n "$extra_path" ]; then
        local found
        found=$(PATH="$extra_path:$PATH" command -v xchesscc 2>/dev/null || true)
        case "$found" in
            "$aietools"/*) ;;  # Vitis launcher — good
            *)
                echo "WARNING (#1913): xchesscc resolves to '$found', which is NOT the" >&2
                echo "  Vitis aietools launcher ($aietools/bin/xchesscc). getAietoolsDir()" >&2
                echo "  may derive the wrong root and the chess flow will break the same way." >&2
                ;;
        esac
    fi

    # Trap 4: essentials-style target/data naming.
    check_chess_aie2ps_aliases "$aietools" || return 1

    # Traps 2 and 3: the runtime wrapper, and whether its acquire form matches
    # the model. The aiecc that will run decides the path, so prefer an explicit
    # argument, then $AIECC, then PATH.
    if [ -z "$aiecc" ]; then
        aiecc="${AIECC:-}"
    fi
    if [ -z "$aiecc" ]; then
        aiecc=$(command -v aiecc 2>/dev/null || true)
    fi
    if [ -z "$aiecc" ] || [ ! -x "$aiecc" ]; then
        echo "WARNING (#3690): no aiecc found to inspect, so the chess runtime lib" >&2
        echo "  (aie_runtime_lib/AIE2P/chess_intrinsic_wrapper.ll) was NOT checked." >&2
        return 0
    fi

    local form model
    form=$(chess_wrapper_form "$aiecc")
    model=$(chess_model_acquire_form "$aietools")

    if [ "$form" = "absent" ]; then
        echo "ERROR (#3690): no chess_intrinsic_wrapper.ll in the aiecc tree:" >&2
        echo "  $(dirname "$(dirname "$aiecc")")/aie_runtime_lib/AIE2P/chess_intrinsic_wrapper.ll" >&2
        echo "  aiecc resolves aie_runtime_lib relative to ITSELF (not --aietools), so with" >&2
        echo "  this aiecc the chess-llvm-link step is SKIPPED SILENTLY and the compile dies" >&2
        echo "  later with an opaque 'xchesscc Failed No such device'." >&2
        echo "  Use the aiecc whose tree has the file — e.g. <mlir-aie>/install/bin/aiecc" >&2
        echo "  rather than <mlir-aie>/build_tmp/bin/aiecc." >&2
        return 1
    fi

    if [ "$model" = "plain" ] && [ "$form" = "guarded" ]; then
        echo "ERROR (#3690): the installed wrapper emits the GUARDED acquire, but this" >&2
        echo "  chess model declares only the PLAIN one ($aietools). chesscc then reports" >&2
        echo "  'unrecognised intrinsic ... model inconsistency', DSFG errors, and a noodle" >&2
        echo "  segfault.  Swap the wrapper's 2 calls + 2 declarations to the plain names" >&2
        echo "  for this build and restore them afterwards (bench_compiler_ab.sh does this" >&2
        echo "  with LEGACY_PATCH_WRAPPER=1)." >&2
        return 1
    fi

    if [ "$model" = "guarded" ] && [ "$form" = "plain" ]; then
        echo "WARNING (#3690): the wrapper emits the PLAIN acquire while this model declares" >&2
        echo "  only guarded/no_fence — a 2025.2/2026.1 chesscc may reject it.  The plain" >&2
        echo "  form is what the 2024 vitis_aie_essentials model provides." >&2
    elif [ "$form" = "guarded" ]; then
        echo "WARNING (#3690): this build emits the GUARDED acquire.  It compiles, but on" >&2
        echo "  AIE2P silicon a core stuck on that encoding never completes (all-zero output" >&2
        echo "  / ERT timeout).  The plain form needs the 2024 vitis_aie_essentials model." >&2
    fi
    return 0
}

# ── standalone mode ──────────────────────────────────────────────────────────
if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    AIETOOLS_ARG="${AIETOOLS:-}"
    AIECC_ARG="${AIECC:-}"
    USE_CHESS="true"
    while [ $# -gt 0 ]; do
        case "$1" in
            --aietools)  AIETOOLS_ARG="${2:-}"; shift 2 ;;
            --aiecc)     AIECC_ARG="${2:-}"; shift 2 ;;
            --use-chess) USE_CHESS="${2:-}"; shift 2 ;;
            -h|--help)   sed -n '2,40p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
            *) echo "unknown argument: $1" >&2; exit 2 ;;
        esac
    done
    if [ -z "$AIETOOLS_ARG" ]; then
        echo "usage: $0 --aietools DIR [--aiecc PATH] [--use-chess true|false]" >&2
        exit 2
    fi
    if check_chess_aietools "$AIETOOLS_ARG" "$USE_CHESS" "" "$AIECC_ARG"; then
        printf 'chess pre-flight OK: aietools=%s aiecc=%s wrapper=%s model=%s\n' \
            "$AIETOOLS_ARG" "${AIECC_ARG:-<from PATH>}" \
            "$(chess_wrapper_form "${AIECC_ARG:-$(command -v aiecc 2>/dev/null || echo none)}")" \
            "$(chess_model_acquire_form "$AIETOOLS_ARG")"
        exit 0
    fi
    exit 1
fi
