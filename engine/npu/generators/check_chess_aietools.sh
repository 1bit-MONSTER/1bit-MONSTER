#!/bin/bash
# check_chess_aietools.sh — pre-flight guard for the aiecc --xchesscc arm
# (issue #1913).
#
# When --xchesscc is requested, aiecc derives the chess toolchain path from
# --aietools and looks for chess-llvm-link at:
#   <aietools>/tps/lnx64/target_aie2p/bin/LNa64bin/chess-llvm-link
# If --aietools points at mlir-aie's own build_tmp (the peano arm tolerates
# that — and check_mm_kernel_2x4.sh's AIETOOLS default does exactly that),
# aiecc SILENTLY skips the chess-llvm-link step and fails later with a
# confusing 'main_input.chesslinked.ll' missing error at xchesscc_wrapper.
#
# This guard fails loudly instead. Usage:
#   source check_chess_aietools.sh
#   check_chess_aietools <aietools-dir> <true|false: use_xchesscc> [extra-path]
# Returns 0 (ok) or 1 (fatal) — does not exit on its own so callers can
# add context to the error message.
set -u

check_chess_aietools() {
    local aietools="$1"
    local use_chess="$2"
    local extra_path="${3:-}"

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
    return 0
}
