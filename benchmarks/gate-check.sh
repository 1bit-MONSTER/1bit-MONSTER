#!/usr/bin/env bash
# gate-check.sh — verify the native engine's first decoded token against FLM's reference.
#
# WHY THIS EXISTS
#   A hand-rolled inline `sed` in this session matched only `[1] <digits>` and produced an
#   EMPTY value for models that print a different shape — Nanbeige prints `[0] boot=5938 (24ms)`
#   and `[1] batch=1 toks: ...`, so the pattern never matched, the value came back empty, and it
#   was read as a REGRESSION. Three commits were reverted on that basis. The engine was fine.
#   So: one parser, both formats, and it PRINTS what it matched.
#
# THE TWO FORMATS THE ENGINE USES
#   runlist path, prefill boot :   [0] boot=5938 (24ms)         -> take the boot= number
#   runlist path, decoded token:   [1] 220                      -> take the bare number
#   batch decode               :   [1] batch=1 toks: 152470     -> NOT a token; must not match
#
# USAGE
#   benchmarks/gate-check.sh            # all gates
#   benchmarks/gate-check.sh <model> <ctx> <engine> <ref>
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# The runlist resolves its per-context ELF dir RELATIVE to the CWD (npu_runlist_bridge.cpp:
# "npu-infer/captures/txn-elfs-8b"), so the harness must run from the repo root.
cd "$ROOT" || exit 1
# The repo's OWN xclbins are the default, and an inherited NPU_XCLBIN_DIR from OUTSIDE this
# worktree is a hazard worth shouting about: this session's shell had it pointing at a sibling
# worktree (`1bit-MONSTER-pi`), so the harness ran against another tree's xclbins and every
# model failed at init. That was misread as a code regression and three verified commits were
# reverted because of it. So: prefer the repo's own directory, and say so when we do not.
if [ -n "${NPU_XCLBIN_DIR:-}" ] && [ "${NPU_XCLBIN_DIR}" != "$ROOT/engine/npu/xclbins" ]; then
    case "${NPU_XCLBIN_DIR}" in
      "$ROOT"/*) : ;;   # inside this repo: fine
      *) echo "WARN: NPU_XCLBIN_DIR is OUTSIDE this worktree:" >&2
         echo "        $NPU_XCLBIN_DIR" >&2
         echo "      using this repo's own xclbins instead ($ROOT/engine/npu/xclbins)" >&2 ;;
    esac
fi
export NPU_XCLBIN_DIR="$ROOT/engine/npu/xclbins"
MODELS="${MODELS:-/home/bcloud/.config/flm/models}"
ENG="$ROOT/engine/npu/build"

# The token extractor. Order matters:
#   1. `boot=<n>`            (prefill boot; the strongest single value)
#   2. a bare `[k] <n>` line (decoded token) -- explicitly NOT `[k] batch=...`
# It prints nothing when neither is present, so an absent value is visible as absent.
extract() {
    local log; log="$(cat)"
    local b; b="$(printf '%s\n' "$log" | sed -n 's/.*boot=\([0-9][0-9]*\).*/\1/p' | head -1)"
    if [ -n "$b" ]; then printf '%s' "$b"; return; fi
    printf '%s\n' "$log" \
      | sed -n 's/^[[:space:]]*\[[0-9][0-9]*\][[:space:]]\{1,\}\([0-9][0-9]*\)[[:space:]]*$/\1/p' \
      | head -1
}

one() {
    local model="$1" ctx="$2" eng="$3" ref="$4" extra="$5"
    local ids="/tmp/ids_${ctx}.txt"
    [ -f "$ids" ] || { printf "  %-22s %-6s %-10s SKIP (no %s)\n" "$model" "$ctx" "-" "$ids"; return; }
    local log v
    # shellcheck disable=SC2086
    log="$(env $extra NPU_RUNLIST=1 timeout 900 "$ENG/npu_engine_$eng" "$MODELS/$model-NPU2/model.q4nx" 4 "$ids" 2>/dev/null)"
    v="$(printf '%s' "$log" | extract)"
    if [ -z "$v" ]; then
        printf "  %-22s %-6s %-10s %-6s  NO TOKEN PARSED\n" "$model" "$ctx" "<none>" "$ref"
        return
    fi
    if [ "$v" = "$ref" ]; then
        printf "  %-22s %-6s %-10s %-6s  MATCH\n" "$model" "$ctx" "$v" "$ref"
        return
    fi
    # A value that was NOT matched may be a device-state artefact of running many models back to
    # back rather than a regression: Nanbeige's gate fails inside this sequence and passes when
    # run alone, reproducibly. Re-check IN ISOLATION before calling it a regression -- a false
    # regression cost this session three reverts of verified work.
    local solo
    # shellcheck disable=SC2086
    solo="$(env $extra NPU_RUNLIST=1 timeout 900 "$ENG/npu_engine_$eng" "$MODELS/$model-NPU2/model.q4nx" 4 "$ids" 2>/dev/null | extract)"
    if [ "$solo" = "$ref" ]; then
        printf "  %-22s %-6s %-10s %-6s  MATCH (in isolation; %s in sequence)%s\n" \
               "$model" "$ctx" "$solo" "$ref" "${v:-<none>}" "${v:+ -- re-check}"
    elif [ -z "$v" ] && [ -z "$solo" ]; then
        printf "  %-22s %-6s %-10s %-6s  NO TOKEN PARSED (sequence and isolation)\n" "$model" "$ctx" "<none>" "$ref"
    else
        printf "  %-22s %-6s %-10s %-6s  ** MISMATCH **\n" "$model" "$ctx" "${solo:-$v}" "$ref"
    fi
}

if [ $# -eq 4 ]; then
    one "$1" "$2" "$3" "$4" ""
    exit 0
fi

echo "gate check -- model ctx measured ref"
one Qwen3-0.6B          1024 qwen3_0_6b     25   ""
one Qwen3-0.6B           256 qwen3_0_6b     1614 ""
one Qwen3-1.7B          1024 qwen3_1_7b     220  ""
one Qwen3-4B             256 qwen3_4b       1614 ""
one Qwen3-4B            1024 qwen3_4b       220  ""
one Qwen3-8B            1024 qwen3_8b       220  ""
one Qwen3-VL-4B-Instruct 1024 qwen3_vl_4b   220  ""
one Nanbeige4.1-3B      1024 nanbeige4_1_3b 1033 ""
one Nanbeige4.1-3B       256 nanbeige4_1_3b 5938 ""
# Llama needs the per-context layer ELFs (benchmarks/gen-layer-elfs.sh); skipped unless present.
one Llama-3.1-8B        1024 llama          220  "NPU_LAYER_ELF_DIR=${NPU_LAYER_ELF_DIR:-/tmp/ll_from_script}"
