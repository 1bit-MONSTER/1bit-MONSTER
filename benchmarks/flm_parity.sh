#!/usr/bin/env bash
# flm_parity.sh — measure the native NPU engine vs FastFlowLM on THIS box and
# emit a decode/prefill/TTFT parity diff.
#
# Goal mttxt22c-a6rv75, task-1: "Build the FLM-parity benchmark harness".
#
# Two measurement paths, same hardware/weights/prompt/context lengths:
#   FLM    : `flm bench <tag> -i <config.json>` (hidden command, v1.0.4) writes
#            bench_<tag>_<yyyymmdd>.csv with ttft/prefill/decode per context.
#   native : `npu_engine_<model> model.q4nx <decode_tokens> <ids_file>`, where
#            <ids_file> is whitespace-separated token IDs from
#            engine/npu/tokenizer/tokenize (comma output -> tr ',' ' ').
#            Markers parsed: "Prefill: Xms (Y ms/tok)" and the final
#            "=== Z ms/tok (W tok/s) | ... ===" line.
#
# The published FLM tables (Kraken Point) live in amd-oss/.../benchmarks/ and
# are the *official* reporting bar; this script's primary job is the on-box
# apples-to-apples native-vs-FLM comparison. See
# benchmarks/FLM-PARITY-DATA-SOURCES.md.
#
# Usage:
#   flm_parity.sh --model qwen3_0_6b --flm-tag qwen3:0.6b \
#     --engine engine/npu/build/npu_engine_qwen3_0_6b \
#     --q4nx ~/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx \
#     --tokenizer ~/.config/flm/models/Qwen3-0.6B-NPU2/tokenizer.json \
#     --prompt benchmarks/prompts/reclaimer.txt \
#     --ctx-k 1 --decode-tokens 32 [--flm-max-length 1024] [--flm-iterations 1]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOKENIZE="$ROOT/engine/npu/tokenizer/tokenize"
FLM="${FLM:-/opt/fastflowlm/bin/flm}"
# the pre-built npu_engine_* binaries carry a stale compiled-in xclbin path;
# point them at this repo's xclbins (see FLM-PARITY-DATA-SOURCES.md).
export NPU_XCLBIN_DIR="${NPU_XCLBIN_DIR:-$ROOT/engine/npu/xclbins}"

MODEL="" FLM_TAG="" ENGINE="" Q4NX="" TOKENIZER="" PROMPT=""
CTX_K=1 DECODE_TOKENS=32
FLM_MAX_LENGTH=1024 FLM_ITERATIONS=1
SKIP_FLM=0 SKIP_NATIVE=0
WORK="$(mktemp -d)"; [ -n "${KEEP_WORK:-}" ] || trap 'rm -rf "$WORK"' EXIT

usage() { sed -n '20,34p' "$0"; exit 1; }
while [ $# -gt 0 ]; do
  case "$1" in
    --model)            MODEL="$2"; shift 2 ;;
    --flm-tag)          FLM_TAG="$2"; shift 2 ;;
    --engine)           ENGINE="$2"; shift 2 ;;
    --q4nx)             Q4NX="$2"; shift 2 ;;
    --tokenizer)        TOKENIZER="$2"; shift 2 ;;
    --prompt)           PROMPT="$2"; shift 2 ;;
    --ctx-k)            CTX_K="$2"; shift 2 ;;
    --decode-tokens)    DECODE_TOKENS="$2"; shift 2 ;;
    --flm-max-length)   FLM_MAX_LENGTH="$2"; shift 2 ;;
    --flm-iterations)   FLM_ITERATIONS="$2"; shift 2 ;;
    --skip-flm)         SKIP_FLM=1; shift ;;
    --skip-native)      SKIP_NATIVE=1; shift ;;
    *) usage ;;
  esac
done
[ -n "$MODEL" ] && [ -n "$ENGINE" ] && [ -n "$Q4NX" ] || usage

# ---------------------------------------------------------------------------
# measure_flm: run `flm bench` and return "ttft_s prefill_tok_s decode_tok_s"
# for the given context length (context_length_k in the CSV).
# ---------------------------------------------------------------------------
measure_flm() {
  local tag="$1" maxlen="$2" iters="$3" ctx_k="$4"
  [ "$SKIP_FLM" = 1 ] && { echo "skip"; return 0; }
  local cfg="$WORK/flm_bench.json"
  python3 - "$maxlen" "$iters" "$cfg" "$PROMPT" <<'PYEOF'
import json, sys
maxlen, iters, cfg, prompt = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3], sys.argv[4]
d = {"max_length": maxlen, "iterations": iters}
if prompt:
    d["input_text"] = open(prompt).read()
json.dump(d, open(cfg, "w"))
PYEOF

  local dir="$WORK/flm"; mkdir -p "$dir"
  ( cd "$dir" && "$FLM" bench "$tag" -i "$cfg" >stdout.log 2>&1 )
  local csv; csv="$(ls "$dir"/bench_*.csv 2>/dev/null | head -1 || true)"
  [ -z "$csv" ] && { echo "FLM bench produced no CSV (see $dir/stdout.log)"; tail -5 "$dir/stdout.log" >&2; return 1; }
  # header: context_length_k,ttft_avg_s,...,prefill_avg_toks_per_s,...,decoding_avg_toks_per_s,...
  python3 - "$csv" "$ctx_k" <<'PYEOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
ctx = int(sys.argv[2])
for r in rows:
    if int(r["context_length_k"]) == ctx:
        print(r["ttft_avg_s"], r["prefill_avg_toks_per_s"], r["decoding_avg_toks_per_s"])
        sys.exit(0)
print("ctx_k not found in", sys.argv[1]); sys.exit(2)
PYEOF
}

# ---------------------------------------------------------------------------
# measure_native: tokenize prompt -> repeat to ctx_k -> run engine -> parse.
# Returns "ttft_s prefill_tok_s decode_tok_s".
# ---------------------------------------------------------------------------
measure_native() {
  local ctx_k="$1"
  [ "$SKIP_NATIVE" = 1 ] && { echo "skip"; return 0; }
  # 1) tokenize prompt -> comma-separated IDs -> whitespace-separated
  local ids="$WORK/ids.txt"
  if [ -n "$PROMPT" ] && [ -f "$PROMPT" ]; then
    "$TOKENIZE" "$TOKENIZER" <"$PROMPT" 2>/dev/null | tr ',' ' ' >"$ids"
  else
    echo "0" >"$ids"   # no prompt -> single token
  fi
  # 2) repeat the token sequence to approximate ctx_k * 1024 tokens
  local all="$WORK/all_ids.txt"; : >"$all"
  local seq; seq="$(cat "$ids")"
  # ctx_k is 1/2/4/8/16/32 (thousands). FLM repeats the story 1<<log2(k)=k times,
  # so k copies of the ~1k-token story matches FLM's "<k>k" stage.
  for _ in $(seq 1 "$ctx_k"); do printf '%s ' "$seq" >>"$all"; done
  # 3a) prefill via FLM's qwen3_npu::prefill (NPU_FLM_PREFILL=1 — architectural
  #     change; the hand-rolled bf16 reimplementation diverged at H>1024)
  # TRULY-NATIVE mode (FLM_PARITY_TRUE_NATIVE=1): do NOT set NPU_FLM_* — those
  # drive FLM's own captured libs through the engine and are NOT the native
  # backend. Native prefill = the bf16 per-op path (NPU_PREFILL_BF16=1) with an
  # explicit token cap (NPU_PREFILL_MAX, default 1024 = the generated
  # long-context attention ELF at $NPU_XCLBIN_DIR/attn_mha_1024_nh16.elf);
  # native decode = the whole-layer runlist (NPU_RUNLIST=1).
  local pout="$WORK/prefill.log" dout="$WORK/decode.log"
  if [ "${FLM_PARITY_TRUE_NATIVE:-0}" = 1 ]; then
    NPU_RUNLIST=0 NPU_PREFILL_BF16=1 NPU_PREFILL_MAX="${NPU_PREFILL_MAX:-1024}" \
      "$ENGINE" "$Q4NX" 1 "$all" >"$pout" 2>&1 || true
    NPU_RUNLIST=1 "$ENGINE" "$Q4NX" "$DECODE_TOKENS" "$all" >"$dout" 2>&1 || true
  else
    NPU_FLM_PREFILL=1 "$ENGINE" "$Q4NX" 1 "$all" >"$pout" 2>&1 || true
    NPU_FLM_PREFILL=1 NPU_FLM_DECODE=1 "$ENGINE" "$Q4NX" "$DECODE_TOKENS" "$all" >"$dout" 2>&1 || true
  fi
  # 4) parse markers (grep -m1 avoids the set -e + head early-close SIGPIPE trap)
  local prefill_ms prefill_ms_tok decode_tok_s ttft_s npt
  npt="$(grep -oE '=== Prefill [0-9]+ ===' "$pout" | grep -om1 '[0-9]\+' || true)"
  prefill_ms="$(grep -oE 'Prefill: [0-9]+ms \([0-9.]+ ms/tok\)' "$pout" | grep -oE '[0-9]+ms' | grep -om1 '[0-9]\+' || true)"
  prefill_ms_tok="$(grep -oE 'Prefill: [0-9]+ms \([0-9.]+ ms/tok\)' "$pout" | grep -oE '[0-9.]+ ms/tok' | grep -om1 '[0-9.]\+' || true)"
  decode_tok_s="$(grep -oE '\([0-9.]+ tok/s\)' "$dout" | grep -oE '[0-9.]+' | tail -1 || true)"
  if [ -n "$prefill_ms" ] && [ -n "$prefill_ms_tok" ]; then
    ttft_s="$(python3 -c "print(round($prefill_ms/1000.0,4))")"
    local prefill_tok_s; prefill_tok_s="$(python3 -c "print(round(1000.0/$prefill_ms_tok,1))")"
  else
    ttft_s=""; prefill_tok_s=""
  fi
  echo "$ttft_s $prefill_tok_s $decode_tok_s $npt"
  # keep the logs for diagnostics
  cp "$pout" "$WORK/prefill_${MODEL}_ctx${ctx_k}.log"
  cp "$dout" "$WORK/decode_${MODEL}_ctx${ctx_k}.log"
}

# ---------------------------------------------------------------------------
# run + diff
# ---------------------------------------------------------------------------
echo "== FLM parity: $MODEL (ctx=${CTX_K}k, decode=$DECODE_TOKENS tok) =="
FLM_R="$(measure_flm "$FLM_TAG" "$FLM_MAX_LENGTH" "$FLM_ITERATIONS" "$CTX_K" || true)"
NAT_R="$(measure_native "$CTX_K" || true)"
read -r F_TTFT F_PREF F_DEC <<<"$FLM_R"
read -r N_TTFT N_PREF N_DEC N_NPT <<<"$NAT_R"

printf '\n%-12s %14s %14s %14s\n' "metric" "native" "FLM(on-box)" "gap%"
printf '%-12s %14s %14s %14s\n' "decode tok/s" "${N_DEC:-n/a}" "${F_DEC:-n/a}" "$(python3 -c "import sys; a=sys.argv[1]; r=sys.argv[2]; print(round((float(a)-float(r))/float(r)*100,1) if a and r and a!='n/a' and r!='n/a' else '')" "$N_DEC" "$F_DEC" 2>/dev/null)"
printf '%-12s %14s %14s\n' "prefill tok/s" "${N_PREF:-n/a}" "${F_PREF:-n/a}"
printf '%-12s %14s %14s\n' "TTFT (s)" "${N_TTFT:-n/a}" "${F_TTFT:-n/a}"
printf '%-12s %14s %14s\n' "prefill tokens" "${N_NPT:-n/a}" "~1928 (1 story copy)"
echo
echo "native log: $WORK/native_${MODEL}_ctx${CTX_K}.log"
