#!/usr/bin/env bash
# oracle_accuracy_0_6b.sh — score native Qwen3-0.6B decode arms against the FLM oracle.
#
# Goal mu34scbf-4ffm1o step 1/2. Replaces ad-hoc single-prompt eyeballing with a
# measured per-arm accuracy number over a fixed prompt set.
#
# Reference: FLM (an independent implementation of the same model on the same box).
#   echo "<prompt>" | flm run qwen3:0.6b      ->  its "Model RAW Output"
# Arms:  runlist  NPU_RUNLIST=1  (the fast arm, 64-103 tok/s, what flm_parity.sh measures)
#        dense    NPU_RUNLIST=0  (2 tok/s, first 3 ids correct then gibberish)
#
# Method rules (binding — this line of work has already produced three confident
# wrong results, two of them vacuous comparisons):
#   * decode is forced greedy (NPU_GREEDY=1) so a comparison is a comparison
#   * every extraction is asserted NON-EMPTY before it is scored or compared
#   * results are appended per prompt, so a timeout still leaves usable data
#
# Usage: bash benchmarks/oracle_accuracy_0_6b.sh [prompt_set.tsv] [ntok]
#   prompt_set: lines "prompt|expected-substring"
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SET="${1:-$ROOT/benchmarks/prompts/qwen3_0_6b_oracle_set.txt}"
# NB: NTOK must be accepted from the command line. It previously read only the
# ENVIRONMENT ("${NTOK:-6}"), so every run invoked as `script <set> 32` silently used
# 6 tokens -- which made a 3-prompt templated comparison look like prompt-independent
# degeneration when it was only truncated inside the assistant's shared preamble.
NTOK="${2:-${NTOK:-6}}"

MODEL_DIR="$HOME/.config/flm/models/Qwen3-0.6B-NPU2"
TOK="$ROOT/engine/npu/tokenizer/tokenize"
DET="$ROOT/engine/npu/tokenizer/detokenize"
ENGINE="$ROOT/engine/npu/build/npu_engine_qwen3_0_6b"
Q4NX="$MODEL_DIR/model.q4nx"
TJSON="$MODEL_DIR/tokenizer.json"
FLM="${FLM:-/opt/fastflowlm/bin/flm}"
export NPU_XCLBIN_DIR="${NPU_XCLBIN_DIR:-$ROOT/engine/npu/xclbins}"

OUT="${OUT:-/tmp/oracle_acc_0_6b.tsv}"
LOG="${LOG:-/tmp/oracle_acc_0_6b.log}"

# Text kept per arm for scoring, in characters (0 = unlimited).
#
# This was hard-coded to 90, so every arm was scored on the first 90 characters of
# its output. These models emit a <think> reasoning block before the answer, so a
# 90-char window usually contains only the assistant's shared preamble -- the same
# defect the model-generic harness had at 200 chars, and more aggressive. The note
# above about a previous NTOK truncation "inside the assistant's shared preamble"
# describes exactly this failure mode; the scorer reproduced it a second time.
TXT_CAP="${ORACLE_TEXT_CHARS:-4000}"

# ---- per-run scratch ------------------------------------------------------
# These paths were FIXED ($IDS, $RL_RAW, ...). This harness
# happens to fail LOUDLY on a tokenize failure -- it writes the ids with a shell
# `>` redirect, which truncates, so nids==0 and the row is reported TOKENIZE
# EMPTY -- but two concurrent runs still clobber each other's ids and raw files
# with no error at all. The sibling oracle_accuracy_model.sh had the worse form
# (ids written from inside python, so a failed tokenize left the PREVIOUS run's
# prompt in place and the engine was scored on it silently). Per-invocation
# scratch removes the shared-path class for both.
WORK="$(mktemp -d "${TMPDIR:-/tmp}/oa.XXXXXX")" || exit 2
trap 'rm -rf "$WORK"' EXIT
IDS="$WORK/ids.txt"; RL_RAW="$WORK/rl.raw"; D_RAW="$WORK/d.raw"; FLM_RAW="$WORK/flm.raw"
RL_IDS="$WORK/rl.ids"; D_IDS="$WORK/d.ids"

# Post-reasoning region: the text after the last </think>, else the whole text.
ans_region() {
    python3 -c 'import sys
t = sys.stdin.read()
i = t.rfind("</think")
if i >= 0:
    j = t.find(">", i)
    t = t[j + 1:] if j >= 0 else t[i + 8:]
sys.stdout.write(t)'
}

for f in "$TOK" "$DET" "$ENGINE" "$Q4NX" "$TJSON"; do
    [ -e "$f" ] || { echo "MISSING: $f" >&2; exit 2; }
done

printf 'prompt\texpected\tflm\trunlist\tdense\tflm_ok\trl_ok\tdense_ok\tids\trl_toks\tdense_toks\tflm_ans\trl_ans\tdense_ans\n' > "$OUT"

n=0; flm_ok=0; rl_ok=0; d_ok=0; flm_ans=0; rl_ans=0; d_ans=0; err=0

while IFS='|' read -r prompt expected; do
    [ -z "${prompt:-}" ] && continue
    [ -z "${expected:-}" ] && continue
    n=$((n+1))

    # ---- tokenize (stdin), assert non-empty -------------------------------
    # TPL=1 (default) wraps the prompt in the model's own Qwen chat template,
    # because FLM below is a chat application and applies that same template
    # internally. Without this the native arms are asked a RAW completion while
    # the oracle is asked an ASSISTANT question -- two different tasks. That
    # confound is why the native arm answered every prompt in exam format
    # ('A) ... B)'), with first-step top candidates A/Choices/Options/Which/Answer
    # (see benchmarks/RESULTS-oracle-accuracy-0_6b-2026-09-15.md). Set TPL=0 to
    # reproduce the raw-completion comparison instead.
    # NB: use `echo` semantics -- FLM receives the prompt via echo and therefore
    # sees a trailing newline; a newline-less prompt is not a fair comparison (it
    # flipped the runlist arm from "Paris" to " in the city").
    if [ "${TPL:-1}" = "1" ]; then
        printf '<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n' "$prompt" \
            | "$TOK" "$TJSON" 2>/dev/null | tr ',' ' ' > $IDS
    else
        echo "$prompt" | "$TOK" "$TJSON" 2>/dev/null | tr ',' ' ' > $IDS
    fi
    ids="$(tr -s ' ' < $IDS | sed 's/^ *//; s/ *$//')"
    nids=$(wc -w < $IDS)
    if [ "$nids" -eq 0 ]; then
        echo "[$n] TOKENIZE EMPTY for: $prompt" >&2
        printf '%s\t%s\t%s\t%s\t%s\tERR\tERR\tERR\t0\t0\t0\n' \
            "$prompt" "$expected" "" "" "" >> "$OUT"
        err=$((err+1)); continue
    fi

    # ---- arrow: runlist ----------------------------------------------------
    NPU_RUNLIST=1 NPU_GREEDY=1 timeout 240 "$ENGINE" "$Q4NX" "$NTOK" $IDS \
        >$RL_RAW 2>&1
    grep -aoE '^\s*\[[0-9]+\] [0-9]+' $RL_RAW | awk '{print $2}' > $RL_IDS
    rl_n=$(wc -w < $RL_IDS)
    if [ "$rl_n" -eq 0 ]; then
        rl_txt="<EXTRACTION EMPTY>"
    else
        rl_txt="$("$DET" "$TJSON" < $RL_IDS 2>/dev/null | tr '\n' ' ')"
        [ "${TXT_CAP:-0}" -gt 0 ] && rl_txt="${rl_txt:0:$TXT_CAP}"
    fi

    # ---- arm: dense --------------------------------------------------------
    NPU_RUNLIST=0 NPU_GREEDY=1 timeout 600 "$ENGINE" "$Q4NX" "$NTOK" $IDS \
        >$D_RAW 2>&1
    grep -aoE 'boot=[0-9]+|toks: [0-9]+' $D_RAW | grep -oE '[0-9]+' > $D_IDS
    d_n=$(wc -w < $D_IDS)
    if [ "$d_n" -eq 0 ]; then
        d_txt="<EXTRACTION EMPTY>"
    else
        d_txt="$("$DET" "$TJSON" < $D_IDS 2>/dev/null | tr '\n' ' ')"
        [ "${TXT_CAP:-0}" -gt 0 ] && d_txt="${d_txt:0:$TXT_CAP}"
    fi

    # ---- oracle: FLM -------------------------------------------------------
    echo "$prompt" | timeout 240 "$FLM" run qwen3:0.6b >$FLM_RAW 2>&1
    flm_txt="$(sed -n '/Model RAW Output/,$p' $FLM_RAW \
                 | sed '1d' | sed '/^>>>/,$d' \
                 | tr '\n' ' ' | sed 's/^ *//; s/ *$//')"
    [ "${TXT_CAP:-0}" -gt 0 ] && flm_txt="${flm_txt:0:$TXT_CAP}"
    [ -z "$flm_txt" ] && flm_txt="<EXTRACTION EMPTY>"

    # ---- score (substring, case-insensitive), never on empty extractions ----
    # Two verdicts, both over the FULL text now:
    #   *_ok  -- the expected string appears anywhere in the output (the metric
    #            this script already reported, which only ever saw 90 chars)
    #   *_ans -- the expected string appears OUTSIDE the reasoning block, i.e. in
    #            the text after the last </think>. Stricter, and the one a reader
    #            actually wants when asking whether the model answered.
    fk="-"; rk="-"; dk="-"; fa="-"; ra="-"; da="-"
    case "$flm_txt" in "<EXTRACTION EMPTY>") fk="ERR"; fa="ERR" ;; *)
        echo "$flm_txt" | grep -qiF "$expected" && { fk="Y"; flm_ok=$((flm_ok+1)); } || fk="N"
        echo "$flm_txt" | ans_region | grep -qiF "$expected" && { fa="Y"; flm_ans=$((flm_ans+1)); } || fa="N" ;;
    esac
    case "$rl_txt" in "<EXTRACTION EMPTY>") rk="ERR"; ra="ERR" ;; *)
        echo "$rl_txt" | grep -qiF "$expected" && { rk="Y"; rl_ok=$((rl_ok+1)); } || rk="N"
        echo "$rl_txt" | ans_region | grep -qiF "$expected" && { ra="Y"; rl_ans=$((rl_ans+1)); } || ra="N" ;;
    esac
    case "$d_txt" in "<EXTRACTION EMPTY>") dk="ERR"; da="ERR" ;; *)
        echo "$d_txt" | grep -qiF "$expected" && { dk="Y"; d_ok=$((d_ok+1)); } || dk="N"
        echo "$d_txt" | ans_region | grep -qiF "$expected" && { da="Y"; d_ans=$((d_ans+1)); } || da="N" ;;
    esac

    printf '[%2d] flm=%s/%s rl=%s/%s dense=%s/%s  %s | want=%s\n' "$n" "$fk" "$fa" "$rk" "$ra" "$dk" "$da" "$prompt" "$expected"
    printf '  flm    : %s\n  runlist: %s\n  dense  : %s\n' "$flm_txt" "$rl_txt" "$d_txt"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$prompt" "$expected" "$flm_txt" "$rl_txt" "$d_txt" "$fk" "$rk" "$dk" \
        "$nids" "$rl_n" "$d_n" "$fa" "$ra" "$da" >> "$OUT"
done < "$SET"

echo
echo "================= ORACLE ACCURACY: Qwen3-0.6B ================="
echo "prompts scored         : $n  (extraction errors: $err)"
echo "text kept per arm      : ${TXT_CAP} chars (ORACLE_TEXT_CHARS; 0 = unlimited)"
echo "FLM oracle self-check  : $flm_ok/$n   (should be $n/$n; if not, the scoring or oracle parse is wrong)"
echo "native runlist arm     : $rl_ok/$n"
echo "native dense arm       : $d_ok/$n"
echo "FLM    answer-region   : $flm_ans/$n   (expected string outside the <think> block)"
echo "native runlist answer-region: $rl_ans/$n"
echo "native dense   answer-region: $d_ans/$n"
echo "tsv: $OUT"
