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
NTOK="${NTOK:-6}"

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

for f in "$TOK" "$DET" "$ENGINE" "$Q4NX" "$TJSON"; do
    [ -e "$f" ] || { echo "MISSING: $f" >&2; exit 2; }
done

printf 'prompt\texpected\tflm\trunlist\tdense\tflm_ok\trl_ok\tdense_ok\tids\trl_toks\tdense_toks\n' > "$OUT"

n=0; flm_ok=0; rl_ok=0; d_ok=0; err=0

while IFS='|' read -r prompt expected; do
    [ -z "${prompt:-}" ] && continue
    [ -z "${expected:-}" ] && continue
    n=$((n+1))

    # ---- tokenize (stdin), assert non-empty -------------------------------
    # NB: use `echo`, not `printf '%s'`. FLM below receives the prompt via echo and
    # therefore sees a trailing newline; feeding the natives a newline-less prompt
    # makes this an unfair comparison (it flipped the runlist arm from "Paris" to
    # " in the city" on the France prompt). Same bytes on all three, or it is not a
    # comparison.
    echo "$prompt" | "$TOK" "$TJSON" 2>/dev/null | tr ',' ' ' > /tmp/oa_ids.txt
    ids="$(tr -s ' ' < /tmp/oa_ids.txt | sed 's/^ *//; s/ *$//')"
    nids=$(wc -w < /tmp/oa_ids.txt)
    if [ "$nids" -eq 0 ]; then
        echo "[$n] TOKENIZE EMPTY for: $prompt" >&2
        printf '%s\t%s\t%s\t%s\t%s\tERR\tERR\tERR\t0\t0\t0\n' \
            "$prompt" "$expected" "" "" "" >> "$OUT"
        err=$((err+1)); continue
    fi

    # ---- arrow: runlist ----------------------------------------------------
    NPU_RUNLIST=1 NPU_GREEDY=1 timeout 240 "$ENGINE" "$Q4NX" "$NTOK" /tmp/oa_ids.txt \
        >/tmp/oa_rl.raw 2>&1
    grep -aoE '^\s*\[[0-9]+\] [0-9]+' /tmp/oa_rl.raw | awk '{print $2}' > /tmp/oa_rl.ids
    rl_n=$(wc -w < /tmp/oa_rl.ids)
    if [ "$rl_n" -eq 0 ]; then
        rl_txt="<EXTRACTION EMPTY>"
    else
        rl_txt="$("$DET" "$TJSON" < /tmp/oa_rl.ids 2>/dev/null | tr '\n' ' ' | cut -c1-90)"
    fi

    # ---- arm: dense --------------------------------------------------------
    NPU_RUNLIST=0 NPU_GREEDY=1 timeout 600 "$ENGINE" "$Q4NX" "$NTOK" /tmp/oa_ids.txt \
        >/tmp/oa_d.raw 2>&1
    grep -aoE 'boot=[0-9]+|toks: [0-9]+' /tmp/oa_d.raw | grep -oE '[0-9]+' > /tmp/oa_d.ids
    d_n=$(wc -w < /tmp/oa_d.ids)
    if [ "$d_n" -eq 0 ]; then
        d_txt="<EXTRACTION EMPTY>"
    else
        d_txt="$("$DET" "$TJSON" < /tmp/oa_d.ids 2>/dev/null | tr '\n' ' ' | cut -c1-90)"
    fi

    # ---- oracle: FLM -------------------------------------------------------
    echo "$prompt" | timeout 240 "$FLM" run qwen3:0.6b >/tmp/oa_flm.raw 2>&1
    flm_txt="$(sed -n '/Model RAW Output/,$p' /tmp/oa_flm.raw \
                 | sed '1d' | sed '/^>>>/,$d' \
                 | tr '\n' ' ' | sed 's/^ *//; s/ *$//' | cut -c1-90)"
    [ -z "$flm_txt" ] && flm_txt="<EXTRACTION EMPTY>"

    # ---- score (substring, case-insensitive), never on empty extractions ----
    fk="-"; rk="-"; dk="-"
    case "$flm_txt" in "<EXTRACTION EMPTY>") fk="ERR" ;; *)
        echo "$flm_txt" | grep -qiF "$expected" && { fk="Y"; flm_ok=$((flm_ok+1)); } || fk="N" ;;
    esac
    case "$rl_txt" in "<EXTRACTION EMPTY>") rk="ERR" ;; *)
        echo "$rl_txt" | grep -qiF "$expected" && { rk="Y"; rl_ok=$((rl_ok+1)); } || rk="N" ;;
    esac
    case "$d_txt" in "<EXTRACTION EMPTY>") dk="ERR" ;; *)
        echo "$d_txt" | grep -qiF "$expected" && { dk="Y"; d_ok=$((d_ok+1)); } || dk="N" ;;
    esac

    printf '[%2d] flm=%s rl=%s dense=%s  %s | want=%s\n' "$n" "$fk" "$rk" "$dk" "$prompt" "$expected"
    printf '  flm    : %s\n  runlist: %s\n  dense  : %s\n' "$flm_txt" "$rl_txt" "$d_txt"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$prompt" "$expected" "$flm_txt" "$rl_txt" "$d_txt" "$fk" "$rk" "$dk" \
        "$nids" "$rl_n" "$d_n" >> "$OUT"
done < "$SET"

echo
echo "================= ORACLE ACCURACY: Qwen3-0.6B ================="
echo "prompts scored         : $n  (extraction errors: $err)"
echo "FLM oracle self-check  : $flm_ok/$n   (should be $n/$n; if not, the scoring or oracle parse is wrong)"
echo "native runlist arm     : $rl_ok/$n"
echo "native dense arm       : $d_ok/$n"
echo "tsv: $OUT"
