#!/usr/bin/env bash
# oracle_accuracy_model.sh — score one native NPU engine model arm against the
# FLM oracle on a fixed prompt set.
#
# Generalisation of oracle_accuracy_0_6b.sh (goal mu34scbf-4ffm1o) to the models
# that harness could not reach. Same method rules:
#   * the model's OWN chat template is applied for the native arm, so native and
#     FLM are asked the same question (FLM applies the same template internally)
#   * native decode is forced greedy (NPU_GREEDY=1)
#   * every extraction is asserted NON-EMPTY before scoring
#   * scoring is whole-output, case-insensitive substring match
#
# I1 note: the engine's `tokenizer/tokenize` tool is NOT used here. It prints
# Qwen special-token ids for a Llama vocab, so native ids are built with HF
# `transformers`/`tokenizers` from the model's own tokenizer.json instead.
#
# Usage:
#   oracle_accuracy_model.sh <Model-NPU2> <npu_engine_name> <flm_tag> [ntok] [prompt_set] [out_tsv]
# Env:
#   ENGINE_ENV   extra env for the native invocation, e.g.
#                "NPU_LAYER_ELF_DIR=/tmp/llama-elfs NPU_RUNLIST=1"
set -uo pipefail

MODEL="${1:?usage: oracle_accuracy_model.sh <Model-NPU2> <engine> <flm_tag> [ntok] [set] [out]}"
ENG="${2:?}"
FLM_TAG="${3:?}"
NTOK="${4:-128}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SET="${5:-$ROOT/benchmarks/prompts/qwen3_0_6b_oracle_set.txt}"
OUT="${6:-/tmp/oracle_acc_${MODEL}.tsv}"

D="$HOME/.config/flm/models/$MODEL"
ENGINE="$ROOT/engine/npu/build/$ENG"
FLM="${FLM:-/opt/fastflowlm/bin/flm}"
export NPU_XCLBIN_DIR="${NPU_XCLBIN_DIR:-$ROOT/engine/npu/xclbins}"
ENGINE_ENV="${ENGINE_ENV:-NPU_RUNLIST=1}"

# ---- LAYER_XCLBIN: pin it to this repo's copy ------------------------------
# The runlist arm resolves layer.xclbin from
# /home/bcloud/amd-oss/fastflowlm/src/xclbins/<model>/layer.xclbin, a tree other
# lanes actively rebuild. On 2026-09-18 08:19 that Qwen3-0.6B file was replaced
# (401980 B, md5 fa9f8df2) and every runlist measurement became garbage -- the
# per-ctx ELFs are built against the pinned 339980 B / md5 57431faa file. All
# 20 oracle rows dropped to N (0/20) while FLM still scored 18/20.
# Pinning makes the arm reproducible; set LAYER_XCLBIN yourself to override.
if [ -z "${LAYER_XCLBIN:-}" ] && [ -f "$ROOT/engine/npu/xclbins/flm_models/$MODEL/layer.xclbin" ]; then
    export LAYER_XCLBIN="$ROOT/engine/npu/xclbins/flm_models/$MODEL/layer.xclbin"
    echo "[layer.xclbin] pinned: $LAYER_XCLBIN" >&2
fi

# ---- per-run scratch ------------------------------------------------------
# These paths used to be FIXED (/tmp/oam_ids.txt, /tmp/oam_nat.raw, ...). Two
# consequences, both silent, both measured on 2026-09-16:
#
#  1. A failed tokenize left the PREVIOUS run's ids file in place, and the guard
#     below tested only NON-EMPTY -- so the engine was scored on a stale prompt
#     from an earlier run and the row was recorded as if it were valid.
#     Reproduced: at 11:52 a run asked "2 + 2 =" and "The capital of Italy is",
#     while /tmp/oam_ids.txt still held "The capital of Germany is" from 11:15.
#     Every row came back with the SAME answer, about Germany. The engine was
#     fine (given unique files it answers each prompt differently, diverging at
#     token 5); the harness was feeding it one prompt over and over.
#
#  2. Two concurrent runs -- two models, or another agent on this box -- clobber
#     each other's ids and raw files with no error at all.
#
# A per-invocation directory removes (2), and removing the ids file before the
# tokenize makes (1) impossible to pass silently: a failed tokenize now reports
# TOKENIZE EMPTY instead of falling back to a previous prompt.
WORK="$(mktemp -d "${TMPDIR:-/tmp}/oam.XXXXXX")" || exit 2
trap 'rm -rf "$WORK"' EXIT
IDS="$WORK/ids.txt"; TPL="$WORK/tpl.txt"; NAT_RAW="$WORK/nat.raw"; FLM_RAW="$WORK/flm.raw"

# Text kept per arm for scoring, in characters (0 = unlimited).
#
# This was hard-coded to 200, which truncated EVERY row mid-<think> on the models
# that emit a reasoning block before the answer (measured on the 2026-09-16 runs:
# 20/20 rows for Qwen3-4B, 20/20 native and 19/20 FLM for Qwen3-8B). The score
# was therefore not "did the model answer" but "did it name the answer within its
# first 200 characters" -- a verbosity measurement wearing accuracy's name.
# Observed directly: Qwen3-4B answers "The capital of Germany is" with a <think>
# block whose first 200 chars contain no city, and the string "Berlin is the
# capital of Germany" at ~char 250 -- scored wrong while being right.
TXT_CAP="${ORACLE_TEXT_CHARS:-4000}"

# Post-reasoning region: the text after the last </think>, else the whole text.
# Scoring that separately is the stricter "did it actually answer" signal the
# 200-char cap had been silently proxying for.
ans_region() {
    python3 -c 'import sys
t = sys.stdin.read()
i = t.rfind("</think")
if i >= 0:
    j = t.find(">", i)
    t = t[j + 1:] if j >= 0 else t[i + 8:]
sys.stdout.write(t)'
}

for f in "$ENGINE" "$D/model.q4nx" "$D/tokenizer.json"; do
    [ -e "$f" ] || { echo "MISSING: $f" >&2; exit 2; }
done

printf 'prompt\texpected\tflm\tnative\tflm_ok\tnative_ok\tnids\tnative_toks\tflm_ans\tnative_ans\tflm_nids\ti1\n' > "$OUT"

n=0; flm_ok=0; nat_ok=0; flm_ans=0; nat_ans=0; err=0; i1_ok=0; i1_bad=0
while IFS='|' read -r prompt expected; do
    [ -z "${prompt:-}" ] && continue
    [ -z "${expected:-}" ] && continue
    n=$((n+1))

    # ---- native ids from the model's own chat template --------------------
    # Remove first: the guard below must test "was THIS prompt tokenized", not
    # "is there a file". See the WORK note at the top.
    rm -f "$IDS"
    python3 - "$D" "$prompt" "$IDS" >"$TPL" 2>&1 <<'PY'
import sys
from transformers import AutoTokenizer
d, prompt, out = sys.argv[1], sys.argv[2], sys.argv[3]
tok = AutoTokenizer.from_pretrained(d)
try:
    # Some templates (Qwen3-4B and 1.7B among them) reference optional variables
    # such as user_system_prompt; supply them so the template renders instead of
    # raising UndefinedError.
    text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True,
                                   user_system_prompt="", tools=None)
except Exception as e:
    # Last resort: the plain ChatML form every one of these models accepts.
    print("TEMPLATE_FALLBACK", type(e).__name__, e, file=sys.stderr)
    text = "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n"
ids = tok(text, add_special_tokens=False)["input_ids"]
open(out, "w").write(" ".join(map(str, ids)))
print("TPL", repr(text))
print("NIDS", len(ids))
PY
    if [ ! -s "$IDS" ]; then
        echo "[$n] TOKENIZE EMPTY for: $prompt"; cat "$TPL" >&2
        printf '%s\t%s\t%s\t%s\tERR\tERR\t0\t0\tERR\tERR\t-\tERR\n' "$prompt" "$expected" "" "" >> "$OUT"
        err=$((err+1)); continue
    fi
    nids=$(wc -w < "$IDS")

    # ---- native arm -------------------------------------------------------
    env $ENGINE_ENV NPU_GREEDY=1 timeout 900 "$ENGINE" "$D/model.q4nx" "$NTOK" "$IDS" \
        >"$NAT_RAW" 2>&1
    nat_txt="$(python3 - "$D" "$NAT_RAW" "$TXT_CAP" <<'PY'
import re, sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1])
raw = open(sys.argv[2], "r", errors="replace").read()
cap = int(sys.argv[3])
boot = [int(m.group(1)) for m in re.finditer(r"\[\d+\]\s+boot=(\d+)", raw)]
ids = boot
ids += [int(m.group(1)) for m in re.finditer(r"^\s*\[\d+\]\s+(\d+)\s*$", raw, re.M)]
ids += [int(m.group(1)) for m in re.finditer(r"\[\d+\] batch=\d+ toks:\s*(\d+)", raw)]
txt = tok.decode(ids, skip_special_tokens=True).replace("\n", " ").strip()
print(txt if cap <= 0 else txt[:cap])
PY
)"
    nat_n=$(python3 -c "import re,sys;r=open(sys.argv[1],errors='replace').read();print(len(re.findall(r'\[\d+\]\s+boot=\d+',r))+len(re.findall(r'^\s*\[\d+\]\s+\d+\s*\$',r,re.M))+len(re.findall(r'\[\d+\] batch=\d+ toks:',r)))" "$NAT_RAW")
    [ -z "$nat_txt" ] && nat_txt="<EXTRACTION EMPTY>"

    # ---- oracle: FLM ------------------------------------------------------
    echo "$prompt" | timeout 600 "$FLM" run "$FLM_TAG" >"$FLM_RAW" 2>&1
    flm_txt="$(sed -n '/Model RAW Output/,$p' "$FLM_RAW" \
                 | sed '1d' | sed '/^>>>/,$d' \
                 | tr '\n' ' ' | sed 's/^ *//; s/ *$//')"
    [ "${TXT_CAP:-0}" -gt 0 ] && flm_txt="${flm_txt:0:$TXT_CAP}"
    [ -z "$flm_txt" ] && flm_txt="<EXTRACTION EMPTY>"

    # ---- I1: same bytes, ASSERTED -----------------------------------------
    # The native arm consumes the model's own chat template rendered from
    # tokenizer.json; FLM applies the same template internally. Assert the two
    # agree by comparing FLM's reported prefill token count with the native nids
    # rather than assuming it. UNPARSED is reported honestly (some FLM builds do
    # not print the count) and is NOT counted as a pass.
    flm_nids="$(grep -aoE 'with [0-9]+ tokens' "$FLM_RAW" | grep -oE '[0-9]+' | head -1)"
    if [ -z "$flm_nids" ]; then i1="UNPARSED"
    elif [ "$flm_nids" = "$nids" ]; then i1="OK"; i1_ok=$((i1_ok+1))
    else i1="MISMATCH"; i1_bad=$((i1_bad+1)); fi
    printf '[%2d] I1 %s  flm_nids=%s native_nids=%s\n' "$n" "$i1" "${flm_nids:-?}" "$nids"

    # ---- score ------------------------------------------------------------
    # Two verdicts, both over the FULL text now:
    #   *_ok  -- the expected string appears anywhere in the output (the series'
    #            existing metric, which previously only ever saw the first 200 chars)
    #   *_ans -- the expected string appears OUTSIDE the reasoning block, i.e. in
    #            the text after the last </think>. Stricter, and the one a reader
    #            actually wants when asking whether the model answered the question.
    fk="-"; nk="-"; fa="-"; na="-"
    case "$flm_txt" in "<EXTRACTION EMPTY>") fk="ERR"; fa="ERR" ;; *)
        echo "$flm_txt" | grep -qiF "$expected" && { fk="Y"; flm_ok=$((flm_ok+1)); } || fk="N"
        echo "$flm_txt" | ans_region | grep -qiF "$expected" && { fa="Y"; flm_ans=$((flm_ans+1)); } || fa="N" ;;
    esac
    case "$nat_txt" in "<EXTRACTION EMPTY>") nk="ERR"; na="ERR" ;; *)
        echo "$nat_txt" | grep -qiF "$expected" && { nk="Y"; nat_ok=$((nat_ok+1)); } || nk="N"
        echo "$nat_txt" | ans_region | grep -qiF "$expected" && { na="Y"; nat_ans=$((nat_ans+1)); } || na="N" ;;
    esac

    printf '[%2d] flm=%s/%s native=%s/%s  %s | want=%s\n' "$n" "$fk" "$fa" "$nk" "$na" "$prompt" "$expected"
    printf '  flm    : %s\n  native : %s\n' "$flm_txt" "$nat_txt"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$prompt" "$expected" "$flm_txt" "$nat_txt" "$fk" "$nk" "$nids" "$nat_n" "$fa" "$na" "${flm_nids:-}" "$i1" >> "$OUT"
done < "$SET"

echo
echo "========= ORACLE ACCURACY: $MODEL (ntok=$NTOK) ========="
echo "prompts scored        : $n  (extraction errors: $err)"
echo "I1 same-bytes assertion: OK=$i1_ok  MISMATCH=$i1_bad  (a MISMATCH voids the row)"
echo "text kept per arm     : ${TXT_CAP} chars (ORACLE_TEXT_CHARS; 0 = unlimited)"
echo "FLM oracle self-check : $flm_ok/$n"
echo "native arm ($ENG)     : $nat_ok/$n"
echo "FLM    answer-region  : $flm_ans/$n   (expected string outside the <think> block)"
echo "native answer-region  : $nat_ans/$n"
echo "tsv: $OUT"
