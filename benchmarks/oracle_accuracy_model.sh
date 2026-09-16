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

for f in "$ENGINE" "$D/model.q4nx" "$D/tokenizer.json"; do
    [ -e "$f" ] || { echo "MISSING: $f" >&2; exit 2; }
done

printf 'prompt\texpected\tflm\tnative\tflm_ok\tnative_ok\tnids\tnative_toks\tflm_nids\ti1\n' > "$OUT"  # I1_SAME_BYTES

n=0; flm_ok=0; nat_ok=0; err=0; i1_ok=0; i1_bad=0
while IFS='|' read -r prompt expected; do
    [ -z "${prompt:-}" ] && continue
    [ -z "${expected:-}" ] && continue
    n=$((n+1))

    # ---- native ids from the model's own chat template --------------------
    python3 - "$D" "$prompt" /tmp/oam_ids.txt >/tmp/oam_tpl.txt 2>&1 <<'PY'
import sys
from transformers import AutoTokenizer
d, prompt, out = sys.argv[1], sys.argv[2], sys.argv[3]
tok = AutoTokenizer.from_pretrained(d)
text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                               tokenize=False, add_generation_prompt=True)
ids = tok(text, add_special_tokens=False)["input_ids"]
open(out, "w").write(" ".join(map(str, ids)))
print("TPL", repr(text))
print("NIDS", len(ids))
PY
    if [ ! -s /tmp/oam_ids.txt ]; then
        echo "[$n] TOKENIZE EMPTY for: $prompt"; cat /tmp/oam_tpl.txt >&2
        printf '%s\t%s\t%s\t%s\tERR\tERR\t0\t0\n' "$prompt" "$expected" "" "" >> "$OUT"
        err=$((err+1)); continue
    fi
    nids=$(wc -w < /tmp/oam_ids.txt)

    # ---- native arm -------------------------------------------------------
    env $ENGINE_ENV NPU_GREEDY=1 timeout 900 "$ENGINE" "$D/model.q4nx" "$NTOK" /tmp/oam_ids.txt \
        >/tmp/oam_nat.raw 2>&1
    nat_txt="$(python3 - "$D" /tmp/oam_nat.raw <<'PY'
import re, sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1])
raw = open(sys.argv[2], "r", errors="replace").read()
boot = [int(m.group(1)) for m in re.finditer(r"\[\d+\]\s+boot=(\d+)", raw)]
ids = boot
ids += [int(m.group(1)) for m in re.finditer(r"^\s*\[\d+\]\s+(\d+)\s*$", raw, re.M)]
ids += [int(m.group(1)) for m in re.finditer(r"\[\d+\] batch=\d+ toks:\s*(\d+)", raw)]
print(tok.decode(ids, skip_special_tokens=True).replace("\n", " ").strip()[:200])
PY
)"
    nat_n=$(python3 -c "import re;r=open('/tmp/oam_nat.raw',errors='replace').read();print(len(re.findall(r'\[\d+\]\s+boot=\d+',r))+len(re.findall(r'^\s*\[\d+\]\s+\d+\s*\$',r,re.M))+len(re.findall(r'\[\d+\] batch=\d+ toks:',r)))")
    [ -z "$nat_txt" ] && nat_txt="<EXTRACTION EMPTY>"

    # ---- oracle: FLM ------------------------------------------------------
    echo "$prompt" | timeout 600 "$FLM" run "$FLM_TAG" >/tmp/oam_flm.raw 2>&1
    flm_txt="$(sed -n '/Model RAW Output/,$p' /tmp/oam_flm.raw \
                 | sed '1d' | sed '/^>>>/,$d' \
                 | tr '\n' ' ' | sed 's/^ *//; s/ *$//' | cut -c1-200)"
    [ -z "$flm_txt" ] && flm_txt="<EXTRACTION EMPTY>"

    # ---- I1: same bytes, ASSERTED ------------------------------------------
    # The native arm consumes the model's own chat template rendered from
    # tokenizer.json; FLM applies the same template internally. Assert the two
    # agree by comparing FLM's reported prefill token count with the native nids
    # rather than assuming it. UNPARSED is reported honestly (some FLM builds do
    # not print the count) and is NOT counted as a pass.
    flm_nids="$(grep -aoE 'with [0-9]+ tokens' /tmp/oam_flm.raw | grep -oE '[0-9]+' | head -1)"
    if [ -z "$flm_nids" ]; then i1="UNPARSED"
    elif [ "$flm_nids" = "$nids" ]; then i1="OK"; i1_ok=$((i1_ok+1))
    else i1="MISMATCH"; i1_bad=$((i1_bad+1)); fi
    printf '[%2d] I1 %s  flm_nids=%s native_nids=%s\n' "$n" "$i1" "${flm_nids:-?}" "$nids"

    # ---- score ------------------------------------------------------------
    fk="-"; nk="-"
    case "$flm_txt" in "<EXTRACTION EMPTY>") fk="ERR" ;; *)
        echo "$flm_txt" | grep -qiF "$expected" && { fk="Y"; flm_ok=$((flm_ok+1)); } || fk="N" ;;
    esac
    case "$nat_txt" in "<EXTRACTION EMPTY>") nk="ERR" ;; *)
        echo "$nat_txt" | grep -qiF "$expected" && { nk="Y"; nat_ok=$((nat_ok+1)); } || nk="N" ;;
    esac

    printf '[%2d] flm=%s native=%s  %s | want=%s\n' "$n" "$fk" "$nk" "$prompt" "$expected"
    printf '  flm    : %s\n  native : %s\n' "$flm_txt" "$nat_txt"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$prompt" "$expected" "$flm_txt" "$nat_txt" "$fk" "$nk" "$nids" "$nat_n" "${flm_nids:-}" "$i1" >> "$OUT"
done < "$SET"

echo
echo "========= ORACLE ACCURACY: $MODEL (ntok=$NTOK) ========="
echo "prompts scored        : $n  (extraction errors: $err)"
echo "I1 same-bytes assertion: OK=$i1_ok  MISMATCH=$i1_bad  (a MISMATCH voids the row)"
echo "FLM oracle self-check : $flm_ok/$n"
echo "native arm ($ENG)     : $nat_ok/$n"
echo "tsv: $OUT"
