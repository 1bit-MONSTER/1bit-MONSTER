#!/bin/bash
# test_e2e_v27_tokens.sh — E2E validation for the v27 (multi-row, 32-core) NPU
# GEMM xclbins.
#
# Two checks per model:
#   1. v26-vs-v27 bit-identical token streams (the #1500 procedure, with
#      NPU_SEED pinning the RNG).  NECESSARY but NOT SUFFICIENT — both xclbin
#      generations share the same host forward, so a broken forward passes it
#      (this was exactly the qwen3_0_6b trap: "bit-identical vs v26" held for
#      garbage output).  See check 2.
#   2. Reference comparison: the engine's greedy/sampled output vs a
#      llama.cpp greedy run of the same prompt (Q8_0 GGUF).  This catches
#      forward-path regressions (e.g. the dense QKV pack bug fixed in
#      #1513: the pack applied the Qwen3.6 fused-q+gate layout to all dense
#      models, reading OOB q_proj rows and ignoring k_proj/v_proj).
#
# Requirements: NPU free (stop flm-whisper/zaya-npu services), model .q4nx
# files under ~/.config/flm/models/, engine binaries built (build_npu.sh),
# v26 xclbins extractable from git (eee6122b^), llama.cpp built in
# third_party/llama.cpp for the reference check.  Runs from the repo root.
#
# Usage:  engine/npu/tests/test_e2e_v27_tokens.sh [model_tag ...]
# Exit 0 = all PASS, 1 = any FAIL, 2 = skipped.

set -u
cd "$(git rev-parse --show-toplevel)" || exit 2
ENG=engine/npu/build/npu_engine
SEED=42
TOKENS=8
PROMPT=/tmp/npu_e2e_prompt.txt
printf "1 48077 4 5\n" > "$PROMPT"      # canonical ids of `"#$%&` (same tokens llama.cpp sees)
V26DIR=/tmp/v26_xclbins
V26_35B=/tmp/v26_xclbins_35b
V26_REF=eee6122b^

MODELS=(
  "qwen3_0_6b|Qwen3-0.6B-NPU2|qwen3_0_6b|Qwen3-0.6B||"
  "qwen3_8b|Qwen3-8B-NPU2|qwen3_8b||mem-blocked|"
  "qwen3_vl_4b|Qwen3-VL-4B-Instruct-NPU2|qwen3_vl_4b||mem-blocked|"
  "gemma4_e2b|Gemma4-E2B-IT-NPU2|gemma4_e2b||gated-hybrid arch unsupported|"
  "llama|Llama-3.1-8B-NPU2|llama|Llama-3.1-8B|mem-blocked|"
  "qwen3_6_35b_a3b|Qwen3.6-35B-A3B-NPU2|qwen3_6_moe_35b|||NPU_MOE=1"
)

# Fields: tag|q4nx dir|binary|gguf|skip reason|extra env

extract_v26() { # tag -> /tmp/v26_xclbins
  mkdir -p "$V26DIR"
  local tag="$1" f
  for f in $(git ls-tree --name-only "$V26_REF" -- engine/npu/xclbins/ | grep -E "final_i8_(QKV|O|G|U|GU|D)_${tag}\.xclbin|insts_i8_(QKV|O|G|U|GU|D)_${tag}\.txt"); do
    local b; b=$(basename "$f")
    [ -s "$V26DIR/$b" ] || git show "$V26_REF:engine/npu/xclbins/$b" > "$V26DIR/$b" 2>/dev/null
  done
}
build_v26_35b() {
  [ -d "$V26_35B" ] && return 0
  mkdir -p "$V26_35B"
  for f in engine/npu/xclbins/final_i8_{QKV,O}_qwen3_6_35b_a3b.xclbin \
           engine/npu/xclbins/insts_i8_{QKV,O}_qwen3_6_35b_a3b.txt \
           engine/npu/xclbins/final_i8_{GU,D}_qwen3_6_35b_a3b.xclbin \
           engine/npu/xclbins/insts_i8_{GU,D}_qwen3_6_35b_a3b.txt; do
    cp -f "$f" "$V26_35B/" 2>/dev/null
  done
  for f in engine/npu/xclbins/v26_backup/final_i8_MOE_{GU,D,SGU,SD}_qwen3_6_35b_a3b.xclbin \
           engine/npu/xclbins/v26_backup/insts_i8_MOE_{GU,D,SGU,SD}_qwen3_6_35b_a3b.txt; do
    cp -f "$f" "$V26_35B/" 2>/dev/null
  done
}
stream() { grep -E "^  \[[0-9]+\] boot=|^  \[[0-9]+\] batch=" "$1" | grep -Eo "boot=[0-9]+|tok=[0-9]+" | sed 's/boot=//;s/tok=//'; }
run_once() { # xclbin_dir q4nx bin extra tokens out
  env NPU_SEED=$SEED NPU_XCLBIN_DIR="$1" $4 timeout 1200 \
      "$3" "$2" "$5" "$PROMPT" > "$6" 2> "$6.err"
}

fails=0; skipped=0; total=0
declare -A R
for entry in "${MODELS[@]}"; do
  IFS='|' read -r tag dir bin gguf note extra <<< "$entry"
  q4nx="$HOME/.config/flm/models/$dir/model.q4nx"
  [ -s "$q4nx" ] || { echo "SKIP  $tag: missing $q4nx"; skipped=$((skipped+1)); continue; }
  [ -x "$ENG"_"$bin" ] || { echo "SKIP  $tag: binary not built"; skipped=$((skipped+1)); continue; }
  if [ -n "$note" ]; then
    echo "SKIP  $tag: $note (cannot e2e on current NPU stack)"
    R[$tag]=SKIP; skipped=$((skipped+1)); continue
  fi
  if [ "$tag" = "qwen3_6_35b_a3b" ]; then build_v26_35b; v26d="$V26_35B"; extra="NPU_MOE=1";
  else extract_v26 "$tag"; v26d="$V26DIR"; extra=""; fi
  [ -z "$(ls engine/npu/xclbins/final_i8_*_${tag}.xclbin 2>/dev/null)" ] && \
    [ -z "$(ls engine/npu/xclbins/final_i8_*_qwen3_6_35b_a3b.xclbin 2>/dev/null)" ] && \
    { echo "SKIP  $tag: no v27 xclbins"; skipped=$((skipped+1)); continue; }

  total=$((total+1)); echo "==== $tag ===="
  run_once "engine/npu/xclbins" "$q4nx" "$ENG"_"$bin" "$extra" "$TOKENS" /tmp/e2e_${tag}_v27.log
  v27rc=$?
  run_once "$v26d" "$q4nx" "$ENG"_"$bin" "$extra" "$TOKENS" /tmp/e2e_${tag}_v26.log
  v26rc=$?
  if [ $v27rc -ne 0 ] || [ $v26rc -ne 0 ]; then
    echo "FAIL  $tag: engine exit v27=$v27rc v26=$v26rc"; fails=$((fails+1)); R[$tag]=FAIL; continue
  fi
  s27=$(stream /tmp/e2e_${tag}_v27.log); s26=$(stream /tmp/e2e_${tag}_v26.log)
  if [ "$s27" = "$s26" ] && [ -n "$s27" ]; then
    echo "  v26-vs-v27: PASS (bit-identical tokens)"
  else
    echo "  v26-vs-v27: FAIL"; fails=$((fails+1)); R[$tag]=FAIL; continue
  fi
  # Reference check (dense models): llama.cpp greedy on the same prompt
  if [ -n "$gguf" ] && [ -x third_party/llama.cpp/build/bin/llama-completion ]; then
    gg=$(find /home/bcloud/models -iname "*q8_0*.gguf" 2>/dev/null | grep -i "$gguf" | head -1)
    if [ -n "$gg" ]; then
      timeout 300 third_party/llama.cpp/build/bin/llama-completion -m "$gg" -p '"#$%&' \
        -n $TOKENS --temp 0 --top-k 1 -no-cnv --single-turn 2>/dev/null | grep -v '^[[:space:]]*$' | tail -1 > /tmp/e2e_${tag}_ref.txt
      # engine: decode ids -> text (best-effort with the q4nx tokenizer)
      python3 - "$tag" <<'PYEOF' > /tmp/e2e_${tag}_decoded.txt 2>/dev/null
import sys, re
from tokenizers import Tokenizer
tag = sys.argv[1]
tok = Tokenizer.from_file(f"{__import__('os').path.expanduser('~/.config/flm/models')}/Qwen3-0.6B-NPU2/tokenizer.json")
toks = [int(x) for x in re.findall(r'(?:boot|tok)=(\d+)', open(f"/tmp/e2e_{tag}_v27.log").read())]
print(tok.decode(toks))
PYEOF
      ref=$(cat /tmp/e2e_${tag}_ref.txt); got=$(cat /tmp/e2e_${tag}_decoded.txt)
      if [ -n "$ref" ] && [ -n "$got" ] && { [[ "$ref" == *"$got"* ]] || [[ "$got" == *"$ref"* ]]; }; then
        echo "  reference: PASS  (engine='$got' llama.cpp='$ref')"
        R[$tag]=PASS
      else
        echo "  reference: FAIL  (engine='$got' llama.cpp='$ref')"
        fails=$((fails+1)); R[$tag]=FAIL
      fi
    else
      echo "  reference: SKIP (no GGUF found)"; R[$tag]=PASS
    fi
  else
    R[$tag]=PASS
  fi
done

echo ""
echo "════════ SUMMARY ════════"
for entry in "${MODELS[@]}"; do
  IFS='|' read -r tag _ <<< "$entry"
  printf "  %-16s %s\n" "$tag" "${R[$tag]:-SKIP}"
done
echo "total=$total fail=$fails skipped=$skipped"
[ $fails -eq 0 ]
