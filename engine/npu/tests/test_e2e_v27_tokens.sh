#!/bin/bash
# test_e2e_v27_tokens.sh — E2E token validation for the v27 (multi-row, 32-core)
# NPU GEMM xclbins against the v26 (single-row) originals.
#
# For each tracked model: load the same .q4nx through the engine twice — once
# with the repo's v27 xclbin set, once with the v26 set from git history
# (eee6122b^) — with NPU_SEED pinned so the RNG stream is identical, and
# require a bit-identical sampled token stream. This is the same procedure
# that validated qwen3_0_6b in #1500; it catches header-dims↔xclbin
# disagreement (wrong K/N wiring → garbage logits → divergent tokens) and any
# numeric regression introduced by the v27 rebuild.
#
# Requirements: NPU free (stop flm-whisper/zaya-npu services first), model
# .q4nx files under ~/.config/flm/models/, engine binaries built
# (engine/npu/build_npu.sh). Runs from the repo root.
#
# Usage:  engine/npu/tests/test_e2e_v27_tokens.sh [model_tag ...]
# Exit 0 = all PASS, 1 = any FAIL, 2 = skipped (missing model/xclbin).

set -u
cd "$(git rev-parse --show-toplevel)" || exit 2
ENG=engine/npu/build/npu_engine
SEED=42
TOKENS_DENSE=12
TOKENS_MOE=8
PROMPT=/tmp/npu_e2e_prompt.txt
printf "1 2 3 4 5\n" > "$PROMPT"
V26DIR=/tmp/v26_xclbins
V26_35B=/tmp/v26_xclbins_35b
V26_REF=eee6122b^           # commit whose engine/npu/xclbins held the v26 sets

mkdir -p "$V26DIR"

# Models: tag|q4nx dir under ~/.config/flm/models|extra env|decode tokens
MODELS=(
  "qwen3_0_6b|Qwen3-0.6B-NPU2||$TOKENS_DENSE"
  "qwen3_8b|Qwen3-8B-NPU2||$TOKENS_DENSE"
  "qwen3_vl_4b|Qwen3-VL-4B-Instruct-NPU2||$TOKENS_DENSE"
  "gemma4_e2b|Gemma4-E2B-IT-NPU2||$TOKENS_DENSE"
  "llama|Llama-3.1-8B-NPU2|NPU_MODEL_TAG=llama|$TOKENS_DENSE"
  "qwen3_6_35b_a3b|Qwen3.6-35B-A3B-NPU2|NPU_MOE=1|$TOKENS_MOE"
)

# Extract one model's v26 xclbin+insts pair from git history.
extract_v26() { # tag
  local tag="$1" f
  for f in $(git ls-tree --name-only "$V26_REF" -- engine/npu/xclbins/ | grep -E "final_i8_(QKV|O|G|U|GU|D)_${tag}\.xclbin|insts_i8_(QKV|O|G|U|GU|D)_${tag}\.txt"); do
    local b; b=$(basename "$f")
    [ -s "$V26DIR/$b" ] || git show "$V26_REF:engine/npu/xclbins/$b" > "$V26DIR/$b" 2>/dev/null
  done
}

# 35B: v26 MOE ops come from the committed v26_backup; QKV/O are unchanged
# between the v26 and v27 eras, so the comparison isolates the MOE rebuild.
build_v26_35b() {
  [ -d "$V26_35B" ] && return 0
  mkdir -p "$V26_35B"
  local f
  for f in engine/npu/xclbins/final_i8_{QKV,O}_qwen3_6_35b_a3b.xclbin \
           engine/npu/xclbins/insts_i8_{QKV,O}_qwen3_6_35b_a3b.txt; do
    cp -f "$f" "$V26_35B/" 2>/dev/null
  done
  for f in engine/npu/xclbins/v26_backup/final_i8_MOE_{GU,D,SGU,SD}_qwen3_6_35b_a3b.xclbin \
           engine/npu/xclbins/v26_backup/insts_i8_MOE_{GU,D,SGU,SD}_qwen3_6_35b_a3b.txt; do
    cp -f "$f" "$V26_35B/" 2>/dev/null
  done
}

run_once() { # xclbin_dir model_q4nx extra_env tag tokens out_log
  local xd="$1" q4nx="$2" extra="$3" tag="$4" tok="$5" out="$6"
  env NPU_SEED=$SEED NPU_XCLBIN_DIR="$xd" $extra timeout 900 \
      "$ENG"_"$tag" "$q4nx" "$tok" "$PROMPT" > "$out" 2> "$out.err"
}

stream() { # log → boot + decode token ids, one per line
  grep -Eo "boot=[0-9]+" "$1" | sed 's/boot=//'
  grep -Eo "tok=[0-9]+" "$1" | sed 's/tok=//'
}

fails=0; skipped=0; total=0
declare -A RESULTS
for entry in "${MODELS[@]}"; do
  IFS='|' read -r tag dir extra tok <<< "$entry"
  q4nx="$HOME/.config/flm/models/$dir/model.q4nx"
  [ -s "$q4nx" ] || { echo "SKIP  $tag: missing $q4nx"; skipped=$((skipped+1)); continue; }
  [ -x "$ENG"_"$tag" ] || { echo "SKIP  $tag: binary ${ENG}_${tag} not built"; skipped=$((skipped+1)); continue; }
  if [ "$tag" = "qwen3_6_35b_a3b" ]; then build_v26_35b; v26d="$V26_35B";
  else extract_v26 "$tag"; v26d="$V26DIR"; fi
  # both xclbin sets present?
  local_missing=$(ls engine/npu/xclbins/final_i8_*_${tag}.xclbin 2>/dev/null | wc -l)
  v26_missing=$(ls "$v26d"/final_i8_*_${tag}.xclbin 2>/dev/null | wc -l)
  if [ "$local_missing" -eq 0 ] || [ "$v26_missing" -eq 0 ]; then
    echo "SKIP  $tag: xclbin set incomplete (v27=$local_missing v26=$v26_missing)"; skipped=$((skipped+1)); continue
  fi

  total=$((total+1))
  echo "==== $tag: v27 xclbins (repo) vs v26 xclbins ($v26d) ===="
  run_once "engine/npu/xclbins" "$q4nx" "$extra" "$tag" "$tok" /tmp/e2e_${tag}_v27.log
  v27rc=$?
  run_once "$v26d" "$q4nx" "$extra" "$tag" "$tok" /tmp/e2e_${tag}_v26.log
  v26rc=$?
  if [ $v27rc -ne 0 ] || [ $v26rc -ne 0 ]; then
    echo "FAIL  $tag: engine exit v27=$v27rc v26=$v26rc"
    grep -E "ERR|FAIL" /tmp/e2e_${tag}_v27.log.err /tmp/e2e_${tag}_v26.log.err | head -4
    fails=$((fails+1)); RESULTS[$tag]=FAIL; continue
  fi
  v27s=$(stream /tmp/e2e_${tag}_v27.log); v26s=$(stream /tmp/e2e_${tag}_v26.log)
  if [ "$v27s" = "$v26s" ] && [ -n "$v27s" ]; then
    echo "PASS  $tag: tokens $(echo "$v27s" | tr '\n' ' ')"
    RESULTS[$tag]=PASS
  else
    echo "FAIL  $tag: token stream mismatch"
    echo "  v27: $(echo "$v27s" | tr '\n' ' ')"
    echo "  v26: $(echo "$v26s" | tr '\n' ' ')"
    fails=$((fails+1)); RESULTS[$tag]=FAIL
  fi
done

echo ""
echo "════════ SUMMARY ════════"
for entry in "${MODELS[@]}"; do
  IFS='|' read -r tag _ <<< "$entry"
  printf "  %-16s %s\n" "$tag" "${RESULTS[$tag]:-SKIP}"
done
echo "total=$total pass=$((total-fails)) fail=$fails skipped=$skipped"
[ $fails -eq 0 ]
