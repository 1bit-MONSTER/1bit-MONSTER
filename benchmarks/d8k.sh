#!/usr/bin/env bash
# d8k.sh - decode rate at the TOP of the supported window (decode context up to 8192).
#
# Why this exists: the per-ctx ELFs bake MAX_L=8192, so a prompt of 8192 tokens with ng>=2
# needs ctx=8194 and fails. But a prompt of 8192-ng leaves every decode forward at ctx<=8192,
# so the decode clause of criterion (c) IS measurable at the top of the range:
#   prompt 8160 tokens + ng=32 -> forwards at ctx 8161..8192, all inside the window.
# Use a warm-up-free decode count (>=32): at 8 tokens the native arm read 31.5 tok/s against
# FLM's 33.83 (0.93x) and at 32 tokens 35.5 against 33.86 (1.05x) - the same transient the 1k
# rows had (see RESULTS-decode-window-warmup).
#
# usage: d8k.sh <Model-NPU2> <flm_tag> <engine> <abmodel> [runs] [extra native env ...]
set -u
M="${1:?usage: d8k.sh <Model-NPU2> <flm_tag> <engine> <abmodel> [runs] [env ...]}"
TAG="${2:?}"; ENG="${3:?}"; ABM="${4:?}"; RUNS="${5:-2}"; shift 5 || true
D="$HOME/.config/flm/models/$M"
E="./engine/npu/build/$ENG"
P=/tmp/p_8160.txt
[ -f "$P" ] || { echo "missing $P" >&2; exit 2; }
echo "== $M : decode at ~8.2k ctx (prompt 8160 + ng 32), $RUNS pairs =="
for i in $(seq 1 "$RUNS"); do
    pre="$(for p in /proc/[0-9]*; do
               ls -l "$p/fd" 2>/dev/null | grep -q accel0 || continue
               pid="${p#/proc/}"
               cmd="$(tr '\0' ' ' < "$p/cmdline" 2>/dev/null | cut -c1-60)"
               case "$cmd" in *"qwen3.6-moe:35b-a3b"*) continue ;; esac
               echo "$pid:$cmd"
           done | tr '\n' ';')"
    raw="/tmp/d8k-$(basename "$ENG")-$i.log"
    timeout 900 env "$@" NPU_PREFILL_BF16=1 NPU_BF16=1 NPU_GREEDY=1 NPU_PREFILL_MAX=8192 \
        NPU_PROMPT_MAX=8192 "$E" "$D/model.q4nx" 32 "$P" > "$raw" 2>&1
    nrc=$?
    nline="$(grep -aE 'ms/tok \(' "$raw" | tail -1)"
    flog="/tmp/d8k-flm-$(basename "$ENG")-$i.log"
    timeout 900 bash "$HOME/npu-ab/npu_ab.sh" --model "$ABM" --flm-tag "$TAG" --engine "$E" \
        --q4nx "$D/model.q4nx" --tokenizer "$D/tokenizer.json" --prompt "$P" \
        --ctx-k 8 --decode-tokens 32 --reps 1 --skip-native > "$flog" 2>&1
    fline="$(grep -aE 'rep 1' "$flog" | head -1)"
    nt="$(printf '%s' "$nline" | sed -n 's/.*=== \([0-9.]*\) ms\/tok.*/\1/p')"
    ns="$(printf '%s' "$nline" | sed -n 's/.*(\([0-9]*\) tok\/s).*/\1/p')"
    ft="$(printf '%s' "$fline" | sed -n 's/.*decode=\([0-9.]*\)t\/s.*/\1/p')"
    echo "  pair $i: native rc=$nrc ${nt:-?} ms/tok (${ns:-?} tok/s) | FLM decode ${ft:-?} t/s | holders=[${pre:-none}]"
    awk -v a="${ns:-0}" -v b="${ft:-0}" 'BEGIN{if (a>0 && b>0) printf "           native/FLM decode ratio = %.3fx\n", a/b; else print "           native/FLM decode ratio = ?"}'
done
