#!/usr/bin/env bash
# c8k_guarded.sh — criterion-(c) 8k prefill/TTFT campaign WITH a contention guard.
#
# Why this exists: the 8k rows moved 3.6x between runs of an identical command
# (RESULTS-8k-campaign-variance-2026-09-18.md), and nothing in this lane's harnesses records
# whether a foreign process held the NPU while a number was taken. Runs taken with a foreign
# holder are DISCARDED here rather than averaged in.
#
# usage: c8k_guarded.sh <Model-NPU2> <flm_tag> <engine_name> <runs> [extra native env ...]
#   e.g. c8k_guarded.sh Qwen3-0.6B-NPU2 qwen3:0.6b npu_engine_qwen3_0_6b 5
#        c8k_guarded.sh Llama-3.1-8B-NPU2 llama3.1:8b npu_engine_llama 5 NPU_UNIFIED=1 NPU_LAYER_ELF_DIR=/tmp/llama-elfs
#
# Native is ng=1 on purpose: at 8k the SECOND decode forward needs ctx=8194, which fails
# against the baked MAX_L=8192 per-ctx ELF window, so only prefill/TTFT is a valid 8k row.
# The production `flm serve` holds accel0 permanently and is expected; it is excluded from the
# "foreign holder" set. The campaign engine's own pid appears in `post` only if it leaked.
set -u
M="${1:?usage: c8k_guarded.sh <Model-NPU2> <flm_tag> <engine> <runs> [env ...]}"
TAG="${2:?}"; ENG="${3:?}"; RUNS="${4:?}"; shift 4
D="$HOME/.config/flm/models/$M"
E="./engine/npu/build/$ENG"
PROMPT=/tmp/p_8k.txt
LOGDIR="./benchmarks/c8k-guarded-$M-$(date -u +%Y%m%dT%H%M%SZ)"
[ -f "$PROMPT" ] || { echo "missing $PROMPT" >&2; exit 2; }
mkdir -p "$LOGDIR"

# PIDs holding accel0, excluding the persistent production `flm serve`.
foreign_holders() {
    local out=""
    for p in /proc/[0-9]*; do
        ls -l "$p/fd" 2>/dev/null | grep -q accel0 || continue
        local pid="${p#/proc/}" cmd
        cmd="$(tr '\0' ' ' < "$p/cmdline" 2>/dev/null | cut -c1-60)"
        case "$cmd" in *"flm serve"*) continue ;; esac
        out="$out${pid}:${cmd}; "
    done
    printf '%s' "$out"
}
runner_lines() { wc -l < /tmp/runner.log 2>/dev/null || echo 0; }

echo "== $M : $RUNS runs, guard on =="
n_ok=0
declare -a N_PREFILL=() N_TTFT=() F_PREFILL=() F_TTFT=()
for i in $(seq 1 "$RUNS"); do
    pre_f="$(foreign_holders)"; pre_r="$(runner_lines)"
    raw="$LOGDIR/native-$i.log"
    timeout 900 env "$@" NPU_PREFILL_BF16=1 NPU_BF16=1 NPU_GREEDY=1 NPU_PREFILL_MAX=8192 \
        NPU_PROMPT_MAX=8192 "$E" "$D/model.q4nx" 1 "$PROMPT" > "$raw" 2>&1
    np_line="$(grep -aE 'Prefill:' "$raw" | tail -1)"
    post_f="$(foreign_holders)"; post_r="$(runner_lines)"
    flog="$LOGDIR/flm-$i.log"
    timeout 900 bash "$HOME/npu-ab/npu_ab.sh" --model "$ENG" --flm-tag "$TAG" --engine "$E" \
        --q4nx "$D/model.q4nx" --tokenizer "$D/tokenizer.json" --prompt "$PROMPT" \
        --ctx-k 8 --decode-tokens 1 --reps 1 --skip-native > "$flog" 2>&1
    f_line="$(grep -aE 'rep 1' "$flog" | head -1)"
    ci=""
    [ "$pre_r" = "$post_r" ] || ci=" (runner.log grew: $pre_r -> $post_r)"

    status="ACCEPT"
    [ -n "$pre_f" ] && status="DISCARD(pre-holder: $pre_f)"
    [ -n "$post_f" ] && status="DISCARD(post-holder: $post_f)"
    [ -n "$ci" ] && [ "$status" = ACCEPT ] && status="SUSPECT(ci$ci)"

    np_t="$(printf '%s' "$np_line" | sed -n 's/.*(\([0-9.]*\) ms\/tok).*/\1/p')"
    np_ms="$(printf '%s' "$np_line" | sed -n 's/Prefill: \([0-9]*\)ms.*/\1/p')"
    f_t="$(printf '%s' "$f_line" | sed -n 's/.*ttft=\([0-9.]*\)s.*/\1/p')"
    f_p="$(printf '%s' "$f_line" | sed -n 's/.*prefill=\([0-9.]*\)t\/s.*/\1/p')"
    echo "run $i [$status] native ${np_ms:-?}ms (${np_t:-?} ms/tok) | FLM ttft ${f_t:-?}s prefill ${f_p:-?}t/s"
    if [ "$status" = ACCEPT ]; then
        n_ok=$((n_ok+1))
        [ -n "$np_t" ] && N_TTFT+=("$np_t")
        [ -n "$f_t" ] && F_TTFT+=("$f_t")
        [ -n "$f_p" ] && F_PREFILL+=("$f_p")
    fi
done

median() {  # median of the arguments, numeric
    [ "$#" -gt 0 ] || { echo "n/a"; return; }
    printf '%s\n' "$@" | sort -g | awk '{a[NR]=$1} END{print (NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2}'
}
echo "-- accepted runs: $n_ok/$RUNS"
echo "-- native ms/tok   median: $(median "${N_TTFT[@]:-}")   (all: ${N_TTFT[*]:-none})"
echo "-- FLM   ttft s    median: $(median "${F_TTFT[@]:-}")   (all: ${F_TTFT[*]:-none})"
echo "-- FLM   prefill t/s median: $(median "${F_PREFILL[@]:-}")   (all: ${F_PREFILL[*]:-none})"
echo "-- logs: $LOGDIR"
