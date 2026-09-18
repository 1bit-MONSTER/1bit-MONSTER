#!/bin/bash
# sweep2.sh — more instances, and whether core pinning fixes the 8k (host-bound) anti-scaling.
set -u
cd "$HOME/1bit-MONSTER-goal" || exit 1
declare -A ENG DIR
ENG[q06]=./engine/npu/build/npu_engine_qwen3_0_6b; DIR[q06]=Qwen3-0.6B-NPU2
ENG[q17]=./engine/npu/build/npu_engine_qwen3_1_7b; DIR[q17]=Qwen3-1.7B-NPU2
ENG[q4b]=./engine/npu/build/npu_engine_qwen3_4b;   DIR[q4b]=Qwen3-4B-NPU2
rate() { grep -aoE "\([0-9]+ tok/s\) \| tokens=" "$1" | tail -1 | grep -oE "[0-9]+" | head -1; }
pref() { grep -aoE "Prefill: [0-9]+ms" "$1" | tail -1 | grep -oE "[0-9]+"; }
tk() { grep -aoE "^  \[[0-9]+\] [0-9]+" "$1" | tr -d ' ' | tr '\n' ' '; }

launch() {  # $1 tag $2 log $3 pin $4 ng $5 prompt $6.. extra env
    local tag=$1 log=$2 pin=$3 ng=$4 prompt=$5; shift 5
    timeout 900 $pin env NPU_NO_DEVICE_LOCK=1 NPU_HOST_THREADS=${THR:-8} "$@" NPU_PREFILL_BF16=1 NPU_BF16=1 NPU_GREEDY=1 \
        NPU_PREFILL_MAX=8192 NPU_PROMPT_MAX=8192 "${ENG[$tag]}" \
        "$HOME/.config/flm/models/${DIR[$tag]}/model.q4nx" "$ng" "$prompt" > "$log" 2>&1
}

multi() {  # $1 label ; then "tag:pin" descriptors ; env via EXTRA
    local label="$1"; shift
    local i=0 logs=() tags=()
    for d in "$@"; do
        local t=${d%%:*}; logs+=("/tmp/s2_${label// /_}_$i.log"); tags+=("$t"); i=$((i+1))
    done
    i=0; local pids=()
    for d in "$@"; do
        local t=${d%%:*}; local pin=${d#*:}; [ "$pin" = "$t" ] && pin=""
        launch "$t" "${logs[$i]}" "$pin" "$NG" "$PR" & pids+=($!)
        i=$((i+1))
    done
    wait "${pids[@]}"
    local sum=0 out="" j=0
    for t in "${tags[@]}"; do
        local r=$(rate "${logs[$j]}"); sum=$((sum+${r:-0})); out="$out $t=${r:-?}"
        j=$((j+1))
    done
    printf "%-40s%s  aggregate=%s tok/s\n" "$label" "$out" "$sum"
}

NG=128; PR=/tmp/p_1k.txt; THR=8
echo "== 1k context, ng=128 =="
multi "4x0.6B default 8thr" q06: q06: q06: q06:
multi "4x0.6B pinned 4thr" "q06:taskset -c 0-3" "q06:taskset -c 4-7" "q06:taskset -c 8-11" "q06:taskset -c 12-15"
THR=4
multi "4x0.6B pinned 4thr (explicit)" "q06:taskset -c 0-3" "q06:taskset -c 4-7" "q06:taskset -c 8-11" "q06:taskset -c 12-15"
THR=8
multi "4 mixed 0.6+0.6+1.7+4B pinned" "q06:taskset -c 0-3" "q06:taskset -c 4-7" "q17:taskset -c 8-11" "q4b:taskset -c 12-15"

echo
echo "== 8160-token context, ng=32 (prefill is host-bound) =="
NG=32; PR=/tmp/p_8160.txt; THR=8
for cfg in "default" "pinned"; do
    if [ "$cfg" = "pinned" ]; then
        P1="taskset -c 0-7"; P2="taskset -c 8-15"
    else
        P1=""; P2=""
    fi
    t0=$(date +%s.%N)
    launch q06 /tmp/s2_8k_a.log "$P1" "$NG" "$PR" & a=$!
    launch q17 /tmp/s2_8k_b.log "$P2" "$NG" "$PR" & b=$!
    wait $a $b
    t1=$(date +%s.%N)
    python3 -c "print('%-8s wall=%6.2fs  0.6B prefill=%sms decode=%s | 1.7B prefill=%sms decode=%s' % ('$cfg', $t1-$t0, '$(pref /tmp/s2_8k_a.log)', '$(rate /tmp/s2_8k_a.log)', '$(pref /tmp/s2_8k_b.log)', '$(rate /tmp/s2_8k_b.log)'))"
    [ "$(tk /tmp/s2_8k_a.log)" = "$(tk /tmp/d8k_s_q06.log)" ] && x=IDENTICAL || x=DIFFER
    echo "         0.6B tokens vs serial: $x"
done
