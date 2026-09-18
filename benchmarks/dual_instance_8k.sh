#!/bin/bash
# dual8k.sh — same multi-instance test at 8160 tokens of context, where the prefill is host-bound
# (so the interesting question is whether a second engine helps or hurts on the host side).
set -u
cd "$HOME/1bit-MONSTER-goal" || exit 1
P=/tmp/p_8160.txt
NG=32
declare -A ENG DIR
ENG[q06]=./engine/npu/build/npu_engine_qwen3_0_6b; DIR[q06]=Qwen3-0.6B-NPU2
ENG[q17]=./engine/npu/build/npu_engine_qwen3_1_7b; DIR[q17]=Qwen3-1.7B-NPU2
pf() { grep -aoE "Prefill: [0-9]+ms" "$1" | tail -1 | grep -oE "[0-9]+"; }
dk() { grep -aoE "\([0-9]+ tok/s\) \| tokens=" "$1" | tail -1 | grep -oE "[0-9]+" | head -1; }
tk() { grep -aoE "^  \[[0-9]+\] [0-9]+" "$1" | tr -d ' ' | tr '\n' ' '; }
launch() { timeout 900 env $3 NPU_PREFILL_BF16=1 NPU_BF16=1 NPU_GREEDY=1 NPU_PREFILL_MAX=8192 \
    NPU_PROMPT_MAX=8192 "${ENG[$1]}" "$HOME/.config/flm/models/${DIR[$1]}/model.q4nx" "$NG" "$P" > "$2" 2>&1; }

echo "== 8160-token prompt, ng=$NG: prefill is host-bound at this context =="
for t in q06 q17; do
    launch "$t" "/tmp/d8k_s_$t.log" ""
    echo "serial $t: prefill=$(pf /tmp/d8k_s_$t.log)ms decode=$(dk /tmp/d8k_s_$t.log) tok/s"
done
t0=$(date +%s.%N)
launch q06 /tmp/d8k_c_q06.log "NPU_NO_DEVICE_LOCK=1" & a=$!
launch q17 /tmp/d8k_c_q17.log "NPU_NO_DEVICE_LOCK=1" & b=$!
wait $a $b
t1=$(date +%s.%N)
python3 -c "print('concurrent wall = %.2f s' % ($t1-$t0))"
for t in q06 q17; do
    [ "$(tk /tmp/d8k_s_$t.log)" = "$(tk /tmp/d8k_c_$t.log)" ] && v=IDENTICAL || v=DIFFER
    echo "  $t concurrent: prefill=$(pf /tmp/d8k_c_$t.log)ms decode=$(dk /tmp/d8k_c_$t.log) tok/s tokens=$v"
done
grep -aoiE "ERT_CMD_STATE_TIMEOUT|timed out|TDR" /tmp/d8k_c_*.log | sort -u | head -3
echo "(no error lines above = clean)"
