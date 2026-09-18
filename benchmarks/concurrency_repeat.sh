#!/bin/bash
# repeat.sh — the two headline configurations, 3 repetitions each, because single-instance rate on
# this box moved 61..77 tok/s between runs and the difference between configs must be larger than
# that.
set -u
cd "$HOME/1bit-MONSTER-goal" || exit 1
P=/tmp/p_1k.txt; NG=128
declare -A ENG DIR
ENG[q06]=./engine/npu/build/npu_engine_qwen3_0_6b; DIR[q06]=Qwen3-0.6B-NPU2
ENG[q17]=./engine/npu/build/npu_engine_qwen3_1_7b; DIR[q17]=Qwen3-1.7B-NPU2
rate() { grep -aoE "\([0-9]+ tok/s\) \| tokens=" "$1" | tail -1 | grep -oE "[0-9]+" | head -1; }
launch() { timeout 900 env NPU_NO_DEVICE_LOCK=1 NPU_HOST_THREADS=8 NPU_PREFILL_BF16=1 NPU_BF16=1 \
    NPU_GREEDY=1 NPU_PREFILL_MAX=2048 "${ENG[$1]}" "$HOME/.config/flm/models/${DIR[$1]}/model.q4nx" \
    "$NG" "$P" > "$2" 2>&1; }
echo "cfg,rep,per-instance,aggregate,loadavg_before"
for rep in 1 2 3; do
    L0=$(cut -d' ' -f1 /proc/loadavg)
    launch q06 /tmp/rp_s.log
    echo "single-0.6B,$rep,$(rate /tmp/rp_s.log),$(rate /tmp/rp_s.log),$L0"
done
for rep in 1 2 3; do
    L0=$(cut -d' ' -f1 /proc/loadavg)
    launch q06 /tmp/rp_a.log & a=$!
    launch q17 /tmp/rp_b.log & b=$!
    wait $a $b
    r1=$(rate /tmp/rp_a.log); r2=$(rate /tmp/rp_b.log)
    echo "pair-0.6B+1.7B,$rep,$r1+$r2,$(( ${r1:-0} + ${r2:-0} )),$L0"
done
