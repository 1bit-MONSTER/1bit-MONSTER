#!/bin/bash
# sweep3.sh — aggregate throughput vs number of concurrent instances, and the best mixed set,
# on the 1k decode-bound workload. Answers "how many engines should be loaded to maximise the box".
set -u
cd "$HOME/1bit-MONSTER-goal" || exit 1
P=/tmp/p_1k.txt
NG=${NG:-128}
declare -A ENG DIR
ENG[q06]=./engine/npu/build/npu_engine_qwen3_0_6b; DIR[q06]=Qwen3-0.6B-NPU2
ENG[q17]=./engine/npu/build/npu_engine_qwen3_1_7b; DIR[q17]=Qwen3-1.7B-NPU2
ENG[q4b]=./engine/npu/build/npu_engine_qwen3_4b;   DIR[q4b]=Qwen3-4B-NPU2
rate() { grep -aoE "\([0-9]+ tok/s\) \| tokens=" "$1" | tail -1 | grep -oE "[0-9]+" | head -1; }
launch() { timeout 900 env NPU_NO_DEVICE_LOCK=1 NPU_HOST_THREADS=${THR:-8} NPU_PREFILL_BF16=1 NPU_BF16=1 \
    NPU_GREEDY=1 NPU_PREFILL_MAX=2048 "${ENG[$1]}" "$HOME/.config/flm/models/${DIR[$1]}/model.q4nx" \
    "$NG" "$P" > "$2" 2>&1; }

set_of() {  # $1 label ; $2.. tags
    local label="$1"; shift
    local i=0 logs=() tags=()
    for t in "$@"; do logs+=("/tmp/s3_${label// /_}_$i.log"); tags+=("$t"); i=$((i+1)); done
    i=0; local pids=()
    for t in "${tags[@]}"; do launch "$t" "${logs[$i]}" & pids+=($!); i=$((i+1)); done
    wait "${pids[@]}"
    local sum=0 out="" j=0
    for t in "${tags[@]}"; do local r=$(rate "${logs[$j]}"); sum=$((sum+${r:-0})); out="$out ${r:-?}"; j=$((j+1)); done
    printf "%-26s n=%d  per-instance:%s  AGGREGATE=%s tok/s\n" "$label" "$#" "$out" "$sum"
}

echo "== Qwen3-0.6B, 1k prompt, ng=$NG: aggregate vs instance count (default 8 threads each) =="
THR=8
set_of "0.6B x1" q06
set_of "0.6B x2" q06 q06
set_of "0.6B x3" q06 q06 q06
set_of "0.6B x4" q06 q06 q06 q06
set_of "0.6B x6" q06 q06 q06 q06 q06 q06
echo
echo "== best mixed candidates =="
set_of "mixed 0.6+1.7" q06 q17
set_of "mixed 0.6+1.7+4B" q06 q17 q4b
set_of "mixed 0.6+0.6+1.7+4B" q06 q06 q17 q4b
set_of "mixed 0.6x2+1.7x2+4B" q06 q06 q17 q17 q4b
echo
echo "== same, 4 threads each (host thread budget split) =="
THR=4
set_of "0.6B x4 (4thr)" q06 q06 q06 q06
set_of "mixed .6x2+1.7+4B (4thr)" q06 q06 q17 q4b
