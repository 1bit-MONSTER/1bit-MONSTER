#!/bin/bash
# dual2.sh — NPU multi-instance scaling with the engine's OWN decode rate (which excludes process
# startup), plus per-instance output correctness against the serial runs of the same model.
# The engine flocks /tmp/1bit-npu-device.lock for its lifetime, so concurrent legs opt out with
# NPU_NO_DEVICE_LOCK=1 deliberately; serial baselines keep the lock.
set -u
cd "$HOME/1bit-MONSTER-goal" || exit 1
P=${P:-/tmp/p_1k.txt}
NG=${NG:-128}
declare -A ENG DIR SER
ENG[q06]=./engine/npu/build/npu_engine_qwen3_0_6b; DIR[q06]=Qwen3-0.6B-NPU2
ENG[q17]=./engine/npu/build/npu_engine_qwen3_1_7b; DIR[q17]=Qwen3-1.7B-NPU2
ENG[q4b]=./engine/npu/build/npu_engine_qwen3_4b;   DIR[q4b]=Qwen3-4B-NPU2

rate() { grep -aoE "\([0-9]+ tok/s\) \| tokens=" "$1" | tail -1 | grep -oE "[0-9]+" | head -1; }
pref() { grep -aoE "Prefill: [0-9]+ms" "$1" | tail -1 | grep -oE "[0-9]+"; }
toks() { grep -aoE "^  \[[0-9]+\] [0-9]+" "$1" | tr -d ' ' | tr '\n' ' '; }
launch() { timeout 900 env $3 NPU_PREFILL_BF16=1 NPU_BF16=1 NPU_GREEDY=1 NPU_PREFILL_MAX=2048 \
    "${ENG[$1]}" "$HOME/.config/flm/models/${DIR[$1]}/model.q4nx" "$NG" "$P" > "$2" 2>&1; }

echo "== prompt=$P (1024 tok) ng=$NG =="
for t in q06 q17 q4b; do
    L="/tmp/d2_s_$t.log"; launch "$t" "$L" ""
    SER[$t]=$(rate "$L")
    echo "serial $t: prefill=$(pref $L)ms decode=${SER[$t]} tok/s"
done

pair() {
    local label="$1"; shift
    local i=0 pids=() tags=() logs=()
    for t in "$@"; do
        L="/tmp/d2_c_${label// /_}_${i}_$t.log"; logs+=("$L"); tags+=("$t"); i=$((i+1))
    done
    i=0
    for t in "${tags[@]}"; do launch "$t" "${logs[$i]}" "NPU_NO_DEVICE_LOCK=1" & pids+=($!); i=$((i+1)); done
    wait "${pids[@]}"
    local sum=0 out="" j=0
    for t in "${tags[@]}"; do
        local r=$(rate "${logs[$j]}")
        sum=$((sum + ${r:-0})); out="$out $t=${r:-?}"
        j=$((j+1))
    done
    echo "$label: $out  aggregate=$sum tok/s"
    j=0
    for t in "${tags[@]}"; do
        [ "${SER[$t]:-x}" = "$(rate "${logs[$j]}")" ] && vr="rate=serial" || vr="rate=SHARED($(rate ${logs[$j]}) vs ${SER[$t]})"
        [ "$(toks /tmp/d2_s_$t.log)" = "$(toks "${logs[$j]}")" ] && vt="tokens=IDENTICAL" || vt="tokens=DIFFER"
        printf "    %-4s %s %s\n" "$t" "$vt" "$vr"
        j=$((j+1))
    done
}

pair "2x0.6B" q06 q06
pair "0.6B+1.7B" q06 q17
pair "0.6B+1.7B+4B" q06 q17 q4b
for f in /tmp/d2_c_*.log; do
    m=$(grep -aoiE "ERT_CMD_STATE_TIMEOUT|timed out|TDR|ERROR" "$f" | sort -u | tr '\n' ',')
    [ -n "$m" ] && echo "  $(basename $f): $m"
done
echo "(no per-log error lines above = no device errors)"
