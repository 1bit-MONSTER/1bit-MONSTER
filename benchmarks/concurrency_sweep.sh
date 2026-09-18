#!/bin/bash
# sweep.sh — configuration sweep for CONCURRENT engine instances: per-instance host-thread count,
# CPU pinning (disjoint core sets), and OpenMP placement. Metric = each instance's own decode rate
# (excludes startup) plus their aggregate.
set -u
cd "$HOME/1bit-MONSTER-goal" || exit 1
P=/tmp/p_1k.txt
NG=${NG:-128}
declare -A ENG DIR
ENG[q06]=./engine/npu/build/npu_engine_qwen3_0_6b; DIR[q06]=Qwen3-0.6B-NPU2
ENG[q17]=./engine/npu/build/npu_engine_qwen3_1_7b; DIR[q17]=Qwen3-1.7B-NPU2
rate() { grep -aoE "\([0-9]+ tok/s\) \| tokens=" "$1" | tail -1 | grep -oE "[0-9]+" | head -1; }

run() {  # $1 tag  $2 log  $3 pin(optional "taskset -c ...")  $4 env...
    local tag=$1 log=$2 pin=$3; shift 3
    timeout 900 $pin env NPU_NO_DEVICE_LOCK=1 "$@" NPU_PREFILL_BF16=1 NPU_BF16=1 NPU_GREEDY=1 \
        NPU_PREFILL_MAX=2048 "${ENG[$tag]}" "$HOME/.config/flm/models/${DIR[$tag]}/model.q4nx" "$NG" "$P" > "$log" 2>&1
}

pair() {  # $1 label  $2 pin1  $3 pin2  $4.. env
    local label="$1" p1="$2" p2="$3"; shift 3
    run q06 "/tmp/sw_a.log" "$p1" "$@" & a=$!
    run q06 "/tmp/sw_b.log" "$p2" "$@" & b=$!
    wait $a $b
    local r1=$(rate /tmp/sw_a.log) r2=$(rate /tmp/sw_b.log)
    printf "%-46s %s + %s = %s tok/s\n" "$label" "${r1:-?}" "${r2:-?}" "$(( ${r1:-0} + ${r2:-0} ))"
}

echo "== 2x Qwen3-0.6B, $P (1024 tok), ng=$NG =="
pair "A  no pinning, 8 threads each (default)" "" "" 
pair "B  pinned 0-7 / 8-15, 8 threads each" "taskset -c 0-7" "taskset -c 8-15" 
pair "C  pinned 0-7 / 8-15, 4 threads each" "taskset -c 0-7" "taskset -c 8-15" NPU_HOST_THREADS=4
pair "D  pinned 0-3 / 4-7, 4 threads each" "taskset -c 0-3" "taskset -c 4-7" NPU_HOST_THREADS=4
pair "E  no pinning, 4 threads each" "" "" NPU_HOST_THREADS=4
pair "F  no pinning, 16 threads each" "" "" NPU_HOST_THREADS=16
pair "G  OMP_PLACES=cores PROC_BIND=close, 8 thr" "" "" OMP_PLACES=cores OMP_PROC_BIND=close
pair "H  pinned 0-7/8-15, 8thr, OMP close" "taskset -c 0-7" "taskset -c 8-15" OMP_PLACES=cores OMP_PROC_BIND=close
echo
echo "== mixed pair 0.6B + 1.7B, same prompt =="
mix() { local label="$1" p1="$2" p2="$3"; shift 3
    run q06 /tmp/sw_a.log "$p1" "$@" & a=$!
    run q17 /tmp/sw_b.log "$p2" "$@" & b=$!
    wait $a $b
    local r1=$(rate /tmp/sw_a.log) r2=$(rate /tmp/sw_b.log)
    printf "%-46s 0.6B=%s + 1.7B=%s = %s tok/s\n" "$label" "${r1:-?}" "${r2:-?}" "$(( ${r1:-0} + ${r2:-0} ))"
}
mix "I  no pinning (default)" "" ""
mix "J  pinned 0-7 / 8-15" "taskset -c 0-7" "taskset -c 8-15"
mix "K  pinned 0-3 / 4-7, 4 threads" "taskset -c 0-3" "taskset -c 4-7" NPU_HOST_THREADS=4
