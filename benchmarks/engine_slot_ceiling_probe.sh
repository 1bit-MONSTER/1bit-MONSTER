#!/bin/bash
# engine_slot_ceiling_probe.sh — how many concurrent npu_engine_qwen3_* instances
# (this lane's own engine) can hold a device context and serve at once.
#
# Launches N instances at once with NPU_NO_DEVICE_LOCK=1, then polls until every
# instance has either produced its full token stream (serving) or been refused with
# CREATE_HWCTX err=-2, or the deadline passes. Classifies each by its own log.
# Cleans up by exact process name (comm prefix), never `pkill -f` (whose pattern
# would also match the ssh command line).
#
# Run on strixhalo from the repo root:
#   N=8  bash benchmarks/engine_slot_ceiling_probe.sh
#   N=10 bash benchmarks/engine_slot_ceiling_probe.sh
# Raw per-instance logs: /tmp/engslot_<TAG>_<i>.log
set -u
cd "$HOME/1bit-MONSTER-goal" || exit 1
N=${N:-8}
NG=${NG:-64}
TAG=${TAG:-q06}
P=${P:-/tmp/p_1k.txt}
DEADLINE=${DEADLINE:-420}
LOG=/tmp/engine_slot_ceiling_${TAG}_${N}.log
: > "$LOG"

declare -A BIN DIR
BIN[q06]=./engine/npu/build/npu_engine_qwen3_0_6b; DIR[q06]=Qwen3-0.6B-NPU2
BIN[q17]=./engine/npu/build/npu_engine_qwen3_1_7b; DIR[q17]=Qwen3-1.7B-NPU2
BIN[q4b]=./engine/npu/build/npu_engine_qwen3_4b;   DIR[q4b]=Qwen3-4B-NPU2

MODEL=$HOME/.config/flm/models/${DIR[$TAG]}/model.q4nx

# count HW contexts across both partitions (Ctx ID rows, one per context)
hwctx() { xrt-smi examine -r aie-partitions 2>/dev/null | awk -F'|' '
    NF>=3 && $2 ~ /^[0-9]+[ ]*$/ && $3 ~ /Active|Inactive|/ { c++ } END { print c+0 }'; }

cleanup() {
    for pid in $(ps -eo pid,comm | awk '$2 ~ /^npu_engine/ {print $1}'); do kill "$pid" 2>/dev/null; done
}

pids=()
echo "== engine slot ceiling: N=$N tag=$TAG ng=$NG prompt=$P ==" | tee -a "$LOG"
echo "== baseline hwctx=$(hwctx) ==" | tee -a "$LOG"

for i in $(seq 1 "$N"); do
    timeout $((DEADLINE+60)) env NPU_NO_DEVICE_LOCK=1 NPU_PREFILL_BF16=1 NPU_BF16=1 NPU_GREEDY=1 NPU_PREFILL_MAX=2048 \
        "${BIN[$TAG]}" "$MODEL" "$NG" "$P" > "/tmp/engslot_${TAG}_$i.log" 2>&1 &
    pids+=($!)
done

peak=0
for t in $(seq 1 $((DEADLINE/5))); do
    sleep 5
    pending=0
    for i in $(seq 1 "$N"); do
        L="/tmp/engslot_${TAG}_$i.log"
        if grep -aqE "tokens=" "$L"; then :
        elif grep -aq "CREATE_HWCTX" "$L"; then :
        else pending=$((pending+1)); fi
    done
    c=$(hwctx); [ "$c" -gt "$peak" ] && peak=$c
    [ "$pending" -eq 0 ] && break
done

echo "== after ${t}x5s: hwctx_now=$(hwctx) peak_hwctx=$peak ==" | tee -a "$LOG"
ok=0; fail=0; pending=0
for i in $(seq 1 "$N"); do
    L="/tmp/engslot_${TAG}_$i.log"
    pre=$(grep -aoE "Prefill: [0-9]+ ?ms|prefill [0-9]+ ?ms" "$L" | tail -1)
    r=$(grep -aoE "[0-9.]+ ms/tok \([0-9]+ tok/s\)" "$L" | tail -1)
    if grep -aq "CREATE_HWCTX" "$L"; then
        err=$(grep -aoE "CREATE_HWCTX[^\"]*" "$L" | tail -1)
        echo "  slot $i: FAIL  $err" | tee -a "$LOG"; fail=$((fail+1))
    elif grep -aqE "tokens=" "$L"; then
        echo "  slot $i: SERVED  $r  $pre  tokens=${NG}" | tee -a "$LOG"; ok=$((ok+1))
    else
        echo "  slot $i: PENDING  last=$(tail -1 "$L")" | tee -a "$LOG"; pending=$((pending+1))
    fi
done
rss=$(ps -eo rss,comm | awk '$2 ~ /^npu_engine/ {s+=$1} END {printf "%.2f", s/1048576}')
echo "== SERVED=$ok FAIL=$fail PENDING=$pending (N=$N), engine RSS now=${rss:-?} GB ==" | tee -a "$LOG"
echo "== driver refusals ==" | tee -a "$LOG"
grep -ah "CREATE_HWCTX" /tmp/engslot_${TAG}_*.log 2>/dev/null | sed 's/^/   /' | sort | uniq -c | tee -a "$LOG"

cleanup
sleep 4
echo "== after cleanup hwctx=$(hwctx) ==" | tee -a "$LOG"
