#!/bin/bash
# engine_slot_stagger_probe.sh — resident engine-slot ceiling, one slot at a time, kept alive.
#
# Engine analogue of the flm one-at-a-time slot probe. Launch slot 1 and wait until it is
# actually serving (first decoded token), then slot 2, and so on. Every slot stays alive
# (huge ng; cleanup at exit kills them by exact comm prefix), so the first slot the driver
# refuses with DRM_IOCTL_AMDXDNA_CREATE_HWCTX err=-2 IS the ceiling.
#
# A tiny prompt is deliberate: prefill is what contends on the host, and for a *residency*
# ceiling we want each slot to start serving fast and then just hold its context. Use
# P=/tmp/p_1k.txt instead if you want the contended prefill times too.
#
# Run on strixhalo from the repo root:  MAXN=8 bash benchmarks/engine_slot_stagger_probe.sh
set -u
cd "$HOME/1bit-MONSTER-goal" || exit 1
MAXN=${MAXN:-8}
TAG=${TAG:-q06}
NG=${NG:-100000000}
P=${P:-/tmp/tiny.txt}
WAIT=${WAIT:-240}
LOG=/tmp/engine_slot_stagger_${TAG}.log
: > "$LOG"

declare -A BIN DIR
BIN[q06]=./engine/npu/build/npu_engine_qwen3_0_6b; DIR[q06]=Qwen3-0.6B-NPU2
BIN[q17]=./engine/npu/build/npu_engine_qwen3_1_7b; DIR[q17]=Qwen3-1.7B-NPU2
MODEL=$HOME/.config/flm/models/${DIR[$TAG]}/model.q4nx

# count HW-context rows: a row whose 2nd pipe-field is a numeric PID (see aie-partitions table;
# the rows are indented, hence the leading-space tolerance)
hwctx() { xrt-smi examine -r aie-partitions 2>/dev/null | awk -F'|' '$2 ~ /^[ ]*[0-9]+[ ]*$/ {c++} END {print c+0}'; }
cleanup() { for pid in $(ps -eo pid,comm | awk '$2 ~ /^npu_engine/ {print $1}'); do kill "$pid" 2>/dev/null; done; }
trap cleanup EXIT

echo "== engine slot stagger: tag=$TAG ng=$NG prompt=$P maxn=$MAXN ==" | tee -a "$LOG"
echo "== baseline hwctx=$(hwctx) ==" | tee -a "$LOG"

served=0
for i in $(seq 1 "$MAXN"); do
    L="/tmp/engstag_${TAG}_$i.log"
    env NPU_NO_DEVICE_LOCK=1 NPU_PREFILL_BF16=1 NPU_BF16=1 NPU_GREEDY=1 NPU_PREFILL_MAX=2048 \
        "${BIN[$TAG]}" "$MODEL" "$NG" "$P" > "$L" 2>&1 &
    pid=$!
    state=pending
    for t in $(seq 1 "$WAIT"); do
        if grep -aq "CREATE_HWCTX" "$L" 2>/dev/null; then state=fail; break; fi
        if grep -aqE '^  \[[0-9]+\] ' "$L" 2>/dev/null; then state=serving; break; fi
        kill -0 $pid 2>/dev/null || { state=dead; break; }
        sleep 1
    done
    c=$(hwctx)
    if [ "$state" = serving ]; then
        served=$((served+1))
        pre=$(grep -aoE "Prefill: [0-9]+ ?ms" "$L" | tail -1)
        echo "  slot $i: SERVING  $pre  hwctx=$c" | tee -a "$LOG"
    elif [ "$state" = fail ]; then
        echo "  slot $i: REFUSED  $(grep -aoE 'CREATE_HWCTX[^"]*' "$L" | tail -1)  hwctx=$c" | tee -a "$LOG"
        break
    elif [ "$state" = dead ]; then
        echo "  slot $i: DIED  $(tail -1 "$L")  hwctx=$c" | tee -a "$LOG"
        break
    else
        echo "  slot $i: TIMEOUT after ${WAIT}s  last=$(tail -1 "$L")  hwctx=$c" | tee -a "$LOG"
        break
    fi
done

rss=$(ps -eo rss,comm | awk '$2 ~ /^npu_engine/ {s+=$1} END {printf "%.2f", s/1048576}')
echo == | tee -a "$LOG"
echo "== engine slots serving: $served  |  hwctx now: $(hwctx)  |  total engine RSS: ${rss:-?} GB ==" | tee -a "$LOG"
cleanup
sleep 4
echo "== after cleanup hwctx=$(hwctx) ==" | tee -a "$LOG"
