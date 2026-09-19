#!/bin/bash
# engine_slot_identity_bench.sh — validate concurrent engine slots: every slot's token
# stream must be byte-identical to the serial reference of the same model/prompt/ng.
#
# The reference is run twice (greedy, deterministic) so a non-deterministic reference is
# caught before it is used as ground truth. Concurrent slots run with NPU_NO_DEVICE_LOCK=1
# (the engine otherwise flocks the device and serialises); LAUNCH=stagger starts each slot
# only after the previous one is serving (the real "agents join over time" slot pattern),
# LAUNCH=simul starts them all at once.
#
# Run on strixhalo from the repo root:
#   N=8 P=/tmp/p_1k.txt NG=64 LAUNCH=stagger bash benchmarks/engine_slot_identity_bench.sh
#   N=4 P=/tmp/p_1k.txt NG=64 LAUNCH=simul   bash benchmarks/engine_slot_identity_bench.sh
set -u
cd "$HOME/1bit-MONSTER-goal" || exit 1
N=${N:-8}
TAG=${TAG:-q06}
NG=${NG:-64}
P=${P:-/tmp/p_1k.txt}
LAUNCH=${LAUNCH:-stagger}
WAIT=${WAIT:-300}
OUTDIR=${OUTDIR:-/tmp/eng_identity}
mkdir -p "$OUTDIR"; rm -f "$OUTDIR"/*.log

declare -A BIN DIR
BIN[q06]=./engine/npu/build/npu_engine_qwen3_0_6b; DIR[q06]=Qwen3-0.6B-NPU2
BIN[q17]=./engine/npu/build/npu_engine_qwen3_1_7b; DIR[q17]=Qwen3-1.7B-NPU2
BIN[q4b]=./engine/npu/build/npu_engine_qwen3_4b;   DIR[q4b]=Qwen3-4B-NPU2
MODEL=$HOME/.config/flm/models/${DIR[$TAG]}/model.q4nx

toks() { grep -aoE '^  \[[0-9]+\] [0-9]+' "$1" 2>/dev/null | awk '{print $2}' | paste -sd' ' -; }
rate() { grep -aoE '[0-9.]+ ms/tok \([0-9]+ tok/s\)' "$1" | tail -1; }
pref() { grep -aoE 'Prefill: [0-9]+ ?ms' "$1" | tail -1; }
cleanup() { for pid in $(ps -eo pid,comm | awk '$2 ~ /^npu_engine/ {print $1}'); do kill "$pid" 2>/dev/null; done; }
trap cleanup EXIT

run_serial() {  # $1 log
    timeout 900 env NPU_PREFILL_BF16=1 NPU_BF16=1 NPU_GREEDY=1 NPU_PREFILL_MAX=2048 \
        "${BIN[$TAG]}" "$MODEL" "$NG" "$P" > "$1" 2>&1
}

echo "== engine slot identity: N=$N tag=$TAG ng=$NG launch=$LAUNCH prompt=$P =="
cleanup; sleep 3
run_serial "$OUTDIR/ref1.log"
run_serial "$OUTDIR/ref2.log"
REF1=$(toks "$OUTDIR/ref1.log"); REF2=$(toks "$OUTDIR/ref2.log")
echo "serial reference:  $(rate "$OUTDIR/ref1.log")  $(pref "$OUTDIR/ref1.log")"
if [ "$REF1" = "$REF2" ]; then echo "reference determinism: IDENTICAL across two serial runs"; else echo "reference determinism: DIFFERENT -- refusing to use it as ground truth"; exit 2; fi

pids=()
for i in $(seq 1 "$N"); do
    L="$OUTDIR/slot_$i.log"
    env NPU_NO_DEVICE_LOCK=1 NPU_PREFILL_BF16=1 NPU_BF16=1 NPU_GREEDY=1 NPU_PREFILL_MAX=2048 \
        "${BIN[$TAG]}" "$MODEL" "$NG" "$P" > "$L" 2>&1 &
    pids+=($!)
    if [ "$LAUNCH" = stagger ]; then
        for t in $(seq 1 "$WAIT"); do
            grep -aqE '^  \[[0-9]+\] ' "$L" 2>/dev/null && break
            grep -aq "CREATE_HWCTX" "$L" 2>/dev/null && break
            kill -0 ${pids[$((${#pids[@]}-1))]} 2>/dev/null || break
            sleep 1
        done
    fi
done
for p in "${pids[@]}"; do wait "$p" 2>/dev/null; done

ok=0; differ=0; refused=0; sum=0
for i in $(seq 1 "$N"); do
    L="$OUTDIR/slot_$i.log"
    t=$(toks "$L"); r=$(rate "$L"); pf=$(pref "$L")
    if grep -aq "CREATE_HWCTX" "$L"; then
        echo "  slot $i: REFUSED (no context)"; refused=$((refused+1)); continue
    fi
    rt=$(echo "$r" | grep -oE '[0-9]+ tok/s' | grep -oE '[0-9]+')
    sum=$((sum + ${rt:-0}))
    if [ "$t" = "$REF1" ]; then
        echo "  slot $i: tokens=IDENTICAL  $r  $pf"; ok=$((ok+1))
    else
        echo "  slot $i: tokens=DIFFER  $r  $pf"; differ=$((differ+1))
    fi
done
echo "== identical=$ok differ=$differ refused=$refused of N=$N  |  aggregate decode=${sum} tok/s =="
echo "== device errors =="
grep -lahE "CREATE_HWCTX|ERT_CMD_STATE_TIMEOUT|TDR|timed out" "$OUTDIR"/*.log 2>/dev/null | sed 's/^/   /' | sort -u | head
cleanup
