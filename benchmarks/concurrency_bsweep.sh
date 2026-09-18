#!/bin/bash
# bsweep.sh — the in-process alternative to multiple engine processes: NPU_BS=B decodes B copies
# of the same sequence per step (shared prompt prefill, batched GEMMs, M=B). Short prompt so the
# auto rule for the unified path does not fire and the batched-decode path is reached.
set -u
cd "$HOME/1bit-MONSTER-goal" || exit 1
E=./engine/npu/build/npu_engine_qwen3_0_6b
M="$HOME/.config/flm/models/Qwen3-0.6B-NPU2"
P=${P:-/tmp/pp.txt}     # 13 prompt tokens
NG=${NG:-128}
echo "== Qwen3-0.6B, prompt=$P ng=$NG, NPU_BS sweep (single process) =="
for BS in 1 2 4 8; do
    L=/tmp/bs_$BS.log
    timeout 900 env NPU_BS=$BS NPU_HOST_THREADS=${THR:-8} NPU_PREFILL_BF16=1 NPU_BF16=1 NPU_GREEDY=1 \
        NPU_PREFILL_MAX=2048 "$E" "$M/model.q4nx" "$NG" "$P" > "$L" 2>&1
    line=$(grep -aoE "=== [0-9.]+ ms/tok \([0-9]+ tok/s\) \| boot=[0-9]+ms batches=[0-9]+ tokens=[0-9]+ ===" "$L" | tail -1)
    if [ -z "$line" ]; then
        line="(no batched-decode line) $(grep -aoE '=== M=[0-9]+ Batch Decode' "$L" | tail -1)"
    fi
    steps=$(printf '%s' "$line" | sed -n 's/.*(\([0-9]*\) tok\/s).*/\1/p')
    eff=$(python3 -c "print('%.0f'%($BS*${steps:-0}))" 2>/dev/null)
    printf "BS=%-2s %s\n        effective total tokens/s = %s\n" "$BS" "$line" "${eff:-?}"
done
echo
echo "== same total work, compared: 4 processes x BS=1 vs 1 process x BS=4 (13-token prompt) =="
t0=$(date +%s.%N)
for i in 1 2 3 4; do timeout 900 env NPU_NO_DEVICE_LOCK=1 NPU_HOST_THREADS=${THR:-8} NPU_PREFILL_BF16=1 NPU_BF16=1 NPU_GREEDY=1 NPU_PREFILL_MAX=2048 "$E" "$M/model.q4nx" "$NG" "$P" > /tmp/bs_proc_$i.log 2>&1 & done
wait
t1=$(date +%s.%N)
python3 -c "print('4 processes x BS=1: wall=%.2fs  (512 tokens total)' % ($t1-$t0))"
for i in 1 2 3 4; do printf "  proc%s %s\n" "$i" "$(grep -aoE '=== [0-9.]+ ms/tok \([0-9]+ tok/s\).*tokens=[0-9]+ ===' /tmp/bs_proc_$i.log | tail -1)"; done
