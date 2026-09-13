#!/bin/bash
# Attention-correctness sweep: NPU (embedded captured ELF) vs CPU attention,
# same prompt, several lengths. Gate: boot tokens must match.
set -u
ROOT=/home/bcloud/1bit-MONSTER-goal
ENG=$ROOT/engine/npu/build/npu_engine_qwen3_0_6b
Q4NX=/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx
P=/home/bcloud/npu-build/parity
cd $ROOT || exit 1
for n in "$@"; do
  for mode in npu cpu; do
    extra=""
    [ "$mode" = cpu ] && extra="NPU_ATTN_CPU=1"
    env $extra NPU_PREFILL_MAX=$n NPU_RUNLIST=0 NPU_PREFILL_BF16=1 \
      timeout 3600 $ENG $Q4NX 1 $P/ids$n.txt > $P/sweep_${mode}_$n.log 2>&1
    boot=$(grep -oE 'boot=[0-9]+' $P/sweep_${mode}_$n.log | tail -1)
    pre=$(grep -oE 'Prefill: [0-9]+ms' $P/sweep_${mode}_$n.log | tail -1)
    echo "n=$n $mode $boot $pre"
  done
done
