#!/bin/bash
# run_bench.sh — goal mu8oyo0n: PTQ1_0 decode-speed A/B (legacy pre-fix vs default post-fix) over
# the real 27B HIP forward, with Q1_0/PQ2_0 controls, in ONE triad-bracketed window.
set -u
exec 9>/tmp/prism-bench.lock
if ! flock -n 9; then echo "another bench runner holds the lock" >&2; exit 1; fi

BIN=/tmp/pfhip
BW=/tmp/hip_bw_probe
P=$HOME/models/prism/1bp
IDS=/home/bcloud/1bit-MONSTER-dddf9e/tests/prism/ppl/slice200.ids.txt
PROMPT=$(head -32 "$IDS" | tr '\n' ' ')
REPS=${REPS:-5}
NGEN=${NGEN:-128}
LOG=/tmp/ppl_work/bench_full.log
: > "$LOG"

LOCK=/tmp/1bit-npu-device.lock
TOOK=0
if [ ! -s "$LOCK" ]; then printf 'prism-bench pid=%s start=%s\n' "$$" "$(date -u +%FT%TZ)" > "$LOCK" 2>/dev/null && TOOK=1; fi
trap 'if [ "$TOOK" = 1 ] && grep -q "prism-bench pid=$$ " "$LOCK" 2>/dev/null; then : > "$LOCK"; fi' EXIT

echo "runner pid=$$ start=$(date -u +%FT%TZ) head=$(git -C /home/bcloud/1bit-MONSTER-dddf9e rev-parse --short HEAD)" | tee -a "$LOG"
echo "prompt=first 32 ids of tests/prism/ppl/slice200.ids.txt  ngen=$NGEN reps=$REPS" | tee -a "$LOG"
echo "cmd: $BIN <pack.1bp> <32 ids> --predict $NGEN   [PRISM_LEGACY_FOLD_GROUP=1 for the pre-fix grouping]" | tee -a "$LOG"
echo "--- TRIAD start ---" | tee -a "$LOG"
"$BW" 128 256 512 1024 2>&1 | tee -a "$LOG"

for name in Ternary-Bonsai-2-27B-PTQ1_0 Bonsai-27B-Q1_0 Ternary-Bonsai-27B-PQ2_0; do
    for cfg in default legacy; do
        for rep in $(seq 1 "$REPS"); do
            if [ "$cfg" = legacy ]; then
                line=$(PRISM_LEGACY_FOLD_GROUP=1 "$BIN" "$P/$name.1bp" $PROMPT --predict "$NGEN" 2>&1 | tail -1)
            else
                line=$("$BIN" "$P/$name.1bp" $PROMPT --predict "$NGEN" 2>&1 | tail -1)
            fi
            echo "$name  $cfg  rep$rep : $line" | tee -a "$LOG"
        done
    done
done

echo "--- TRIAD end ---" | tee -a "$LOG"
"$BW" 128 256 512 1024 2>&1 | tee -a "$LOG"
echo "=== bench done $(date -u +%FT%TZ) ===" | tee -a "$LOG"
