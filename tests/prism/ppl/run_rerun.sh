#!/bin/bash
# run_rerun.sh — goal mu8i2k3x: independent re-run of the PTQ1_0 fold-group investigation at the
# frozen HEAD 323bcc683. One binary, two knobs:
#   PRISM_LEGACY_FOLD_GROUP=1 reproduces the pre-fix grouping (the committed A/B binary's behaviour)
#   PRISM_FORCE_EXACT_DOT=1  selects the exact float dot (the A/B's other arm)
# Post-fix numbers are the default (no env), i.e. the committed fix.
set -u
exec 9>/tmp/ppl-rerun.lock
if ! flock -n 9; then echo "another rerun runner holds the lock" >&2; exit 1; fi

BIN=/tmp/ppl_ids_rerun
IDS=/home/bcloud/1bit-MONSTER-dddf9e/tests/prism/ppl/slice200.ids.txt
P=$HOME/models/prism/1bp
LOG=/tmp/ppl_work/rerun_full.log
: > "$LOG"

LOCK=/tmp/1bit-npu-device.lock
TOOK=0
if [ ! -s "$LOCK" ]; then printf 'ppl-rerun pid=%s start=%s\n' "$$" "$(date -u +%FT%TZ)" > "$LOCK" 2>/dev/null && TOOK=1; fi
trap 'if [ "$TOOK" = 1 ] && grep -q "ppl-rerun pid=$$ " "$LOCK" 2>/dev/null; then : > "$LOCK"; fi' EXIT

echo "runner pid=$$ start=$(date -u +%FT%TZ) head=$(git -C /home/bcloud/1bit-MONSTER-dddf9e rev-parse --short HEAD)" | tee -a "$LOG"
echo "stream md5=$(md5sum "$IDS" | cut -d' ' -f1) ids=$(wc -l < "$IDS")" | tee -a "$LOG"

echo "=== A/B EXACT (legacy grouping) ===" | tee -a "$LOG"
PRISM_LEGACY_FOLD_GROUP=1 PRISM_FORCE_EXACT_DOT=1 /usr/bin/time -f "  wall=%es" stdbuf -oL "$BIN" "$P/Ternary-Bonsai-2-27B-PTQ1_0.1bp" "$IDS" 2048 0 2>&1 | tee -a "$LOG"
echo | tee -a "$LOG"

echo "=== A/B DP4A (legacy grouping) ===" | tee -a "$LOG"
PRISM_LEGACY_FOLD_GROUP=1 /usr/bin/time -f "  wall=%es" stdbuf -oL "$BIN" "$P/Ternary-Bonsai-2-27B-PTQ1_0.1bp" "$IDS" 2048 0 2>&1 | tee -a "$LOG"
echo | tee -a "$LOG"

for name in Ternary-Bonsai-2-27B-PTQ1_0 Bonsai-27B-Q1_0 Ternary-Bonsai-27B-PQ2_0; do
    echo "=== POST-FIX $name ===" | tee -a "$LOG"
    /usr/bin/time -f "  wall=%es" stdbuf -oL "$BIN" "$P/$name.1bp" "$IDS" 2048 0 2>&1 | tee -a "$LOG"
    echo | tee -a "$LOG"
done
echo "=== rerun done $(date -u +%FT%TZ) ===" | tee -a "$LOG"
