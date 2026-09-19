#!/bin/bash
# run_ab.sh — full 6-chunk PTQ1_0 A/B: exact (float tile) vs dp4a (int8 activation), same stream.
set -u
exec 9>/tmp/ppl-ab-runner.lock
if ! flock -n 9; then echo "another ab runner holds the lock" >&2; exit 1; fi

PACK=/home/bcloud/models/prism/1bp/Ternary-Bonsai-2-27B-PTQ1_0.1bp
IDS=/home/bcloud/models/prism/eval/slice200.ids.txt
BIN=/tmp/ppl_ids_ab
LOG=/tmp/ppl_work/ab_full.log
: > "$LOG"

LOCK=/tmp/1bit-npu-device.lock
TOOK=0
if [ ! -s "$LOCK" ]; then printf 'ppl-ab pid=%s\n' "$$" > "$LOCK" 2>/dev/null && TOOK=1; fi
trap 'if [ "$TOOK" = 1 ] && grep -q "ppl-ab pid=$$" "$LOCK" 2>/dev/null; then : > "$LOCK"; fi' EXIT

echo "runner pid=$$ start=$(date -u +%FT%TZ)" | tee -a "$LOG"
echo "=== EXACT (PRISM_FORCE_EXACT_DOT=1) 6 chunks ===" | tee -a "$LOG"
PRISM_FORCE_EXACT_DOT=1 /usr/bin/time -f "  wall=%es" stdbuf -oL "$BIN" "$PACK" "$IDS" 2048 0 2>&1 | tee -a "$LOG"
echo | tee -a "$LOG"
echo "=== DP4A (default) 6 chunks ===" | tee -a "$LOG"
/usr/bin/time -f "  wall=%es" stdbuf -oL "$BIN" "$PACK" "$IDS" 2048 0 2>&1 | tee -a "$LOG"
echo "=== A/B done $(date -u +%FT%TZ) ===" | tee -a "$LOG"
