#!/bin/bash
# run_fix.sh — post-fix PTQ1_0 (and Q1_0/PQ2_0 regression) PPL over the canonical stream.
# Same frozen P6 protocol (n_ctx=2048, first=1024, 6 chunks, 6138 scored). Binary:
# /tmp/ppl_ids_fix (built from include/prism_engine.h with the fold-group correction).
set -u
exec 9>/tmp/ppl-fix-runner.lock
if ! flock -n 9; then echo "another fix runner holds the lock" >&2; exit 1; fi

BIN=/tmp/ppl_ids_fix
IDS=/home/bcloud/1bit-MONSTER-dddf9e/tests/prism/ppl/slice200.ids.txt
LOG=/tmp/ppl_work/fix_full.log
: > "$LOG"

LOCK=/tmp/1bit-npu-device.lock
TOOK=0
if [ ! -s "$LOCK" ]; then printf 'ppl-fix pid=%s start=%s\n' "$$" "$(date -u +%FT%TZ)" > "$LOCK" 2>/dev/null && TOOK=1; fi
trap 'if [ "$TOOK" = 1 ] && grep -q "ppl-fix pid=$$ " "$LOCK" 2>/dev/null; then : > "$LOCK"; fi' EXIT

echo "runner pid=$$ start=$(date -u +%FT%TZ) bin=$BIN ids=$IDS" | tee -a "$LOG"
# PTQ1_0 first (the pack under investigation), then the two regression packs.
for name in Ternary-Bonsai-2-27B-PTQ1_0 Bonsai-27B-Q1_0 Ternary-Bonsai-27B-PQ2_0; do
    echo "=== $name  pack=$HOME/models/prism/1bp/$name.1bp  $(date -u +%FT%TZ) ===" | tee -a "$LOG"
    /usr/bin/time -f "  wall=%es maxrss=%MkB" stdbuf -oL "$BIN" "$HOME/models/prism/1bp/$name.1bp" "$IDS" 2048 0 2>&1 | tee -a "$LOG"
    echo | tee -a "$LOG"
done
echo "=== fix run done $(date -u +%FT%TZ) ===" | tee -a "$LOG"
