#!/bin/bash
# run_ppl_ids.sh — column 1 (our engine) PPL on the three Prism packs over ONE fixed id stream.
# Mirrors the fork's default perplexity protocol (see test_prism_ppl_ids.hip header).
# A single-instance flock guards against the duplicate-runner incident of 01:21 (two runners,
# one shared log); a second invocation now exits instead of contending for the GPU.
set -u
exec 9>/tmp/ppl-ids-runner.lock
if ! flock -n 9; then
  echo "another ppl-ids runner holds /tmp/ppl-ids-runner.lock; refusing to start a second" >&2
  exit 1
fi

BIN=/tmp/ppl_ids_v2
IDS=/home/bcloud/models/prism/eval/slice200.ids.txt
OUT=/tmp/ppl_work
mkdir -p "$OUT"
LOG="$OUT/our_ppl.log"
: > "$LOG"

# Declare the run in the shared advisory device lock (etiquette), and release on exit.
LOCK=/tmp/1bit-npu-device.lock
TOOK=0
if [ ! -s "$LOCK" ]; then
  printf 'ppl-ids pid=%s start=%s\n' "$$" "$(date +%H:%M:%S)" > "$LOCK" 2>/dev/null && TOOK=1
fi
trap 'if [ "$TOOK" = 1 ] && grep -q "ppl-ids pid=$$ " "$LOCK" 2>/dev/null; then : > "$LOCK"; fi' EXIT

echo "runner pid=$$ start=$(date -u +%FT%TZ) ids=$IDS" | tee -a "$LOG"
PACKS="Bonsai-27B-Q1_0 Ternary-Bonsai-2-27B-PTQ1_0 Ternary-Bonsai-27B-PQ2_0"
for name in $PACKS; do
  pack="$HOME/models/prism/1bp/$name.1bp"
  echo "=== $name  pack=$pack  $(date -u +%FT%TZ) ===" | tee -a "$LOG"
  /usr/bin/time -f "  wall=%es maxrss=%MkB" stdbuf -oL "$BIN" "$pack" "$IDS" 2048 0 2>&1 | tee -a "$LOG"
  echo | tee -a "$LOG"
done
echo "=== column 1 done $(date -u +%FT%TZ) ===" | tee -a "$LOG"
