#!/usr/bin/env bash
# update_stats.sh
#
# Reference script: run this periodically (e.g. via cron) on the machine
# that rsyncs the TUH EEG Corpus, AFTER each sync completes. It writes
# aggregate, non-identifying counts to stats.json and pushes it, which
# triggers an automatic Cloudflare Pages redeploy of eeg-medical.1bit.systems.
#
# IMPORTANT: this script must never copy, list, or publish any raw EEG
# files, filenames, or patient/subject metadata — only aggregate counts.
# This is required by Temple University's TUH EEG data use agreement,
# which prohibits redistributing the corpus itself.
#
# Recommended: run this at most a few times per day. Every push triggers
# this repo's full CI suite (C++ build, CodeQL, etc.), so avoid pushing
# on every rsync run if that happens frequently.

set -euo pipefail

REPO_DIR="${REPO_DIR:-$HOME/1bit-systems}"
DATA_DIR="${DATA_DIR:-/data/tuh_raw}"
TOTAL_AVAILABLE="${TOTAL_AVAILABLE:-26846}"
STATS_FILE="$REPO_DIR/site/eeg-medical/stats.json"

cd "$REPO_DIR"
git fetch origin main
git checkout main
git pull --ff-only

RECORDINGS=$(find "$DATA_DIR" -type f -name '*.edf' 2>/dev/null | wc -l | tr -d ' ')
GB=$(du -sBG "$DATA_DIR" 2>/dev/null | cut -f1 | tr -d 'G' || echo 0)
NOW=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

cat > "$STATS_FILE" <<EOF
{
  "recordings_synced": $RECORDINGS,
  "total_available": $TOTAL_AVAILABLE,
  "gb_synced": $GB,
  "last_sync_utc": "$NOW"
}
EOF

git add "$STATS_FILE"
if git diff --cached --quiet; then
  echo "No change in stats, skipping commit."
  exit 0
fi

git commit -m "chore(eeg-medical): update sync stats ($RECORDINGS recordings, $NOW)"
git push origin main
