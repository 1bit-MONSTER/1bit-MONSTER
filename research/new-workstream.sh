#!/usr/bin/env bash
# Scaffold a new research workstream from the template.
# Usage: ./new-workstream.sh wsNN-<name>  (e.g. ./new-workstream.sh ws11-fp4-format)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAME="${1:?usage: new-workstream.sh wsNN-<name>}"

if [[ ! "$NAME" =~ ^ws[0-9]{2}- ]]; then
  echo "error: name must look like wsNN-<slug> (got '$NAME')" >&2
  exit 1
fi

TARGET="$ROOT/$NAME"
if [ -e "$TARGET" ]; then
  echo "error: $TARGET already exists" >&2
  exit 1
fi

cp -r "$ROOT/templates/workstream" "$TARGET"
sed -i '/TEMPLATE-NOTES/d' "$TARGET/README.md"
sed -i "s/^# WS-NN — <Name>/# $NAME — <Name>/" "$TARGET/README.md"

TRACKING="$ROOT/TRACKING.md"
if [ -f "$TRACKING" ]; then
  sed -i "s/^## Task detail/## Task detail\n\n### $NAME\n- [ ] P0: —\n- [ ] P1: —/" "$TRACKING"
  echo "registered in TRACKING.md"
fi

echo "created $TARGET"
echo "next: fill in README.md (goal, papers, theory, tasks, validation); update research/README.md + PLAN.md"
