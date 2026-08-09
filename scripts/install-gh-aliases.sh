#!/usr/bin/env bash
# Recreate the 4 local gh aliases (idempotent — re-running overwrites).
set -euo pipefail
gh alias set --shell ci 'run list --workflow=ci.yml --limit 10'
gh alias set --shell ciw 'run watch'
gh alias set --shell co 'pr checkout'
gh alias set --shell ship 'pr merge --squash --delete-branch'
echo "gh aliases installed: ci, ciw, co, ship"
