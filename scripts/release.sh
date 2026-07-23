#!/usr/bin/env bash
#
# release.sh — cut a new 1bit.systems release.
#
# Versioning is CalVer: YYYY.MM.DD, with an optional .N suffix when there is
# more than one release on the same day. VERSION is the single source of truth;
# this script bumps it, rolls the CHANGELOG, stamps every packaging manifest
# (via version-sync.sh), then commits and tags vYYYY.MM.DD.
#
# Usage:
#   scripts/release.sh              # version = today's date (auto .N on collision)
#   scripts/release.sh 2026.08.01   # explicit version
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Require a clean tree so the release commit contains only release changes.
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "release: working tree not clean — commit or stash first" >&2
  exit 1
fi

# Resolve the version.
if [ $# -ge 1 ]; then
  V="$1"
else
  base="$(date +%Y.%m.%d)"
  V="$base"; n=1
  while git rev-parse -q --verify "refs/tags/v$V" >/dev/null 2>&1; do
    V="$base.$n"; n=$((n + 1))
  done
fi

printf '%s' "$V" | grep -Eq '^[0-9]{4}\.[0-9]{2}\.[0-9]{2}(\.[0-9]+)?$' \
  || { echo "release: '$V' is not valid CalVer (YYYY.MM.DD[.N])" >&2; exit 1; }
if git rev-parse -q --verify "refs/tags/v$V" >/dev/null 2>&1; then
  echo "release: tag v$V already exists" >&2; exit 1
fi

today="$(date +%Y-%m-%d)"
echo "Releasing v$V ($today)"

# 1. Bump the single source of truth.
printf '%s\n' "$V" > VERSION

# 2. Roll the CHANGELOG: freeze "## Unreleased" into a dated section and open a
#    fresh empty Unreleased above it.
awk -v ver="$V" -v day="$today" '
  !done && /^## Unreleased[[:space:]]*$/ {
    print "## Unreleased"
    print ""
    print "_Nothing yet._"
    print ""
    print "## " ver " — " day
    done = 1
    next
  }
  { print }
' CHANGELOG.md > CHANGELOG.md.tmp && mv CHANGELOG.md.tmp CHANGELOG.md

# 3. Stamp packaging manifests from VERSION.
scripts/version-sync.sh

# 4. Commit and tag.
git add -A
git commit -q -m "release: v$V"
git tag -a "v$V" -m "1bit.systems v$V"

cat <<EOF

✔ Released v$V — committed and tagged.
  Review:  git show v$V --stat
  Publish: git push origin main && git push origin v$V
EOF
