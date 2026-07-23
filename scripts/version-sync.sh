#!/usr/bin/env bash
#
# version-sync.sh — stamp the root VERSION into every packaging manifest.
#
# VERSION is the single source of truth (issue #117). This script is the
# mechanism that keeps deb/snap/aur/npm/docker manifests from drifting away
# from it. It is idempotent: run it any time. CI can run
#   scripts/version-sync.sh --check
# to fail the build if any manifest has drifted from VERSION.
#
# Scheme: CalVer  YYYY.MM.DD  (optional .N suffix for a same-day re-release).
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CHECK=0
[ "${1:-}" = "--check" ] && CHECK=1

[ -f VERSION ] || { echo "version-sync: VERSION file not found" >&2; exit 1; }
V="$(tr -d '[:space:]' < VERSION)"

if ! printf '%s' "$V" | grep -Eq '^[0-9]{4}\.[0-9]{2}\.[0-9]{2}(\.[0-9]+)?$'; then
  echo "version-sync: '$V' is not valid CalVer (YYYY.MM.DD[.N])" >&2
  exit 1
fi

rc=0

# stamp <file> <sed-expr> [<sed-expr> ...]
# In --check mode, just assert the file mentions the current version.
stamp() {
  local file="$1"; shift
  [ -f "$file" ] || { echo "version-sync: skip (missing) $file" >&2; return 0; }
  if [ "$CHECK" = 1 ]; then
    if ! grep -Fq -- "$V" "$file"; then
      echo "version-sync: DRIFT — $file does not contain $V" >&2
      rc=1
    fi
    return 0
  fi
  local args=() e
  for e in "$@"; do args+=( -e "$e" ); done
  sed -i -E "${args[@]}" "$file"
}

# npm package
stamp package.json \
  "s/(\"version\"[[:space:]]*:[[:space:]]*)\"[^\"]*\"/\1\"$V\"/"

# Debian
stamp packaging/deb/DEBIAN/control \
  "s/^Version:.*/Version: $V/"
stamp packaging/deb/DEBIAN/postinst \
  "s/(1bit\.systems v)[0-9][0-9A-Za-z._-]*/\1$V/"

# Snapcraft (build + legacy manifest) and its source-tag
stamp packaging/snap/snapcraft.yaml \
  "s/^version:.*/version: '$V'/" \
  "s/^([[:space:]]*source-tag:).*/\1 v$V/"
stamp snap/snapcraft.yaml \
  "s/^version:.*/version: '$V'/"

# Arch (AUR)
stamp packaging/aur/PKGBUILD \
  "s/^pkgver=.*/pkgver=$V/"

# Packaging README: title, release-download refs, docker image tags
stamp packaging/README.md \
  "s/v[0-9]{4}\.[0-9]{2}\.[0-9]{2}[0-9A-Za-z._-]*/v$V/g" \
  "s/:[0-9]{4}\.[0-9]{2}\.[0-9]{2}[0-9A-Za-z._-]*/:$V/g"

if [ "$CHECK" = 1 ]; then
  if [ "$rc" = 0 ]; then
    echo "version-sync: all manifests match VERSION ($V)"
  fi
  exit "$rc"
fi

echo "version-sync: stamped $V into all packaging manifests"
