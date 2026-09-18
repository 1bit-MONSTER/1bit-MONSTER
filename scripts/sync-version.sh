#!/usr/bin/env bash
set -euo pipefail
# sync-version.sh — propagate the canonical version (root VERSION file) into
# every packaging manifest so they can't drift apart again (issue #117).
#
# Usage:
#   scripts/sync-version.sh            # rewrite all manifests from VERSION
#   scripts/sync-version.sh --check    # fail if any manifest is out of sync (CI)
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

VERSION="$(tr -d '[:space:]' < VERSION)"
[ -n "$VERSION" ] || { echo "VERSION file is empty" >&2; exit 1; }

MODE="${1:-write}"
fail=0

# file : sed-expression that rewrites the version line in place
sync_file() {
  local file="$1" expr="$2"
  if [ ! -f "$file" ]; then
    # In --check mode a missing manifest is a FAILURE, not a skip. This used to be
    # `[ -f "$file" ] || return 0`, so any manifest that moved out of the tree took
    # its version check with it, silently — measured before this fix: moving
    # packaging/aur/PKGBUILD away left `sync-version.sh --check` at exit 0 while a
    # stale-but-present one correctly failed. A required check that loses a target
    # without a signal is the #2476 failure mode.
    if [ "$MODE" = "--check" ]; then
      echo "MISSING: $file (the tree should carry it, at version $VERSION)"
      fail=1
    fi
    return 0
  fi
  if [ "$MODE" = "--check" ]; then
    if ! grep -Eq "$3" "$file"; then
      echo "OUT OF SYNC: $file (expected version $VERSION)"
      fail=1
    fi
  else
    sed -i -E "$expr" "$file"
    echo "synced: $file -> $VERSION"
  fi
}

# package.json  ->  "version": "<VERSION>"
sync_file package.json \
  "s/(\"version\"[[:space:]]*:[[:space:]]*\")[^\"]*(\")/\1${VERSION}\2/" \
  "\"version\"[[:space:]]*:[[:space:]]*\"${VERSION}\""

# snap manifests  ->  version: '<VERSION>'
for f in snap/snapcraft.yaml packaging/snap/snapcraft.yaml; do
  sync_file "$f" \
    "s/^(version:[[:space:]]*').*(')/\1${VERSION}\2/" \
    "^version:[[:space:]]*'${VERSION}'"
done

# deb control  ->  Version: <VERSION>
sync_file packaging/deb/DEBIAN/control \
  "s/^(Version:[[:space:]]*).*/\1${VERSION}/" \
  "^Version:[[:space:]]*${VERSION}$"

# AUR PKGBUILD  ->  pkgver=<VERSION>
sync_file packaging/aur/PKGBUILD \
  "s/^(pkgver=).*/\1${VERSION}/" \
  "^pkgver=${VERSION}$"

# Homebrew formula — the two entries that lived here are REMOVED. They checked
# `packaging/homebrew/1bit-monster.rb`, which has never existed in this repository:
# there is no .rb anywhere in the tree, nothing is tracked under packaging/homebrew/,
# and `git log --diff-filter=A --all` finds no add (the path arrived with a vendored
# packaging script, not with the formula). So both entries resolved to nothing and
# the check skipped them in silence. Now that a missing manifest is a failure they
# would fail instead — restore these two entries and the formula together, not one
# without the other:
#
#   sync_file packaging/homebrew/1bit-monster.rb \
#     "s/^(  version \")[^\"]*(\")/\1${VERSION}\2/" \
#     "^  version \"${VERSION}\"$"
#   sync_file packaging/homebrew/1bit-monster.rb \
#     "s|(tags/v)[0-9]{4}\.[0-9]{2}\.[0-9]{2}[A-Za-z0-9.-]*|\1${VERSION}|" \
#     "tags/v${VERSION}"

# deb postinst banner  ->  v<VERSION>
sync_file packaging/deb/DEBIAN/postinst \
  "s/v[0-9]{4}\.[0-9]{2}\.[0-9]{2}[A-Za-z0-9.-]*/v${VERSION}/" \
  "v${VERSION}"

# Version strings compiled into the binary — what `1bit --version` and the chat
# banner print. A drift here ships an artifact that misreports itself, and
# nothing else can see it: tools/onebit.cpp had sat at 2026.07.22 while VERSION
# and every package said 2026.08.04, because this script did not know about
# either file and a compiled constant is invisible to every other check.
sync_file tools/onebit.cpp \
  "s/(kVersion = \")[^\"]*(\")/\1${VERSION}\2/" \
  "kVersion = \"${VERSION}\""

sync_file src/onebit_c.cpp \
  "s/(kOneBitVersion = \")[^\"]*(\")/\1${VERSION}\2/" \
  "kOneBitVersion = \"${VERSION}\""

if [ "$MODE" = "--check" ] && [ "$fail" -ne 0 ]; then
  echo "Run scripts/sync-version.sh to fix." >&2
  exit 1
fi
