#!/usr/bin/env bash
# version_sync_selfcheck.sh — the REQUIRED "Version consistency" check must fail when a
# manifest it covers moves out of the tree, instead of skipping it (issue #2488).
#
# Why: sync_file() began with `[ -f "$file" ] || return 0`, so a manifest that left the tree
# took its version check with it, silently. Measured before the fix: moving
# packaging/aur/PKGBUILD away left `sync-version.sh --check` at exit 0, while a
# stale-but-present manifest correctly failed. Two of the script's nine entries named
# packaging/homebrew/1bit-monster.rb, which has never existed here, so a required check was
# green over a set smaller than the one it names — and with every manifest gone it still
# exited 0, having checked nothing.
#
# The fixture is a scratch tree with the same relative layout, built from the real manifests
# and the real VERSION value, so each case is one move or one edit and no case depends on
# what the current version happens to be.
#
# Run: bash Testing/version_sync_selfcheck.sh
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$REPO/scripts/sync-version.sh"
[ -f "$SCRIPT" ] || { echo "FAIL: no $SCRIPT"; exit 1; }

# The manifests this check must cover. Literal, not parsed out of the script: two of them are
# wired through a `for f in snap/... packaging/snap/...` loop, so grepping for sync_file
# lines finds six of eight. Each is asserted below to exist AND to be named by the script, so
# dropping one from either side is reported here rather than shrinking the gate quietly.
MANIFESTS=(
    package.json
    snap/snapcraft.yaml
    packaging/snap/snapcraft.yaml
    packaging/deb/DEBIAN/control
    packaging/aur/PKGBUILD
    packaging/deb/DEBIAN/postinst
    tools/onebit.cpp
    src/onebit_c.cpp
)

T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
FIX="$T/tree"; fail=0; present=0

build() {
    rm -rf "$FIX"; mkdir -p "$FIX/scripts"
    cp "$REPO/VERSION" "$FIX/VERSION"
    cp "$SCRIPT" "$FIX/scripts/sync-version.sh"
    present=0
    local f
    for f in "${MANIFESTS[@]}"; do
        [ -f "$REPO/$f" ] || continue
        mkdir -p "$FIX/$(dirname "$f")"
        cp "$REPO/$f" "$FIX/$f"
        present=$((present+1))
    done
}

run() { bash "$FIX/scripts/sync-version.sh" "$@" 2>&1; }

expect() { # expect <name> <pass|fail> [marker in the output]
    local name="$1" want="$2" marker="${3:-}" out rc verdict=ok
    out="$(run --check)"; rc=$?
    local got=pass; [ "$rc" -eq 0 ] || got=fail
    [ "$got" = "$want" ] || verdict=BAD
    if [ "$want" = fail ] && ! printf '%s' "$out" | grep -qF "$marker"; then verdict=BAD; fi
    if [ "$verdict" = ok ]; then
        printf '  ok   %-52s rc=%s\n' "$name" "$rc"
    else
        printf '  FAIL %-52s rc=%s (wanted %s matching %s)\n' "$name" "$rc" "$want" "$marker"
        printf '%s\n' "$out" | head -3 | sed 's/^/         /'
        fail=1
    fi
}

echo "version sync: a manifest that moves away must be reported, not skipped"

# --- floor: the gate must have something to look at, on both sides ---
gone=(); uncovered=()
for f in "${MANIFESTS[@]}"; do
    [ -f "$REPO/$f" ] || gone+=("$f")
    grep -qF "$f" "$SCRIPT" || uncovered+=("$f")
done
build
if [ "${#gone[@]}" -eq 0 ] && [ "${#uncovered[@]}" -eq 0 ] \
   && [ "$present" -eq "${#MANIFESTS[@]}" ]; then
    printf '  ok   %-52s %s manifests, all named by the script\n' "coverage floor" "$present"
else
    printf '  FAIL %-52s %s/%s in the tree, uncovered: [%s], absent: [%s]\n' \
        "coverage floor" "$present" "${#MANIFESTS[@]}" "${uncovered[*]:-}" "${gone[*]:-}"
    fail=1
fi

expect "every manifest present and in sync"                pass

sed -i 's/"version": "[^"]*"/"version": "0.0.0"/' "$FIX/package.json"
expect "a drifted manifest still fails"                    fail "OUT OF SYNC: package.json"

build
rm -f "$FIX/packaging/aur/PKGBUILD"
expect "a manifest that moved away fails"                  fail "MISSING: packaging/aur/PKGBUILD"
# ...and the writer must not "repair" it by inventing an empty manifest
run >/dev/null
if [ ! -e "$FIX/packaging/aur/PKGBUILD" ]; then
    echo "  ok   write mode leaves the absent manifest absent"
else
    echo "  FAIL write mode created the absent manifest"; fail=1
fi

build
for f in "${MANIFESTS[@]}"; do rm -f "$FIX/$f"; done
expect "no manifests at all fails (nothing checked)"       fail "MISSING: package.json"

build
sed -i 's/"version": "[^"]*"/"version": "0.0.0"/' "$FIX/package.json"
run >/dev/null          # write mode
expect "write mode repairs drift (round trip)"             pass

exit "$fail"
