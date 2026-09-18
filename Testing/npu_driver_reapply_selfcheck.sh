#!/usr/bin/env bash
# npu_driver_reapply_selfcheck.sh — the driver-reapply script must name the tree it installs
# from, and refuse to install the one upstream deleted (issues #2459, #2517).
#
# Why: scripts/npu-driver-reapply.sh looked for `amdxdna.ko.zst` while the running kernel
# ships `amdxdna.ko`, so its first check failed on every run and the "run this after a kernel
# update" safety net did nothing; and the tree it would have rebuilt from is the legacy
# out-of-tree one (`src/driver/amdxdna`, HEAD 2026-08-27) that upstream removed in 813e0bf on
# 2026-09-01 — the same tree the module running on this box was built from (identified by the
# compile path in its own bytes; see #2459). Nothing in the script recorded either fact.
#
# The pure helpers are sourced from the script itself, so these cases exercise the real
# code path rather than a copy: a classifier, a discovery function, and the gate that decides
# whether a source checkout is the legacy tree. The last case checks the helpers are actually
# wired into the script — a helper nobody calls is how this class of defect hides.
#
# Run: bash Testing/npu_driver_reapply_selfcheck.sh
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$REPO/scripts/npu-driver-reapply.sh"
[ -f "$SCRIPT" ] || { echo "FAIL: no $SCRIPT"; exit 1; }

T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
fail=0; skipped=0

ok()   { printf '  ok   %-56s %s\n' "$1" "$2"; }
bad()  { printf '  FAIL %-56s %s\n' "$1" "$2"; fail=1; }
skip() { printf '  skip %-56s %s\n' "$1" "$2"; skipped=$((skipped+1)); }

expect() { # expect <name> <got> <want>
    if [ "$2" = "$3" ]; then ok "$1" "$2"; else bad "$1" "$2 != $3"; fi
}

# The script must be sourceable: without its guard the host work below runs inside this
# shell (and its `exit 1` from the module check would end the selfcheck before a single case).
if ! grep -qE 'BASH_SOURCE\[0\].*!= *"\$0"' "$SCRIPT"; then
    echo "FAIL: $SCRIPT has no sourcing guard — nothing here can exercise its helpers"
    exit 1
fi
# shellcheck disable=SC1090
source "$SCRIPT" || { echo "FAIL: cannot source $SCRIPT"; exit 1; }

# --- floor: the helpers must exist before anything below means anything ---
missing=""
for fn in module_tree_of find_installed_module installed_matches_raw source_tree_of; do
    declare -F "$fn" >/dev/null || missing="$missing $fn"
done
if [ -z "$missing" ]; then
    ok "helper floor" "4/4 present"
else
    bad "helper floor" "missing:$missing"
fi

# --- module provenance, read out of the module's own bytes ---
printf 'blah\x00src/driver/amdxdna/aie2_ctx.c\x00blah'  > "$T/legacy.ko"
printf 'blah\x00drivers/accel/amdxdna/aie2_ctx.c\x00blah' > "$T/upstream.ko"
printf 'blah\x00nothing/at/all.c\x00blah'                > "$T/neither.ko"
printf 'src/driver/amdxdna and drivers/accel/amdxdna'    > "$T/both.ko"
expect "legacy compile path -> legacy"    "$(module_tree_of "$T/legacy.ko")"   legacy
expect "upstream compile path -> upstream" "$(module_tree_of "$T/upstream.ko")" upstream
expect "no compile path -> unknown"       "$(module_tree_of "$T/neither.ko")"  unknown
expect "both paths -> legacy (refusal wins)" "$(module_tree_of "$T/both.ko")"  legacy

# the classifier must read a real module, not only fixtures: this is the read that found
# the running driver's provenance in #2459.#
real_mod="$(modinfo -n amdxdna 2>/dev/null || true)"
if [ -n "$real_mod" ] && [ -f "$real_mod" ]; then
    got="$(module_tree_of "$real_mod")"
    if [ "$got" != unknown ]; then
        ok "installed module classifies" "$got  ($real_mod)"
    else
        bad "installed module classifies" "unknown for $real_mod"
    fi
else
    skip "installed module classifies" "no amdxdna module on this runner (host-only case)"
fi

# --- discovery of the installed module, either compression ---
mkdir -p "$T/onlyko" "$T/onlyzst" "$T/bothdir" "$T/none"
: > "$T/onlyko/amdxdna.ko"; : > "$T/onlyzst/amdxdna.ko.zst"
: > "$T/bothdir/amdxdna.ko"; : > "$T/bothdir/amdxdna.ko.zst"
expect "uncompressed module found"  "$(find_installed_module "$T/onlyko")"   "$T/onlyko/amdxdna.ko"
expect "compressed module found"    "$(find_installed_module "$T/onlyzst")"  "$T/onlyzst/amdxdna.ko.zst"
expect "compressed preferred"       "$(find_installed_module "$T/bothdir")"  "$T/bothdir/amdxdna.ko.zst"
if ! find_installed_module "$T/none" >/dev/null 2>&1; then
    ok "no module -> non-zero" "returned non-zero"
else
    bad "no module -> non-zero" "returned 0"
fi

# --- unsigned-install verification understands the installed form ---
cp "$T/upstream.ko" "$T/same.ko"
if installed_matches_raw "$T/same.ko" "$T/upstream.ko"; then ok "byte compare: match" "returned 0"; else bad "byte compare: match" "non-zero"; fi
if installed_matches_raw "$T/same.ko" "$T/legacy.ko"; then bad "byte compare: mismatch" "returned 0"; else ok "byte compare: mismatch" "non-zero"; fi
if command -v zstd >/dev/null 2>&1; then
    zstd -q -f -o "$T/same.ko.zst" "$T/upstream.ko"
    if installed_matches_raw "$T/same.ko.zst" "$T/upstream.ko"; then ok "zst compare: match" "returned 0"; else bad "zst compare: match" "non-zero"; fi
else
    skip "zst compare: match" "no zstd on this runner"
fi

# --- the gate: which tree a checkout is ---
mkdir -p "$T/legacy-src/src/driver/amdxdna" "$T/upstream-src/drivers/accel/amdxdna" "$T/tarball-src/src/driver/amdxdna"
: > "$T/legacy-src/src/driver/amdxdna/a.c"; : > "$T/upstream-src/drivers/accel/amdxdna/a.c"
: > "$T/tarball-src/src/driver/amdxdna/a.c"
git -C "$T/legacy-src" init -q;   git -C "$T/legacy-src" add -A >/dev/null
git -C "$T/upstream-src" init -q; git -C "$T/upstream-src" add -A >/dev/null
expect "checkout tracking src/driver -> legacy"  "$(source_tree_of "$T/legacy-src")"   legacy
expect "checkout without it -> upstream"         "$(source_tree_of "$T/upstream-src")"  upstream
expect "not a git tree -> unknown"               "$(source_tree_of "$T/tarball-src")"   unknown

# --- and the helpers must be wired into the script, or all of the above guards nothing ---
wired_ok=1
grep -q 'source_tree_of "\$SRC"' "$SCRIPT"        || { bad "gate is wired: source tree" "source_tree_of not called on \$SRC"; wired_ok=0; }
grep -q 'module_tree_of "\$KO_RAW"' "$SCRIPT"     || { bad "gate is wired: built module" "module_tree_of not called on \$KO_RAW"; wired_ok=0; }
grep -q -- '--allow-legacy-tree' "$SCRIPT"        || { bad "gate is wired: override flag" "no --allow-legacy-tree"; wired_ok=0; }
if grep -q -A3 'legacy)' "$SCRIPT" && grep -q 'exit 2' "$SCRIPT"; then
    :
else
    bad "gate is wired: cannot-determine exits 2" "no exit 2 for unknown provenance"; wired_ok=0
fi
[ "$wired_ok" -eq 1 ] && ok "gate is wired into the script" "3 call sites + override"

[ "$skipped" -gt 0 ] && printf '  (%s host-only case(s) skipped)\n' "$skipped"
exit "$fail"
