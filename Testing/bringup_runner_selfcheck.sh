#!/usr/bin/env bash
# bringup_runner_selfcheck.sh — the family runner must not report a pass when it ran nothing
# (issue #2520).
#
# Why: Testing/bringup_runner.sh backs the 29 `validated` statuses in
# Testing/models_manifest.json, and nothing invokes it. On a box without the host fixtures it
# printed
#
#     0/4 generation gates passed
#     4 FAILURES
#
# where the other 25 families were never in the denominator, and with no `gate` commands and
# no fixtures at all it printed `0/0 generation gates passed` and exited **0** — "nothing ran"
# reading as "clean". Skips are now counted and printed, and no-gate-ran is exit 2.
#
# The runner's verdict and summary are pure functions so they can be exercised here without a
# manifest, a fixture, or a compiler; the last cases assert the runner actually calls them, so
# a fix that lives only in these tests cannot pass.
#
# Run: bash Testing/bringup_runner_selfcheck.sh
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNNER="$REPO/Testing/bringup_runner.sh"
[ -f "$RUNNER" ] || { echo "FAIL: no $RUNNER"; exit 1; }

fail=0
ok()  { printf '  ok   %-52s %s\n' "$1" "$2"; }
bad() { printf '  FAIL %-52s %s\n' "$1" "$2"; fail=1; }
expect() { if [ "$2" = "$3" ]; then ok "$1" "$2"; else bad "$1" "$2 != $3"; fi; }

# Sourcing must not run the host work (it compiles e2e_seq and walks the manifest).
if ! grep -qE 'BASH_SOURCE\[0\].*!= *"\$0"' "$RUNNER"; then
    echo "FAIL: $RUNNER has no sourcing guard — its helpers cannot be tested without a run"
    exit 1
fi
# shellcheck disable=SC1090
source "$RUNNER" || { echo "FAIL: cannot source $RUNNER"; exit 1; }

missing=""
for fn in gate_verdict gate_summary; do declare -F "$fn" >/dev/null || missing="$missing $fn"; done
if [ -z "$missing" ]; then ok "helper floor" "2/2 present"; else bad "helper floor" "missing:$missing"; fi

# --- the verdict: nothing ran is not a pass ---
expect "no gate ran -> exit 2"            "$(gate_verdict 0 0)"   2
expect "all that ran passed -> exit 0"    "$(gate_verdict 4 0)"   0
expect "one gate ran and passed -> 0"     "$(gate_verdict 1 0)"   0
expect "a gate failed -> exit 1"          "$(gate_verdict 4 4)"   1
expect "one failure among passes -> 1"    "$(gate_verdict 25 1)"  1

# --- the summary line must show the skips, which is where the 25 families went ---
expect "skips are in the summary" \
       "$(gate_summary 4 4 25)" "0/4 generation gates passed, 25 skipped (no fixture)"
expect "a clean run still reports skips" \
       "$(gate_summary 4 0 25)" "4/4 generation gates passed, 25 skipped (no fixture)"
expect "0/0 is possible only as a report, never a pass" \
       "$(gate_summary 0 0 29)" "0/0 generation gates passed, 29 skipped (no fixture)"

# --- and the runner must be the thing that calls them ---
wired=1
grep -q 'gate_verdict "$total" "$fail"' "$RUNNER"            || { bad "runner calls gate_verdict" "not found"; wired=0; }
grep -q 'gate_summary "$total" "$fail" "$skip"' "$RUNNER"    || { bad "runner calls gate_summary" "not found"; wired=0; }
grep -q 'skip=$((skip+1))' "$RUNNER"                         || { bad "skips are counted" "no counter"; wired=0; }
grep -q 'exit 2' "$RUNNER"                                   || { bad "nothing-ran exits 2" "no exit 2"; wired=0; }
if grep -q 'echo "$((total-fail))/$total generation gates passed"' "$RUNNER"; then
    bad "old unqualified summary is gone" 'still prints "$((total-fail))/$total"'
    wired=0
fi
[ "$wired" -eq 1 ] && ok "runner is wired to the helpers" "4 checks + the old line gone"

exit "$fail"
