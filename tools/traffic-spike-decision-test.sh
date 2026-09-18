#!/usr/bin/env bash
# traffic-spike-decision-test.sh — replay the traffic-alert spike decision.
#
# Why this exists (issue #2420): the workflow computed CLONES_PER_RUN for the
# alert body but never consulted it in the decision, so it filed "traffic spike"
# issues for clone rises that were themselves CI run-count rises — and then
# explained that in the same body. #2378 (3027 clones / 776 runs = 3.9 per run)
# and #2420 (3063 / 670 = 4.6) are both inside the 2.1-5.0 clones/run band that
# docs/traffic-attribution.md measured.
#
# The rule lives between the SPIKE DECISION BEGIN/END markers in
# .github/workflows/traffic-alert.yml and is pure arithmetic (no network), so
# this test EXECUTES THAT TEXT rather than re-implementing it: if the workflow's
# rule and this test ever disagree, that is impossible by construction. It only
# proves the rule — it cannot prove the workflow's API plumbing.
#
# The cases below use the numbers recorded in docs/traffic-attribution.md
# (2026-08-31..09-13) and in issue #2420.
#
# Run: bash tools/traffic-spike-decision-test.sh
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WF="$REPO_ROOT/.github/workflows/traffic-alert.yml"

# 10 spaces is the YAML `run: |` indent; strip it so the block is valid bash.
# Anchor on the full marker line: a bare /SPIKE DECISION END/ also matches prose
# in the block's own comment, which silently truncated the extraction once.
DECISION="$(awk '
    /^ *# --- SPIKE DECISION BEGIN ---/ { f=1; next }
    /^ *# --- SPIKE DECISION END ---/   { f=0 }
    f { sub(/^ {10}/, ""); print }
' "$WF")"

if [ -z "$DECISION" ]; then
    echo "FAIL: no SPIKE DECISION BEGIN/END block found in $WF"
    exit 1
fi

HARNESS="$(mktemp)"
trap 'rm -f "$HARNESS"' EXIT
{
    echo '#!/usr/bin/env bash'
    printf '%s\n' "$DECISION"
    echo 'printf "DECISION_SPIKE=%s\nDECISION_ATTR=%s\n" "$SPIKE" "$SPIKE_ATTR"'
} > "$HARNESS"

pass=0; fail=0

# run_case <name> <want_spike> <want_attr:yes|no> <VAR=VALUE>...
run_case() {
    local name="$1" want="$2" want_attr="$3"; shift 3
    local out spike attr
    out="$(env "$@" bash "$HARNESS" 2>&1)"
    spike="$(printf '%s\n' "$out" | sed -n 's/^DECISION_SPIKE=//p')"
    attr="$(printf '%s\n' "$out" | sed -n 's/^DECISION_ATTR=//p')"

    local ok=1 why=""
    [ "$spike" = "$want" ] || { ok=0; why="spike=$spike want=$want"; }
    if [ "$want_attr" = "yes" ] && [ -z "$attr" ]; then ok=0; why="$why attr=empty want=set"; fi
    if [ "$want_attr" = "no" ]  && [ -n "$attr" ]; then ok=0; why="$why attr=set want=empty"; fi

    if [ "$ok" = 1 ]; then
        pass=$((pass + 1)); printf '  ok   %-52s -> SPIKE=%s\n' "$name" "$spike"
    else
        fail=$((fail + 1)); printf '  FAIL %-52s -> %s\n' "$name" "$why"
    fi
}

echo "spike-decision cases (rule read from $(basename "$WF")):"

# The two false positives this change is about. Both had a raw ratio >= 2.5x,
# and both are flat in clones-per-run terms.
run_case "#2420 3063 clones / 4.6 per run (trailing 3.54)" false yes \
    YESTERDAY=3063 DAY_BEFORE=2900 AVG7=1000 CLONES_PER_RUN=4.6 TRAILING_CPR=3.54 TRAILING_RUNS=3400
run_case "#2378 3027 clones / 3.9 per run (trailing 3.54)" false yes \
    YESTERDAY=3027 DAY_BEFORE=2277 AVG7=1000 CLONES_PER_RUN=3.9 TRAILING_CPR=3.54 TRAILING_RUNS=3400

# A real spike: the per-run rate itself is elevated, so it still files.
run_case "genuine spike 4500 clones / 6.4 per run" true no \
    YESTERDAY=4500 DAY_BEFORE=1200 AVG7=1000 CLONES_PER_RUN=6.4 TRAILING_CPR=3.54 TRAILING_RUNS=3400

# No run data (attribution block failed): old behaviour must stand.
run_case "no run data -> raw rule only" true no \
    YESTERDAY=3063 DAY_BEFORE=2900 AVG7=1000 CLONES_PER_RUN= TRAILING_CPR= TRAILING_RUNS=

# HIGH_VOLUME is an absolute guard and must survive the attribution gate.
run_case ">5000 clones always files, even with flat per-run" true no \
    YESTERDAY=6000 DAY_BEFORE=2900 AVG7=1000 CLONES_PER_RUN=4.6 TRAILING_CPR=3.54 TRAILING_RUNS=3400

# Raw ratio below threshold: nothing to attribute, nothing filed.
run_case "quiet day, raw ratio 1.2x" false no \
    YESTERDAY=1200 DAY_BEFORE=1100 AVG7=1000 CLONES_PER_RUN=3.4 TRAILING_CPR=3.54 TRAILING_RUNS=3400

# The env override must be honoured (default is 1.5).
run_case "override TRAFFIC_CPR_MIN=1.2 files the #2420 case" true no \
    YESTERDAY=3063 DAY_BEFORE=2900 AVG7=1000 CLONES_PER_RUN=4.6 TRAILING_CPR=3.54 TRAILING_RUNS=3400 TRAFFIC_CPR_MIN=1.2

echo
echo "SUMMARY: pass=$pass fail=$fail"
[ "$fail" -eq 0 ] || exit 1
