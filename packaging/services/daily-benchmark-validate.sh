#!/usr/bin/env bash
# daily-benchmark-validate.sh — re-measure the 5 hardware-verified benchmark
# claims and publish the result as a PR against origin/main.
#
# Runs entirely inside a dedicated worktree/branch (benchmarks/auto-update)
# so it never touches whatever is checked out in the main working copy.
# Force-pushes a single fresh commit each run: rebuilt from current
# origin/main plus today's measurement, so the branch never accumulates
# history and a still-open PR just gets its diff updated.
#
# Within tolerance  -> commit + push + open/refresh the PR.
# Drift or a failed  -> no publish; open/comment on a tracking issue instead.
#   benchmark run
# No GPU present     -> log and exit clean (not a claims failure).
set -euo pipefail

REPO="${REPO:-$HOME/1bit-MONSTER-benchmarks}"
TOOLS_SRC="${TOOLS_SRC:-$HOME/1bit-benchmark-tools}"
BRANCH="benchmarks/auto-update"
TODAY="$(date -u +%F)"

# Files this pipeline owns, relative to $REPO. None of these exist on
# origin/main yet, so `git checkout -B ... origin/main` below wipes them from
# the worktree every run -- $TOOLS_SRC/repo-files mirrors this same relative
# layout as a durable copy (plain files, not a git checkout) that survives
# the reset and gets restaged fresh each run.
MANAGED_FILES=(
    tools/validate_claims.py
    tools/sync_numbers.py
    packaging/services/daily-benchmark-validate.sh
    packaging/services/1bit-benchmark-validate.service
    packaging/services/1bit-benchmark-validate.timer
)

cd "$REPO"
echo "=== $(date -u +%FT%TZ) daily benchmark validate ==="

git fetch origin main
# -f: discard any leftover local modifications to tracked files (e.g. a
# stray manual test run) rather than aborting -- this worktree is dedicated
# to this job, nothing here is meant to be preserved across runs.
git checkout -f -B "$BRANCH" origin/main

cp -r "$TOOLS_SRC/repo-files/." .

cmake --build build --target bench_sherry bench_prefill_variants -j"$(nproc)"

set +e
python3 tools/validate_claims.py --build-dir build --repeat 3 --warmup 1 --update --json /tmp/1bit-validate-report.json
rc=$?
set -e

if [ "$rc" -eq 2 ]; then
    echo "hardware unavailable this run -- nothing measured, nothing published"
    exit 0
fi

if [ "$rc" -ne 0 ]; then
    echo "drift or benchmark failure (exit $rc) -- opening/updating an issue instead of publishing"
    body="$(python3 - <<'PY'
import json
r = json.load(open("/tmp/1bit-validate-report.json"))
lines = [
    "Automated daily re-measure (tools/validate_claims.py) found a published",
    "benchmark claim drifted beyond the 15% tolerance, or a benchmark run failed.",
    "",
]
for d in r.get("drifted", []):
    lines.append(f"- DRIFT: {d}")
for f in r.get("failures", []):
    lines.append(f"- FAILURE: {f}")
if r.get("crashes"):
    lines.append("")
    lines.append("Crashes (retried once each; did not by themselves cause this):")
    for c in r["crashes"]:
        lines.append(f"- {c}")
print("\n".join(lines))
PY
)"
    existing="$(gh issue list --search '"[benchmark drift]" in:title is:open' --json number --jq '.[0].number' 2>/dev/null || true)"
    if [ -n "$existing" ]; then
        gh issue comment "$existing" --body "$body"
    else
        gh issue create --title "[benchmark drift] daily re-measure $TODAY" --body "$body"
    fi
    exit 0
fi

python3 tools/sync_numbers.py

if git diff --quiet -- benchmarks/latest.json site/numbers.json site/index.html site/benchmarks.html; then
    echo "no change vs current origin/main -- nothing to publish"
    exit 0
fi

git add benchmarks/latest.json site/numbers.json site/index.html site/benchmarks.html "${MANAGED_FILES[@]}"
git commit -m "chore(benchmarks): daily re-measure $TODAY

Automated via tools/validate_claims.py --update. All published claims
re-measured within the 15% tolerance this run. See benchmarks/latest.json
for per-key medians and spread."

git push --force-with-lease origin "$BRANCH:$BRANCH"

# gh pr view/create's branch-name matching is unreliable from a detached,
# non-interactive context; rather than gate on it, just try to create and
# swallow the (expected, common) "already exists" failure -- the push above
# already updated any existing PR's diff, so a failed create here is a no-op.
gh pr create \
    --title "chore(benchmarks): daily auto re-measure" \
    --body "Automated daily re-measurement of the 5 hardware-verified benchmark claims (tools/validate_claims.py). This branch is force-pushed fresh off current main every run, so the diff always reflects the latest measurement — safe to merge whenever, or let tomorrow's run supersede it." \
    --base main --head "$BRANCH" 2>&1 | tee /tmp/1bit-gh-pr-create.log || true
if grep -q "already exists" /tmp/1bit-gh-pr-create.log 2>/dev/null; then
    echo "PR already exists for $BRANCH -- push above already updated it"
fi

echo "done"
