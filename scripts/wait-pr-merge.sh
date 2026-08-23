#!/usr/bin/env bash
set -euo pipefail
# wait-pr-merge.sh — Poll a PR until it is ready to merge or no longer relevant
#
# Replaces ad-hoc `while true; ... mergeStateStatus=CLEAN` background loops.
# Those loops hung for hours after the PR auto-merged because a merged PR
# reports mergeStateStatus=UNKNOWN, never CLEAN (see PR #1787 incident,
# Aug 2026). This script exits on ALL terminal states and has a hard timeout,
# so it can never run forever.
#
# Exit codes:
#   0  PR is ready to merge (mergeStateStatus=CLEAN) or already MERGED
#   1  PR is CLOSED without merge (aborted) — nothing to do
#   2  Timed out before reaching a terminal state
#   3  Usage / gh error
#
# Usage:
#   ./wait-pr-merge.sh <PR> [--timeout-min N] [--interval N]
#
# Examples:
#   ./wait-pr-merge.sh 1787                       # poll until CLEAN/MERGED/CLOSED
#   ./wait-pr-merge.sh 1787 --timeout-min 180     # give CI up to 3h
#   ./wait-pr-merge.sh 1787 --interval 30         # poll every 30s

PR=""
TIMEOUT_MIN=120
INTERVAL=55

while [ "$#" -gt 0 ]; do
    case "$1" in
        --timeout-min) TIMEOUT_MIN="${2:?missing value}"; shift 2 ;;
        --interval)    INTERVAL="${2:?missing value}";    shift 2 ;;
        -h|--help)     echo "usage: $0 <PR> [--timeout-min N] [--interval N]"; exit 0 ;;
        *) [ -z "$PR" ] && PR="$1"; shift ;;
    esac
done

if ! [[ "$PR" =~ ^[0-9]+$ ]]; then
    echo "usage: $0 <PR> [--timeout-min N] [--interval N]" >&2
    exit 3
fi

# Hard ceiling so the loop can never run indefinitely even if gh misbehaves.
MAX_ITER=$(( TIMEOUT_MIN * 60 / INTERVAL ))
[ "$MAX_ITER" -lt 1 ] && MAX_ITER=1

echo "[wait-pr] polling PR #$PR (timeout ${TIMEOUT_MIN}min, interval ${INTERVAL}s)"

for (( i = 1; i <= MAX_ITER; i++ )); do
    # Single gh call per iteration — one field fetch is enough for all states.
    OUT="$(gh pr view "$PR" --json state,mergeStateStatus --jq '.state + " " + .mergeStateStatus' 2>/dev/null || true)"
    STATE="${OUT%% *}"
    MERGE="${OUT##* }"

    echo "[wait-pr] $(date +%H:%M:%S) #$PR state=$STATE mergeStateStatus=${MERGE:-?}"

    case "$STATE" in
        MERGED)
            echo "[wait-pr] PR #$PR already merged — done"
            exit 0
            ;;
        CLOSED)
            echo "[wait-pr] PR #$PR closed without merge — aborting wait" >&2
            exit 1
            ;;
    esac

    case "$MERGE" in
        CLEAN)
            echo "[wait-pr] PR #$PR is CLEAN — ready to merge"
            exit 0
            ;;
        BEHIND)
            # Not terminal: branch is behind main. Report it, keep polling.
            echo "[wait-pr] PR #$PR is BEHIND main — consider updating the branch" >&2
            ;;
    esac

    sleep "$INTERVAL"
done

echo "[wait-pr] timed out after ${TIMEOUT_MIN}min — PR #$PR state=$STATE mergeStateStatus=${MERGE:-?}" >&2
exit 2
