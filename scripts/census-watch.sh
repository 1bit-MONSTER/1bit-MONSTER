#!/usr/bin/env bash
# census-watch.sh — daily HF new-model watcher entry point
#
# Wraps Testing/hf_new_models.py so the daily census watch is a single
# command, usable from cron, the systemd units
# (scripts/1bit-census-watch.{service,timer}) and the GitHub Actions workflow
# (.github/workflows/census-watch.yml). It exists so the automation is
# reproducible from the repo instead of living ad-hoc on one box.
#
# What it does: polls HuggingFace for the newest causal-LM / VLM models,
# fetches each config.json, strips the architecture class, and probes the REAL
# engine registry (rcpp_arch_from_string via a g++-compiled probe). Any new
# class the registry doesn't map is what silently breaks the census 100% claim
# — that is the alert.
#
# Exit codes (same contract as hf_new_models.py):
#   0  no uncovered classes among the newest models
#   1  a new model carries an architecture class the registry doesn't map
#   *  runtime failure (no network, probe compile error, ...)
#
# Usage:
#   scripts/census-watch.sh [--limit N]     # N newest models, default 120
#
# Every run is appended to $CENSUS_WATCH_LOG_DIR (default ~/.1bit/logs) AND
# echoed to stdout, so cron/systemd/journald/CI all see the same output.
# State (seen model ids) persists in Testing/hf_new_models_state.json.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${CENSUS_WATCH_LOG_DIR:-${HOME}/.1bit/logs}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="${LOG_DIR}/census-watch-${STAMP}.log"

mkdir -p "${LOG_DIR}"

echo "== census-watch $(date -u +%FT%TZ) ==" | tee "${LOG}"

# ── Optional self-refresh (opt-in: CENSUS_WATCH_REFRESH=1, set by the unit) ──
# WHY: a census checkout that drifts silently reproduces an OLD decision. On
# 2026-09-11 ryzen's timer was running a feature-branch worktree 55 commits
# behind main, so the daily alert printed the retired "add to bitnet_model.h"
# hint for four architectures that main classifies as SIGNIFICANT ("NOT an
# alias"). The automation was calling the family-variant lane on arrivals that
# need real engine architecture work.
# The guards are what make it safe to leave enabled on every box:
#   * DETACHED HEAD only — `reset --hard` would otherwise move a branch out
#     from under the lane working in it. A named branch means this is
#     someone's checkout, so we refuse and say so.
#   * no local changes apart from the census's own run state, so a detached
#     tree someone is experimenting in is never wiped.
# A refused refresh is a note, not a failure: the census still runs.
if [ "${CENSUS_WATCH_REFRESH:-0}" = "1" ]; then
    ok_refresh=1
    if [ -n "$(git -C "${ROOT}" symbolic-ref -q HEAD || true)" ]; then
        echo "[census-watch] refresh skipped — ${ROOT} is on branch" \
             "$(git -C "${ROOT}" branch --show-current); the census needs a" \
             "detached main checkout" | tee -a "${LOG}"
        ok_refresh=0
    elif [ -n "$(git -C "${ROOT}" status --porcelain -- . | grep -v -E 'Testing/(hf_new_models_state|significant_arrivals)\.json$' || true)" ]; then
        echo "[census-watch] refresh skipped — ${ROOT} has local changes" \
             | tee -a "${LOG}"
        ok_refresh=0
    fi
    if [ "${ok_refresh}" = "1" ]; then
        if git -C "${ROOT}" fetch --quiet origin main 2>/dev/null \
           && git -C "${ROOT}" reset --hard --quiet origin/main 2>/dev/null; then
            echo "[census-watch] refreshed to $(git -C "${ROOT}" log --oneline -1)" \
                 | tee -a "${LOG}"
        else
            echo "[census-watch] refresh failed (offline?) — running the" \
                 "current checkout $(git -C "${ROOT}" log --oneline -1)" \
                 | tee -a "${LOG}"
        fi
    fi
fi

set +e
python3 "${ROOT}/Testing/hf_new_models.py" "$@" 2>&1 | tee -a "${LOG}"
RC=${PIPESTATUS[0]}
set -e

if [ "${RC}" -ne 0 ]; then
    echo "census-watch: EXIT ${RC} (uncovered class -> see the class's line" \
         "above: a SIGNIFICANT arrival needs real engine support, a family" \
         "variant needs a bitnet_model.h mapping; full log: ${LOG})" | tee -a "${LOG}"
fi

exit "${RC}"
