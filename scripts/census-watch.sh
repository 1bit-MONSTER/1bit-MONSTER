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

set +e
python3 "${ROOT}/Testing/hf_new_models.py" "$@" 2>&1 | tee -a "${LOG}"
RC=${PIPESTATUS[0]}
set -e

if [ "${RC}" -ne 0 ]; then
    echo "census-watch: EXIT ${RC} (uncovered class -> needs a bitnet_model.h" \
         "mapping; full log: ${LOG})" | tee -a "${LOG}"
fi

exit "${RC}"
