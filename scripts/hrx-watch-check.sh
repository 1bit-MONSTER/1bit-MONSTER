#!/usr/bin/env bash
# hrx-watch-check.sh — the mechanical form of issue #1945's two re-engage signals.
#
# Why this exists: #1945 is a *watch* issue with two named triggers, and it has been
# hand-refreshed about nine times — including one lane correcting its own refresh as
# "duplicating the standing watch refresh", because eight prior comments already did
# the same two API queries. A watch that depends on someone remembering to repeat a
# query is a watch that reports on attention, not on upstream. This makes the query
# the cheap part so a comment can carry only a *change*.
#
# The two signals, as the issue names them:
#   1. a stable ROCm/hrx-system release with an ABI/version story  (last: v0.3.0, 2026-05-30)
#   2. llama.cpp PR #27218 moving past draft
#
# Exit codes, deliberately three-valued — "could not determine" must never be
# reported as "no move", which is the failure mode this whole series of checks exists
# to stop:
#   0  neither signal moved
#   1  a signal moved -> re-engage #1945 (recipe: docs/research/hrx-engine-goal.md §P2)
#   2  could not determine (missing gh, auth, network) — none of which is "no move"
#
# Read-only: two GETs. It posts nothing; a red workflow run is the alert.
set -uo pipefail

# Baselines are recorded, not inferred, so "moved" is a comparison rather than a
# judgement. Update them only when the issue's own state is updated.
HRX_REPO="ROCm/hrx-system"
HRX_BASELINE_TAG="v0.3.0"          # latest release as of 2026-05-30
LLAMA_REPO="ggml-org/llama.cpp"
PR_NUMBER="27218"
PR_BASELINE_DRAFT="true"           # still a draft as of 2026-09-03T19:49:22Z
PR_BASELINE_UPDATED="2026-09-03T19:49:22Z"

say() { printf '%s\n' "$*"; }

if ! command -v gh >/dev/null 2>&1; then
    say "hrx-watch-check: no 'gh' on PATH — cannot judge (exit 2, NOT 'no move')" >&2
    exit 2
fi

# ── Signal 1: a release newer than the baseline ────────────────────────────────
rel_json="$(gh api "repos/${HRX_REPO}/releases/latest" 2>/dev/null)" || rel_json=""
if [ -z "$rel_json" ]; then
    say "hrx-watch-check: could not read ${HRX_REPO}/releases/latest (exit 2)" >&2
    exit 2
fi
rel_tag="$(printf '%s' "$rel_json" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("tag_name",""))')"
rel_date="$(printf '%s' "$rel_json" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("published_at",""))')"
[ -n "$rel_tag" ] || { say "hrx-watch-check: release JSON had no tag_name (exit 2)" >&2; exit 2; }

say "signal 1  ${HRX_REPO} latest release : ${rel_tag}  (${rel_date})"
say "          baseline                  : ${HRX_BASELINE_TAG}"

# ── Signal 2: the draft flag, or a state change ────────────────────────────────
pr_json="$(gh api "repos/${LLAMA_REPO}/pulls/${PR_NUMBER}" 2>/dev/null)" || pr_json=""
if [ -z "$pr_json" ]; then
    say "hrx-watch-check: could not read ${LLAMA_REPO} PR #${PR_NUMBER} (exit 2)" >&2
    exit 2
fi
read -r pr_state pr_draft pr_updated <<EOF
$(printf '%s' "$pr_json" | python3 -c '
import json,sys
d = json.load(sys.stdin)
print(d.get("state",""), str(d.get("draft","")).lower(), d.get("updated_at",""))')
EOF

say "signal 2  ${LLAMA_REPO} PR #${PR_NUMBER}     : state=${pr_state} draft=${pr_draft} updated=${pr_updated}"
say "          baseline                  : state=open draft=${PR_BASELINE_DRAFT} updated=${PR_BASELINE_UPDATED}"

moved=""
# Signal 1 fires on any release that is not the baseline one.
[ "$rel_tag" = "$HRX_BASELINE_TAG" ] || moved="${moved} release:${rel_tag}"
# Signal 2 fires on the substantive change the issue names — out of draft, or no
# longer open. A *touch* (updated_at moving) is reported but does not fire: comments
# on a draft are not "moving past draft", and firing on them would make the watch as
# noisy as the hand-refreshes it replaces.
[ "$pr_draft" = "$PR_BASELINE_DRAFT" ] || moved="${moved} pr-draft:${pr_draft}"
[ "$pr_state" = "open" ] || moved="${moved} pr-state:${pr_state}"

say ""
if [ -n "$moved" ]; then
    say "MOVED:${moved}"
    say "  -> re-engage #1945: per-family probe + in-process benchmark"
    say "     recipe: docs/research/hrx-engine-goal.md §P2"
    exit 1
fi
if [ "$pr_updated" != "$PR_BASELINE_UPDATED" ]; then
    say "note: PR #${PR_NUMBER} was touched since the baseline (updated=${pr_updated}) —"
    say "      not a trigger; the draft flag is what the issue names."
fi
say "no move: neither signal has fired (this is the expected steady state)"
exit 0
