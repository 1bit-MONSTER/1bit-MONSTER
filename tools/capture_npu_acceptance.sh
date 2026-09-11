#!/usr/bin/env bash
# capture_npu_acceptance.sh — run the goal mtvd3pmx literal acceptance ON AN NPU BOX
# and KEEP THE WHOLE TRANSCRIPT. Run this instead of a bare `1bit unified ...`.
#
# WHY THIS EXISTS
# ADR §9.8.2 recorded this run as three lines:
#
#     content: ''            finish_reason: error
#     Active backend: npu_xrt (AMD XDNA NPU via native worker engine)   ~0 tok/s
#
# That was enough to declare the literal clause unsatisfiable, and not enough to say
# WHY. The backend-selection lines that would answer it — "BackendManager: trying X...",
# "→ creation failed", "→ skipped (issue #1427)", "Router: <reason>", "→ not registered
# with per-token router" — were never captured. Two later attempts to explain the
# outcome from those three lines were both wrong (ADR §9.10.5 and its §9.10.6
# correction), because the three lines do not identify a mechanism: the quoted string
# has no "✓" prefix, so it is the manager's status field (backend_manager.cpp:1597),
# not the startup banner (tools/unified_server.cpp:1764). Capture the transcript first;
# infer afterwards.
#
# WHAT IT ANSWERS
#   - did `BackendManager: trying npu_flm...` appear, and what did it print after it?
#   - which lane did `Router:` name, and did the post-init selection loop find it?
#   - is the run's `Active backend` a selection or a default?
#
# Usage:  ./tools/capture_npu_acceptance.sh <model.q4nx|model-id> [port] [binary]
# Output: <outdir>/<timestamp>/ ; prints the path of SUMMARY.md at the end.
#
# NOT run by the author: written on a box with no NPU and no build tree. Syntax-checked
# with `bash -n` only. Read it before trusting it.
set -uo pipefail

MODEL="${1:?usage: $0 <model.q4nx|model-id> [port] [binary]}"
PORT="${2:-8183}"
BIN="${3:-./build/1bit}"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="${OUTDIR:-$HOME/acceptance-captures}/$STAMP"

# Check the binary BEFORE creating the output directory, so a typo does not leave an
# empty capture dir behind that a later reader could mistake for a run that happened.
if [ ! -x "$BIN" ]; then
    echo "FATAL: binary '$BIN' not executable. Build first, or pass it as \$3." >&2
    exit 2
fi

mkdir -p "$OUT"
LOG="$OUT/server.log"
: > "$LOG"

echo "== capture: $OUT"
echo "== model  : $MODEL"
echo "== port   : $PORT"
echo "== binary : $BIN"

# ── Preflight: the paths that silently decide WHICH LANE COMES UP ──────────
# ADR §9.10.8. On strixhalo a bare `1bit unified -m <native .q4nx>` loses the FLM lane
# before any routing happens: the engine binary bakes
# FLM_CONFIG_PATH=/opt/fastflowlm/etc/flm/model_list.json, while the FastFlowLM package
# ships that registry at /opt/fastflowlm/share/flm/model_list.json. npu_flm::init does
# access(flm_config_, R_OK) and returns false when it is missing, so the legacy npu_xrt
# lane becomes the only functional accelerator and is selected. Its worker then
# resolves as `./npu_engine_universal` (or `build/`), while the box builds it at
# `build/engine/npu/npu_engine_universal` — so the worker exec fails at request time.
# Neither failure is a routing decision, and both present as "served at ~0 tok/s".
# Print the candidates BEFORE launching so the log explains itself.
{
    echo "== preflight (paths that decide the lane) =="
    for p in "${NPU_FLM_BIN:-/opt/fastflowlm/bin/flm}" \
             "${NPU_FLM_CONFIG:-/opt/fastflowlm/etc/flm/model_list.json}" \
             /opt/fastflowlm/share/flm/model_list.json \
             "${NPU_FLM_XCLBINS:-/opt/fastflowlm/share/flm/xclbins}" \
             "${NPU_ENGINE_BIN:-./npu_engine_universal}" \
             build/npu_engine_universal \
             build/engine/npu/npu_engine_universal \
             /opt/rocm/bin/flm /opt/rocm/etc/flm/model_list.json; do
        [ -e "$p" ] && st=EXISTS || st=MISSING
        printf '   %-58s %s\n' "$p" "$st"
    done
    echo "   cwd: $(pwd)"
    [ -n "${NPU_FLM_CONFIG:-}" ] || echo "   NOTE: NPU_FLM_CONFIG unset — if the FLM config above is MISSING, npu_flm::init returns false and the legacy lane is selected instead."
    [ -n "${NPU_ENGINE_BIN:-}" ] || echo "   NOTE: NPU_ENGINE_BIN unset — the npu_xrt worker resolves by cwd; check the candidates above."
} | tee -a "$LOG"

# The request must name the model the SAME way the server was told to load it (R8, and
# the refusal predicate in §9.8.1: a request whose basename equals the loaded model's
# path basename is satisfied, not refused). Using the basename keeps the two spellings
# from diverging, which is the other thing §9.8.1 was about.
REQ_MODEL="$(basename "$MODEL")"

# NOTE ON THE CHILD PROCESSES: npu_flm spawns an `flm serve` child, and those have been
# observed to orphan (see the log entries about five orphaned `flm serve` on strixhalo).
# We kill the PID we started, then REPORT any survivors rather than pkill-ing a pattern
# that may match a server some other lane is using.
pre_flm="$(pgrep -fc 'flm serve' 2>/dev/null || echo 0)"

echo "== starting server (transcript -> $LOG)"
# APPEND, not truncate: the preflight above is already in $LOG, and the whole point of
# this script is that the evidence survives.
"$BIN" unified --port "$PORT" -m "$MODEL" >>"$LOG" 2>&1 &
PID=$!
echo "   pid $PID" | tee -a "$LOG"

cleanup() {
    if kill -0 "$PID" 2>/dev/null; then
        echo "== stopping pid $PID" | tee -a "$LOG"
        kill "$PID" 2>/dev/null
        for _ in $(seq 1 20); do kill -0 "$PID" 2>/dev/null || break; sleep 0.5; done
        kill -9 "$PID" 2>/dev/null
    fi
}
trap cleanup EXIT

# Wait for readiness without assuming how long an NPU model load takes.
ready=0
for _ in $(seq 1 180); do
    if ! kill -0 "$PID" 2>/dev/null; then
        echo "== server exited before becoming ready (see $LOG)" | tee -a "$LOG"
        break
    fi
    if curl -sf -m 3 "http://127.0.0.1:$PORT/v1/health" -o "$OUT/health.json" 2>/dev/null; then
        ready=1; break
    fi
    sleep 1
done
echo "== health ready: $ready" | tee -a "$LOG"

if [ "$ready" = 1 ]; then
    curl -s -m 20 "http://127.0.0.1:$PORT/v1/models" -o "$OUT/models.json" 2>/dev/null

    # The literal clause, asked for by the SAME spelling the loader was given.
    curl -s -m 300 "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"$REQ_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"The capital of France is\"}],\"max_tokens\":16,\"temperature\":0}" \
        -o "$OUT/completion.json" 2>/dev/null

    # ── NEGATIVE CONTROL 1 — refusal must exist ————————————————————
    # An id that cannot exist must be REFUSED, not answered by whatever happens to be
    # loaded. This is the fault §9.8.1 fixed; without this line a run cannot tell
    # "served the artifact" from "served something".
    curl -s -m 60 -o "$OUT/refusal.json" -w '%{http_code}\n' \
        "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
        -d '{"model":"__no_such_artifact__.q4nx","messages":[{"role":"user","content":"x"}],"max_tokens":1}' \
        > "$OUT/refusal.code" 2>/dev/null

    # ── NEGATIVE CONTROL 2 — the weight-discriminating one (ADR §9.10.9) ————
    # Content agreement is NOT evidence of artifact-level service on a lane that
    # receives a TAG instead of the file (npu_flm execs `flm serve <tag>`:
    # src/backend_npu_flm.cpp:321). Set CONTROL_MODEL to a native artifact whose tag
    # maps to DIFFERENT weights — on strixhalo, zaya1-8b.q4nx → qwen3:1.7b — and an
    # ANSWER here means some lane served weights other than the requested ones. Only a
    # refusal demonstrates artifact-level service.
    if [ -n "${CONTROL_MODEL:-}" ]; then
        curl -s -m 300 -o "$OUT/control.json" -w '%{http_code}\n' \
            "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
            -d "{\"model\":\"$(basename "$CONTROL_MODEL")\",\"messages\":[{\"role\":\"user\",\"content\":\"The capital of France is\"}],\"max_tokens\":16,\"temperature\":0}" \
            > "$OUT/control.code" 2>/dev/null
    fi

    # The status field that §9.8.2 quoted — captured from the live server, so we can
    # compare it against the startup banner in the same run.
    { echo "--- /v1/models"; cat "$OUT/models.json"; echo; } > "$OUT/status.txt" 2>/dev/null
fi

# The part that was lost last time, extracted so it cannot be missed.
{
    echo "# Acceptance capture $STAMP"
    echo
    echo "- model: \`$MODEL\`  (requested as \`$REQ_MODEL\`)"
    echo "- port: $PORT   binary: \`$BIN\`   health ready: $ready"
    echo "- full transcript: \`server.log\` (this is the artifact §9.8.2 did not keep)"
    echo
    echo "## Backend selection (the lines that answer \"why this lane\")"
    echo '```'
    grep -nE 'Router:|BackendManager: trying|→|->|skipped|creation failed|not registered|Active backend|Primary:|init (threw|timed out)|❌' "$LOG" | head -80
    echo '```'
    echo
    echo "## Completion (the literal clause)"
    echo '```'
    head -c 1200 "$OUT/completion.json" 2>/dev/null || echo "(no completion captured)"
    echo
    echo '```'
    echo
    echo "## Controls (a passing completion proves less than it looks like)"
    echo '```'
    printf 'nonexistent id  -> HTTP %s   (must be 404 model_not_loadable_on_this_face)\n' "$(cat "$OUT/refusal.code" 2>/dev/null || echo 'not captured')"
    if [ -n "${CONTROL_MODEL:-}" ]; then
        printf 'control artifact %s -> HTTP %s\n' "$CONTROL_MODEL" "$(cat "$OUT/control.code" 2>/dev/null || echo 'not captured')"
        echo "   must ALSO be a refusal: an answer means a lane served weights other than"
        echo "   the requested file (ADR §9.10.9 — npu_flm execs a tag, never a path)."
    else
        echo 'CONTROL_MODEL unset — the weight-discriminating control did NOT run.'
        echo '   Set it to a native artifact whose FLM tag maps elsewhere (strixhalo:'
        echo '   zaya1-8b.q4nx -> qwen3:1.7b). Without it, an agreeing completion cannot'
        echo '   distinguish "the artifact was served" from "a lane served its own model".'
    fi
    echo '```'
    echo
    echo "## Reading guide"
    echo "- If \`trying npu_flm\` is absent, the route never asked for it — read \`Router:\`."
    echo "- If \`trying npu_flm\` is present and followed by a failure, that failure IS the"
    echo "  answer to §9.8.2's open question (\"why was npu_flm not functional?\")."
    echo "- If npu_flm succeeded but npu_xrt is still active, the selection loop after init"
    echo "  is the seam (tools/unified_server.cpp:1735-1760: route order, then \"any"
    echo "  functional backend\" with no route re-check)."
    echo
    echo "## Orphan check"
    echo "- \`flm serve\` count before: $pre_flm ; now: $(pgrep -fc 'flm serve' 2>/dev/null || echo 0)"
    echo "- survivors are reported, not killed: a pattern kill could take down another lane's server."
} > "$OUT/SUMMARY.md"

echo
echo "== wrote:"
echo "   $OUT/SUMMARY.md"
echo "   $LOG"
