#!/usr/bin/env bash
# jarvis_mesh_smoke.sh — JARVIS fleet dispatch smoke test.
#
# Proves "DSH awareness": a JARVIS install with NO local model announces
# itself on the mesh, discovers sibling installs, and dispatches every LLM
# turn to the machine that serves the requested model. Also proves the DSH
# brain path: POST /v1/jarvis/turn drives JARVIS from outside, and
# jarvis-brain.js routes capability-aware across the fleet.
#
# Usage: bash Testing/jarvis_mesh_smoke.sh [path/to/1bit] [path/to/mesh_peer]
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ONEBIT="${1:-$ROOT/build/1bit}"
MESH_PEER="${2:-$ROOT/build/mesh_peer}"
ALICE_PORT=18088
BOB_PORT=18089
JARVIS_PORT=18081
STATE_A="$(mktemp -d /tmp/jarvis-smoke-a.XXXXXX)"
STATE_B="$(mktemp -d /tmp/jarvis-smoke-b.XXXXXX)"
STATE_J="$(mktemp -d /tmp/jarvis-smoke-j.XXXXXX)"
LOG_J="$(mktemp /tmp/jarvis-smoke-j.log.XXXXXX)"
PIDS=()

pass() { echo "  ✅ $1"; }
fail() { echo "  ❌ $1"; exit 1; }

cleanup() {
    for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null; done
    wait 2>/dev/null
    rm -rf "$STATE_A" "$STATE_B" "$STATE_J" "$LOG_J"
}
trap cleanup EXIT

echo "== JARVIS fleet dispatch smoke test =="

if [ ! -x "$ONEBIT" ] || [ ! -x "$MESH_PEER" ]; then
    echo "  building..."
    (cd "$ROOT" && cmake --build build --target onebin mesh_peer -j"$(nproc)") || fail "build failed"
fi

# Fleet: alice serves Qwen3-4B, bob serves ZAYA1-74B (stub chat servers).
"$MESH_PEER" --name alice --port "$ALICE_PORT" --state-dir "$STATE_A" \
    --announce 1 --ttl 10 --no-agent --stub-chat --models "Qwen3-4B:stub" &
PIDS+=($!)
"$MESH_PEER" --name bob --port "$BOB_PORT" --state-dir "$STATE_B" \
    --announce 1 --ttl 10 --no-agent --stub-chat --models "ZAYA1-74B:stub" &
PIDS+=($!)
sleep 2

# JARVIS: thin fleet node, no local model, brain on the mesh.
( sleep 25 ) | timeout 30 "$ONEBIT" jarvis --mesh-dispatch --text \
    --model Qwen3-4B --port "$JARVIS_PORT" > "$LOG_J" 2>&1 &
PIDS+=($!)

# ── 1. JARVIS is on the mesh and sees the fleet ─────────────────────────
echo "== phase 1: discovery =="
for i in $(seq 1 15); do
    J_STATUS=$(curl -sf "http://127.0.0.1:$JARVIS_PORT/v1/jarvis/status" 2>/dev/null || echo "")
    SAW_ALICE=$(echo "$J_STATUS" | grep -q '"name":"alice"' && echo yes || echo no)
    SAW_BOB=$(echo "$J_STATUS" | grep -q '"name":"bob"' && echo yes || echo no)
    [ "$SAW_ALICE" = yes ] && [ "$SAW_BOB" = yes ] && break
    sleep 1
done
[ "$SAW_ALICE" = yes ] || fail "JARVIS never discovered alice — see $LOG_J"
[ "$SAW_BOB" = yes ] || fail "JARVIS never discovered bob — see $LOG_J"
pass "JARVIS discovered both fleet installs"

# ── 2. Dispatch via the DSH brain path (/v1/jarvis/turn) ────────────────
echo "== phase 2: dispatch =="
for i in $(seq 1 10); do
    TURN=$(curl -sf -X POST "http://127.0.0.1:$JARVIS_PORT/v1/jarvis/turn" \
        -H "Content-Type: application/json" \
        -d '{"text":"what can you do?"}' 2>/dev/null || echo "")
    HIT_ALICE=$(echo "$TURN" | grep -q '"node":"alice"' && echo yes || echo no)
    [ "$HIT_ALICE" = yes ] && break
    sleep 1
done
[ "$HIT_ALICE" = yes ] || fail "/v1/jarvis/turn did not dispatch to alice — $TURN"
echo "$TURN" | grep -q "stub:alice" || fail "alice's reply missing — $TURN"
pass "JARVIS dispatched a turn to alice (Qwen3-4B)"

# ── 3. Capability routing: the DSH brain picks bob for ZAYA1-74B ───────
echo "== phase 3: capability routing =="
BRAIN_OUT=$(node "$ROOT/integrations/dsh/jarvis-brain.js" --node "http://127.0.0.1:$JARVIS_PORT" \
    --say "hello from the fleet" --model ZAYA1-74B 2>/dev/null)
echo "$BRAIN_OUT" | grep -q "bob" || fail "jarvis-brain did not route ZAYA1-74B to bob — $BRAIN_OUT"
pass "jarvis-brain routed ZAYA1-74B to bob"

# ── 4. JARVIS announced itself on the mesh (other installs see it) ─────
echo "== phase 4: JARVIS is a fleet citizen =="
for _ in $(seq 1 10); do
    A_PEERS=$(curl -sf "http://127.0.0.1:$ALICE_PORT/v1/mesh/peers" 2>/dev/null || echo "")
    # JARVIS advertises the "jarvis" feature — that's its fingerprint.
    JARVIS_SEEN=$(echo "$A_PEERS" | grep -q '"features":\[[^]]*"jarvis"' && echo yes || echo no)
    [ "$JARVIS_SEEN" = yes ] && break
    sleep 1
done
[ "$JARVIS_SEEN" = yes ] || fail "alice never saw JARVIS announce itself"
pass "JARVIS announced itself; alice sees it as a peer"

echo ""
echo "════════════════════════════════════════════"
echo "  JARVIS FLEET SMOKE TEST: PASS — 4/4"
echo "════════════════════════════════════════════"
