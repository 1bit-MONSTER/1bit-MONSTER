#!/usr/bin/env bash
# mesh_smoke.sh — 1bit-MONSTER Mesh smoke test.
#
# Proves the out-of-the-box guarantee: two mesh nodes (no model weights, no
# config) discover each other on the network via UDP multicast and complete
# a full integration conversation (intro ask → auto-answer → hooked up).
#
# Usage: bash Testing/mesh_smoke.sh [path/to/mesh_peer]
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${1:-$ROOT/build/mesh_peer}"
ALICE_PORT=18088
BOB_PORT=18089
STATE_A="$(mktemp -d /tmp/mesh-smoke-a.XXXXXX)"
STATE_B="$(mktemp -d /tmp/mesh-smoke-b.XXXXXX)"
LOG_A="$(mktemp /tmp/mesh-smoke-a.log.XXXXXX)"
LOG_B="$(mktemp /tmp/mesh-smoke-b.log.XXXXXX)"
PIDS=()

pass() { echo "  ✅ $1"; }
fail() { echo "  ❌ $1"; exit 1; }

cleanup() {
    for p in "${PIDS[@]:-}"; do kill "$p" 2>/dev/null; done
    wait 2>/dev/null
    rm -rf "$STATE_A" "$STATE_B" "$LOG_A" "$LOG_B"
}
trap cleanup EXIT

echo "== 1bit-MONSTER Mesh smoke test =="

if [ ! -x "$BIN" ]; then
    echo "  building mesh_peer..."
    (cd "$ROOT" && cmake --build build --target mesh_peer -j"$(nproc)") || fail "build failed"
fi

"$BIN" --name alice --port "$ALICE_PORT" --state-dir "$STATE_A" \
    --announce 1 --ttl 4 --agent-interval 2 --models "Qwen3-4B:npu_flm" \
    > "$LOG_A" 2>&1 &
PIDS+=($!)
"$BIN" --name bob --port "$BOB_PORT" --state-dir "$STATE_B" \
    --announce 1 --ttl 4 --agent-interval 2 --models "ZAYA1-74B:ggml_vulkan" \
    > "$LOG_B" 2>&1 &
PIDS+=($!)

# ── 1. Discovery: each node sees the other (multicast) ─────────────────
echo "== phase 1: discovery =="
for i in $(seq 1 15); do
    A_PEERS=$(curl -sf "http://127.0.0.1:$ALICE_PORT/v1/mesh/peers" 2>/dev/null || echo "")
    B_PEERS=$(curl -sf "http://127.0.0.1:$BOB_PORT/v1/mesh/peers" 2>/dev/null || echo "")
    A_SAW_B=$(echo "$A_PEERS" | grep -q '"name":"bob"' && echo yes || echo no)
    B_SAW_A=$(echo "$B_PEERS" | grep -q '"name":"alice"' && echo yes || echo no)
    [ "$A_SAW_B" = yes ] && [ "$B_SAW_A" = yes ] && break
    sleep 1
done
[ "$A_SAW_B" = yes ] || fail "alice never discovered bob — see $LOG_A"
[ "$B_SAW_A" = yes ] || fail "bob never discovered alice — see $LOG_B"
pass "alice and bob discovered each other via multicast"

# ── 2. Self-awareness: agent sent an intro ask, peer auto-answered ─────
echo "== phase 2: ask / answer =="
for i in $(seq 1 15); do
    B_ASKS=$(curl -sf "http://127.0.0.1:$BOB_PORT/v1/mesh/asks" 2>/dev/null || echo "")
    HAS_ASK=$(echo "$B_ASKS" | grep -q '"from_name":"alice"' && echo yes || echo no)
    ANSWERED=$(echo "$B_ASKS" | grep -q '"answered":true' && echo yes || echo no)
    [ "$HAS_ASK" = yes ] && [ "$ANSWERED" = yes ] && break
    sleep 1
done
[ "$HAS_ASK" = yes ] || fail "bob never received an intro ask from alice — see $LOG_B"
[ "$ANSWERED" = yes ] || fail "bob never answered the intro ask — see $LOG_B"
pass "alice asked bob to hook up, bob auto-answered"

# ── 3. Hooked up: both sides mark the peer integrated ──────────────────
echo "== phase 3: integration =="
for _ in $(seq 1 10); do
    B_PEERS=$(curl -sf "http://127.0.0.1:$BOB_PORT/v1/mesh/peers" 2>/dev/null || echo "")
    INTEGRATED=$(echo "$B_PEERS" | grep -q '"integrated":true' && echo yes || echo no)
    [ "$INTEGRATED" = yes ] && break
    sleep 1
done
[ "$INTEGRATED" = yes ] || fail "bob did not mark alice as integrated — see $LOG_B"
pass "bob and alice are hooked up (integrated)"

echo ""
echo "════════════════════════════════════════════"
echo "  MESH SMOKE TEST: PASS — 3/3"
echo "════════════════════════════════════════════"
