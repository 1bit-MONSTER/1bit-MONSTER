#!/usr/bin/env bash
# zoo-smoke.sh — one-command engine health check: every model in the zoo
# must answer "What is 2+2?" with something containing 4/Four.
#
# Usage: scripts/zoo-smoke.sh [port]   (default 8088)
# Starts the unified server if it isn't already listening.
set -u
PORT="${1:-8088}"
BASE="http://127.0.0.1:${PORT}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PASS=0; FAIL=0

if ! curl -sf -m 3 "$BASE/v1/health" > /dev/null 2>&1; then
    echo "starting unified server on :$PORT ..."
    (cd "$ROOT" && nohup ./build/1bit unified -p "$PORT" -w ./models \
        > /tmp/zoo-smoke-server.log 2>&1 &)
    for _ in $(seq 1 60); do
        curl -sf -m 3 "$BASE/v1/health" > /dev/null 2>&1 && break
        sleep 2
    done
    curl -sf -m 3 "$BASE/v1/health" > /dev/null 2>&1 || {
        echo "FAIL: server did not come up"; tail -5 /tmp/zoo-smoke-server.log; exit 1; }
fi

check() {
    local model="$1" pat="$2"
    local out
    out=$(curl -sf -m 150 "$BASE/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 2+2? Answer in one word.\"}],\"max_tokens\":200}" 2>/dev/null \
        | python3 -c "import json,sys; print(json.load(sys.stdin)['choices'][0]['message']['content'])" 2>/dev/null)
    if echo "$out" | grep -qiE "$pat"; then
        echo "PASS  $model  ->  $(echo "$out" | head -c 60)"
        PASS=$((PASS+1))
    else
        echo "FAIL  $model  ->  $(echo "$out" | head -c 60)"
        FAIL=$((FAIL+1))
    fi
}

check "Llama 3.2 1B Instruct"       "four|\\b4\\b"
check "Qwen3 0.6B Instruct"         "four|\\b4\\b"
check "Bonsai-1.7B-TQ2"             "four|\\b4\\b"
check "Zamba2-1.2B-Instruct-v2.Q8_0" "four|\\b4\\b"
check "Qwen3-4B"                    "four|\\b4\\b"
check "ZAYA1-8B"                  "four|\\b4\\b"

echo "----------------------------------------"
echo "zoo-smoke: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ] || exit 1
