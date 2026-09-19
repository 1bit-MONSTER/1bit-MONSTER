#!/bin/bash
# eight_models.sh — load N qwen3:0.6b FLM servers at once and measure what the box gives you:
# context accounting, per-request latency with everything resident but idle, and aggregate vs
# per-request latency at 1/2/4/8 concurrent requests. This is the multi-agent capacity question.
set -u
N=${N:-8}
BASE_PORT=8099
PROMPT="$(cat /tmp/flm_prompt1k.txt)"
MAXTOK=${MAXTOK:-64}
CLIENT=/tmp/flm_client.py

ctx() { xrt-smi examine -r aie-partitions 2>/dev/null | awk -F'|' '
    NF>3 && $2 ~ /^[0-9]+[ ]*$/ { gsub(/ /,"",$2); n[$2]++ } END { t=0; for (p in n) { t+=n[p]; printf "%s:%d ", p, n[p] } printf "= %d live\n", t }'; }
req() {  # $1 port -> prints "tok/s" on stdout
    python3 "$CLIENT" "$1" qwen3:0.6b "$PROMPT" "$MAXTOK" 2>/dev/null | python3 -c "
import json,sys
try:
    j=json.load(sys.stdin); print('%.1f' % (j.get('usage') or {}).get('decoding_speed_tps', 0) or j.get('decode_tps') or 0)
except Exception: print('0')
"
}

echo "== loading $N qwen3:0.6b servers (:${BASE_PORT}..) =="
pids=()
for i in $(seq 0 $((N-1))); do
    p=$((BASE_PORT + i))
    /opt/fastflowlm/bin/flm serve qwen3:0.6b --port $p > /tmp/flm8_$p.log 2>&1 &
    pids+=($!)
done
ok=0
for i in $(seq 0 $((N-1))); do
    p=$((BASE_PORT + i))
    for t in $(seq 1 90); do curl -sf -m 2 "http://127.0.0.1:$p/v1/models" >/dev/null 2>&1 && { ok=$((ok+1)); break; }; sleep 2; done
done
echo "  servers ready: $ok/$N"
echo "  NPU contexts now: $(ctx)"
echo "  RSS of the $N servers: $(ps -o rss= -p $(IFS=,; echo "${pids[*]}") 2>/dev/null | awk '{s+=$1} END {printf "%.1f GB", s/1048576}')"

echo
echo "== 1 model, solo (baseline) =="
solo=$(req $BASE_PORT)
echo "  decode: ${solo} tok/s"

echo
echo "== all $N resident but IDLE: one request to each, sequential =="
for i in $(seq 0 1); do
    p=$((BASE_PORT + i))
    echo "  :$p -> $(req $p) tok/s"
done
echo "  (idle peers do not cost throughput if these match solo)"

echo
for k in 2 4 8; do
    [ "$k" -gt "$N" ] && continue
    echo "== $k concurrent requests =="
    t0=$(date +%s.%N)
    for i in $(seq 0 $((k-1))); do req $((BASE_PORT + i)) > /tmp/e8_$i.out & done
    wait
    t1=$(date +%s.%N)
    sum=$(cat /tmp/e8_*.out | paste -sd+ | bc)
    per=$(paste -sd' ' /tmp/e8_*.out)
    python3 -c "
k=$k; wall=$t1-$t0; s=float('$solo' or 0); agg=float('$sum' or 0)
print('  per-request: $per tok/s   aggregate=%d tok/s' % agg)
print('  aggregate vs solo = %.2fx ; wall for %d x %d tokens = %.2fs' % (agg/s if s else 0, k, $MAXTOK, wall))
"
done

echo
echo "== stopping $N servers =="
kill "${pids[@]}" 2>/dev/null; sleep 3
echo "  NPU contexts after: $(ctx)"
