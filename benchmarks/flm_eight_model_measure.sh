#!/bin/bash
# measure8.sh — the 8 already-loaded qwen3:0.6b servers: solo, idle-peer check, and 2/4/8-way
# concurrent aggregate + per-request latency.
set -u
P="$(cat /tmp/flm_prompt1k.txt)"; MAXTOK=${MAXTOK:-64}; BASE=8099; N=${N:-8}
req() { python3 /tmp/flm_client.py "$1" qwen3:0.6b "$P" "$MAXTOK" 2>/dev/null | python3 -c "
import json,sys
try:
    j=json.load(sys.stdin); u=j.get('usage') or {}
    print('%.1f %.2f' % (u.get('decoding_speed_tps') or 0, j.get('wall') or 0))
except Exception: print('0 0')"; }
ctx() { xrt-smi examine -r aie-partitions 2>/dev/null | awk -F'|' 'NF>3 && $2 ~ /^[0-9]+[ ]*$/ {gsub(/ /,"",$2); n[$2]++} END{t=0; for(p in n) t+=n[p]; printf "%.0f", t}'; }
echo "== state: 8 x qwen3:0.6b resident + production MoE; NPU contexts = $(ctx) (hwctx_limit=16) =="
echo "== solo (one server, others idle) =="
read -r solo w < <(req $BASE); echo "  decode ${solo} tok/s (wall ${w}s)"
echo "== one request to each of 3 servers, sequential (peers idle) =="
for i in 0 1 2; do read -r r _ < <(req $((BASE+i))); echo "  :$((BASE+i)) -> ${r} tok/s"; done
for k in 2 4 8; do
  [ "$k" -gt "$N" ] && continue
  t0=$(date +%s.%N)
  for i in $(seq 0 $((k-1))); do req $((BASE+i)) > /tmp/m8_$i.out & done
  wait
  t1=$(date +%s.%N)
  python3 - "$k" "$solo" "$t0" "$t1" <<'PY'
import sys, statistics
k, solo, t0, t1 = int(sys.argv[1]), float(sys.argv[2] or 0), float(sys.argv[3]), float(sys.argv[4])
vals=[l.split() for l in (open(f"/tmp/m8_{i}.out").read().strip() for i in range(k)) if l]
rates=[float(v[0]) for v in vals if v]; walls=[float(v[1]) for v in vals if len(v)>1 and float(v[1])>0]
agg=sum(rates)
print(f"  {k}-way: per-request {[round(r,1) for r in rates]}  aggregate={agg:.0f} tok/s"
      f"  ({agg/solo:.2f}x solo)" + (f"  median per-request wall={statistics.median(walls):.2f}s" if walls else ""))
PY
done
echo "== contexts while 8 are resident = $(ctx) =="
