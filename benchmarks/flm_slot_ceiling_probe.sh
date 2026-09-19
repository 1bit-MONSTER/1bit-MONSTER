#!/bin/bash
# slot_ceiling.sh — how many models can be RESIDENT at once, one slot per agent. Loads
# qwen3:0.6b servers one at a time (each becomes an agent slot), and at each step records the live
# hwctx count, whether the new server actually serves a request (not merely starts), and the RSS.
# Stops at the first server that fails to come up or fails its probe.
set -u
BASE=8199            # keep clear of 8099-8106 used earlier
MAXN=${MAXN:-14}
TAG=qwen3:0.6b
LOG=/tmp/slot_ceiling.log
: > $LOG

ctxcount() { xrt-smi examine -r aie-partitions 2>/dev/null | awk -F'|' '
    NF>3 && $2 ~ /^[0-9]+[ ]*$/ { gsub(/ /,"",$2); n[$2]++ } END { t=0; for (p in n) t+=n[p]; print t+0 }'; }
ctxdetail() { xrt-smi examine -r aie-partitions 2>/dev/null | awk -F'|' '
    NF>3 && $2 ~ /^[0-9]+[ ]*$/ { gsub(/ /,"",$2); n[$2]++ } END { for (p in n) printf "%s:%d ", p, n[p] }'; }
probe() {  # $1 port -> "ok <tps>" or "FAIL"
    python3 /tmp/flm_client.py "$1" "$TAG" "Say OK" 4 2>/dev/null | python3 -c "
import json,sys
try:
    j=json.load(sys.stdin)
    if 'error' in j: print('FAIL', j['error'][:40])
    else: print('ok %.1f' % ((j.get('usage') or {}).get('decoding_speed_tps') or 0))
except Exception as e: print('FAIL', str(e)[:40])"
}

pids=(); ports=(); n=0
echo "== baseline: $(ctxdetail) (production MoE already resident)  hwctx_limit=$(cat /sys/module/amdxdna/parameters/hwctx_limit) ==" | tee -a $LOG
for i in $(seq 1 $MAXN); do
    p=$((BASE + i))
    /opt/fastflowlm/bin/flm serve "$TAG" --port $p > /tmp/slot_$p.log 2>&1 &
    pid=$!
    ready=no
    for t in $(seq 1 60); do curl -sf -m 2 "http://127.0.0.1:$p/v1/models" >/dev/null 2>&1 && { ready=yes; break; }; sleep 1
    	kill -0 $pid 2>/dev/null || break
    done
    if [ "$ready" != yes ]; then
        echo "  slot $i (:${p}) FAILED to start" | tee -a $LOG
        err=$(grep -aiE "error|fail|hwctx|context" /tmp/slot_$p.log | tail -2 | tr '\n' ' ')
        echo "     log: ${err:0:180}" | tee -a $LOG
        kill $pid 2>/dev/null; break
    fi
    res=$(probe $p)
    c=$(ctxcount); d=$(ctxdetail)
    rss=$(ps -o rss= -p $pid 2>/dev/null | awk '{printf "%.2f GB", $1/1048576}')
    echo "  slot $i (:${p}) start=ok probe=${res}  hwctx=$c  rss=$rss" | tee -a $LOG
    if [ "${res%% *}" != ok ]; then
        echo "     probe failed -> this is the ceiling for serving" | tee -a $LOG
        pids+=($pid); ports+=($p); n=$i; break
    fi
    pids+=($pid); ports+=($p); n=$i
done

echo | tee -a $LOG
echo "== resident slots: $n  |  hwctx now: $(ctxcount)  ($(ctxdetail)) ==" | tee -a $LOG
echo "== total RSS of my $n servers: $(ps -o rss= -p $(IFS=,; echo "${pids[*]}") 2>/dev/null | awk '{s+=$1} END {printf "%.1f GB", s/1048576}') ==" | tee -a $LOG
echo "== driver/firmware errors during the run ==" | tee -a $LOG
dmesg -T 2>/dev/null | tail -5 | sed 's/^/   /' | tee -a $LOG || echo "   (dmesg needs root)" | tee -a $LOG
for p in "${ports[@]}"; do grep -aiE "hwctx|context limit|Cannot allocate|error" /tmp/slot_$p.log | tail -1 | sed "s/^/   :$p /" | tee -a $LOG; done

echo | tee -a $LOG
echo "== not stopping the slots (caller decides); PIDs: ${pids[*]} ==" | tee -a $LOG
