#!/bin/bash
# flm_slot_ceiling_idle_probe.sh — resident slot ceiling with the NPU otherwise IDLE
# (production MoE stopped). Loads qwen3:0.6b servers one at a time, probes that each
# actually serves a request (not merely starts), records live hwctx + per-slot RSS.
# Stops at the first slot that cannot serve. Complements flm_slot_ceiling_probe.sh,
# which measured the same thing with the 35B MoE resident.
#
# Run on strixhalo:  MAXN=22 bash flm_slot_ceiling_idle_probe.sh
# Raw log: /tmp/slot_ceiling_idle.log
#
# NOTE on the probe verdict below: the driver failure on an exhausted slot surfaces on
# the FIRST INFERENCE, not on startup — the server still answers GET /v1/models (200)
# and returns a JSON body with decoding_speed_tps=0. So a 0.0 reading is a FAILURE,
# and the authoritative evidence is the server log line:
#   [ERROR] Failed to load model: DRM_IOCTL_AMDXDNA_CREATE_HWCTX IOCTL failed (err=-2)
set -u
BASE=8219
MAXN=${MAXN:-22}
TAG=qwen3:0.6b
LOG=/tmp/slot_ceiling_idle.log
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
echo "== baseline: $(ctxdetail) hwctx=$(ctxcount) (NPU IDLE, production MoE stopped) hwctx_limit=$(cat /sys/module/amdxdna/parameters/hwctx_limit) ==" | tee -a $LOG
for i in $(seq 1 $MAXN); do
    p=$((BASE + i))
    /opt/fastflowlm/bin/flm serve "$TAG" --port $p > /tmp/slotidle_$p.log 2>&1 &
    pid=$!
    ready=no
    for t in $(seq 1 60); do curl -sf -m 2 "http://127.0.0.1:$p/v1/models" >/dev/null 2>&1 && { ready=yes; break; }; sleep 1
    	kill -0 $pid 2>/dev/null || break
    done
    if [ "$ready" != yes ]; then
        echo "  slot $i (:${p}) FAILED to start" | tee -a $LOG
        err=$(grep -aiE "error|fail|hwctx|context" /tmp/slotidle_$p.log | tail -2 | tr '\n' ' ')
        echo "     log: ${err:0:200}" | tee -a $LOG
        kill $pid 2>/dev/null; break
    fi
    res=$(probe $p)
    c=$(ctxcount); d=$(ctxdetail)
    rss=$(ps -o rss= -p $pid 2>/dev/null | awk '{printf "%.2f GB", $1/1048576}')
    echo "  slot $i (:${p}) start=ok probe=${res}  hwctx=$c  rss=$rss" | tee -a $LOG
    if [ "${res%% *}" != ok ] || [ "${res##* }" = "0.0" ]; then
        echo "     probe failed (0.0 tok/s or error) -> ceiling for serving" | tee -a $LOG
        pids+=($pid); ports+=($p); n=$i; break
    fi
    pids+=($pid); ports+=($p); n=$i
done

echo | tee -a $LOG
echo "== resident slots that serve: $n | hwctx now: $(ctxcount) ($(ctxdetail)) ==" | tee -a $LOG
echo "== total RSS of my $n servers: $(ps -o rss= -p $(IFS=,; echo "${pids[*]}") 2>/dev/null | awk '{s+=$1} END {printf "%.1f GB", s/1048576}') ==" | tee -a $LOG
for p in "${ports[@]}"; do grep -aiE "hwctx|context limit|Cannot allocate|error" /tmp/slotidle_$p.log | tail -1 | sed "s/^/   :$p /" | tee -a $LOG; done
echo "== not stopping the slots (caller decides); PIDs: ${pids[*]} ==" | tee -a $LOG
