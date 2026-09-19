#!/bin/bash
# flm_slot_identity_probe.sh — token-identity validation for the flm slot fleet.
#
# Complements flm_slot_ceiling_idle_probe.sh (which only proves each flm slot *serves*).
# Starts flm qwen3:0.6b slots one at a time and sends each the SAME deterministic request
# (temperature 0), then compares every slot's generated text byte-for-byte with the
# serial reference (slot 1). Reports per-slot decode rate + RSS and the live hwctx count.
#
# Stopping is by exact PID (`pgrep -x flm`), never `pkill -f` (whose pattern would match
# the ssh command line). Run on strixhalo:  N=16 bash flm_slot_identity_probe.sh
set -u
BASE=8259
N=${N:-16}
TAG=qwen3:0.6b
MAXTOK=${MAXTOK:-24}
PROMPT=${PROMPT:-"Reply with exactly the two characters OK and nothing else."}
LOG=/tmp/flm_slot_identity.log
: > "$LOG"

cat > /tmp/flm_gen.py <<'PY'
import json, sys, time, urllib.request
port, model, prompt, maxtok = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                   "max_tokens": maxtok, "temperature": 0, "stream": False}).encode()
try:
    r = urllib.request.urlopen(urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"}), timeout=600)
    j = json.load(r)
    u = j.get("usage", {}) or {}
    content = (j.get("choices", [{}])[0].get("message", {}) or {}).get("content")
    print(json.dumps({"content": content, "completion_tokens": u.get("completion_tokens"),
                      "decode_tps": u.get("decoding_speed_tps")}))
except Exception as e:
    print(json.dumps({"error": repr(e)}))
PY

hwctx() { xrt-smi examine -r aie-partitions 2>/dev/null | awk -F'|' '
    NF>3 && $2 ~ /^[0-9]+[ ]*$/ { gsub(/ /,"",$2); n[$2]++ } END { t=0; for (p in n) t+=n[p]; print t+0 }'; }
cleanup() { for pid in $(pgrep -x flm); do kill "$pid" 2>/dev/null; done; }
trap cleanup EXIT

gen() { python3 /tmp/flm_gen.py "$1" "$TAG" "$PROMPT" "$MAXTOK" 2>/dev/null; }

pids=(); ports=(); ok=0; differ=0; ref=""; n=0
for i in $(seq 1 "$N"); do
    p=$((BASE + i))
    /opt/fastflowlm/bin/flm serve "$TAG" --port "$p" > "/tmp/flmid_$p.log" 2>&1 &
    pid=$!
    ready=no
    for t in $(seq 1 60); do curl -sf -m 2 "http://127.0.0.1:$p/v1/models" >/dev/null 2>&1 && { ready=yes; break; }; sleep 1
        kill -0 $pid 2>/dev/null || break
    done
    if [ "$ready" != yes ]; then
        echo "  slot $i (:${p}) FAILED to start" | tee -a "$LOG"; kill $pid 2>/dev/null; break
    fi
    r1=$(gen "$p")
    r2=$(gen "$p")
    c1=$(echo "$r1" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("content"))' 2>/dev/null)
    c2=$(echo "$r2" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("content"))' 2>/dev/null)
    tps=$(echo "$r1" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("decode_tps"))' 2>/dev/null)
    rss=$(ps -o rss= -p "$pid" 2>/dev/null | awk '{printf "%.2f GB", $1/1048576}')
    if [ $i -eq 1 ]; then
        ref="$c1"
        if [ "$c1" = "$c2" ]; then det="deterministic"; else det="NON-DETERMINISTIC"; fi
        echo "  slot 1 (reference) text=$(printf %q "$c1") tps=${tps} rss=${rss} hwctx=$(hwctx) [$det]" | tee -a "$LOG"
    else
        if [ "$c1" = "$ref" ]; then verdict=IDENTICAL; ok=$((ok+1)); else verdict=DIFFER; differ=$((differ+1)); fi
        echo "  slot $i (:${p}) text=$(printf %q "$c1") tps=${tps} rss=${rss} hwctx=$(hwctx) -> $verdict" | tee -a "$LOG"
    fi
    pids+=($pid); ports+=($p); n=$i
done

echo "== slots started: $n | identical to reference: $ok | differ: $differ ==" | tee -a "$LOG"
echo "== total RSS: $(ps -o rss= -p $(IFS=,; echo "${pids[*]}") 2>/dev/null | awk '{s+=$1} END {printf "%.1f GB", s/1048576}') ==" | tee -a "$LOG"
cleanup; sleep 5
echo "== after cleanup hwctx=$(hwctx) ==" | tee -a "$LOG"
