#!/bin/bash
# flmdual.sh — two FLM servers loaded at once: single-server baselines, then both serving at the
# same time. Answers: does FastFlowLM aggregate across two loaded models, and how does that compare
# with our engine's ~1.4-1.6x (RESULTS-npu-concurrency-config-sweep)?
set -u
cd /tmp
PROMPT_FILE=/tmp/flm_prompt.txt
python3 - <<'PY'
text = ("The history of computing is a long arc of abstraction. "
        "Mechanical calculators gave way to relay machines, then to vacuum tubes, then to transistors, "
        "then to integrated circuits, and each transition moved the same work further from the "
        "physical details that made it possible. A programmer of the 1950s toggled switches and read "
        "blinking lights; a programmer of the 2020s types into a model that predicts the next token. "
        "Between those two points sit compilers, operating systems, networks, and a great deal of "
        "specialised silicon. What has not changed is the shape of the problem: something must decide "
        "which operations to perform, in what order, and where to keep the intermediate results. "
        "Hardware accelerators make that decision concrete, because their performance depends not only "
        "on raw arithmetic but on how well the software feeding them matches their memory hierarchy "
        "and scheduling. ")
open("/tmp/flm_prompt.txt", "w").write(text * 3)
print("prompt chars:", len(text) * 3)
PY
MAXTOK=${MAXTOK:-64}
echo "== waiting for a quiet box (<=600s) =="
for i in $(seq 1 60); do
    busy=$(ps -eo args | grep -cE "c8k_guarded|npu_engine_|flm bench" )
    l=$(cut -d' ' -f1 /proc/loadavg)
    awk -v l="$l" 'BEGIN{exit !(l<8)}' && [ "$busy" -le 0 ] && { echo "  quiet after ${i} polls (load $l)"; break; }
    sleep 10
done

start() { /opt/fastflowlm/bin/flm serve "$2" --port "$1" > "/tmp/flmserve_$1.log" 2>&1 & echo $!; }
ready() { for i in $(seq 1 90); do curl -sf -m 2 "http://127.0.0.1:$1/v1/models" >/dev/null 2>&1 && return 0; sleep 2; done; return 1; }

A=$(start 8099 qwen3:0.6b); B=$(start 8100 qwen3:1.7b)
echo "== loading: A=qwen3:0.6b:8099 (pid $A)  B=qwen3:1.7b:8100 (pid $B) =="
ready 8099 && echo "  A ready" || echo "  A FAILED"
ready 8100 && echo "  B ready" || echo "  B FAILED"
xrt-smi examine -r aie-partitions 2>/dev/null | awk -F'|' 'NF>3 && $2 ~ /^[0-9]+[ ]*$/ {gsub(/ /,"",$2); n[$2]++} END{for (p in n) printf "  ctx %s:%d\n", p, n[p]}'

echo "== serial A =="; python3 /tmp/flm_client.py 8099 qwen3:0.6b "$(cat $PROMPT_FILE)" $MAXTOK
echo "== serial B =="; python3 /tmp/flm_client.py 8100 qwen3:1.7b "$(cat $PROMPT_FILE)" $MAXTOK
echo "== concurrent A+B =="
python3 /tmp/flm_client.py 8099 qwen3:0.6b "$(cat $PROMPT_FILE)" $MAXTOK > /tmp/flm_A.json &
pa=$!
python3 /tmp/flm_client.py 8100 qwen3:1.7b "$(cat $PROMPT_FILE)" $MAXTOK > /tmp/flm_B.json &
pb=$!
wait $pa $pb
cat /tmp/flm_A.json; cat /tmp/flm_B.json
python3 - <<'PY'
import json
try:
    a=json.load(open("/tmp/flm_A.json")); b=json.load(open("/tmp/flm_B.json"))
    def tps(x): return x.get("decode_tps") or 0
    print("AGGREGATE decode: %.1f tok/s  (A %.1f + B %.1f)  wall A=%.2fs B=%.2fs"%(
        tps(a)+tps(b), tps(a), tps(b), a.get("wall",0), b.get("wall",0)))
except Exception as e:
    print("aggregate parse failed:", e)
PY
kill $A $B 2>/dev/null; sleep 3
echo "servers stopped"
