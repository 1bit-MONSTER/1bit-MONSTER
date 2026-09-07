#!/bin/bash
# four_stream_decode.sh — run FOUR concurrent full decodes on one NPU, each
# on its own 2-column quarter (cols 0-1 | 2-3 | 4-5 | 6-7) via the fused
# GU→SiLU→D single-context kernels (xclbins built by
# build_zaya_fused_quarter.sh). Tests whether the runqueue co-schedules four
# disjoint quarter-array contexts (issue #2128 arc: halves co-schedule,
# full-array serializes).
#
# Requires (built on strixhalo, npu-verify tree):
#   - final_i8_MOE_FUSED_zaya_q{0..3}.xclbin + insts (build_zaya_fused_quarter.sh)
#   - npu_engine_zr1 rebuilt with the fused-mode 1-ctx patch
#     (~/1bit-MONSTER/engine/npu/build/npu_engine_zr1)
#   - zaya q4nx model
#
# Env overrides: XCLBIN_DIR BIN MODEL ITERS PROMPT
#   STOP_FLM=1 (default when flm-35b is active): stop flm for the test window
#   and restore it after. KEEP_FLM=1 disables that.
#
# Exit: 0 all streams completed with output; 1 partial; 2 fail.
set -u

XCLBIN_DIR="${XCLBIN_DIR:-$HOME/npu-verify/1bit-MONSTER/engine/npu/xclbins}"
BIN="${BIN:-$HOME/1bit-MONSTER/engine/npu/build/npu_engine_zr1}"
MODEL="${MODEL:-/home/bcloud/models/zaya1-8b-fresh.q4nx}"
IT="${ITERS:-8}"
PROMPT="${PROMPT:-9079 236761 107 2717 108 1882}"
LOG=${LOG:-/tmp}
QUARTS=(q0 q1 q2 q3)
COLS=("cols 0-1" "cols 2-3" "cols 4-5" "cols 6-7")

for q in "${QUARTS[@]}"; do
    [ -f "$XCLBIN_DIR/final_i8_MOE_FUSED_zaya_$q.xclbin" ] || { echo "MISSING xclbin $q"; exit 2; }
    [ -f "$XCLBIN_DIR/insts_i8_MOE_FUSED_zaya_$q.txt" ] || { echo "MISSING insts $q"; exit 2; }
done
[ -f "$BIN" ] || { echo "MISSING: $BIN"; exit 2; }
[ -f "$MODEL" ] || { echo "MISSING: $MODEL"; exit 2; }

restore_flm=0
if systemctl --user is-active flm-35b >/dev/null 2>&1; then
    if [ "${KEEP_FLM:-0}" = "1" ]; then
        echo "note: flm-35b active and KEEP_FLM=1 — decode ctxs share the partition with flm."
    else
        echo "stopping flm-35b for the test window..."
        systemctl --user stop flm-35b && restore_flm=1
        sleep 2
    fi
fi

echo "== four-stream decode: q0(0-1) || q1(2-3) || q2(4-5) || q3(6-7), one NPU =="
echo "model: $MODEL  iters=$IT"

pids=()
logs=()
for i in "${!QUARTS[@]}"; do
    q="${QUARTS[$i]}"
    xp="$XCLBIN_DIR/final_i8_MOE_FUSED_zaya_$q.xclbin"
    fi="$XCLBIN_DIR/insts_i8_MOE_FUSED_zaya_$q.txt"
    log="$LOG/four_$q.log"
    logs+=("$log")
    ( cd "$(dirname "$BIN")" && exec env NPU_FUSED=1 \
        NPU_FUSED_XCLBIN="$xp" NPU_FUSED_INSTS="$fi" \
        timeout "$(( 90 + IT * 45 ))" "$BIN" "$MODEL" $PROMPT \
        > "$log" 2>&1 ) &
    pids+=($!)
    echo "started $q pid ${pids[$i]} → $log"
done

total="${#QUARTS[@]}"
ok=0
for i in $(seq 1 60); do
    sleep 10
    n=0
    for log in "${logs[@]}"; do
        grep -q "perf]" "$log" 2>/dev/null && n=$((n+1))
    done
    [ "$n" -eq "$total" ] && { ok=1; break; }
    alive=0
    for pid in "${pids[@]}"; do
        kill -0 "$pid" 2>/dev/null && alive=$((alive+1))
    done
    if [ "$alive" -eq 0 ] && [ "$n" -lt "$total" ]; then
        echo "all died early ($n/$total done) — see logs"
        break
    fi
done

for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null; done
pkill -9 -f "npu_engine_zr1" 2>/dev/null   # clear any stragglers

echo
echo "== per-stream results =="
done=0
for i in "${!QUARTS[@]}"; do
    q="${QUARTS[$i]}"
    log="${logs[$i]}"
    if grep -q "perf]" "$log" 2>/dev/null; then
        done=$((done+1))
        echo "-- $q (${COLS[$i]}) --"
        grep -E "fused mode|corr|perf]" "$log" | tail -2
        grep -E "^[0-9]+ " "$log" | tail -1 | head -c 120; echo
    else
        echo "-- $q (${COLS[$i]}): NO perf marker --"
        tail -2 "$log" | head -c 300; echo
    fi
done

if [ "$restore_flm" = 1 ]; then
    echo "restoring flm-35b..."
    systemctl --user start flm-35b
fi
[ "$ok" = 1 ] && [ "$done" -eq "$total" ] && echo "RESULT: ALL $total STREAMS DECODED CONCURRENTLY" && exit 0
echo "RESULT: PARTIAL — done=$done/$total"
exit 1
