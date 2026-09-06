#!/bin/bash
# two_stream_decode.sh — run TWO concurrent full decodes on one NPU, each on
# its own 4-column half (cols 0-3 | 4-7) via the single-context fused
# GU→SiLU→D kernels. Validates the issue-#2128 column-sliced co-schedule
# (2 contexts per partition) end to end with the real Zaya model.
#
# Requires (built on strixhalo):
#   - column-sliced fused xclbins: final_i8_MOE_FUSED_zaya_{h0,h1}.xclbin
#     + insts  (generators/build_zaya_fused_half.sh, run in the npu-verify tree)
#   - npu_engine_zr1 rebuilt with the fused-mode 1-ctx patch
#     (~/1bit-MONSTER/engine/npu/build/npu_engine_zr1)
#   - zaya q4nx model
#
# Env overrides: XCLBIN_DIR BIN MODEL PROMPT_A PROMPT_B ITERS_A ITERS_B
#   (PROMPT_* = space-separated token ids; default = the oracle "capital of
#    France" prefix)
#   STOP_FLM=1 (default when flm-35b is active): stop flm for the test window
#   and restore it after. KEEP_FLM=1 disables that.
#
# Exit: 0 both streams completed with output; 1 partial; 2 fail.
set -u

XCLBIN_DIR="${XCLBIN_DIR:-$HOME/npu-verify/1bit-MONSTER/engine/npu/xclbins}"
BIN="${BIN:-$HOME/1bit-MONSTER/engine/npu/build/npu_engine_zr1}"
MODEL="${MODEL:-/home/bcloud/models/zaya1-8b-fresh.q4nx}"
PA="${PROMPT_A:-9079 236761 107 2717 108 1882}"
PB="${PROMPT_B:-9079 236761 107 2717 108 1882}"
IA="${ITERS_A:-8}"; IB="${ITERS_B:-8}"
LOG=${LOG:-/tmp}
H0F="$XCLBIN_DIR/final_i8_MOE_FUSED_zaya_h0.xclbin"; H0I="$XCLBIN_DIR/insts_i8_MOE_FUSED_zaya_h0.txt"
H1F="$XCLBIN_DIR/final_i8_MOE_FUSED_zaya_h1.xclbin"; H1I="$XCLBIN_DIR/insts_i8_MOE_FUSED_zaya_h1.txt"

for f in "$BIN" "$MODEL" "$H0F" "$H0I" "$H1F" "$H1I"; do
    [ -f "$f" ] || { echo "MISSING: $f"; exit 2; }
done

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

echo "== two-stream decode: A(cols 0-3) || B(cols 4-7), one NPU =="
echo "model: $MODEL"
echo "stream A: $H0F  (${IA} tok)"
echo "stream B: $H1F  (${IB} tok)"

common="NPU_FUSED=1"
run_one() { # $1=h/f suffix  $2=out log  $3=iters  rest=prompt
    local suffix="$1" log="$2" iters="$3"; shift 3
    local xp fi
    if [ "$suffix" = h0 ]; then xp="$H0F"; fi="$H0I"; else xp="$H1F"; fi="$H1I"; fi
    ( cd "$(dirname "$BIN")" && env $common \
        NPU_FUSED_XCLBIN="$xp" NPU_FUSED_INSTS="$fi" \
        timeout "$(( 90 + iters * 45 ))" "$BIN" "$MODEL" "$@" \
        > "$log" 2>&1 & echo $! )
}
pA=$(run_one h0 "$LOG/two_A.log" "$IA" $PA)
pB=$(run_one h1 "$LOG/two_B.log" "$IB" $PB)
echo "pids: A=$pA B=$pB  (model load ~60s, then decode)"
okA=0; okB=0
for i in $(seq 1 40); do
    sleep 10
    dA=$(grep -c "perf]" "$LOG/two_A.log" 2>/dev/null || true)
    dB=$(grep -c "perf]" "$LOG/two_B.log" 2>/dev/null || true)
    [ "$dA" -ge 1 ] && okA=1
    [ "$dB" -ge 1 ] && okB=1
    if [ "$okA" = 1 ] && [ "$okB" = 1 ]; then break; fi
    if ! kill -0 "$pA" 2>/dev/null && ! kill -0 "$pB" 2>/dev/null && \
       [ "$dA" = 0 ] && [ "$dB" = 0 ]; then echo "both died early — see logs"; break; fi
done
wait "$pA" 2>/dev/null; wait "$pB" 2>/dev/null
pkill -9 -f "npu_engine_zr1" 2>/dev/null   # clear any stragglers

echo "== stream A (cols 0-3) =="
grep -E "fused mode|corr|perf|tokens" "$LOG/two_A.log" | tail -3
echo "== stream B (cols 4-7) =="
grep -E "fused mode|corr|perf|tokens" "$LOG/two_B.log" | tail -3

if [ "$restore_flm" = 1 ]; then
    echo "restoring flm-35b..."
    systemctl --user start flm-35b
fi
[ "$okA" = 1 ] && [ "$okB" = 1 ] && echo "RESULT: BOTH STREAMS DECODED CONCURRENTLY" && exit 0
echo "RESULT: PARTIAL — A=$okA B=$okB"
exit 1
