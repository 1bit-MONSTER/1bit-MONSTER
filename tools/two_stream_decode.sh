#!/bin/bash
# two_stream_decode.sh — NPU decode driver for the Zaya fused GU→SiLU→D path.
#
#   MODE=single (--single) : ONE stream on the FULL 8-column fused kernel.
#                            This is the numerically exact path: measured
#                            corr=0.998469, 22–25 tok/s (main @ 2026-09-18,
#                            NPU_EMB_INT8=1). Gate: MIN_CORR_SINGLE (0.99).
#   MODE=two    (--two)    : TWO concurrent streams, one per 4-column half
#                            (cols 0-3 | 4-7), the issue-#2128 column-sliced
#                            co-schedule. Gate: MIN_CORR (0.80) — even a good
#                            half-array run only reaches ~0.8719.
#   MODE=auto   (default)  : two if the h0/h1 halves exist, else single.
#
# GATES (2026-09-18): the success claim is REFUSED unless every stream passes
# the numeric gate. Before this the success test was `grep -c "perf]"`, which
# printed "BOTH STREAMS DECODED CONCURRENTLY" and exit 0 on a corr=-0.0016 run
# (a host↔xclbin ABI mismatch). A gate that cannot fail is not a gate.
#
# Requirements:
#   - fused xclbins + insts:
#       single: <engine-xclbin-dir>/final_i8_MOE_FUSED_zaya.xclbin (+insts)
#       two:    <XCLBIN_DIR>/final_i8_MOE_FUSED_zaya_{h0,h1}.xclbin (+insts)
#               NOTE the halves must be built from the SAME revision as the
#               binary: generators/build_zaya_fused_cols.sh. The h0/h1 pair in
#               ~/npu-verify (Aug 25 lineage) is ABI-stale and reads corr
#               ≈ -0.0016 — the gate exists to catch exactly that.
#   - npu_engine_zr1 (fused-mode build), revision recorded into the run dir
#   - the engine's full-array kernel set (GU/D m16 + attn) resolvable
#   - zaya q4nx model
#
# Env overrides: BIN MODEL XCLBIN_DIR NPU_XCLBIN_DIR LOG ITERS_A ITERS_B
#   PROMPT_A PROMPT_B MIN_CORR MIN_CORR_SINGLE MIN_EMB_CORR
#   STOP_FLM=1 (default when flm-35b is active): stop flm for the test window
#   and restore it after. KEEP_FLM=1 disables that.
#   NPU_EMB_INT8=1 etc. are inherited by the child as usual.
#
# Exit: 0 both/either stream(s) decoded AND all gates passed
#       1 partial — a stream produced no [perf] line
#       2 precondition failure (missing artifact / unusable env)
#       3 WRONG MODE — decoded but a gate failed, or the gate line was absent
set -u

MODE="auto"
for a in "$@"; do
    case "$a" in
        --single) MODE=single ;;
        --two)    MODE=two ;;
        -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
        *) echo "unknown argument: $a" >&2; exit 2 ;;
    esac
done
[ -n "${MODE_ENV:-}" ] && MODE="$MODE_ENV"

XCLBIN_DIR="${XCLBIN_DIR:-$HOME/npu-verify/1bit-MONSTER/engine/npu/xclbins}"
BIN="${BIN:-$HOME/1bit-MONSTER/engine/npu/build/npu_engine_zr1}"
LOG=${LOG:-/tmp}
IA="${ITERS_A:-8}"; IB="${ITERS_B:-8}"
PA="${PROMPT_A:-9079 236761 107 2717 108 1882}"
PB="${PROMPT_B:-9079 236761 107 2717 108 1882}"
MIN_CORR="${MIN_CORR:-0.80}"                 # half-array expectation ~0.8719
MIN_CORR_SINGLE="${MIN_CORR_SINGLE:-0.99}"   # full-array expectation 0.998469
MIN_EMB_CORR="${MIN_EMB_CORR:-0.99}"

# --- model: the old default named zaya1-8b-fresh.q4nx, which no longer exists
# --- (the dedup kept one name; both were byte-identical).
if [ -z "${MODEL:-}" ]; then
    for c in "$HOME/models/zaya1-8b.q4nx" "$HOME/models/zaya1-8b-fresh.q4nx"; do
        [ -f "$c" ] && { MODEL="$c"; break; }
    done
fi
MODEL="${MODEL:-}"

# --- resolve the engine's full-array kernel dir. Its own default
# --- (~/.local/share/1bit-monster/xclbins) does not exist on this box, and the
# --- failure surfaced deep inside the engine as "GU ctx init failed" long after
# --- this script's pre-check had passed.
ENGINE_XCLBIN_DIR=""
if [ -n "${NPU_XCLBIN_DIR:-}" ]; then
    if [ -d "$NPU_XCLBIN_DIR" ]; then
        ENGINE_XCLBIN_DIR="$NPU_XCLBIN_DIR"
    else
        echo "note: NPU_XCLBIN_DIR='$NPU_XCLBIN_DIR' is not a directory on this" >&2
        echo "      machine — ignoring it (an override naming a missing dir is" >&2
        echo "      not an override, cf. #2350)" >&2
    fi
fi
for d in "$HOME/1bit-MONSTER/engine/npu/xclbins" \
         "/usr/local/share/1bit-monster/xclbins" \
         "$HOME/.local/share/1bit-monster/xclbins"; do
    [ -n "$ENGINE_XCLBIN_DIR" ] && break
    [ -d "$d" ] && ENGINE_XCLBIN_DIR="$d"
done
ENGINE_XCLBIN_DIR="${ENGINE_XCLBIN_DIR:-}"

FULLF="$ENGINE_XCLBIN_DIR/final_i8_MOE_FUSED_zaya.xclbin"
FULLI="$ENGINE_XCLBIN_DIR/insts_i8_MOE_FUSED_zaya.txt"
H0F="$XCLBIN_DIR/final_i8_MOE_FUSED_zaya_h0.xclbin"; H0I="$XCLBIN_DIR/insts_i8_MOE_FUSED_zaya_h0.txt"
H1F="$XCLBIN_DIR/final_i8_MOE_FUSED_zaya_h1.xclbin"; H1I="$XCLBIN_DIR/insts_i8_MOE_FUSED_zaya_h1.txt"

if [ "$MODE" = auto ]; then
    if [ -f "$H0F" ] && [ -f "$H0I" ] && [ -f "$H1F" ] && [ -f "$H1I" ]; then MODE=two; else MODE=single; fi
    echo "mode: $MODE (auto — halves $([ "$MODE" = two ] && echo present || echo absent) in $XCLBIN_DIR)"
fi

missing=0
for f in "$BIN" "$MODEL"; do [ -f "$f" ] || { echo "MISSING: $f"; missing=1; }; done
if [ -z "$ENGINE_XCLBIN_DIR" ]; then
    echo "MISSING: no engine xclbin dir found (tried \$NPU_XCLBIN_DIR,"
    echo "         \$HOME/1bit-MONSTER/engine/npu/xclbins,"
    echo "         /usr/local/share/1bit-monster/xclbins,"
    echo "         \$HOME/.local/share/1bit-monster/xclbins)"
    missing=1
else
    for f in final_i8_MOE_GU_zaya_m16.xclbin insts_i8_MOE_GU_zaya_m16.txt \
             final_i8_MOE_D_zaya_m16.xclbin  insts_i8_MOE_D_zaya_m16.txt \
             attn.xclbin attn_insts.txt; do
        [ -f "$ENGINE_XCLBIN_DIR/$f" ] || {
            echo "MISSING: $ENGINE_XCLBIN_DIR/$f  (engine full-array kernel set)"; missing=1; }
    done
fi
if [ "$MODE" = single ]; then
    for f in "$FULLF" "$FULLI"; do
        [ -f "$f" ] || { echo "MISSING: $f  (full-array fused pair; build it with"
                         echo "         generators/build_zaya_fused.sh)"; missing=1; }
    done
else
    for f in "$H0F" "$H0I" "$H1F" "$H1I"; do [ -f "$f" ] || { echo "MISSING: $f"; missing=1; }; done
fi
[ "$missing" = 1 ] && exit 2

restore_flm=0
pA=""; pB=""
cleanup() {
    local rc=$?
    [ -n "$pA" ] && kill "$pA" 2>/dev/null
    [ -n "$pB" ] && kill "$pB" 2>/dev/null
    wait "$pA" "$pB" 2>/dev/null
    # only self-started children are killed (invariant I4, npu-ab-yardstick)
    if [ "$restore_flm" = 1 ]; then
        echo "restoring flm-35b..."
        systemctl --user start flm-35b
    fi
    return $rc
}
trap cleanup EXIT

if systemctl --user is-active flm-35b >/dev/null 2>&1; then
    if [ "${KEEP_FLM:-0}" = "1" ]; then
        echo "note: flm-35b active and KEEP_FLM=1 — decode ctxs share the partition with flm."
    else
        echo "stopping flm-35b for the test window..."
        systemctl --user stop flm-35b && restore_flm=1
        sleep 2
    fi
fi

echo "== $MODE decode on one NPU =="
echo "model:             $MODEL"
echo "binary:            $BIN"
echo "engine kernels:    $ENGINE_XCLBIN_DIR"
[ "$MODE" = single ] && echo "full fused pair:   $FULLF" || echo "fused halves from: $XCLBIN_DIR"
if [ "$MODE" = single ]; then
    echo "gate:              moe_corr >= $MIN_CORR_SINGLE, emb_corr >= $MIN_EMB_CORR"
else
    echo "gates:             moe_corr >= $MIN_CORR, emb_corr >= $MIN_EMB_CORR"
fi

# --- provenance: which binary + which kernel pair produced these numbers.
# --- #2172 is a host↔xclbin ABI pairing, so the pair IS the evidence.
{
    echo "utc=$(date -u +%FT%TZ)"
    echo "mode=$MODE"
    echo "binary=$BIN sha256=$(sha256sum "$BIN" | cut -d" " -f1)"
    echo "model=$MODEL sha256=$(sha256sum "$MODEL" | cut -d" " -f1)"
    if [ "$MODE" = single ]; then
        for f in "$FULLF" "$FULLI"; do echo "$f sha256=$(sha256sum "$f" | cut -d" " -f1)"; done
    else
        for f in "$H0F" "$H0I" "$H1F" "$H1I"; do echo "$f sha256=$(sha256sum "$f" | cut -d" " -f1)"; done
    fi
    for f in final_i8_MOE_GU_zaya_m16.xclbin insts_i8_MOE_GU_zaya_m16.txt \
             final_i8_MOE_D_zaya_m16.xclbin insts_i8_MOE_D_zaya_m16.txt; do
        echo "$ENGINE_XCLBIN_DIR/$f sha256=$(sha256sum "$ENGINE_XCLBIN_DIR/$f" | cut -d" " -f1)"
    done
    rev=$(git -C "$(dirname "$BIN")/../.." rev-parse HEAD 2>/dev/null) && \
        echo "tree_HEAD=$rev" || echo "tree_HEAD=(not a git checkout)"
} > "$LOG/provenance.txt" 2>&1
echo "provenance:        $LOG/provenance.txt"

common="NPU_FUSED=1"
run_one() { # $1=out log  $2=iters  $3=xclbin  $4=insts  rest=prompt
    local log="$1" iters="$2" xp="$3" ip="$4"; shift 4
    ( cd "$(dirname "$BIN")" && env $common NPU_XCLBIN_DIR="$ENGINE_XCLBIN_DIR" \
        NPU_FUSED_XCLBIN="$xp" NPU_FUSED_INSTS="$ip" \
        timeout "$(( 90 + iters * 45 ))" "$BIN" "$MODEL" "$@" \
        > "$log" 2>&1 & echo $! )
}

if [ "$MODE" = single ]; then
    LA="$LOG/single.log"; LB=""
    pA=$(run_one "$LA" "$IA" "$FULLF" "$FULLI" $PA)
    echo "pid: A=$pA  (model load ~60s, then decode)"
else
    LA="$LOG/two_A.log"; LB="$LOG/two_B.log"
    pA=$(run_one "$LA" "$IA" "$H0F" "$H0I" $PA)
    pB=$(run_one "$LB" "$IB" "$H1F" "$H1I" $PB)
    echo "pids: A=$pA B=$pB  (model load ~60s, then decode)"
fi

okA=0; okB=0
[ "$MODE" = single ] && okB=1
for _ in $(seq 1 40); do
    sleep 10
    dA=$(grep -c "perf]" "$LA" 2>/dev/null || true)
    [ "$dA" -ge 1 ] && okA=1
    dB=1
    if [ -n "$LB" ]; then dB=$(grep -c "perf]" "$LB" 2>/dev/null || true); fi
    [ "$dB" -ge 1 ] && okB=1
    if [ "$okA" = 1 ] && [ "$okB" = 1 ]; then break; fi
    if ! kill -0 "$pA" 2>/dev/null && { [ -z "$pB" ] || ! kill -0 "$pB" 2>/dev/null; } && \
       [ "$dA" = 0 ] && [ "$dB" = 0 ]; then echo "died early — see $LA"; break; fi
done
wait "$pA" 2>/dev/null; [ -n "$pB" ] && wait "$pB" 2>/dev/null
pA=""; pB=""

echo "== stream A $([ "$MODE" = single ] && echo '(full 8 cols)' || echo '(cols 0-3)') =="
grep -E "MoE path|fused mode|corr|perf|tokens" "$LA" | tail -3
if [ "$MODE" = two ]; then
    echo "== stream B (cols 4-7) =="
    grep -E "MoE path|fused mode|corr|perf|tokens" "$LB" | tail -3
fi

# --- GATES ---------------------------------------------------------------
# moe_corr: [MoE L1 single dbg] corr. Full-array 0.998469; half-array ~0.8719;
#           a host↔xclbin ABI mismatch reads ~ -0.0015.
# emb_corr: [EMB dbg] corr — exact (1.0) in every run seen.
# Absent line = FAIL: a missing gate line must never read as a pass.
num() { grep -oE "$2" "$1" 2>/dev/null | tail -1 | sed "s/.*=//"; }
ge() { awk -v a="$1" -v b="$2" 'BEGIN{exit !(a+0>=b+0)}'; }

LIM_MOE="$MIN_CORR"; [ "$MODE" = single ] && LIM_MOE="$MIN_CORR_SINGLE"
mA=$(num "$LA" '\[MoE L1 single dbg\] corr=[-0-9.eE]+')
eA=$(num "$LA" '\[EMB dbg\] corr=[-0-9.eE]+')
mB=""; eB=""
if [ "$MODE" = two ]; then
    mB=$(num "$LB" '\[MoE L1 single dbg\] corr=[-0-9.eE]+')
    eB=$(num "$LB" '\[EMB dbg\] corr=[-0-9.eE]+')
fi

gate_fail=""
chk() { # $1=label $2=value $3=min
    if [ -z "$2" ]; then gate_fail="$gate_fail $1=ABSENT(need>=$3)"
    elif ! ge "$2" "$3"; then gate_fail="$gate_fail $1=$2(need>=$3)"; fi
}
chk moeA "$mA" "$LIM_MOE"
chk embA "$eA" "$MIN_EMB_CORR"
if [ "$MODE" = two ]; then chk moeB "$mB" "$LIM_MOE"; chk embB "$eB" "$MIN_EMB_CORR"; fi

tA=$(grep -E '^[0-9][0-9 ]*$' "$LA" 2>/dev/null | tail -1)
tB=""; [ "$MODE" = two ] && tB=$(grep -E '^[0-9][0-9 ]*$' "$LB" 2>/dev/null | tail -1)
if [ "$MODE" = two ] && [ -n "$tA" ] && [ "$tA" = "$tB" ]; then
    echo "note: A and B token streams are identical (determinism, NOT correctness)"
fi
{ echo "mode=$MODE moeA=$mA moeB=$mB embA=$eA embB=$eB min_moe=$LIM_MOE min_emb=$MIN_EMB_CORR"
  echo "tokens_A=$tA"; [ "$MODE" = two ] && echo "tokens_B=$tB"; } > "$LOG/gates.txt"

if [ "$okA" != 1 ] || [ "$okB" != 1 ]; then
    echo "RESULT: PARTIAL — A=$okA B=$okB (see $LA${LB:+ / $LB})"
    echo "gates: $LOG/gates.txt"
    exit 1
fi

if [ -n "$gate_fail" ]; then
    echo "RESULT: WRONG MODE — decoded, but the numeric gate FAILED:"
    echo "       $gate_fail"
    echo "       SUCCESS CLAIM REFUSED. This is a fast wrong mode: completion and"
    echo "       identical A/B token streams both pass on numerically invalid output."
    echo "       Known cause of ~ -0.0015: a host<->xclbin ABI mismatch (#2163 vs"
    echo "       #2172) — the binary and the fused xclbin must be a matched pair."
    echo "       For MODE=two the halves must be built from the same revision as"
    echo "       the binary (generators/build_zaya_fused_cols.sh)."
    echo "       See $LOG/provenance.txt."
    exit 3
fi

if [ "$MODE" = single ]; then
    echo "RESULT: SINGLE-STREAM FULL-ARRAY DECODE (gates passed: moeA=$mA embA=$eA)"
else
    echo "RESULT: BOTH STREAMS DECODED CONCURRENTLY (gates passed: moeA=$mA moeB=$mB embA=$eA embB=$eB)"
fi
exit 0
