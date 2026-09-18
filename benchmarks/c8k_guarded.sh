#!/usr/bin/env bash
# c8k_guarded.sh — criterion-(c) 8k prefill/TTFT campaign WITH both contamination guards.
#
# Why this exists: an identical 8k native prefill command produced 5.6 s .. 20.0 s
# (RESULTS-8k-campaign-variance-2026-09-18.md), and two independent contamination axes were
# found (RESULTS-8k-contention-source-2026-09-18.md):
#   1. a foreign process holding /dev/accel/accel0, which the engine's own
#      /tmp/1bit-npu-device.lock does NOT exclude (LEVERS 6.7);
#   2. host CPU load — the native prefill's "conv+other" term is host work and tracks load
#      (0.6B 8k: 4078 ms at load 16.8, 7286 ms at 20.9, 11238 ms at 23.2, all device-clean).
# Runs are DISCARDED/flagged on either axis, and medians are printed over the accepted ones.
#
# usage: c8k_guarded.sh <Model-NPU2> <flm_tag> <engine_name> <runs> [extra native env ...]
#   e.g. c8k_guarded.sh Qwen3-0.6B-NPU2 qwen3:0.6b npu_engine_qwen3_0_6b 5
#        c8k_guarded.sh Llama-3.1-8B-NPU2 llama3.1:8b npu_engine_llama 5 NPU_UNIFIED=1 NPU_LAYER_ELF_DIR=/tmp/llama-elfs
#
# Env knobs: C8K_MAX_LOAD (default 18)  C8K_MAX_LOAD_DELTA (default 4)
#            C8K_IGNORE_PREEXISTING=1   run even if a holder is present before the campaign
#
# Native is ng=1 on purpose: at 8k the SECOND decode forward needs ctx=8194, which fails
# against the baked MAX_L=8192 per-ctx ELF window, so only prefill/TTFT is a valid 8k row.
set -u
M="${1:?usage: c8k_guarded.sh <Model-NPU2> <flm_tag> <engine> <runs> [env ...]}"
TAG="${2:?}"; ENG="${3:?}"; RUNS="${4:?}"; shift 4
D="$HOME/.config/flm/models/$M"
E="./engine/npu/build/$ENG"
PROMPT=/tmp/p_8k.txt
LOGDIR="./benchmarks/c8k-guarded-$M-$(date -u +%Y%m%dT%H%M%SZ)"
[ -f "$PROMPT" ] || { echo "missing $PROMPT" >&2; exit 2; }
mkdir -p "$LOGDIR"

# Every PID holding accel0, one "pid:cmd" per line, EXCLUDING the persistent production
# server only. A transient `flm serve` started by an FLM measurement leg MUST count: the
# broad *"flm serve"* exclusion this function used to have hid a lingering leg instance and
# reported 8970/6927/11956 ms for runs that are ~4078 ms clean.
foreign_holders() {
    local p pid cmd
    for p in /proc/[0-9]*; do
        ls -l "$p/fd" 2>/dev/null | grep -q accel0 || continue
        pid="${p#/proc/}"
        cmd="$(tr '\0' ' ' < "$p/cmdline" 2>/dev/null | cut -c1-70)"
        case "$cmd" in *"qwen3.6-moe:35b-a3b"*) continue ;; esac
        printf '%s:%s\n' "$pid" "$cmd"
    done
}

# Human detail for the pre-flight abort: age, and device I/O (a long-lived holder with
# read_bytes=0 write_bytes=0 is spinning, not measuring — @agent-dddf9e's /tmp/attrib leak).
foreign_holders_detail() {
    local h pid cmd age io
    while IFS= read -r h; do
        [ -n "$h" ] || continue
        pid="${h%%:*}"; cmd="${h#*:}"
        age="$(ps -o etime= -p "$pid" 2>/dev/null | tr -d ' ')"
        io="$(awk '/^(read_bytes|write_bytes):/{printf "%s ", $0}' "/proc/$pid/io" 2>/dev/null)"
        printf '  pid=%s age=%s %scmd=%s\n' "$pid" "${age:-?}" "$io" "$cmd"
    done
}

load1()  { cut -d' ' -f1 /proc/loadavg 2>/dev/null || echo 0; }
# Top CPU consumer as "pcpu comm pid parent", so a foreign hog can be attributed to the
# lane that owns it (2026-09-18: `pf` -> tests/prism/check_oracle_agreement.py ->
# tests/prism/run_prism_tests.sh, cwd ~/1bit-MONSTER-dddf9e).
topcpu() {
    local line pid ppid pcmd
    line="$(ps -eo pcpu=,comm=,pid= --sort=-pcpu 2>/dev/null | head -1)"
    pid="$(printf '%s' "$line" | awk '{print $3}')"
    if [ -n "$pid" ]; then
        ppid="$(ps -o ppid= -p "$pid" 2>/dev/null | tr -d ' ')"
        pcmd="$(ps -o cmd= -p "$ppid" 2>/dev/null | cut -c1-60)"
        printf '%s parent=%s:%s' "$(printf '%s' "$line" | awk '{print $1, $2, "pid="$3}')" "${ppid:-?}" "${pcmd:-?}"
    else
        printf '%s' "$line"
    fi
}
runner_lines() { wc -l < /tmp/runner.log 2>/dev/null || echo 0; }
# Load ceiling calibrated on 2026-09-18 (0.6B 8k native, device clean):
#   load 16.83 -> 4078 ms(0.498)  |  20.89 -> 7268 ms(0.887)  |  23.23 -> 11238 ms(1.372)
MAX_LOAD="${C8K_MAX_LOAD:-18}"; MAX_DL="${C8K_MAX_LOAD_DELTA:-4}"

# ---- pre-flight: a holder already present BEFORE the campaign --------------------------
if [ -n "$(foreign_holders)" ] && [ -z "${C8K_IGNORE_PREEXISTING:-}" ]; then
    echo "ABORT: foreign holder(s) on /dev/accel/accel0 are already present before the campaign:"
    foreign_holders_detail < <(foreign_holders)
    echo "  A long-lived holder with read_bytes=0 write_bytes=0 is spinning, not measuring."
    echo "  Wait for it to exit, or set C8K_IGNORE_PREEXISTING=1 to run anyway (runs are still"
    echo "  flagged or discarded individually)."
    exit 3
fi

echo "== $M : $RUNS runs, guard on (max load $MAX_LOAD, delta $MAX_DL) =="
n_ok=0
declare -a N_TTFT=() F_TTFT=() F_PREFILL=()
for i in $(seq 1 "$RUNS"); do
    # Settle: wait for any transient holder to release the device before measuring.
    for _w in $(seq 1 20); do [ -z "$(foreign_holders)" ] && break; sleep 1; done
    # C8K_WAIT_QUIET=1: wait for BOTH axes to be clear before sampling, instead of measuring
    # into a rising load and discarding. In a contended environment the right protocol is
    # "wait for the window, then take one sample" -- 2026-09-18: 5/5 1.7B runs were rejected
    # because the load went 18 -> 48 inside the campaign while the Prism test suite ran.
    # Bounded by C8K_WAIT_MAX (default 600 s); on timeout the run proceeds and is flagged.
    if [ -n "${C8K_WAIT_QUIET:-}" ]; then
        for _q in $(seq 1 "${C8K_WAIT_MAX:-600}"); do
            [ -n "$(foreign_holders)" ] && { sleep 1; continue; }
            awk -v a="$(load1)" -v b="$MAX_LOAD" 'BEGIN{exit !(a<b)}' && break
            sleep 1
        done
    fi
    pre_f="$(foreign_holders | tr '\n' ';')"; pre_r="$(runner_lines)"; pre_l="$(load1)"

    raw="$LOGDIR/native-$i.log"
    timeout 900 env "$@" NPU_PREFILL_BF16=1 NPU_BF16=1 NPU_GREEDY=1 NPU_PREFILL_MAX=8192 \
        NPU_PROMPT_MAX=8192 "$E" "$D/model.q4nx" 1 "$PROMPT" > "$raw" 2>&1
    np_line="$(grep -aE 'Prefill:' "$raw" | tail -1)"
    post_f="$(foreign_holders | tr '\n' ';')"; post_r="$(runner_lines)"; post_l="$(load1)"

    flog="$LOGDIR/flm-$i.log"
    timeout 900 bash "$HOME/npu-ab/npu_ab.sh" --model "$ENG" --flm-tag "$TAG" --engine "$E" \
        --q4nx "$D/model.q4nx" --tokenizer "$D/tokenizer.json" --prompt "$PROMPT" \
        --ctx-k 8 --decode-tokens 1 --reps 1 --skip-native > "$flog" 2>&1
    f_line="$(grep -aE 'rep 1' "$flog" | head -1)"

    status="ACCEPT"
    [ -n "$pre_f" ]  && status="DISCARD(pre-holder: $pre_f)"
    [ -n "$post_f" ] && status="DISCARD(post-holder: $post_f)"
    if [ "$status" = ACCEPT ]; then
        [ "$pre_r" != "$post_r" ] && status="SUSPECT(ci $pre_r->$post_r)"
    fi
    if [ "$status" = ACCEPT ]; then
        awk -v a="$pre_l" -v b="$MAX_LOAD" 'BEGIN{exit !(a>b)}' \
            && status="SUSPECT(pre-load=$pre_l>$MAX_LOAD)"
    fi
    if [ "$status" = ACCEPT ]; then
        awk -v a="$pre_l" -v b="$post_l" -v d="$MAX_DL" 'BEGIN{exit !((b-a)>d)}' \
            && status="SUSPECT(load rose $pre_l->$post_l)"
    fi

    np_t="$(printf '%s' "$np_line" | sed -n 's/.*(\([0-9.]*\) ms\/tok).*/\1/p')"
    np_ms="$(printf '%s' "$np_line" | sed -n 's/Prefill: \([0-9]*\)ms.*/\1/p')"
    f_t="$(printf '%s' "$f_line" | sed -n 's/.*ttft=\([0-9.]*\)s.*/\1/p')"
    f_p="$(printf '%s' "$f_line" | sed -n 's/.*prefill=\([0-9.]*\)t\/s.*/\1/p')"
    echo "run $i [$status] native ${np_ms:-?}ms (${np_t:-?} ms/tok) | FLM ttft ${f_t:-?}s prefill ${f_p:-?}t/s | load ${pre_l}->${post_l} topcpu=$(topcpu)"
    if [ "$status" = ACCEPT ]; then
        n_ok=$((n_ok+1))
        [ -n "$np_t" ] && N_TTFT+=("$np_t")
        [ -n "$f_t" ]  && F_TTFT+=("$f_t")
        [ -n "$f_p" ]  && F_PREFILL+=("$f_p")
    fi
done

median() {
    [ "$#" -gt 0 ] || { echo "n/a"; return; }
    printf '%s\n' "$@" | sort -g | awk '{a[NR]=$1} END{print (NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2}'
}
echo "-- accepted runs: $n_ok/$RUNS"
echo "-- native ms/tok    median: $(median "${N_TTFT[@]:-}")   (all: ${N_TTFT[*]:-none})"
echo "-- FLM   ttft s     median: $(median "${F_TTFT[@]:-}")   (all: ${F_TTFT[*]:-none})"
echo "-- FLM   prefill t/s median: $(median "${F_PREFILL[@]:-}")   (all: ${F_PREFILL[*]:-none})"
echo "-- logs: $LOGDIR"
