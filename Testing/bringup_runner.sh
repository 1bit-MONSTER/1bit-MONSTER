#!/bin/bash
# bringup_runner.sh — MONSTER family bring-up runner (Phase 3 skeleton, 2026-08-14).
# Reads Testing/models_manifest.json; for each family verifies the arch mapping
# is wired and runs the real-checkpoint generation gate (20/20 vs torch) when
# the fixture dir exists. Add a family = manifest entry + fixture, then run.
#
# What a run actually establishes (measured 2026-09-18 on a box with no fixtures, #2520):
#   * a family with a `gate` command is RUN — its exit status is the verdict;
#   * a family whose fixture is absent is SKIPPED, counted and printed as such, and never
#     counted as passing. The manifest's `validated` status records a one-time measurement
#     on the box that made that fixture; this run does not re-establish it;
#   * a run in which nothing could run exits 2 — "0/0 passed" is not a pass.
# Nothing in CI invokes this script (the mapping gate it shares with run_all.sh is
# `run arch`), and the fixture families need $FIXTURE_ROOT/<family> to exist, which no
# checkout carries.

# ── pure helpers: Testing/bringup_runner_selfcheck.sh sources this file to exercise them ──
# gate_verdict <ran> <failed> — 0 = every gate that ran passed, 1 = something failed,
# 2 = nothing ran at all (which must never be reported as success).
gate_verdict() {
    if [ "$1" -eq 0 ]; then printf '2\n'
    elif [ "$2" -gt 0 ]; then printf '1\n'
    else printf '0\n'
    fi
}

# gate_summary <ran> <failed> <skipped> — the line a reader takes the verdict from.
gate_summary() {
    printf '%s/%s generation gates passed, %s skipped (no fixture)\n' \
        "$(( $1 - $2 ))" "$1" "$3"
}

if [ "${BASH_SOURCE[0]}" != "$0" ]; then return 0; fi

set -u
cd "$(dirname "$0")/.." || exit 1
CXX="${CXX:-g++}"; FLAGS="-std=c++17 -O2 -Iinclude -Isrc"
BIN=/tmp/onebit_bringup; mkdir -p "$BIN"
FIXTURE_ROOT="${ONEBIT_E2E_DIR:-/tmp/onebit-e2e}"
fail=0; total=0; skip=0

[ -f Testing/models_manifest.json ] || { echo "missing manifest"; exit 1; }

# 1) mapping gate: every hf_arch_string of every family must resolve to its
#    mapping_target in rcpp_arch_from_string (compiled via the arch self-check).
echo "== mapping gate (arch_mapping_selfcheck) =="
if ! "$CXX" $FLAGS Testing/arch_mapping_selfcheck.cpp -o "$BIN/arch" 2>/dev/null; then
    echo "✗ arch selfcheck COMPILE FAILED"; exit 1
fi
"$BIN/arch" >/dev/null 2>&1 && echo "✓ arch mapping (41 checks)" || { echo "✗ arch mapping"; fail=$((fail+1)); }

# 2) compile the generation binary once
"$CXX" $FLAGS src/backend_generic.cpp src/model_discovery.cpp src/gguf_reader.cpp \
    src/q4nx_reader.cpp src/safetensors_reader.cpp Testing/e2e_seq_gen.cpp -o "$BIN/e2e_seq" 2>/dev/null \
    || { echo "✗ e2e_seq COMPILE FAILED"; exit 1; }
[ -x /tmp/e2e_seq ] || ln -sf "$BIN/e2e_seq" /tmp/e2e_seq

# 3) per-family generation gate (20/20 tokens vs torch)
python3 - <<'EOF'
import json
m = json.load(open('Testing/models_manifest.json'))
for f in m['families']:
    print(f"{f['family']:16s} [{f['status']:9s}] target={f['mapping_target']}")
EOF
echo
echo "fixture root: $FIXTURE_ROOT — families without a fixture are reported as skipped, never as"
echo "passing (the manifest's 'validated' records the measurement that made the fixture, #2520)."
echo
for fam in $(python3 -c "
import json; print(' '.join(f['family'] for f in json.load(open('Testing/models_manifest.json'))['families'] if f['status']=='validated'))"); do
    # Families with a manifest-level gate command (custom harness/oracle: numpy
    # refs, torch remote-code oracles, bespoke backends). The gate is the
    # exact command that must exit 0 — "add a family" = manifest entry with a
    # gate, no runner surgery.
    gate_cmd=$(python3 -c "
import json
for f in json.load(open('Testing/models_manifest.json'))['families']:
    if f['family']=='$fam' and f.get('gate'): print(f['gate']); break")
    if [ -n "$gate_cmd" ]; then
        total=$((total+1))
        if eval "$gate_cmd" 2>/dev/null; then
            echo "  $fam [gate]: PASS"
        else
            echo "  $fam [gate]: FAIL"
            fail=$((fail+1))
        fi
        continue
    fi
    dir=$FIXTURE_ROOT/$fam
    if [ -f "$dir/oracle-q8.gguf" ] && [ -f "$dir/config.json" ]; then
        # families whose torch oracle is unavailable (archs dropped from
        # transformers 5.x) use the llama.cpp reference instead
        oracle=$(python3 -c "
import json
for f in json.load(open('Testing/models_manifest.json'))['families']:
    if f['family']=='$fam':
        if f.get('numpy_ref'): print('numpy');
        else:
            o=f.get('oracle','torch'); print('llamacpp' if o.startswith('llamacpp') else 'torch'); break")
        total=$((total+1))
        if [ "$oracle" = "numpy" ]; then
            out=$(E2E_FULL_LOGITS=/tmp/onebit_ref_logits.txt timeout 570 python3 Testing/e2e_numpy_ref.py "$dir" "$fam" 2>/dev/null | head -1)
        elif [ "$oracle" = "llamacpp" ]; then
            out=$(timeout 570 python3 Testing/e2e_gen_check_llamacpp.py "$dir" 2>/dev/null | head -1)
        else
            out=$(timeout 570 python3 Testing/e2e_gen_check.py "$dir" 2>/dev/null | head -1)
        fi
        echo "  $fam [$oracle]: ${out:-GATE FAILED}"
        case "$out" in
            *"20/20"*|*"MATCHES"*) ;;
            *) echo "  ✗ $fam gate failed ($out)"; fail=$((fail+1));;
        esac
    else
        echo "  $fam: fixture absent ($dir), skipped"
        skip=$((skip+1))
    fi
done

echo "======================================"
echo "  $(gate_summary "$total" "$fail" "$skip")"
case "$(gate_verdict "$total" "$fail")" in
    0) echo "OK: every generation gate that ran passed; $skip family(ies) skipped for lack of a fixture." ;;
    1) echo "$fail FAILURES"; exit 1 ;;
    *) echo "NOTHING RAN: no family had a gate command and none had a fixture under $FIXTURE_ROOT." >&2
       echo "  This is not a pass (#2520). Generate a fixture (Testing/make_mini_*.py) or add a" >&2
       echo "  'gate' command to Testing/models_manifest.json for the family you want checked." >&2
       exit 2 ;;
esac
