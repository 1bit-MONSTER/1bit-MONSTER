#!/bin/bash
# bringup_runner.sh — MONSTER family bring-up runner (Phase 3 skeleton, 2026-08-14).
# Reads Testing/models_manifest.json; for each family verifies the arch mapping
# is wired and runs the real-checkpoint generation gate (20/20 vs torch) when
# the fixture dir exists. Add a family = manifest entry + fixture, then run.
set -u
cd "$(dirname "$0")/.."
CXX="${CXX:-g++}"; FLAGS="-std=c++17 -O2 -Iinclude -Isrc"
BIN=/tmp/onebit_bringup; mkdir -p "$BIN"
fail=0; total=0

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
for fam in $(python3 -c "
import json; print(' '.join(f['family'] for f in json.load(open('Testing/models_manifest.json'))['families'] if f['status']=='validated'))"); do
    dir=/tmp/onebit-e2e/$fam
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
    fi
done

echo "======================================"
echo "$((total-fail))/$total generation gates passed"
[ "$fail" -eq 0 ] || { echo "$fail FAILURES"; exit 1; }
