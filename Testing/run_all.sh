#!/usr/bin/env bash
# run_all.sh — One Bit Systems: run the full HF-native bring-up test suite.
# Compiles + runs every self-check and the real-checkpoint e2e families.
# (mirrors the bring-up arc documented in ~/onebit-modular-research.md §1-23)
set -u
cd "$(dirname "$0")/.." || exit 1   # repo root
CXX="${CXX:-g++}"; FLAGS="-std=c++17 -Iinclude -Isrc -O2"
BIN=/tmp/onebit_tests; mkdir -p "$BIN"
fail=0; total=0

run() {  # run <name> <compile-args...> -- <run-args...>
    local name="$1"; shift
    local src=(); local runargs=()
    while [ "${1:-}" != "--" ] && [ $# -gt 0 ]; do src+=("$1"); shift; done
    [ $# -gt 0 ] && shift
    while [ $# -gt 0 ]; do runargs+=("$1"); shift; done
    total=$((total+1))
    if ! "$CXX" $FLAGS "${src[@]}" -o "$BIN/$name" 2>/dev/null; then
        echo "✗ $name: COMPILE FAILED"; fail=$((fail+1)); return
    fi
    if "$BIN/$name" "${runargs[@]}" >/dev/null 2>&1; then
        echo "✓ $name"; else echo "✗ $name: CHECK FAILED"; fail=$((fail+1)); fi
}

echo "== fixture self-checks =="
run arch      Testing/arch_mapping_selfcheck.cpp --
run discovery Testing/discovery_selfcheck.cpp src/model_discovery.cpp src/gguf_reader.cpp src/q4nx_reader.cpp src/safetensors_reader.cpp
run router    Testing/router_selfcheck.cpp src/model_router.cpp
run dtypes    Testing/safetensors_weights_selfcheck.cpp src/safetensors_reader.cpp src/q4nx_reader.cpp
run sharded   Testing/sharded_reader_selfcheck.cpp src/safetensors_reader.cpp src/q4nx_reader.cpp
run rotation  Testing/rotation_table_selfcheck.cpp
run iq1       Testing/iq1_selfcheck.cpp --
run tq2nz     Testing/tq2nz_e4m3_selfcheck.cpp --

# v4 dedup e2e: synthetic GGUF with duplicated tensors -> converter -> loaders
DEDUP_DIR=/tmp/onebit_dedup; mkdir -p "$DEDUP_DIR"
total=$((total+1))
if python3 Testing/make_mini_gguf.py "$DEDUP_DIR/mini.gguf" >/dev/null 2>&1 && \
   "$CXX" $FLAGS tools/gguf_to_onebp.cpp src/gguf_reader.cpp src/q4nx_reader.cpp src/safetensors_reader.cpp \
       -o "$BIN/g2o" 2>/dev/null; then
    conv_out=$("$BIN/g2o" "$DEDUP_DIR/mini.gguf" "$DEDUP_DIR/mini.1bp" 2>&1)
    if [ $? -eq 0 ] && printf '%s' "$conv_out" | grep -q 'dedup: blk.1.attn_q.weight'; then
        echo "✓ dedup converter (alias emitted)"
        run dedup_e2e Testing/dedup_loader_check.cpp src/onebp_model.cpp -- "$DEDUP_DIR/mini.1bp"
    else
        echo "✗ dedup converter: no alias emitted"; fail=$((fail+1))
    fi
else
    echo "✗ dedup converter: build/generate failed"; fail=$((fail+1))
fi

echo "== backend compile =="
total=$((total+1))
if "$CXX" $FLAGS -c src/backend_generic.cpp -o "$BIN/bg.o" 2>/dev/null; then
    echo "✓ backend_generic.cpp"; else echo "✗ backend_generic.cpp"; fail=$((fail+1)); fi

echo "== e2e (needs model fixtures in /tmp/onebit-e2e — skipped if absent) =="
e2e() {  # e2e <name> <model_dir> <oracle.gguf> [expect-torch-string]
    local name="$1" dir="$2" gguf="$3"
    if [ ! -f "$gguf" ]; then echo "  - $name: fixtures absent, skipped"; return; fi
    total=$((total+1))
    if ! "$CXX" $FLAGS src/backend_generic.cpp src/model_discovery.cpp src/gguf_reader.cpp \
        src/q4nx_reader.cpp src/safetensors_reader.cpp Testing/e2e_safetensors_selfcheck.cpp \
        -o "$BIN/e2e" 2>/dev/null; then echo "✗ $name e2e: COMPILE FAILED"; fail=$((fail+1)); return; fi
    if "$BIN/e2e" "$dir" "$gguf" >/dev/null 2>&1; then
        echo "✓ $name e2e"; else echo "✗ $name e2e: loader mismatch"; fail=$((fail+1)); fi
}
e2e llama  /tmp/onebit-e2e/smollm  /tmp/onebit-e2e/smollm/oracle-q8.gguf
e2e qwen2  /tmp/onebit-e2e/qwen2    /tmp/onebit-e2e/qwen2/oracle-q8.gguf
e2e gemma  /tmp/onebit-e2e/gemma    /tmp/onebit-e2e/gemma/oracle-q8.gguf
e2e qwen3  /tmp/onebit-e2e/qwen3    /tmp/onebit-e2e/qwen3/oracle-q8.gguf

echo "======================================"
echo "$((total-fail))/$total passed"
[ "$fail" -eq 0 ] || { echo "$fail FAILURES"; exit 1; }

run rni-bf16 Testing/aie2p_bf16_rni_selfcheck.cpp --
