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

# Instella-MoE (DeepSeek-V3 clone): gated MLA + FarSkip dual-residual + sigmoid
# router engine gate — mini fixture (real tokenizer, mini dims) vs HF logits.
# Fixtures live in 1bit-monster/models/kl-test/ (mini-full-f16.gguf + mini-full-hf.pt).
instella_mini=/tmp/onebit-instella/mini-full-f16.gguf
instella_ref=/tmp/onebit-instella/hf.npy
if [ -f "$instella_mini" ] && [ -f "$instella_ref" ]; then
    total=$((total+1))
    if ! "$CXX" $FLAGS Testing/cmp_instella.cpp src/deepseek.cpp src/gguf_reader.cpp \
        -o "$BIN/cmp_instella" 2>/dev/null; then echo "✗ instella: COMPILE FAILED"; fail=$((fail+1));
    elif "$BIN/cmp_instella" "$instella_mini" /tmp/onebit-instella/ids.txt "$instella_ref" 20 18 >/dev/null 2>&1; then
        echo "✓ instella engine (gated MLA + FarSkip)";
    else echo "✗ instella engine: top-20 mismatch vs HF"; fail=$((fail+1)); fi
else
    echo "  - instella: fixtures absent, skipped (cp -r 1bit-monster/models/kl-test/mini-full* /tmp/onebit-instella/)"
fi


# Instella 1bp loader gate (Gate 5): convert mini GGUF -> Q4NX 1bp -> load via
# DeepSeekModel::load_from_1bp, assert MLA dims + gated/farskip flags round-trip
# and the forward runs. Golden top1 = the fp16 GGUF engine's own output (mini
# weights are random ~0.01 so quantization drifts top-1 — the STRUCTURE is the
# gate here; the real-16B Q4NX top1=128804 == fp16 is the manual validation).
instella_mini=/tmp/onebit-instella/mini-full-f16.gguf
instella_1bp=/tmp/onebit-instella/mini-full-q4nx.1bp
if [ -f "$instella_mini" ]; then
    total=$((total+1))
    if ! "$CXX" $FLAGS tools/gguf_to_onebp.cpp src/gguf_reader.cpp -o "$BIN/gguf_to_onebp" 2>/dev/null; then
        echo "✗ instella-1bp: converter COMPILE FAILED"; fail=$((fail+1))
    else
        "$BIN/gguf_to_onebp" "$instella_mini" "$instella_1bp" >/dev/null 2>&1
        if ! "$CXX" $FLAGS Testing/cmp_instella_1bp.cpp src/deepseek.cpp src/gguf_reader.cpp \
            -o "$BIN/cmp_instella_1bp" 2>/dev/null; then echo "✗ instella-1bp: COMPILE FAILED"; fail=$((fail+1));
        elif "$BIN/cmp_instella_1bp" "$instella_1bp" /tmp/onebit-instella/ids.txt 178 >/dev/null 2>&1; then
            echo "✓ instella 1bp loader (MLA dims + gated + farskip round-trip)";
        else echo "✗ instella 1bp loader: structure/top1 mismatch"; fail=$((fail+1)); fi
    fi
else
    echo "  - instella-1bp: fixture absent, skipped"
fi

# ── DeepSeek V4 gate (mini fixture, HF safetensors oracle) ──
total=$((total+1))
dsv4_dir=/tmp/onebit-dsv4
if [ -f "$dsv4_dir/logits_last.npy" ] && [ -f "$dsv4_dir/model.safetensors" ]; then
    echo "5 7 9 11 3" > /tmp/onebit-dsv4-ids.txt
    if ! "$CXX" $FLAGS Testing/cmp_deepseek_v4.cpp src/deepseek_v4.cpp src/safetensors_reader.cpp src/q4nx_reader.cpp \
        -o "$BIN/cmp_dsv4" 2>/dev/null; then echo "✗ deepseek_v4: COMPILE FAILED"; fail=$((fail+1));
    elif "$BIN/cmp_dsv4" "$dsv4_dir" /tmp/onebit-dsv4-ids.txt "$dsv4_dir/logits_last.npy" 20 18 >/dev/null 2>&1; then
        echo "✓ deepseek_v4 engine (Shared-KV MQA + mHC + hash-MoE)";
    else echo "✗ deepseek_v4 engine: top-20 mismatch vs HF"; fail=$((fail+1)); fi
else
    echo "  - deepseek_v4: fixture absent, skipped (python3 Testing/make_mini_deepseek_v4.py /tmp/onebit-dsv4)"
fi

# ── GLM-MoE-DSA gate (mini fixture, HF safetensors oracle) ──
total=$((total+1))
glmdsa_dir=/tmp/onebit-glmdsa
if [ -f "$glmdsa_dir/logits_last.npy" ] && [ -f "$glmdsa_dir/model.safetensors" ]; then
    echo "5 7 9 11 3" > /tmp/onebit-glmdsa-ids.txt
    if ! "$CXX" $FLAGS Testing/cmp_glm_moe_dsa.cpp src/glm_moe_dsa.cpp src/safetensors_reader.cpp src/q4nx_reader.cpp \
        -o "$BIN/cmp_glmdsa" 2>/dev/null; then echo "✗ glm_moe_dsa: COMPILE FAILED"; fail=$((fail+1));
    elif "$BIN/cmp_glmdsa" "$glmdsa_dir" /tmp/onebit-glmdsa-ids.txt "$glmdsa_dir/logits_last.npy" 20 18 >/dev/null 2>&1; then
        echo "✓ glm_moe_dsa engine (V3-MLA + DSA indexer + group-topk MoE)";
    else echo "✗ glm_moe_dsa engine: top-20 mismatch vs HF"; fail=$((fail+1)); fi
else
    echo "  - glm_moe_dsa: fixture absent, skipped (python3 Testing/make_mini_glm_moe_dsa.py /tmp/onebit-glmdsa)"
fi

# ── MiMo-V2 gate (mini fixture, vendored remote modeling oracle) ──
total=$((total+1))
mimo_dir=/tmp/onebit-mimo
if [ -f "$mimo_dir/logits_last.npy" ] && [ -f "$mimo_dir/model.safetensors" ]; then
    echo "5 7 9 11 3" > /tmp/onebit-mimo-ids.txt
    if ! "$CXX" $FLAGS Testing/cmp_mimo_v2.cpp src/mimo_v2.cpp src/safetensors_reader.cpp src/q4nx_reader.cpp \
        -o "$BIN/cmp_mimo" 2>/dev/null; then echo "✗ mimo_v2: COMPILE FAILED"; fail=$((fail+1));
    elif "$BIN/cmp_mimo" "$mimo_dir" /tmp/onebit-mimo-ids.txt "$mimo_dir/logits_last.npy" 20 18 >/dev/null 2>&1; then
        echo "✓ mimo_v2 engine (MoD hybrid: SWA+full GQA, sigmoid group-topk MoE)";
    else echo "✗ mimo_v2 engine: top-20 mismatch vs HF"; fail=$((fail+1)); fi
else
    echo "  - mimo_v2: fixture absent, skipped (python3 Testing/make_mini_mimo_v2.py /tmp/onebit-mimo)"
fi

# ── Qwen3_5 text gate (mini fixture, HF oracle) ──
total=$((total+1))
q35_dir=/tmp/onebit-q35
if [ -f "$q35_dir/logits_last.npy" ] && [ -f "$q35_dir/model.safetensors" ]; then
    echo "5 7 9 11 3" > /tmp/onebit-q35-ids.txt
    if ! "$CXX" $FLAGS Testing/cmp_qwen3_5.cpp src/qwen3_5.cpp src/safetensors_reader.cpp src/q4nx_reader.cpp \
        -o "$BIN/cmp_q35" 2>/dev/null; then echo "✗ qwen3_5: COMPILE FAILED"; fail=$((fail+1));
    elif "$BIN/cmp_q35" "$q35_dir" /tmp/onebit-q35-ids.txt "$q35_dir/logits_last.npy" 20 18 >/dev/null 2>&1; then
        echo "✓ qwen3_5 text engine (GatedDeltaNet + gated GQA hybrid)";
    else echo "✗ qwen3_5 text engine: top-20 mismatch vs HF"; fail=$((fail+1)); fi
else
    echo "  - qwen3_5: fixture absent, skipped (python3 Testing/make_mini_qwen3_5.py /tmp/onebit-q35)"
fi

# ── Mesh: self-aware network substrate (optional — needs the CMake build) ──
total=$((total+1))
if [ -x build/mesh_peer ]; then
    if bash Testing/mesh_smoke.sh build/mesh_peer >/dev/null 2>&1; then
        echo "✓ mesh (peer discovery + ask/answer)";
    else echo "✗ mesh (peer discovery + ask/answer)"; fail=$((fail+1)); fi
else
    echo "  - mesh: mesh_peer binary absent, skipped (cmake --build build --target mesh_peer)"
fi

# ── JARVIS fleet dispatch (optional — needs build/1bit + build/mesh_peer) ──
total=$((total+1))
if [ -x build/1bit ] && [ -x build/mesh_peer ]; then
    if bash Testing/jarvis_mesh_smoke.sh build/1bit build/mesh_peer >/dev/null 2>&1; then
        echo "✓ jarvis fleet dispatch (mesh-aware, DSH brain path)";
    else echo "✗ jarvis fleet dispatch (mesh-aware, DSH brain path)"; fail=$((fail+1)); fi
else
    echo "  - jarvis fleet: binaries absent, skipped (cmake --build build --target onebin mesh_peer)"
fi


echo "======================================"
echo "$((total-fail))/$total passed"
[ "$fail" -eq 0 ] || { echo "$fail FAILURES"; exit 1; }

run rni-bf16 Testing/aie2p_bf16_rni_selfcheck.cpp --
