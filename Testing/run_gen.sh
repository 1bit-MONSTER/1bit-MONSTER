#!/bin/bash
# PILOT #22: multi-token generation gate — the engine's full autoregressive
# loop (KV cache + position advancement + argmax) vs torch, 20 tokens, all
# validated families. Requires the fixture dirs + torch + transformers.
set -e
cd "$(dirname "$0")/.."
g++ -std=c++17 -O2 -Iinclude -Isrc src/backend_generic.cpp src/model_discovery.cpp \
    src/gguf_reader.cpp src/q4nx_reader.cpp src/safetensors_reader.cpp \
    Testing/e2e_seq_gen.cpp -o /tmp/e2e_seq_gen
echo "=== generation gate (20 tokens vs torch):"
for m in smollm qwen2 gemma granite qwen3 olmo; do
    python3 Testing/e2e_gen_check.py /tmp/onebit-e2e/$m 2>/dev/null | head -1
done
for m in hf-Mistral-7B-v0.1 hf-Phi-3-mini-4k-instruct; do
    python3 Testing/e2e_gen_check_hf.py /tmp/$m 2>/dev/null | head -1
done
