#!/usr/bin/env bash
# run-hrx-gfx1151.sh — serve a GGUF model on the custom gfx1151 HRX build
# (GET_ROWS fixed; large prompts work — the shipped b59/b66 bundles cannot).
#
# Build:  bash scripts/build-hrx-gfx1151.sh   (one-time, ~30-60 min)
# Run:    bash scripts/run-hrx-gfx1151.sh [model.gguf] [port]
#   default model: Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf (HF cache)
#   default port:  8080
# Then:   curl http://127.0.0.1:8080/v1/chat/completions ... (OpenAI API)

set -euo pipefail

HRX_DIR="${HRX_DIR:-$HOME/hrx-gfx1151}"
MODEL="${1:-$HOME/.cache/huggingface/hub/models--unsloth--Qwen3-30B-A3B-Instruct-2507-GGUF/snapshots/eea7b2be5805a5f151f8847ede8e5f9a9284bf77/Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf}"
PORT="${2:-8080}"

SERVER="$HRX_DIR/llama-build/bin/llama-server"
LIBDIR="$HRX_DIR/hrx-runtime/lib"

if [ ! -x "$SERVER" ]; then
    echo "ERROR: $SERVER not found — run scripts/build-hrx-gfx1151.sh first (or set HRX_DIR)" >&2
    exit 1
fi
if [ ! -f "$MODEL" ]; then
    echo "ERROR: model not found: $MODEL" >&2
    exit 1
fi

echo "HRX gfx1151 (GET_ROWS fixed) — $SERVER"
echo "Model: $MODEL"
echo "Serving on http://127.0.0.1:$PORT ..."

LD_LIBRARY_PATH="$LIBDIR:$HRX_DIR/llama-build/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
GGML_DISABLE_VULKAN=1 \
exec "$SERVER" -m "$MODEL" --device HRX0 --port "$PORT" \
    --jinja --parallel 1 --ctx-size 8192
