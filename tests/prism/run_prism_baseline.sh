#!/bin/bash
# run_prism_baseline.sh — reproduce the P0.5 outside baseline on strixhalo.
#
# Runs Prism ML's own llama.cpp fork (Vulkan/RADV on gfx1151) against the two un-folded
# and folded 27B packs, and writes logs to ~/prism/. This is the yardstick and the
# token-level correctness oracle for our own engine — never a runtime dependency.
#
# Docs: docs/research/prism-bonsai-27b/P0.5-baseline.md
# Prereq: build the fork (see docs) at ~/prism/llama.cpp, Vulkan/RADV present.
#
# Usage: tests/prism/run_prism_baseline.sh [fork_dir] [model_dir]
set -euo pipefail

FORK="${1:-$HOME/prism/llama.cpp}"
MDIR="${2:-$HOME/models/prism}"
OUT="${HOME}/prism"
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/radeon_icd.json

CLI="$FORK/build/bin/llama-cli"
BENCH="$FORK/build/bin/llama-bench"
for b in "$CLI" "$BENCH"; do
  [ -x "$b" ] || { echo "missing $b — build the fork first (see P0.5-baseline.md)"; exit 1; }
done

PROMPT="The capital of France is"
SEED=20260918

echo "== device =="
"$CLI" --list-devices

# 1-bit Bonsai-27B (Qwen3.6, Q1_0 g128, NOT Hadamard-folded)
Q1="$MDIR/onebit-gguf/Bonsai-27B-Q1_0.gguf"
if [ -f "$Q1" ]; then
  echo "== llama-bench: $(basename "$Q1") =="
  "$BENCH" -m "$Q1" -ngl 99 -t 16 -p 64 -n 32 2>&1 | tail -6 | tee "$OUT/bench-Q1_0.log"
  echo "== llama-cli greedy: $(basename "$Q1") =="
  "$CLI" -m "$Q1" -ngl 99 -t 16 -c 4096 -n 24 --temp 0 --top-k 1 -s "$SEED" -st -p "$PROMPT" \
    > "$OUT/Q1_0-vulkan.log" 2>&1
  grep -E "Prompt:|Generation:" "$OUT/Q1_0-vulkan.log" || true
else
  echo "skip Q1_0 (not downloaded): $Q1"
fi

# Ternary-Bonsai-2-27B (Qwen3.8, PTQ1_0, Hadamard-folded — exercises the prism.hadamard path)
PTQ="$MDIR/ternary2-gguf/Ternary-Bonsai-2-27B-PTQ1_0.gguf"
if [ -f "$PTQ" ]; then
  echo "== llama-cli greedy: $(basename "$PTQ") =="
  "$CLI" -m "$PTQ" -ngl 99 -t 16 -c 2048 -n 8 --temp 0 --top-k 1 -s "$SEED" -st -p "$PROMPT" \
    > "$OUT/PTQ1_0-vulkan.log" 2>&1
  grep -E "Prompt:|Generation:" "$OUT/PTQ1_0-vulkan.log" || true
else
  echo "skip PTQ1_0 (not downloaded): $PTQ"
fi

echo "logs written to $OUT/"
