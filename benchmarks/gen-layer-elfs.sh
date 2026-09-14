#!/usr/bin/env bash
# gen-layer-elfs.sh — build the per-context layer ELFs the runlist decode needs.
#
# WHY THIS EXISTS
#   The runlist path needs one layer ELF per CONTEXT, and `ctx` counts TOKENS
#   PROCESSED. So a model exercised at a 1024-token prompt needs ctx>=1025 for the
#   FIRST decode step — an ELF set sized for the prefill alone does not cover the
#   decode, and the run fails with:
#       [runlist] decode forward ctx=1025 failed
#       [runlist] whole-layer path failed (rc=1); falling back to split path
#   That failure looks like a tooling gap and is an off-by-range. Generate with
#   HEADROOM (prompt + decode tokens, rounded up), not exactly the prompt length.
#
# WHY IT RELINKS
#   gen_layer_elfs.cpp references every family's sequence class, so it must link all
#   the family libraries — and it must link them from ONE library tree. Linked against
#   a second FLM install instead, the same binary throws std::bad_alloc for EVERY
#   family, including qwen3, which reads as "the generator crashes on <family>".
#   The library path below is the one the tool was verified against.
#
# USAGE
#   benchmarks/gen-layer-elfs.sh <model_dir> <out_dir> <max_ctx> [family]
#
#   family is the argv[6] selector of gen_layer_elfs: qwen3 | llama | nanbeige |
#   phi4 | gemma_text. Default qwen3.
#
# THEN RUN THE ENGINE WITH
#   NPU_LAYER_ELF_DIR=<out_dir> NPU_RUNLIST=1 <engine> <model.q4nx> <n_tokens> <ids>
#
# Verified: Llama-3.1-8B @1024 with ctx 1..1100 gives
#   Prefill 1024 [runlist] -> [1] 220   and   68.9 ms/tok (15 tok/s)
set -euo pipefail

MODEL_DIR="${1:?usage: gen-layer-elfs.sh <model_dir> <out_dir> <max_ctx> [family]}"
OUT_DIR="${2:?usage: gen-layer-elfs.sh <model_dir> <out_dir> <max_ctx> [family]}"
MAX_CTX="${3:?usage: gen-layer-elfs.sh <model_dir> <out_dir> <max_ctx> [family]}"
FAMILY="${4:-qwen3}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$ROOT/npu-infer/tools/gen_layer_elfs.cpp"
BIN="${TMPDIR:-/tmp}/gen_layer_elfs_$FAMILY"

# ONE library tree. Do not mix trees: a mismatched tree crashes every family.
FLM_LIB="${FLM_LIB:-/home/bcloud/amd-oss/fastflowlm/src/lib/xrt}"
FLM_INC="${FLM_INC:-/home/bcloud/amd-oss/fastflowlm/src/include}"

[ -f "$SRC" ]        || { echo "missing $SRC" >&2; exit 1; }
[ -d "$FLM_LIB" ]    || { echo "missing lib tree $FLM_LIB" >&2; exit 1; }
[ -d "$MODEL_DIR" ]  || { echo "missing model dir $MODEL_DIR" >&2; exit 1; }

echo "== building the generator (lib tree: $FLM_LIB) =="
g++ -O2 -std=c++17 -include climits "$SRC" -o "$BIN" \
  -I"$FLM_INC" -I"$FLM_INC/npu_utils" -I/usr/include/aiebu \
  -L"$FLM_LIB" \
  -lqwen3_npu -lllama_npu -lnanbeige_npu -lphi4_npu -lqwen3_6_moe_npu \
  -lgemma4e_npu -lgemma_text_npu -llfm2_npu \
  -lgemm -lmha -lq4_npu_eXpress -L/usr/local/lib -laiebu -lxrt_coreutil -lxrt_core \
  -Wl,-rpath,"$FLM_LIB" 2>&1 | grep -E 'undefined reference|cannot find' && {
    echo "link failed — check the family libs exist in $FLM_LIB" >&2; exit 1; }
[ -x "$BIN" ] || { echo "generator build produced no binary" >&2; exit 1; }

echo "== generating ctx 1..$MAX_CTX as family=$FAMILY =="
mkdir -p "$OUT_DIR"
"$BIN" "$MODEL_DIR" "$OUT_DIR" 1 "$MAX_CTX" 32768 "$FAMILY"

n=$(ls "$OUT_DIR" 2>/dev/null | grep -c 'layer_ctx.*\.elf$' || true)
echo
echo "layer ELFs: $n  (expected $MAX_CTX)"
[ "$n" -eq "$MAX_CTX" ] || echo "WARNING: count mismatch — check the log above for aiebu failures"
cat <<EOF

Run the engine with:
  NPU_LAYER_ELF_DIR=$OUT_DIR NPU_RUNLIST=1 <engine> $MODEL_DIR/model.q4nx <n_tokens> <ids_file>

If the first decode step still fails, the ctx range was too small: ctx counts TOKENS
PROCESSED, so max_ctx must exceed prompt + decode tokens.
EOF
