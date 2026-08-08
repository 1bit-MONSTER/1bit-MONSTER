#!/usr/bin/env bash
# batch_convert_q4nx.sh — Batch convert GGUF models to Q4NX (the FLM/NPU format)
#
# Q4NX pivot: every zoo model ships as Q4NX, converted with ROCm's official
# FLM_Q4NX_Converter (third_party/FLM_Q4NX_Converter). Output lands in
# ~/.config/flm/models/<Name>/model.q4nx — the layout the FLM stack and the
# NPU backend already discover (backend_npu.cpp auto-discovery).
#
# Usage:
#   bash tools/batch_convert_q4nx.sh                    # all GGUFs in models/
#   bash tools/batch_convert_q4nx.sh model.gguf ...     # specific files
#   bash tools/batch_convert_q4nx.sh --force model.gguf # reconvert existing
#
# Requirements: python3 + torch/gguf/safetensors/einops (the converter venv)
#
# Not convertible (no official converter config): zaya, zamba2, mamba,
# mistral, falcon, olmo, granite, deepseek2/3 — those stay on their native
# backends (HIP 1BP / GGML-Vulkan), reported as SKIP.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

CONVERTER="$REPO_DIR/third_party/FLM_Q4NX_Converter/convert.py"
OUT_ROOT="${Q4NX_MODELS_DIR:-$HOME/.config/flm/models}"
FORCE=0
[ "${1:-}" = "--force" ] && { FORCE=1; shift; }

if [ "$#" -gt 0 ]; then
    GGUFS=("$@")
else
    GGUFS=(models/*.gguf)
fi

# ─── Color output ──────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${CYAN}[info]${NC} $1"; }
ok()    { echo -e "${GREEN}[ok]${NC}   $1"; }
warn()  { echo -e "${YELLOW}[warn]${NC} $1"; }
err()   { echo -e "${RED}[err]${NC}  $1"; }

# ─── GGUF architecture → converter arch ────────────────────────────
# Keys: GGUF general.architecture. Values: converter's ModelArchNames
# (third_party/FLM_Q4NX_Converter/q4nx/constants.py). Anything absent has
# no official converter config and cannot be Q4NX-converted.
declare -A ARCH_MAP=(
    ["llama"]="llama"
    ["qwen3"]="qwen3"
    ["qwen2"]="qwen2"
    ["gemma3"]="gemma3"
    ["gemma4"]="gemma4"
    ["phi3"]="phi3"          # converter config phi4.json (PHI4 arch)
    ["gpt-oss"]="gpt-oss"
    ["lfm2"]="lfm2"
    ["nanbeige"]="nanbeige"
    ["qwen2vl"]="qwen2.5-Vl"
    ["qwen3vl"]="qwen3vl"
)

detect_arch() {
    python3 -c "
import struct
with open('$1', 'rb') as f:
    magic = f.read(4)
    if magic != b'GGUF': exit(1)
    f.read(4); f.read(8)
    n_kvs = struct.unpack('Q', f.read(8))[0]
    for i in range(min(n_kvs, 100)):
        klen = struct.unpack('Q', f.read(8))[0]
        key = f.read(klen).decode('utf-8')
        vtype = struct.unpack('I', f.read(4))[0]
        if key == 'general.architecture':
            slen = struct.unpack('Q', f.read(8))[0]
            print(f.read(slen).decode('utf-8'))
            exit(0)
        if vtype == 8:
            sl = struct.unpack('Q', f.read(8))[0]
            if sl > 10000: sl = 0
            f.read(sl)
        else:
            f.read(4)
" 2>/dev/null || echo ""
}

# ─── Convert one GGUF → Q4NX ──────────────────────────────────────
convert_file() {
    local gguf="$1" arch conv_arch name out_dir out_file
    arch=$(detect_arch "$gguf")
    [ -z "$arch" ] && { err "$(basename "$gguf"): cannot read architecture — skipping"; return 1; }
    conv_arch="${ARCH_MAP[$arch]:-}"
    if [ -z "$conv_arch" ]; then
        warn "$(basename "$gguf"): arch '$arch' has no official Q4NX converter — skipping (stays on native backend)"
        return 2
    fi

    # Model dir name: GGUF basename minus quant suffix (Q8_0/Q4_K_M/...)
    name=$(basename "$gguf" .gguf)
    for q in Q8_0 Q4_K_M F16 BF16 Q4_0 Q4_1 Q5_K_M Q6_K; do
        name=${name%.$q}; name=${name%-$q}
    done

    # Q1_0 (ternary) tensors have no dequant in the official converter —
    # skip with a clear reason instead of crashing mid-convert.
    if python3 -c "
from gguf import GGUFReader
r = GGUFReader('$gguf')
qt = {t.tensor_type.name for t in r.tensors}
import sys
sys.exit(0 if 'Q1_0' in qt else 1)" 2>/dev/null; then
        warn "$(basename "$gguf"): contains Q1_0/ternary tensors — official converter has no Q1_0 dequant, skipping (stays on native backend)"
        return 2
    fi

    out_dir="$OUT_ROOT/$name"
    out_file="$out_dir/model.q4nx"
    if [ -f "$out_file" ] && [ "$FORCE" -eq 0 ]; then
        warn "$(basename "$gguf"): already converted at $out_file (--force to reconvert)"
        return 2
    fi

    info "Converting: $(basename "$gguf") [arch=$arch] → $out_dir/"
    mkdir -p "$out_dir"
    local t0 t1 elapsed
    t0=$(date +%s%N)
    # Run from the converter dir — it resolves configs/ relative to CWD.
    # (Fixed upstream debug sys.argv no longer clobbers args.)
    timeout 3600 bash -c 'cd "$1" && exec python3 "$2" -i "$3" -o "$4" -f "$5"' \
        _ "$(dirname "$CONVERTER")" "$CONVERTER" "$gguf" "$out_dir" "$conv_arch" 2>&1 | tail -2
    local rc=${PIPESTATUS[0]}
    t1=$(date +%s%N); elapsed=$(( (t1 - t0) / 1000000 ))
    if [ "$rc" -eq 0 ] && [ -f "$out_file" ]; then
        ok "Converted: $(ls -lh "$out_file" | awk '{print $5}') in $((elapsed / 1000))s"
        return 0
    else
        err "Conversion failed for $(basename "$gguf")"
        return 1
    fi
}

# ─── Main ─────────────────────────────────────────────────────────
[ -f "$CONVERTER" ] || { err "Converter not found: $CONVERTER"; exit 1; }
mkdir -p "$OUT_ROOT"
PASS=0; FAIL=0; SKIP=0
for gguf in "${GGUFS[@]}"; do
    [ -f "$gguf" ] || { warn "not a file: $gguf"; continue; }
    convert_file "$gguf" || rc=$?
    case ${rc:-0} in
        0) PASS=$((PASS+1));;
        2) SKIP=$((SKIP+1));;
        *) FAIL=$((FAIL+1));;
    esac
    unset rc
done

echo ""
echo "────────────────────────────────────────────"
echo "batch_convert_q4nx: $PASS converted, $SKIP skipped, $FAIL failed"
echo "Q4NX models are in $OUT_ROOT  (served by the FLM/NPU backend)"
[ "$FAIL" -eq 0 ] || exit 1
