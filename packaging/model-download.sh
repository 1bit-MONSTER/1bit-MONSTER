#!/usr/bin/env bash
set -euo pipefail
# 1bit.MONSTER — Model Downloader
# Downloads ready-to-run 1BP models from the public Hugging Face mirror
# (bong-water-water-bong). Plain GGUF models run directly and need no download
# script; this is only for the pre-converted 1BP/NPU weights.
#
# Usage:
#   model-download.sh                  # interactive menu
#   model-download.sh list             # list available models
#   model-download.sh qwen3-0.6b       # download specific model
#   model-download.sh all              # download all models
#
# Requires: curl, awk

RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; YELLOW='\033[1;33m'; NC='\033[0m'
say()  { printf "${GREEN}✓${NC} %s\n" "$*"; }
warn() { printf "${YELLOW}!${NC} %s\n" "$*"; }
die()  { printf "${RED}✗${NC} %s\n" "$*"; exit 1; }

# ── Model registry ──
# Format: name|description|size|url|sha256
# The sha256 is Hugging Face's own LFS digest for the file, so a download is
# verified against what the hub serves without trusting the transfer.
# Descriptions carry the size in IEC units (what `numfmt --to=iec` prints).
MODELS=(
  "qwen3-0.6b|Qwen3-0.6B — 356 MB|356M|https://huggingface.co/bong-water-water-bong/Qwen3-0.6B-1BP/resolve/main/Qwen3-0.6B.1bp|a3591eaefc40a028a857743546b2920c1bf6f8f53406dc8c96c721ca3411a66e"
  "qwen3-8b|Qwen3-8B — 4.8 GB|4.8G|https://huggingface.co/bong-water-water-bong/Qwen3-8B-1BP/resolve/main/Qwen3-8B-1BP.1bp|66035955e1533dd87732167b45ffaf5f28b9ef268065d2f0ca1e6852362b7cf2"
  "qwen3-vl-4b|Qwen3-VL-4B — 2.3 GB|2.3G|https://huggingface.co/bong-water-water-bong/Qwen3-VL-4B-Instruct-1BP/resolve/main/Qwen3-VL-4B-Instruct-1BP.1bp|953a9f04d5e99a3d9fdd6743889843606918c3a17e7f47288817dafd86b06702"
  "gemma4-e2b|Gemma4-E2B — 1.3 GB|1.3G|https://huggingface.co/bong-water-water-bong/Gemma4-E2B-1BP/resolve/main/Gemma4-E2B-1BP.1bp|ad618a733f38caac9bbd7c8bfa363903868eab1843b14a3111148d94a6ce9dce"
  "llama-3.1-8b|Llama-3.1-8B — 4.7 GB|4.7G|https://huggingface.co/bong-water-water-bong/Llama-3.1-8B-1BP/resolve/main/Llama-3.1-8B-1BP.1bp|3e3cbd766860bb891152f8da499b92be8659085ac3ba360b46b180d903f922f5"
  # ── Zyphra: the family the engine was tuned against (docs/model-families/zyphra.md) ──
  "zaya1-8b|ZAYA1-8B — 6.1 GB · flagship MoE+CCA|6.1G|https://huggingface.co/bong-water-water-bong/ZAYA1-8B-1BP/resolve/main/ZAYA1-8B.1bp|5abef9b00ab4d349adc241f522c5e5cd41590f9763bd28eca00934618873ebc4"
  "blackmamba-1.5b|BlackMamba-1.5B — 970 MB · fastest e2e|970M|https://huggingface.co/bong-water-water-bong/BlackMamba-1.5B-1BP/resolve/main/BlackMamba-1.5B.1bp|e3d9a651585f0d21509f327b21536f587c733a2b4652461bcf75b541dca9b9b6"
  "zr1-1.5b|ZR1-1.5B — 781 MB · dense reasoning|781M|https://huggingface.co/bong-water-water-bong/ZR1-1.5B-1BP/resolve/main/ZR1-1.5B.1bp|fabe7de3eadbd3853016138d8e980e8092806a8699ff9eb90538cfd177cbd912"
  "zamba2-1.2b|Zamba2-1.2B-v2 — 1.1 GB · Mamba2 hybrid|1.1G|https://huggingface.co/bong-water-water-bong/Zamba2-1.2B-Instruct-v2-1BP/resolve/main/Zamba2-1.2B-Instruct-v2.1bp|ed9673cc86cc90ec43a610157f2be96f615446f473cb3da2f454ca0a143f0ae0"
  "zaya1-74b|ZAYA1-74B-preview — 46.2 GB · preview, HIP|46.2G|https://huggingface.co/bong-water-water-bong/ZAYA1-74B-preview-1BP/resolve/main/ZAYA1-74B-preview.1bp|bb3e27c429cc971b0977fac3b448895645a303ec9c29e21a88140e31f3485e9b"
)

MODEL_DIR="${HOME}/.local/share/1bit/models"
mkdir -p "${MODEL_DIR}"

# ── Parse model data ──
get_field() { echo "$1" | cut -d'|' -f"$2"; }

list_models() {
  echo ""
  printf "${CYAN}Available models:${NC}\n"
  printf "  %-20s %-45s %s\n" "Name" "Description" "Size"
  printf "  %-20s %-45s %s\n" "────" "───────────" "────"
  for M in "${MODELS[@]}"; do
    printf "  ${GREEN}%-20s${NC} %-45s %s\n" "$(get_field "$M" 1)" "$(get_field "$M" 2)" "$(get_field "$M" 3)"
  done
  echo ""
  printf "  ${YELLOW}all${NC} — download all models (about 69 GB total; zaya1-74b alone is 46 GB)\n"
  echo ""
}

download_model() {
  local NAME="$1"
  local URL=""
  local DESC=""

  for M in "${MODELS[@]}"; do
    if [ "$(get_field "$M" 1)" = "$NAME" ]; then
      URL="$(get_field "$M" 4)"
      DESC="$(get_field "$M" 2)"
      break
    fi
  done

  if [ -z "$URL" ]; then
    die "Unknown model: $NAME. Run 'model-download.sh list' to see available models."
  fi

  # Keep the upstream file name (e.g. Qwen3-0.6B.1bp) — the engine keys off
  # the real extension, and a hardcoded .q4nx suffix would be misleading.
  # Declared, then assigned: `local X="$(cmd)"` makes the assignment's status the
  # `local` builtin's, which is always 0, so a failure inside $(...) is masked
  # (SC2155).
  local OUTFILE
  OUTFILE="${MODEL_DIR}/$(basename "${URL}")"
  if [ -f "$OUTFILE" ]; then
    warn "Model already exists at ${OUTFILE}"
    return 0
  fi

  echo ""
  say "Downloading ${DESC}"
  echo "  From: ${URL}"
  echo "  To:   ${OUTFILE}"
  echo ""

  # Download with progress bar
  curl -fL --progress-bar "$URL" -o "$OUTFILE" || die "Download failed"

  # Verify integrity (#1008: multi-GB weights were previously accepted with
  # zero integrity check — a corrupted or mismatched upstream asset would go
  # completely undetected)
  local SHA256
  SHA256="$(get_field "$M" 5)"
  if [ -n "$SHA256" ]; then
    local ACTUAL
    ACTUAL="$(sha256sum "$OUTFILE" | cut -d' ' -f1)"
    if [ "$ACTUAL" != "$SHA256" ]; then
      rm -f "$OUTFILE"
      die "sha256 mismatch for ${NAME}: expected ${SHA256}, got ${ACTUAL}. Deleted corrupted download."
    fi
    say "sha256 verified: ${SHA256}"
  else
    warn "No sha256 recorded for ${NAME} in the model registry — integrity NOT verified."
  fi

  local SIZE
  SIZE=$(stat -c%s "$OUTFILE" 2>/dev/null || stat -f%z "$OUTFILE" 2>/dev/null)
  say "Downloaded: $(numfmt --to=iec $SIZE) — saved to ${OUTFILE}"
  echo ""
}

# ── Interactive menu ──
interactive_menu() {
  list_models
  echo "Enter model name to download (or 'all' for everything, 'q' to quit):"
  read -r CHOICE
  case "$CHOICE" in
    q|quit|exit) exit 0 ;;
    all)
      for M in "${MODELS[@]}"; do
        download_model "$(get_field "$M" 1)"
      done
      ;;
    list) list_models ;;
    "")
      warn "No selection. Run 'model-download.sh list' to see options."
      exit 1
      ;;
    *)
      download_model "$CHOICE"
      ;;
  esac
}

# ── Main ──
case "${1:-}" in
  list|--list|-l)
    list_models
    ;;
  all|--all|-a)
    for M in "${MODELS[@]}"; do
      download_model "$(get_field "$M" 1)"
    done
    ;;
  "")
    interactive_menu
    ;;
  *)
    download_model "$1"
    ;;
esac

echo ""
say "Models are in ${MODEL_DIR}"
echo "  Run: 1bit-npu --auto 16"
echo "  Or:  1bit-npu ${MODEL_DIR}/Qwen3-0.6B.1bp 16"
echo ""
