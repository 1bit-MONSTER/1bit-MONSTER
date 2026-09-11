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
set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; YELLOW='\033[1;33m'; NC='\033[0m'
say()  { printf "${GREEN}✓${NC} %s\n" "$*"; }
warn() { printf "${YELLOW}!${NC} %s\n" "$*"; }
die()  { printf "${RED}✗${NC} %s\n" "$*"; exit 1; }

# ── Model registry ──
# Format: name|description|size|url|sha256
MODELS=(
  "qwen3-0.6b|Qwen3-0.6B — 356 MB|356M|https://huggingface.co/bong-water-water-bong/Qwen3-0.6B-1BP/resolve/main/Qwen3-0.6B.1bp|"
  "qwen3-8b|Qwen3-8B — 4.8 GB|4.8G|https://huggingface.co/bong-water-water-bong/Qwen3-8B-1BP/resolve/main/Qwen3-8B-1BP.1bp|"
  "qwen3-vl-4b|Qwen3-VL-4B — 2.3 GB|2.3G|https://huggingface.co/bong-water-water-bong/Qwen3-VL-4B-Instruct-1BP/resolve/main/Qwen3-VL-4B-Instruct-1BP.1bp|"
  "gemma4-e2b|Gemma4-E2B — 1.3 GB|1.3G|https://huggingface.co/bong-water-water-bong/Gemma4-E2B-1BP/resolve/main/Gemma4-E2B-1BP.1bp|"
  "llama-3.1-8b|Llama-3.1-8B — 4.7 GB|4.7G|https://huggingface.co/bong-water-water-bong/Llama-3.1-8B-1BP/resolve/main/Llama-3.1-8B-1BP.1bp|"
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
  printf "  ${YELLOW}all${NC} — download all models (about 13.5 GB total)\n"
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
  local OUTFILE="${MODEL_DIR}/$(basename "${URL}")"
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
