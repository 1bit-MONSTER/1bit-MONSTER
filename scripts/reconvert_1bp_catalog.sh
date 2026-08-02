#!/usr/bin/env bash
# reconvert_1bp_catalog.sh — batch re-conversion + gate for the legacy 1BP
# catalog (issue #1243). Re-converts broken artifacts with the current C++
# converter (tools/gguf_to_onebp, post-5f407bea2 tensor-cap fix) from Q8_0
# GGUF sources, then gates every output:
#   - qwen-vocab models: ppl_generic on the Qwen gate set (48 samples/708 tok)
#   - other families: structural gate (token_embd.weight present); per-vocab
#     ppl sample sets still needed (see audit conclusion #4)
#
# Usage: scripts/reconvert_1bp_catalog.sh [manifest.txt]
# Manifest format (tab-free '|' rows, '#' comments):
#   output_name|hf_repo|gguf_file|vocab_family(qwen|other)
# Idempotent: skips models whose output is newer than the source.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
CONV="$ROOT/build/gguf_to_onebp"
PPL="$ROOT/build/ppl_generic"
SCAN="$ROOT/build/scan_1bp"
VERIFY="${VERIFY_1BP:-$ROOT/build/verify_1bp}"
GATE_QWEN="$ROOT/research/ws00-baseline/samples/ppl-gate-48.jsonl"
SRC_DIR="$ROOT/models/_src_q8"
REPORT="$ROOT/research/ws05-1bp-v2/RECONVERT-REPORT.md"
MANIFEST="${1:-$ROOT/scripts/reconvert-manifest.txt}"
THREADS="${PPL_THREADS:-16}"

mkdir -p "$SRC_DIR"
[ -x "$SCAN" ] || g++ -O2 -std=c++23 "$ROOT/tools/scan_1bp.cpp" -o "$SCAN" \
    -I"$ROOT/include" -I"$ROOT/src" -I"$ROOT/engine/npu/src"
[ -x "$CONV" ] || { echo "build gguf_to_onebp first (cmake --build build --target gguf_to_onebp)"; exit 1; }
[ -x "$PPL" ]  || { echo "build ppl_generic first"; exit 1; }

{
  echo "# 1BP Catalog Re-conversion Report (issue #1243)"
  echo "Run: $(date -u +%Y-%m-%dT%H:%MZ) — converter @ $(git -C "$ROOT" rev-parse --short HEAD)"
  echo "Gate: Qwen-vocab → ppl_generic $GATE_QWEN; others → structural only"
  echo ""
  echo "| Model | Source GGUF | PPL (Qwen gate) | verify_1bp | token_embd | Status |"
  echo "|---|---|---|---|---|---|"
} > "$REPORT.tmp"

while IFS='|' read -r name repo file vocab; do
  [ -z "$name" ] && continue
  out="$ROOT/models/$name.1bp"; src="$SRC_DIR/$file"
  echo "== $name"
  if [ ! -s "$src" ]; then
    printf '  download %s ... ' "$file"
    if curl -fL --retry 3 -C - -o "$src" "https://huggingface.co/$repo/resolve/main/$file" 2>/dev/null; then
      echo "OK ($(du -h "$src" | cut -f1))"
    else
      echo "FAIL"; echo "| $name | $file | - | - | - | DOWNLOAD FAIL |" >> "$REPORT.tmp"; continue
    fi
  fi
  if [ ! -f "$out" ] || [ "$src" -nt "$out" ]; then
    printf '  convert ... '
    if "$CONV" "$src" "$out" >/dev/null 2>&1; then echo OK; else
      echo "FAIL"; echo "| $name | $file | - | - | - | CONVERT FAIL |" >> "$REPORT.tmp"; continue
    fi
  else
    echo "  output fresh, skip convert"
  fi
  embd=$("$SCAN" "$out" token_embd.weight 2>/dev/null | grep -o 'token_embd.weight=[A-Z]*' | cut -d= -f2)
  verify="-"
  if [ -f "$VERIFY" ] && [ "$embd" = "YES" ]; then
    if "$VERIFY" "$out" "$src" >/dev/null 2>&1; then verify="PASS"; else verify="FAIL"; fi
  fi
  embd="${embd:-NO}"
  ppl="-"
  if [ "$embd" = "YES" ] && [ "$vocab" = "qwen" ]; then
    printf '  ppl gate ... '
    ppl=$("$PPL" "$out" "$GATE_QWEN" "$THREADS" 2>/dev/null | grep -o 'PPL = [0-9.]*' | awk '{print $3}')
    [ -n "$ppl" ] && echo "PPL=$ppl" || { ppl="-"; echo "PPL FAIL"; }
  else
    echo "  structural gate only (vocab=$vocab)"
  fi
  status="OK"
  [ "$embd" != "YES" ] && status="MISSING token_embd"
  echo "| $name | $file | ${ppl:-'-'} | $verify | $embd | $status |" >> "$REPORT.tmp"
done < <(grep -v '^#' "$MANIFEST")

mv "$REPORT.tmp" "$REPORT"
echo ""
echo "report: $REPORT"
