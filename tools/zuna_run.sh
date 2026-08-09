#!/usr/bin/env bash
# End-to-end ZUNA run: raw EEG -> tokens -> 1bit zuna reconstruction -> waveform.
#
#   zuna_run.sh <weights_dir> <eeg.npy> <chan_pos.npy> <out.npy> [rate] [--z z_true.bin]
#   zuna_run.sh <weights_dir> <rec.edf|rec.bdf> <out.npy> [rate] [--z z_true.bin]
#
# weights_dir : dir with weights.bin + weights.json (from zuna_gen_golden.py)
# eeg.npy     : [channels, samples] raw EEG (float64/32)
# chan_pos.npy: [channels, 3] scalp xyz in meters
# rec.edf     : medical EEG recording (EDF/EDF+/BDF); channels mapped via
#               embedded 10-20 table (override: tools/zuna_edf.py --chan-pos)
# out.npy     : written [channels, samples] reconstructed waveform (denormalized)
# --z FILE    : optional initial-noise file (else 1bit zuna draws its own)
set -euo pipefail
BIN="${BIN:-$(dirname "$0")/../build/1bit}"
WD="$1"; EEG="$2"
case "$EEG" in
  *.edf|*.bdf)
    # EDF form: <wd> <rec.edf> <out.npy> [rate] [--z ...]
    OUT="$3"; RATE="${4:-256}"; shift 3
    ;;
  *)
    # legacy .npy form: <wd> <eeg.npy> <chan_pos.npy> <out.npy> [rate] [--z ...]
    POS="$3"; OUT="$4"; RATE="${5:-256}"; shift 4
    ;;
esac
[ "$#" -ge 1 ] && [ "$1" != "--z" ] && shift   # consume optional RATE
Z=""; [ "$#" -ge 2 ] && [ "$1" = "--z" ] && Z="$2"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
case "$EEG" in
  *.edf|*.bdf)
    echo "research-only: reconstruction is imputed, not ground truth" >&2
    python3 "$(dirname "$0")/zuna_edf.py" "$EEG" "$TMP" "$RATE" --filter
    ORIG="$TMP/raw.npy"
    ;;
  *)
    python3 "$(dirname "$0")/zuna_preprocess.py" "$EEG" "$POS" "$TMP" "$RATE"
    ORIG="$EEG"
    ;;
esac
# onebin dispatches on a "zuna" subcommand; the standalone zuna_port build
# takes the weights dir directly. Detect by binary name.
case "$(basename "$BIN")" in
  onebin|1bit)
    if [ -z "$Z" ]; then
      "$BIN" zuna "$WD" "$TMP/tokens.bin" "$TMP/tok_idx.bin" "$TMP/recon.bin" "$TMP/enc.bin" 0
    else
      "$BIN" zuna "$WD" "$TMP/tokens.bin" "$TMP/tok_idx.bin" "$TMP/recon.bin" "$TMP/enc.bin" 0 "$Z"
    fi
    ;;
  *)
    if [ -z "$Z" ]; then
      "$BIN" "$WD" "$TMP/tokens.bin" "$TMP/tok_idx.bin" "$TMP/recon.bin" "$TMP/enc.bin" 0
    else
      "$BIN" "$WD" "$TMP/tokens.bin" "$TMP/tok_idx.bin" "$TMP/recon.bin" "$TMP/enc.bin" 0 "$Z"
    fi
    ;;
esac
python3 "$(dirname "$0")/zuna_invert_recon.py" "$TMP/recon.bin" "$ORIG" "$TMP/meta.json" "$OUT"
echo "reconstructed -> $OUT"
