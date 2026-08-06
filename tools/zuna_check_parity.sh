#!/usr/bin/env bash
# ZUNA CPU port parity check.
# Builds the standalone C++ port (tools/zuna_port.cpp), runs it on the golden
# tokens + initial noise, and compares encoder latent + reconstruction against
# the reference zuna traces with the given tolerances.
#
# Usage: tools/zuna_check_parity.sh <root_dir>
#   root_dir has: weights.bin weights.json tokens.bin tok_idx.bin z_true.bin
#                 enc_out_ref.npy recon_ref.npy
# Output: prints ENC MAE and RECON MAE; exit 0 if within tolerances.
set -euo pipefail
ROOT="$1"
ENC_TOL="${ENC_TOL:-1e-4}"      # encoder MAE tolerance (abs)
RECON_TOL="${RECON_TOL:-1e-4}"  # reconstruction MAE tolerance (abs)
SRC="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$(mktemp -d)/zuna_port"
g++ -O2 -std=c++17 -o "$BIN" "$SRC/tools/zuna_port.cpp"
ENC_CMP="$(mktemp).enc"; RECON_CMP="$(mktemp).recon"
"$BIN" "$ROOT" "$ROOT/tokens.bin" "$ROOT/tok_idx.bin" "$RECON_CMP" "$ENC_CMP" 0 "$ROOT/z_true.bin" >/dev/null
python3 - "$ROOT" "$ENC_CMP" "$RECON_CMP" "$ENC_TOL" "$RECON_TOL" <<'PY'
import sys, numpy as np
root, ec, rc, et, rt = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), float(sys.argv[5])
enc_ref = np.load(f"{root}/enc_out_ref.npy")[0].ravel()
enc_cpp = np.fromfile(ec, dtype=np.float32).ravel()
recon_ref = np.load(f"{root}/recon_ref.npy")[0].ravel()
recon_cpp = np.fromfile(rc, dtype=np.float32).ravel()
e = float(np.abs(enc_ref - enc_cpp).mean())
r = float(np.abs(recon_ref - recon_cpp).mean())
print(f"ENC MAE   = {e:.3e}  (tol {et:.0e})  {'PASS' if e <= et else 'FAIL'}")
print(f"RECON MAE = {r:.3e}  (tol {rt:.0e})  {'PASS' if r <= rt else 'FAIL'}")
print(f"ENC corr  = {np.corrcoef(enc_ref, enc_cpp)[0,1]:.6f}")
print(f"RECON corr= {np.corrcoef(recon_ref, recon_cpp)[0,1]:.6f}")
sys.exit(0 if (e <= et and r <= rt) else 1)
PY
