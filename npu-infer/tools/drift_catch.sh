#!/usr/bin/env bash
# drift_catch.sh — post-boot drift detector for the runtime-drift phenomenon
# (#2065). Runs the Round-37 health gate; if the runtime's signature has
# drifted (argmax != 397 / std != 3.38) it snapshots the NPU firmware/driver
# state to a report dir so the root cause (below kernel visibility, R66) can
# be investigated at the drifting boot.
#
# Usage:   ./drift_catch.sh [model_dir]
# Env:     DRIFT_OUT (default /tmp/drift_catch_<bootid>)
# Register at boot (e.g. after the Round-37 drill) to catch the next drift.
set -u
MODEL_DIR="${1:-/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2}"
HARNESS=/tmp/txn_decode/run_qwen3_npu   # rebuild per the Round-37 drill if absent
OUT="${DRIFT_OUT:-/tmp/drift_catch_$(journalctl --list-boots 2>/dev/null | head -1 | awk '{print $1}')_$(date +%H%M%S)}"

if [ ! -x "$HARNESS" ]; then echo "MISSING harness — run the Round-37 drill first (logits_1000 gate)"; exit 2; fi
mkdir -p "$OUT"

echo "=== drift gate: run_qwen3_npu token-1000 ==="
NPU_PROMPT_IDS=1000 timeout 300 "$HARNESS" "$MODEL_DIR" 1 >/dev/null 2>&1
LG=/tmp/txn_decode/logits_1000.bin
[ -f "$LG" ] || { echo "no logits file"; exit 2; }

python3 - "$LG" "$OUT" <<'PYEOF'
import sys, numpy as np
raw = open(sys.argv[1],'rb').read()
u16 = np.frombuffer(raw, dtype='<u2')
f32 = (u16.astype('<u4') << 16).view('<f4')
argmax, std = int(f32.argmax()), float(f32.std())
healthy = (argmax == 397 and 3.2 < std < 3.6)
print(f"argmax={argmax} std={std:.4f} -> {'HEALTHY' if healthy else 'DRIFTED'}")
open(sys.argv[2]+'/gate_result.txt','w').write(f"argmax={argmax} std={std:.4f} healthy={healthy}\n")
sys.exit(0 if healthy else 3)
PYEOF
RC=$?
if [ $RC -eq 3 ]; then
  echo "=== DRIFT DETECTED — capturing firmware/driver state to $OUT ==="
  for f in dump_fw_trace dump_fw_trace_buffer dump_fw_log ctx_rq get_app_health powerstate telemetry_health; do
    sudo cat "/sys/kernel/debug/accel/0000:c6:00.1/$f" > "$OUT/$f.txt" 2>/dev/null
  done
  sudo dmesg > "$OUT/dmesg.txt" 2>/dev/null
  xrt-smi examine > "$OUT/xrt-smi.txt" 2>/dev/null
  cat /proc/cmdline > "$OUT/cmdline.txt"
  cp "$LG" "$OUT/logits_1000.bin" 2>/dev/null
  echo "report saved to $OUT — investigate per #2065 / R66 (drift below kernel visibility)"
fi
exit $RC
