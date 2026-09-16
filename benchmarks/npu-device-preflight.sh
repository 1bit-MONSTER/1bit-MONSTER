#!/usr/bin/env bash
# NPU measurement pre-flight: is anyone holding /dev/accel/accel0 right now?
# A non-empty output means any NPU number you take is suspect.
# (Recipe from @agent-7f1cce, 2026-09-16.)
#
# GAP THIS CLOSES: the engine takes /tmp/1bit-npu-device.lock itself, but
# standalone benches (npu-infer/tools/moe_smoke) and the xclbin generators
# do NOT — those are the runs that can silently overlap someone else's.
set -u
holders=""
for p in /proc/[0-9]*; do
  ls -l "$p/fd" 2>/dev/null | grep -q accel0 && holders="$holders ${p#/proc/}"
done
if [ -n "$holders" ]; then
  echo "BUSY — accel0 open by:$holders"
  for pid in $holders; do tr '\0' ' ' < "$pid/cmdline" 2>/dev/null | cut -c1-160; echo; done
  [ -e /tmp/1bit-npu-device.lock ] && echo "lock present: $(ls -l /tmp/1bit-npu-device.lock)"
  exit 1
fi
echo "IDLE — no PID holds accel0; lock: $([ -e /tmp/1bit-npu-device.lock ] && echo present || echo absent)"
exit 0
