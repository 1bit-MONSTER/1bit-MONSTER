#!/usr/bin/env bash
set -euo pipefail
# packaging/live/test-live.sh — boots the live .img in headless QEMU (UEFI/
# OVMF; no install phase — the image boots straight to systemd), waits for
# SSH, then verifies the appliance came up: engine, unified service, GTT
# cmdline, held packages, HRX bundle, /v1/health.
#
# Usage: test-live.sh /path/to/1bit-monster-live-26.04-amd64.img /path/to/test_key

LIVE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMG="${1:?usage: test-live.sh <img> <test_key>}"
TEST_KEY="${2:?usage: test-live.sh <img> <test_key>}"
PIDFILE="${LIVE_DIR}/build/qemu-live.pid"
SSH_PORT=2223

# pipe-free OVMF detection: `ls | head -1` races SIGPIPE under set -o pipefail
# (head exits, ls dies, substitution fails, set -e kills the script silently)
OVMF_CODE=""
for f in /usr/share/OVMF/OVMF_CODE_4M.fd /usr/share/ovmf/OVMF_CODE.fd; do
  [ -f "$f" ] && { OVMF_CODE="$f"; break; }
done
OVMF_VARS=""
for f in /usr/share/OVMF/OVMF_VARS_4M.fd /usr/share/ovmf/OVMF_VARS.fd; do
  [ -f "$f" ] && { OVMF_VARS="$f"; break; }
done
[ -n "$OVMF_CODE" ] && [ -n "$OVMF_VARS" ] || { echo "FATAL: OVMF not installed (apt install ovmf)" >&2; exit 1; }
sudo -n true 2>/dev/null || { echo "needs passwordless sudo (kvm access for QEMU)" >&2; exit 1; }
VARS="${LIVE_DIR}/build/ovmf-vars.fd"
sudo cp "$OVMF_VARS" "$VARS"
sudo chown "$(id -u):$(id -g)" "$VARS"

echo "Booting live image in QEMU (UEFI/OVMF) — expect 2-5 min to SSH..."
sudo qemu-system-x86_64 \
  -m 8G -smp 4 -enable-kvm -cpu host \
  -drive file="$IMG",if=virtio,format=raw \
  -drive if=pflash,format=raw,unit=0,readonly=on,file="$OVMF_CODE" \
  -drive if=pflash,format=raw,unit=1,file="$VARS" \
  -netdev user,id=net0,hostfwd=tcp::${SSH_PORT}-:22 -device virtio-net,netdev=net0 \
  -display none -daemonize -pidfile "$PIDFILE"

cleanup() { sudo kill "$(cat "$PIDFILE")" 2>/dev/null || true; }
trap cleanup EXIT

echo "Waiting for SSH..."
UP=0
for i in $(seq 1 60); do
  if ssh -i "$TEST_KEY" -p "$SSH_PORT" -o StrictHostKeyChecking=no \
      -o UserKnownHostsFile=/dev/null -o ConnectTimeout=5 \
      monster@localhost true 2>/dev/null; then
    echo "SSH is up after ~$((i*10))s"; UP=1; break
  fi
  sleep 10
done
[ "$UP" -eq 1 ] || { echo "FAIL: SSH never came up after 10 minutes"; exit 1; }

RUN() { ssh -i "$TEST_KEY" -p "$SSH_PORT" -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/dev/null monster@localhost "$@"; }

FAIL=0
echo "-- kernel --"
RUN "uname -r" || { echo "FAIL: no kernel"; FAIL=1; }
echo "-- engine installed --"
RUN "dpkg -l | grep -i 1bit" >/dev/null || { echo "FAIL: engine .deb not installed"; FAIL=1; }
echo "-- unified service active --"
RUN "systemctl is-active 1bit-unified.service" || { echo "FAIL: 1bit-unified.service not active"; FAIL=1; }
echo "-- GTT kernel params --"
RUN "cat /proc/cmdline | grep -q 'ttm.pages_limit=31457280' && echo ok" || { echo "FAIL: GTT params missing"; FAIL=1; }
echo "-- held packages --"
RUN "apt-mark showhold | grep -qE 'mesa-vulkan-drivers|libamdhip64-7|bolt' && echo ok" || { echo "FAIL: holds missing"; FAIL=1; }
echo "-- HRX bundle present --"
RUN "test -x /opt/hrx/bin/llama-server && ls /opt/hrx/lib/libhrx.so* >/dev/null && echo ok" || { echo "FAIL: HRX bundle missing"; FAIL=1; }
echo "-- HIP runtime libs resolve --"
RUN "ldconfig -p | grep -qE 'libamdhip64.so.7|libhipblas.so.3|libomp.so' && echo ok" || { echo "FAIL: HIP runtime libs not found"; FAIL=1; }
echo "-- API health --"
RUN "curl -sf localhost:8088/v1/health" || { echo "FAIL: /v1/health not responding"; FAIL=1; }
echo "-- model-fetch unit enabled --"
RUN "systemctl is-enabled 1bit-model-fetch.service" >/dev/null || { echo "FAIL: model-fetch not enabled"; FAIL=1; }

if [ "$FAIL" -eq 0 ]; then echo "PASS"; else echo "One or more checks FAILED"; exit 1; fi
