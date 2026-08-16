#!/usr/bin/env bash
set -euo pipefail
# packaging/iso/test-qemu.sh — boots the built ISO in headless QEMU, waits
# for unattended autoinstall to finish, then verifies the appliance came up
# correctly over SSH.
#
# Usage: test-qemu.sh /path/to/1bit-monster-26.04-amd64.iso /path/to/test_key

ISO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISO="${1:?usage: test-qemu.sh /path/to/1bit-monster-26.04-amd64.iso /path/to/test_key}"
TEST_KEY="${2:?usage: test-qemu.sh /path/to/1bit-monster-26.04-amd64.iso /path/to/test_key}"
DISK="${ISO_DIR}/build/test-disk.qcow2"
PIDFILE="${ISO_DIR}/build/qemu.pid"
SSH_PORT=2222

rm -f "$DISK"
qemu-img create -f qcow2 "$DISK" 40G

echo "Booting installer — unattended autoinstall runs now, expect 10-20 min..."
# -boot once=d (not a persistent -boot d): the CD must only be the boot
# device for the FIRST boot. After autoinstall finishes, curtin reboots
# the guest — with a persistent -boot d that reboot would boot the ISO
# again instead of the freshly-installed disk, sending the VM back into
# the installer (or worse, re-triggering autoinstall against an
# already-partitioned disk) instead of ever reaching the installed
# target system this script is trying to test.
# -cpu host: without this, QEMU's default virtual CPU model (a conservative,
# portable baseline) does NOT expose this host's actual instruction set —
# -enable-kvm only selects the acceleration backend, it does not imply the
# guest sees the host's real CPU features. unified_server crashed with
# SIGILL (illegal instruction) under the default model; -cpu host passes
# through the real underlying CPU's full feature set, matching what the
# appliance would see on actual Strix Halo hardware. Found by booting the
# resulting disk directly and bisecting the crash against QEMU CLI flags.
qemu-system-x86_64 \
  -m 8G -smp 4 -enable-kvm -cpu host \
  -drive file="$DISK",if=virtio \
  -cdrom "$ISO" \
  -boot once=d \
  -netdev user,id=net0,hostfwd=tcp::${SSH_PORT}-:22 -device virtio-net,netdev=net0 \
  -display none -daemonize -pidfile "$PIDFILE"

cleanup() { [ -f "$PIDFILE" ] && kill "$(cat "$PIDFILE")" 2>/dev/null || true; }
trap cleanup EXIT

echo "Waiting for SSH (install + first boot)..."
UP=0
for i in $(seq 1 90); do
  if ssh -i "$TEST_KEY" -p "$SSH_PORT" -o StrictHostKeyChecking=no \
      -o UserKnownHostsFile=/dev/null -o ConnectTimeout=5 \
      monster@localhost true 2>/dev/null; then
    echo "SSH is up after ~$((i*20))s"
    UP=1
    break
  fi
  sleep 20
done
[ "$UP" -eq 1 ] || { echo "FAIL: SSH never came up after 30 minutes"; exit 1; }

RUN() { ssh -i "$TEST_KEY" -p "$SSH_PORT" -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/dev/null monster@localhost "$@"; }

FAIL=0
echo "-- engine installed --"
RUN "dpkg -l | grep -i 1bit" || { echo "FAIL: engine .deb not installed"; FAIL=1; }
echo "-- unified service active --"
RUN "systemctl is-active 1bit-unified.service" || { echo "FAIL: 1bit-unified.service not active"; FAIL=1; }
echo "-- GTT kernel params present --"
RUN "cat /proc/cmdline | grep -o 'ttm.pages_limit=[0-9]*'" || { echo "FAIL: GTT kernel params missing"; FAIL=1; }
echo "-- driver packages held --"
RUN "apt-mark showhold | grep -q mesa-vulkan-drivers" || { echo "FAIL: mesa-vulkan-drivers not held"; FAIL=1; }
RUN "apt-mark showhold | grep -q bolt" || { echo "FAIL: bolt not held"; FAIL=1; }
echo "-- Thunderbolt daemon available --"
# bolt.service is D-Bus/udev-activated (Type=dbus, no [Install] section) —
# it only transitions to "active" when a Thunderbolt controller actually
# enumerates. QEMU's guest has no such hardware, so "is-active" would always
# read "inactive" here regardless of whether bolt itself is correctly
# installed. There's also no NOPASSWD sudo for the test user, so we can't
# force-start it as a workaround. "is-enabled" (unprivileged, read-only)
# confirms the unit is present and correctly registered for activation
# ("static" for a unit with no [Install] section, which is bolt's normal
# state) without requiring real Thunderbolt hardware to prove it.
RUN "systemctl is-enabled bolt.service" || { echo "FAIL: bolt.service not enabled/available"; FAIL=1; }
echo "-- API health --"
RUN "curl -sf localhost:8088/v1/health" || { echo "FAIL: /v1/health not responding"; FAIL=1; }

if [ "$FAIL" -eq 0 ]; then
  echo "PASS"
else
  echo "One or more checks FAILED — see output above"
  exit 1
fi
