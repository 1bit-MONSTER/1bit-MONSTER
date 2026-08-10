#!/usr/bin/env bash
# reboot.sh — reboot this box with hardware-watchdog recovery.
#
# Arms the AMD FCH watchdog (sp5100_tco) before rebooting. If the reboot
# hangs — shutdown, firmware, POST, whatever — the chipset hard-resets the
# box after the timeout, so it always comes back without a manual cold
# reboot. Requires `watchdog.stop_on_reboot=0` on the kernel cmdline
# (set in /etc/default/grub) so the timer is not disarmed at reboot.
#
# Usage:
#   sudo ./scripts/reboot.sh            # clean reboot, watchdog-armed (default 180s)
#   sudo WATCHDOG_REBOOT_TIMEOUT=60 ./scripts/reboot.sh
#   sudo ./scripts/reboot.sh --force    # skip systemd: sync, arm, let the
#                                       # watchdog do the reset (systemd wedged)
set -euo pipefail

TIMEOUT="${WATCHDOG_REBOOT_TIMEOUT:-180}"   # seconds before hard reset
WD=/dev/watchdog
WDSYS=/sys/class/watchdog/watchdog0

arm() {
    if [ ! -e "$WD" ]; then
        echo "reboot.sh: WARNING: no watchdog device — plain reboot, no recovery"
        return
    fi
    echo "reboot.sh: arming hardware watchdog (${TIMEOUT}s hard reset if hung)"
    # set timeout: sysfs is usually read-only, so use the WDIOC_SETTIMEOUT ioctl
    python3 -c "
import os, fcntl, struct
WDIOC_SETTIMEOUT = 0xC0045706
fd = os.open('$WD', os.O_RDWR)
fcntl.ioctl(fd, WDIOC_SETTIMEOUT, struct.pack('i', $TIMEOUT))
os.close(fd)
" 2>/dev/null || echo "$TIMEOUT" > "$WDSYS/timeout" 2>/dev/null \
    || echo "reboot.sh: WARNING: could not set timeout, using current value"
    # opening /dev/watchdog arms it; holding fd open guarantees it stays
    # armed and unpetted even after systemd exits. If systemd already holds
    # it (RuntimeWatchdogSec=60), the watchdog core returns EBUSY — that's
    # fine: systemd's own watchdog is then the safety net, so warn and
    # continue to the reboot instead of aborting (set -e would kill the
    # script before `systemctl reboot` ever runs — the 2026-08-10 hang).
    exec 9<>"$WD" 2>/dev/null \
        || echo "reboot.sh: WARNING: $WD busy (systemd holds it) — systemd RuntimeWatchdog covers hangs; continuing"
}

case "${1:-}" in
    --force)
        arm
        sync
        echo "reboot.sh: force path — watchdog resets in ${TIMEOUT}s"
        sleep "$((TIMEOUT + 5))"
        ;;
    *)
        arm
        sync
        systemctl reboot
        ;;
esac
