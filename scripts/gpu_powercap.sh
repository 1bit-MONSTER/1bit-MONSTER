#!/usr/bin/env bash
# gpu_powercap.sh — Strix Halo iGPU SCLK cap for consistent benchmark results.
#
# Problem: Radeon 8060S iGPU boosts to 2900 MHz / 180W / 111°C under sustained
# inference, causing thermal throttling and unreliable benchmark numbers.
#
# Fix: cap SCLK to 2400 MHz via amdgpu overdrive.
#   decode  (memory-bandwidth-bound): flat ~30 tok/s regardless of clock
#   prefill (compute-bound):          ~126 tok/s at 2400 MHz (vs 134 @ 2600 MHz
#                                     which heat-soaks)
#
# Usage:
#   sudo ./scripts/gpu_powercap.sh           # apply cap once
#   sudo ./scripts/gpu_powercap.sh --watch   # apply + re-apply every 60s (survives GPU resets)
#   sudo ./scripts/gpu_powercap.sh --status  # print current clock / temp / power
#   sudo ./scripts/gpu_powercap.sh --restore # remove OD cap, return to driver defaults
#
# Requirements:
#   - amdgpu.ppfeaturemask=0xffffffff  kernel boot param (enables overdrive)
#   - root / sudo
#   - hwmon sysfs exposed by amdgpu driver
#
# Reference: https://github.com/hogeheer499-commits/strix-halo-guide/issues/24
#
# SCLK table at the recommended cap:
#   SCLK  | Temp | Power | Prefill  | Decode
#   2200   | 68°C |  85W  | 117 t/s  | 30 t/s
#   2400   | 70°C | 100W  | 126 t/s  | 30 t/s   ← default cap
#   2500   | 70°C | 107W  | 130 t/s  | 30 t/s
#   2600   | 80°C | 120W  | 134 t/s  | 30 t/s   (heat-soaks over sustained run)

set -euo pipefail

# ── Target SCLK in MHz ────────────────────────────────────────────────────────
SCLK_MIN=600
SCLK_MAX=2400

# ── Locate the Strix Halo iGPU via stable PCI path ───────────────────────────
# card0/card1 enumeration is NOT stable across reboots.  Use the PCI BDF.
PCI_BDF="0000:c5:00.0"
OD_PATH="/sys/bus/pci/devices/${PCI_BDF}/pp_od_clk_voltage"
PERF_PATH="/sys/class/drm/card1/device/power_dpm_force_performance_level"

die() { echo "ERROR: $*" >&2; exit 1; }

check_root() {
    [[ $EUID -eq 0 ]] || die "This script must be run as root (sudo $0 $*)"
}

find_perf_path() {
    # Resolve the correct card sysfs path via PCI device
    local pci_dev="/sys/bus/pci/devices/${PCI_BDF}"
    [[ -d "$pci_dev" ]] || die "PCI device ${PCI_BDF} not found. Check lspci | grep VGA"
    local card
    card=$(basename "$(readlink -f "${pci_dev}/drm/card"* 2>/dev/null | head -1)" 2>/dev/null || true)
    if [[ -n "$card" ]]; then
        PERF_PATH="/sys/class/drm/${card}/device/power_dpm_force_performance_level"
    fi
}

apply_cap() {
    find_perf_path

    # Enable manual performance level so OD writes are accepted
    echo "manual" > "${PERF_PATH}" || die "Failed to set manual perf level (is amdgpu.ppfeaturemask=0xffffffff set?)"

    # OD voltage table: set min first, then max, then commit.
    # ORDER MATTERS — driver rejects max < current min.
    echo "s 0 ${SCLK_MIN}" > "${OD_PATH}" || die "Failed to write SCLK min"
    echo "s 1 ${SCLK_MAX}" > "${OD_PATH}" || die "Failed to write SCLK max"
    echo "c"                > "${OD_PATH}" || die "Failed to commit OD"

    echo "[gpu_powercap] SCLK capped: ${SCLK_MIN}–${SCLK_MAX} MHz"
}

restore() {
    find_perf_path
    echo "r" > "${OD_PATH}" 2>/dev/null || true
    echo "auto" > "${PERF_PATH}" 2>/dev/null || true
    echo "[gpu_powercap] OD reset to driver defaults"
}

status() {
    local hwmon
    hwmon=$(find /sys/bus/pci/devices/${PCI_BDF}/hwmon -name "temp1_input" 2>/dev/null | head -1 || true)
    if [[ -n "$hwmon" ]]; then
        local temp_mc power_uw
        temp_mc=$(cat "$(dirname "$hwmon")/temp1_input" 2>/dev/null || echo "?")
        power_uw=$(cat "$(dirname "$hwmon")/power1_average" 2>/dev/null || echo "?")
        local temp_c power_w
        [[ "$temp_mc" != "?" ]] && temp_c=$(echo "scale=1; $temp_mc / 1000" | bc) || temp_c="?"
        [[ "$power_uw" != "?" ]] && power_w=$(echo "scale=1; $power_uw / 1000000" | bc) || power_w="?"
        echo "Temp:  ${temp_c}°C"
        echo "Power: ${power_w}W"
    fi
    if [[ -f "$OD_PATH" ]]; then
        echo "--- OD table ---"
        cat "$OD_PATH"
    fi
}

# ── Main ──────────────────────────────────────────────────────────────────────
CMD="${1:-}"
case "$CMD" in
    --status)
        status
        ;;
    --restore)
        check_root
        restore
        ;;
    --watch)
        check_root
        echo "[gpu_powercap] Applying cap every 60s (GPU resets revert OD). Ctrl-C to stop."
        while true; do
            apply_cap
            sleep 60
        done
        ;;
    "")
        check_root
        apply_cap
        ;;
    *)
        echo "Usage: $0 [--status | --restore | --watch]"
        exit 1
        ;;
esac
