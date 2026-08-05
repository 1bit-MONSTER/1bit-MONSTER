#!/usr/bin/env bash
# npu-driver-reapply.sh — re-apply the custom amdxdna NPU driver after kernel updates.
#
# Ubuntu's in-tree amdxdna (from the linux-image package) is replaced by a
# custom build from /home/bcloud/xdna-driver, signed with the local MOK key
# ("strixhalo Secure Boot Module Signature key", /var/lib/dkms/mok.key).
# Kernel upgrades silently restore the stock in-tree module — run this after
# every kernel update (or from a post-upgrade hook) to keep the NPU stack
# consistent with the installed XRT.
#
# Idempotent: exits 0 without rebuilding if the running kernel's module is
# already the custom signed build.
#
# Usage: sudo ./scripts/npu-driver-reapply.sh [--reload]
set -euo pipefail

KVER="$(uname -r)"
KO_DIR="/lib/modules/$KVER/kernel/drivers/accel/amdxdna"
KO="$KO_DIR/amdxdna.ko.zst"
SRC="/home/bcloud/xdna-driver"
BUILD="$SRC/build/Release"
KO_RAW="$BUILD/drivers/accel/amdxdna.ko"
SIGN_KEY="/var/lib/dkms/mok.key"
SIGN_CERT="/var/lib/dkms/mok.pub"

if [ ! -f "$KO" ]; then
    echo "ERROR: $KO not found — is kernel $KVER installed?" >&2
    exit 1
fi

# Already custom? The in-tree Ubuntu module is signed by Canonical/Ubuntu's key.
if modinfo "$KO" 2>/dev/null | grep -q 'strixhalo'; then
    echo "OK: amdxdna for $KVER is already the custom signed build — nothing to do."
    exit 0
fi

echo "==> Stock in-tree module detected for $KVER — rebuilding custom driver."

# ── rebuild ──
if [ ! -d "/usr/src/linux-headers-$KVER" ]; then
    echo "ERROR: linux-headers-$KVER not installed — install it first (apt install linux-headers-$KVER)." >&2
    exit 1
fi
if [ ! -d "$BUILD" ]; then
    echo "ERROR: build dir $BUILD missing — configure first (cmake -B $BUILD -DCMAKE_BUILD_TYPE=Release $SRC)." >&2
    exit 1
fi
# Clean first: stale amdxdna.mod/Module.symvers in the copy tree cause
# bogus modpost "undefined symbol" failures on incremental rebuilds.
make -C "$BUILD/drivers/accel" -f drivers/accel/amdxdna/Makefile \
    BUILD_ROOT_DIR="$BUILD/drivers/accel/amdxdna" UMQ_HELLO_TEST=n XDNA_BUS_TYPE=pci clean >/dev/null 2>&1 || true
cmake --build "$BUILD" --target driver -j"$(nproc)"
[ -f "$KO_RAW" ] || { echo "ERROR: build did not produce $KO_RAW" >&2; exit 1; }

# ── sign (required: SecureBoot + lockdown enforce module signatures) ──
SIGN_FILE="/usr/src/linux-headers-$KVER/scripts/sign-file"
"$SIGN_FILE" sha256 "$SIGN_KEY" "$SIGN_CERT" "$KO_RAW"

# ── install ──
if [ ! -f "$KO_DIR/amdxdna.ko.zst.in-tree" ]; then
    cp "$KO" "$KO_DIR/amdxdna.ko.zst.in-tree"   # keep stock for downgrade
fi
zstd -19 -f -q -o "$KO" "$KO_RAW"
chmod 644 "$KO"
depmod -a "$KVER"

# ── verify ──
if modinfo "$KO" 2>/dev/null | grep -q 'strixhalo'; then
    echo "OK: custom amdxdna installed and signed for $KVER."
else
    echo "ERROR: signature verification failed on installed module." >&2
    exit 1
fi

if [ "${1:-}" = "--reload" ]; then
    modprobe -r amdxdna && modprobe amdxdna
    echo "OK: module reloaded. Check 'xrt-smi examine'."
else
    echo "NOTE: reboot (or rmmod/modprobe amdxdna) to load the new module."
fi
