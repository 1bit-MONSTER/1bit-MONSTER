#!/usr/bin/env bash
# npu-driver-reapply.sh — install the custom amdxdna NPU driver for the running kernel.
#
# Ubuntu's in-tree amdxdna (from the linux-image package) can be replaced by a build from
# $SRC (default /home/bcloud/xdna-driver). Kernel upgrades silently restore the stock
# in-tree module, so this exists to put the custom build back afterwards.
#
# WHAT THIS BOX ACTUALLY LOOKS LIKE (discovered, not assumed — #2459, #2517):
#   * the running kernel ships the module UNCOMPRESSED at
#     /lib/modules/$KVER/kernel/drivers/accel/amdxdna/amdxdna.ko, so the `amdxdna.ko.zst`
#     this script used to look for does not exist and its first check failed on every run;
#   * the installed module is UNSIGNED (`modinfo` reports no signer) while Secure Boot on
#     this box is not enforcing signatures — the MOK key it used to sign with
#     (/var/lib/dkms/mok.key) is not present either;
#   * $SRC is the LEGACY out-of-tree tree (`src/driver/amdxdna`), which upstream deleted in
#     813e0bf on 2026-09-01, and the module now running was built from it 2026-09-01 11:50.
# Refusing that tree is the point of the provenance gate below: this script must not be the
# thing that reinstalls a driver upstream has removed. Moving to the upstream staging driver
# is #2459's call, not this script's — pass --allow-legacy-tree to override knowingly.
#
# Usage: sudo ./scripts/npu-driver-reapply.sh [--reload] [--allow-legacy-tree] [--no-sign]
#   --reload             modprobe -r/-a after installing
#   --allow-legacy-tree  install a module built from the deleted legacy tree anyway
#   --no-sign            install without signing (this box's module is unsigned today)

KVER="$(uname -r)"
KO_DIR="/lib/modules/$KVER/kernel/drivers/accel/amdxdna"
SRC="${NPU_DRIVER_SRC:-/home/bcloud/xdna-driver}"
BUILD="$SRC/build/Release"
KO_RAW="$BUILD/drivers/accel/amdxdna.ko"
SIGN_KEY="/var/lib/dkms/mok.key"
SIGN_CERT="/var/lib/dkms/mok.pub"

# ─────────────────────────────────────────────────────────────────────────────
# Pure helpers. Testing/npu_driver_reapply_selfcheck.sh sources this file to
# exercise them; the guard below stops the sourcing before any host work.
# ─────────────────────────────────────────────────────────────────────────────

# module_tree_of <file> — which tree a built module came from, by the compile path baked
# into its bytes. Prints "legacy" | "upstream" | "unknown"; a caller must treat "unknown"
# as cannot-determine (exit 2), never as fine — this is how #2459 identified the running
# module, and a module whose provenance cannot be read is exactly the case that got us here.
module_tree_of() {
    local found
    found="$(grep -aoE 'src/driver/amdxdna|drivers/accel/amdxdna' "$1" 2>/dev/null | sort -u || true)"
    case "$found" in
        *src/driver/amdxdna*)    printf 'legacy\n' ;;
        *drivers/accel/amdxdna*) printf 'upstream\n' ;;
        *)                       printf 'unknown\n' ;;
    esac
}

# find_installed_module <dir> — the module as installed, compressed or not. Prints the
# path; non-zero when there is none.
find_installed_module() {
    local cand
    for cand in "$1/amdxdna.ko.zst" "$1/amdxdna.ko"; do
        [ -f "$cand" ] && { printf '%s\n' "$cand"; return 0; }
    done
    return 1
}

# installed_matches_raw <installed> <built> — byte comparison that understands the
# installed form, used to verify an unsigned install (modinfo can only prove a signature).
installed_matches_raw() {
    case "$1" in
        *.zst) command -v zstd >/dev/null 2>&1 && zstd -dc "$1" 2>/dev/null | cmp -s - "$2" ;;
        *)     cmp -s "$1" "$2" ;;
    esac
}

# source_tree_of <dir> — which tree a checkout is: "legacy" when it still tracks the deleted
# out-of-tree driver, "upstream" when it does not, "unknown" when it is not a git work tree
# at all. The last case matters: a tarball extract of the legacy tree is not a git tree, and
# an inline `git ls-files` test would have read that as "not legacy" and built from it.
source_tree_of() {
    local dir="$1"
    if ! git -C "$dir" rev-parse --git-dir >/dev/null 2>&1; then
        printf 'unknown\n'
    elif git -C "$dir" ls-files --error-unmatch src/driver/amdxdna >/dev/null 2>&1; then
        printf 'legacy\n'
    else
        printf 'upstream\n'
    fi
}

if [ "${BASH_SOURCE[0]}" != "$0" ]; then return 0; fi

# ─────────────────────────────────────────────────────────────────────────────
# Host work. The options are set here, not at the top: sourcing a helper must not
# change the caller's shell options (it did, and it made the selfcheck exit on the
# first intentionally-failing call).
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

RELOAD=0; ALLOW_LEGACY=0; SIGN=1
for arg in "$@"; do
    case "$arg" in
        --reload)            RELOAD=1 ;;
        --allow-legacy-tree) ALLOW_LEGACY=1 ;;
        --no-sign)           SIGN=0 ;;
        -h|--help)           sed -n '2,25p' "$0"; exit 0 ;;
        *) echo "unknown argument: $arg (try --help)" >&2; exit 2 ;;
    esac
done

# The installed module, as this kernel names it. `modinfo -n` is what the kernel itself
# reports; the directory scan is the fallback for a module that is installed but not yet
# in modules.dep.
KO="$(modinfo -n amdxdna 2>/dev/null || true)"
if [ -z "$KO" ] || [ ! -f "$KO" ]; then
    KO="$(find_installed_module "$KO_DIR" || true)"
fi
if [ -z "$KO" ] || [ ! -f "$KO" ]; then
    echo "ERROR: no amdxdna module for $KVER." >&2
    echo "  modinfo -n amdxdna: ${KO:-<nothing>}" >&2
    echo "  looked for: $KO_DIR/amdxdna.ko.zst and $KO_DIR/amdxdna.ko" >&2
    echo "  is kernel $KVER installed?" >&2
    exit 1
fi
echo "installed module: $KO ($(module_tree_of "$KO") tree)"

# Already the custom signed build? (The stock module is signed by the distro key, the
# custom one by the local MOK key. An unsigned custom build cannot be told apart this way
# and is reported rather than assumed.)
if modinfo "$KO" 2>/dev/null | grep -q 'strixhalo'; then
    echo "OK: amdxdna for $KVER is already the custom signed build — nothing to do."
    exit 0
fi
installed_tree="$(module_tree_of "$KO")"
if [ "$installed_tree" = legacy ]; then
    echo "note: the installed module was built from the LEGACY out-of-tree tree ($SRC is that" >&2
    echo "      tree today), which upstream deleted in 813e0bf on 2026-09-01 — see #2459." >&2
fi

# Where the build comes from. This is the gate that keeps the obsolete tree from being
# reinstalled: it fails before anything is built, and says which tree and which commit.
if [ ! -d "$SRC" ]; then
    echo "ERROR: source tree $SRC not found (override with NPU_DRIVER_SRC=/path)." >&2
    exit 1
fi
src_tree="$(source_tree_of "$SRC")"
if [ "$src_tree" = unknown ]; then
    echo "ERROR: cannot tell which driver tree $SRC is (not a git work tree)." >&2
    echo "  Refusing to build: the legacy tree upstream deleted in 813e0bf cannot be told apart" >&2
    echo "  from the upstream one by provenance here. Use a git checkout (#2459)." >&2
    exit 2
fi
if [ "$src_tree" = legacy ]; then
    head_rev="$(git -C "$SRC" log -1 --format='%h %ad %s' --date=short 2>/dev/null || echo unknown)"
    if [ "$ALLOW_LEGACY" -ne 1 ]; then
        echo "ERROR: $SRC is the LEGACY out-of-tree driver tree (src/driver/amdxdna is tracked there)." >&2
        echo "  HEAD: $head_rev" >&2
        echo "  Upstream removed that tree in 813e0bf (2026-09-01); building from it reinstalls the" >&2
        echo "  driver this box is already running and that #2459 says is superseded by the upstream" >&2
        echo "  staging driver. Nothing was built." >&2
        echo "  Re-run with --allow-legacy-tree only if you have decided to stay on it on purpose." >&2
        exit 1
    fi
    echo "WARNING: building from the LEGACY tree on purpose (--allow-legacy-tree): $head_rev" >&2
fi

# ── rebuild ──
if [ ! -d "/usr/src/linux-headers-$KVER" ]; then
    echo "ERROR: linux-headers-$KVER not installed — install it first (apt install linux-headers-$KVER)." >&2
    exit 1
fi
if [ ! -d "$BUILD" ]; then
    echo "ERROR: build dir $BUILD missing — configure first (cmake -B $BUILD -DCMAKE_BUILD_TYPE=Release $SRC)." >&2
    exit 1
fi

echo "==> rebuilding the custom driver from $SRC"
# Clean first: stale amdxdna.mod/Module.symvers in the copy tree cause bogus modpost
# "undefined symbol" failures on incremental rebuilds.
make -C "$BUILD/drivers/accel" -f drivers/accel/amdxdna/Makefile \
    BUILD_ROOT_DIR="$BUILD/drivers/accel/amdxdna" UMQ_HELLO_TEST=n XDNA_BUS_TYPE=pci clean >/dev/null 2>&1 || true
cmake --build "$BUILD" --target driver -j"$(nproc)"
[ -f "$KO_RAW" ] || { echo "ERROR: build did not produce $KO_RAW" >&2; exit 1; }

# The built module must be the upstream one. A module's provenance is readable from its own
# bytes, so this needs no trust in the tree: "unknown" is a refusal too, because a module
# whose compile path cannot be read is the case that produced #2459.
built_tree="$(module_tree_of "$KO_RAW")"
case "$built_tree" in
    upstream)
        echo "provenance: built module is the upstream tree (drivers/accel/amdxdna)" ;;
    legacy)
        if [ "$ALLOW_LEGACY" -ne 1 ]; then
            echo "ERROR: the module just built came from the LEGACY tree (src/driver/amdxdna)." >&2
            echo "  Not installing it. See #2459 for moving to the upstream staging driver." >&2
            exit 1
        fi
        echo "WARNING: installing a LEGACY-tree module on purpose (--allow-legacy-tree)" >&2 ;;
    *)
        echo "ERROR: cannot determine which tree $KO_RAW was built from" >&2
        echo "  (no 'src/driver/amdxdna' or 'drivers/accel/amdxdna' compile path in its bytes)." >&2
        echo "  Refusing to install a module whose provenance cannot be read." >&2
        exit 2 ;;
esac

# ── sign ──
if [ "$SIGN" -eq 1 ]; then
    if [ ! -f "$SIGN_KEY" ] || [ ! -f "$SIGN_CERT" ]; then
        echo "ERROR: $SIGN_KEY / $SIGN_CERT are missing, so the module cannot be signed." >&2
        echo "  The module installed on this box is unsigned and loads (modinfo reports no" >&2
        echo "  signer), so signing is not enforced here — but installing unsigned is an" >&2
        echo "  operator decision: re-run with --no-sign, or provide the MOK key pair." >&2
        exit 1
    fi
    SIGN_FILE="/usr/src/linux-headers-$KVER/scripts/sign-file"
    "$SIGN_FILE" sha256 "$SIGN_KEY" "$SIGN_CERT" "$KO_RAW"
fi

# ── install, in the form this kernel already uses ──
if [ ! -f "$KO.in-tree" ]; then
    cp "$KO" "$KO.in-tree"   # keep what was there for a downgrade
fi
case "$KO" in
    *.ko.zst) zstd -19 -f -q -o "$KO" "$KO_RAW" ;;
    *)        cp -f "$KO_RAW" "$KO" ;;   # this kernel ships modules uncompressed
esac
chmod 644 "$KO"
depmod -a "$KVER"

# ── verify ──
if [ "$SIGN" -eq 1 ]; then
    if modinfo "$KO" 2>/dev/null | grep -q 'strixhalo'; then
        echo "OK: custom amdxdna installed and signed for $KVER."
    else
        echo "ERROR: signature verification failed on installed module." >&2
        exit 1
    fi
elif installed_matches_raw "$KO" "$KO_RAW"; then
    echo "OK: custom amdxdna installed UNSIGNED for $KVER (bytes match the build)."
else
    echo "ERROR: the installed module does not match the build ($KO vs $KO_RAW)." >&2
    exit 1
fi

if [ "$RELOAD" -eq 1 ]; then
    modprobe -r amdxdna && modprobe amdxdna
    echo "OK: module reloaded. Check 'xrt-smi examine'."
else
    echo "NOTE: reboot (or rmmod/modprobe amdxdna) to load the new module."
fi
