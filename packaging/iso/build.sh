#!/usr/bin/env bash
set -euo pipefail
# packaging/iso/build.sh — builds the 1bit.MONSTER appliance ISO.
# See docs/superpowers/specs/2026-08-16-ubuntu-iso-design.md.
#
# Usage: build.sh --ssh-key /path/to/id_ed25519.pub [--out DIR]
#
# Pins come from packaging/rocm-lane-pin.env (single source shared with
# scripts/setup-therock.sh) — the autoinstall seed is guarded below so it
# cannot drift from this file without the build failing loudly.

ISO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ISO_DIR}/../.." && pwd)"
[ -f "$REPO_ROOT/packaging/rocm-lane-pin.env" ] && . "$REPO_ROOT/packaging/rocm-lane-pin.env"

WORK="${ISO_DIR}/build"
UBUNTU_ISO_NAME="ubuntu-${UBUNTU_VER}-live-server-amd64.iso"
UBUNTU_ISO_URL="https://releases.ubuntu.com/${UBUNTU_VER}/${UBUNTU_ISO_NAME}"

SSH_KEY_PATH=""
OUT_DIR="$WORK"
while [ $# -gt 0 ]; do
  case "$1" in
    --ssh-key) SSH_KEY_PATH="$2"; shift 2 ;;
    --out) OUT_DIR="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done
[ -z "$SSH_KEY_PATH" ] && { echo "usage: build.sh --ssh-key /path/to/key.pub [--out DIR]" >&2; exit 1; }
[ -f "$SSH_KEY_PATH" ] || { echo "ssh key not found: $SSH_KEY_PATH" >&2; exit 1; }

mkdir -p "$WORK" "$OUT_DIR"

echo "[1/7] Building the .deb..."
( cd "${REPO_ROOT}/packaging" && make package-deb )
DEB_PATH="$(ls "${REPO_ROOT}"/packaging/build/1bit-systems_*_amd64.deb | head -1)"
[ -f "$DEB_PATH" ] || { echo "FATAL: no .deb produced by 'make package-deb'" >&2; exit 1; }

echo "[2/7] Fetching base Ubuntu ${UBUNTU_VER} ISO..."
if [ ! -f "${WORK}/${UBUNTU_ISO_NAME}" ]; then
  curl -fL --progress-bar "${UBUNTU_ISO_URL}" -o "${WORK}/${UBUNTU_ISO_NAME}"
  curl -fL "https://releases.ubuntu.com/${UBUNTU_VER}/SHA256SUMS" -o "${WORK}/SHA256SUMS"
  ( cd "$WORK" && grep "${UBUNTU_ISO_NAME}\$" SHA256SUMS | sha256sum -c - )
fi

echo "[3/7] Fetching pinned driver payload..."
bash "${ISO_DIR}/fetch-payload.sh"

echo "[3b/7] Verifying autoinstall seed pins match the shared pin file..."
for pat in "therock-${THEROCK_VERSION}-gfx1151.tar.gz" \
           "therock-install.sh" \
           "mesa-vulkan-drivers_${MESA_VER}_amd64.deb" \
           "libvulkan1_${VULKAN1_VER}_amd64.deb"; do
  grep -qF "$pat" "${ISO_DIR}/autoinstall.yaml.tmpl" || {
    echo "FATAL: autoinstall.yaml.tmpl does not reference '${pat}' — the seed drifted from packaging/rocm-lane-pin.env." >&2
    echo "       Regenerate the seed (or bump its versions) so ISO and live setup stay in lockstep." >&2
    exit 1
  }
done
echo "  seed pins match packaging/rocm-lane-pin.env"

echo "[4/7] Extracting base ISO..."
EXTRACT="${WORK}/extract"
rm -rf "$EXTRACT"
mkdir -p "$EXTRACT"
xorriso -osirrox on -indev "${WORK}/${UBUNTU_ISO_NAME}" -extract / "$EXTRACT"
chmod -R u+w "$EXTRACT"

echo "[5/7] Building autoinstall seed + staging payload pool..."
PASSWORD="$(openssl rand -base64 24)"
# python3 on Ubuntu 26.04 is 3.14+, which removed the `crypt` module — use
# openssl's SHA-512 crypt (same $6$ format autoinstall accepts).
PASSWORD_HASH="$(openssl passwd -6 "$PASSWORD")"
SSH_PUBKEY="$(cat "$SSH_KEY_PATH")"
sed \
  -e "s|__PASSWORD_HASH__|${PASSWORD_HASH}|" \
  -e "s|__SSH_PUBLIC_KEY__|${SSH_PUBKEY}|" \
  "${ISO_DIR}/autoinstall.yaml.tmpl" > "${EXTRACT}/autoinstall.yaml"
echo "$PASSWORD" > "${OUT_DIR}/console-recovery-password.txt"
chmod 600 "${OUT_DIR}/console-recovery-password.txt"
echo "  Console recovery password: ${OUT_DIR}/console-recovery-password.txt (NOT copied onto the ISO)"

mkdir -p "${EXTRACT}/pool"
cp "$DEB_PATH" "${EXTRACT}/pool/"
cp "${WORK}/payload/"*.deb "${EXTRACT}/pool/"
cp "${WORK}/payload/therock-"*.tar.gz "${EXTRACT}/pool/"
cp "${ISO_DIR}/therock-install.sh" "${EXTRACT}/pool/"
cp "${REPO_ROOT}/packaging/services/1bit-unified.service" "${EXTRACT}/pool/"
cp "${REPO_ROOT}/packaging/services/1bit-model-fetch.service" "${EXTRACT}/pool/"
cp "${REPO_ROOT}/packaging/model-download.sh" "${EXTRACT}/pool/"

echo "[6/7] Wiring autoinstall boot entry..."
GRUB_CFG="${EXTRACT}/boot/grub/grub.cfg"
[ -f "$GRUB_CFG" ] || { echo "FATAL: ${GRUB_CFG} not found — inspect ${EXTRACT}/boot/grub/ and fix this path" >&2; exit 1; }
if ! grep -q "autoinstall" "$GRUB_CFG"; then
  sed -i 's|linux\t/casper/vmlinuz|linux\t/casper/vmlinuz autoinstall ds=nocloud\\;s=/cdrom/|' "$GRUB_CFG"
fi

echo "[7/7] Repacking ISO..."
OUT_ISO="${OUT_DIR}/1bit-monster-${UBUNTU_VER}-amd64.iso"
xorriso -as mkisofs \
  -r -V "1bit.MONSTER ${UBUNTU_VER}" \
  -o "$OUT_ISO" \
  -J -joliet-long \
  -b boot/grub/i386-pc/eltorito.img \
  -c boot.catalog -no-emul-boot -boot-load-size 4 -boot-info-table \
  -eltorito-alt-boot -e EFI/boot/bootx64.efi -no-emul-boot \
  -isohybrid-gpt-basdat \
  "$EXTRACT"

echo "Built: ${OUT_ISO}"
