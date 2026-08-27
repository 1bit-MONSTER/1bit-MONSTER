#!/usr/bin/env bash
set -euo pipefail
# packaging/iso/build.sh — builds the 1bit.MONSTER appliance ISO.
# See docs/superpowers/specs/2026-08-16-ubuntu-iso-design.md.
#
# Usage: build.sh --ssh-key /path/to/id_ed25519.pub [--out DIR]

ISO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ISO_DIR}/../.." && pwd)"
WORK="${ISO_DIR}/build"
UBUNTU_VER="26.04"
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
DEB_PATH="$(ls "${REPO_ROOT}"/packaging/build/1bit-monster_*_amd64.deb | head -1)"
[ -f "$DEB_PATH" ] || { echo "FATAL: no .deb produced by 'make package-deb'" >&2; exit 1; }

echo "[2/7] Fetching base Ubuntu ${UBUNTU_VER} ISO..."
if [ ! -f "${WORK}/${UBUNTU_ISO_NAME}" ]; then
  curl -fL --progress-bar "${UBUNTU_ISO_URL}" -o "${WORK}/${UBUNTU_ISO_NAME}"
  curl -fL "https://releases.ubuntu.com/${UBUNTU_VER}/SHA256SUMS" -o "${WORK}/SHA256SUMS"
  ( cd "$WORK" && grep "${UBUNTU_ISO_NAME}\$" SHA256SUMS | sha256sum -c - )
fi

echo "[3/7] Fetching pinned driver payload..."
bash "${ISO_DIR}/fetch-payload.sh"

echo "[4/7] Extracting base ISO..."
EXTRACT="${WORK}/extract"
rm -rf "$EXTRACT"
mkdir -p "$EXTRACT"
xorriso -osirrox on -indev "${WORK}/${UBUNTU_ISO_NAME}" -extract / "$EXTRACT"
chmod -R u+w "$EXTRACT"

echo "[5/7] Building autoinstall seed + staging payload pool..."
PASSWORD="$(openssl rand -base64 24)"
# Python's stdlib `crypt` module was removed in 3.13 (PEP 594), so shell out
# to `openssl passwd -6` for a SHA-512 crypt hash instead — it auto-generates
# a random salt when none is given.
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
cp "${REPO_ROOT}/packaging/services/1bit-unified.service" "${EXTRACT}/pool/"
cp "${REPO_ROOT}/packaging/services/1bit-model-fetch.service" "${EXTRACT}/pool/"
cp "${REPO_ROOT}/packaging/model-download.sh" "${EXTRACT}/pool/"

echo "[6/7] Wiring autoinstall boot entry..."
GRUB_CFG="${EXTRACT}/boot/grub/grub.cfg"
[ -f "$GRUB_CFG" ] || { echo "FATAL: ${GRUB_CFG} not found — inspect ${EXTRACT}/boot/grub/ and fix this path" >&2; exit 1; }
if ! grep -q "autoinstall" "$GRUB_CFG"; then
  # NOTE: the real grub.cfg on Ubuntu 26.04's live-server ISO separates
  # "linux" and "/casper/vmlinuz" with plain spaces, not a tab — match on
  # whitespace generically rather than assuming a literal tab (verified by
  # inspecting the extracted tree; a tab-only match here silently no-ops).
  sed -i 's|\(linux[[:space:]]\+/casper/vmlinuz\)|\1 autoinstall ds=nocloud\\;s=/cdrom/|' "$GRUB_CFG"
  grep -q "autoinstall" "$GRUB_CFG" || { echo "FATAL: failed to wire autoinstall into ${GRUB_CFG}" >&2; exit 1; }
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
