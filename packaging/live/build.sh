#!/usr/bin/env bash
set -euo pipefail
# packaging/live/build.sh — lean 1bit.MONSTER LIVE image, assembled from
# Ubuntu packages (debootstrap), NOT downloaded-and-repacked from the
# installer ISO. No Subiquity, no desktop, no 2.8 GB base-ISO download.
#
# Produces a dd-able GPT image: ESP (GRUB, --removable so it boots on any
# UEFI USB) + ext4 root with:
#   kernel (archive generic OR this host's custom 7.2 og) + amdgpu fw
#   1bit-monster engine .deb (single ELF) + HRX bundle (self-contained,
#   AMD "Hip Runtime Extended", no ROCm stack) at /opt/hrx
#   Ubuntu ROCm 7.1 runtime libs + libxrt2 (NPU userspace) + mesa-vulkan/bolt
#   1bit-unified.service + 1bit-model-fetch.service units
# Usage:
#   bash build.sh --ssh-key ~/.ssh/id_ed25519.pub [--size 8G] [--kernel og|archive] [--out DIR]
#   [--bake-model qwen3-0.6b]   # bake a model into the image (offline-first stick)
#
# Gate: bash test-live.sh build/1bit-monster-live-26.04-amd64.img /path/key

LIVE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${LIVE_DIR}/../.." && pwd)"
BUILD="${LIVE_DIR}/build"
PAYLOAD="$(cd "${LIVE_DIR}/../iso/build/payload" && pwd)"   # shared with iso lane

SUITE="resolute"                       # Ubuntu 26.04
MIRROR="${MIRROR:-http://ca.archive.ubuntu.com/ubuntu}"
SSH_KEY=""
SIZE="8G"
KERNEL_SRC="archive"                   # og = copy this host's 7.2 og kernel
OUT_DIR="$BUILD"
BAKE_MODEL=""
while [ $# -gt 0 ]; do
  case "$1" in
    --ssh-key) SSH_KEY="$2"; shift 2 ;;
    --size) SIZE="$2"; shift 2 ;;
    --kernel) KERNEL_SRC="$2"; shift 2 ;;
    --out) OUT_DIR="$2"; shift 2 ;;
    --bake-model) BAKE_MODEL="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done
[ -n "$SSH_KEY" ] && [ -f "$SSH_KEY" ] || { echo "usage: build.sh --ssh-key <key.pub> [--size 8G] [--kernel og|archive]" >&2; exit 1; }

sudo -n true 2>/dev/null || { echo "needs passwordless sudo (debootstrap/chroot/image ops)" >&2; exit 1; }

# ── shared payload (mesa/bolt debs + hrx-b66 bundle), built by the iso lane ──
if [ ! -f "${PAYLOAD}/hrx-b66.tar.gz" ] || [ ! -f "${PAYLOAD}"/mesa-vulkan-drivers_*.deb ]; then
  echo "[payload] missing — running ../iso/fetch-payload.sh ..."
  bash "${LIVE_DIR}/../iso/fetch-payload.sh"
fi
ENGINE_DEB="$(ls "${REPO_ROOT}"/packaging/build/1bit-monster_*_amd64.deb 2>/dev/null | head -1)"
[ -n "$ENGINE_DEB" ] || { echo "FATAL: no engine .deb — run 'make package-deb' first" >&2; exit 1; }

ROOTFS="${BUILD}/rootfs"
IMG="${OUT_DIR}/1bit-monster-live-26.04-amd64.img"
MOUNT="${BUILD}/mnt"
CONSOLE_PW="$(openssl rand -base64 24)"

# One EXIT trap that cleans rootfs chroot binds AND the loop image mounts
# (whichever exist when the script exits — success or failure).
CLEANUP=""
cleanup() { eval "$CLEANUP"; }
trap cleanup EXIT

echo "[1/7] debootstrap ${SUITE} minbase (no desktop)..."
sudo rm -rf "$ROOTFS" "$MOUNT"
sudo debootstrap --variant=minbase --components=main,universe \
  --include=systemd,initramfs-tools,kmod,openssh-server,ca-certificates,curl,apt-utils \
  "$SUITE" "$ROOTFS" "$MIRROR" 2>&1 | tail -1

# Maintainer scripts (systemd's postinst re-runs on later package installs and
# calls posix_openpt) need /dev/pts + friends — a bare chroot dies with
# "Can not write log (Is /dev/pts mounted?)" otherwise. Bind the host mounts.
sudo mount --bind /dev "$ROOTFS/dev"
sudo mount -t devpts devpts "$ROOTFS/dev/pts"
sudo mount --bind /proc "$ROOTFS/proc"
sudo mount --bind /sys "$ROOTFS/sys"
CLEANUP="sudo umount -q "$ROOTFS"/dev/pts "$ROOTFS"/dev "$ROOTFS"/proc "$ROOTFS"/sys 2>/dev/null || true; $CLEANUP"

echo "[2/7] chroot: apt sources + base runtime packages..."
sudo cp /etc/resolv.conf "$ROOTFS/etc/resolv.conf"
sudo tee "$ROOTFS/etc/apt/sources.list" > /dev/null <<EOF
deb ${MIRROR} ${SUITE} main universe
deb ${MIRROR} ${SUITE}-security main universe
EOF
chroot_apt() { sudo chroot "$ROOTFS" env DEBIAN_FRONTEND=noninteractive apt-get "$@"; }
chroot_apt update -qq
chroot_apt install -y --no-install-recommends \
  libamdhip64-7 libhipblas3 librocblas5 librocsolver0 libhsa-runtime64-1 \
  libxrt2 libxrt-npu2 libwebsockets19t64 \
  systemd-sysv \
  libcurl4t64 libssl3t64 libbrotli1 libzstd1 libdrm-amdgpu1 libomp5 \
  dbus polkitd systemd-resolved \
  libvulkan1 mesa-vulkan-drivers bolt \
  alsa-utils grub-efi-amd64 grub2-common 2>&1 | tail -2

echo "[3/7] kernel: ${KERNEL_SRC}"
sudo mkdir -p "$ROOTFS/tmp/pool" "$ROOTFS/lib/modules"
if [ "$KERNEL_SRC" = "og" ]; then
  KVER="$(ls /boot/vmlinuz-7.2.0-next-*ogc* 2>/dev/null | head -1 | sed 's|.*vmlinuz-||')"
  [ -n "$KVER" ] || { echo "FATAL: no og kernel on this host (needs --kernel archive elsewhere)" >&2; exit 1; }
  sudo cp "/boot/vmlinuz-${KVER}" "$ROOTFS/boot/vmlinuz-${KVER}"
  sudo cp -a "/lib/modules/${KVER}" "$ROOTFS/lib/modules/"
  # amdgpu + amd firmware only (28 MB), enough for gfx1151
  sudo mkdir -p "$ROOTFS/lib/firmware"
  sudo cp -a /lib/firmware/amdgpu /lib/firmware/amd "$ROOTFS/lib/firmware/"
  sudo chroot "$ROOTFS" depmod -a "$KVER"
  sudo chroot "$ROOTFS" update-initramfs -c -k "$KVER" 2>&1 | tail -1
else
  chroot_apt install -y --no-install-recommends linux-image-generic 2>&1 | tail -1
  KVER="$(sudo chroot "$ROOTFS" ls /boot/vmlinuz-* | head -1 | sed 's|.*vmlinuz-||')"
  sudo chroot "$ROOTFS" update-initramfs -c -k "$KVER" 2>&1 | tail -1
fi
echo "   kernel: ${KVER}"

echo "[4/7] chroot: engine .deb + HRX bundle + units + user..."
sudo cp "$ENGINE_DEB" "${PAYLOAD}"/mesa-vulkan-drivers_*.deb "${PAYLOAD}"/libvulkan1_*.deb "${PAYLOAD}"/bolt_*.deb "${PAYLOAD}/hrx-b66.tar.gz" "$ROOTFS/tmp/pool/"
sudo chroot "$ROOTFS" bash -c 'cd /tmp/pool && env DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends ./1bit-monster_*_amd64.deb ./mesa-vulkan-drivers_*.deb ./libvulkan1_*.deb ./bolt_*.deb 2>&1 | tail -2'
sudo chroot "$ROOTFS" mkdir -p /opt/hrx
sudo chroot "$ROOTFS" tar xzf /tmp/pool/hrx-b66.tar.gz --strip-components=1 -C /opt/hrx
sudo chroot "$ROOTFS" sh -c "printf '/opt/hrx/lib\\n' > /etc/ld.so.conf.d/hrx.conf"
sudo chroot "$ROOTFS" ldconfig
# engine NEEDED libomp.so (unversioned) — Ubuntu provides libomp.so.5
sudo chroot "$ROOTFS" ln -sf libomp.so.5 /usr/lib/x86_64-linux-gnu/libomp.so
sudo chroot "$ROOTFS" sh -c "ln -sf /opt/hrx/bin/llama-server /usr/local/bin/llama-server"
# freeze the tested stack (same doctrine as the installer lane)
sudo chroot "$ROOTFS" apt-mark hold libamdhip64-7 libhipblas3 librocblas5 librocsolver0 libhsa-runtime64-1 mesa-vulkan-drivers libvulkan1 bolt linux-image-generic 2>/dev/null | tail -1
# units + model downloader
sudo cp "${REPO_ROOT}/packaging/services/1bit-unified.service" "${REPO_ROOT}/packaging/services/1bit-model-fetch.service" "$ROOTFS/etc/systemd/system/"
sudo cp "${REPO_ROOT}/packaging/model-download.sh" "$ROOTFS/usr/share/1bit/model-download.sh"
sudo chroot "$ROOTFS" chmod +x /usr/share/1bit/model-download.sh
sudo chroot "$ROOTFS" systemctl enable 1bit-unified.service 1bit-model-fetch.service ssh systemd-networkd systemd-resolved 2>&1 | tail -1
# resolv.conf must track systemd-resolved (static host copy points at a dead stub)
sudo chroot "$ROOTFS" ln -sf /run/systemd/resolve/stub-resolv.conf /etc/resolv.conf
# monster user + baked-in ssh key + console password
sudo chroot "$ROOTFS" useradd -m -s /bin/bash monster
sudo chroot "$ROOTFS" mkdir -p /home/monster/.ssh
sudo cp "$SSH_KEY" "$ROOTFS/home/monster/.ssh/authorized_keys"
sudo chroot "$ROOTFS" chown -R monster:monster /home/monster/.ssh
sudo chroot "$ROOTFS" chmod 700 /home/monster/.ssh
# journal access for the appliance account (gate + support)
sudo chroot "$ROOTFS" usermod -aG adm,systemd-journal monster
echo "monster:${CONSOLE_PW}" | sudo chroot "$ROOTFS" chpasswd
echo "$CONSOLE_PW" > "${OUT_DIR}/console-recovery-password.txt"; chmod 600 "${OUT_DIR}/console-recovery-password.txt"
# network (dhcp via systemd-networkd), hostname, motd
sudo tee "$ROOTFS/etc/systemd/network/10-dhcp.network" > /dev/null <<'EOF'
[Match]
Name=en*
[Network]
DHCP=yes
EOF
sudo chroot "$ROOTFS" bash -c "echo monster > /etc/hostname"
sudo chroot "$ROOTFS" bash -c "printf '1bit.MONSTER live appliance — API on :8088, HRX at /opt/hrx\\n' > /etc/motd"

if [ -n "$BAKE_MODEL" ]; then
  echo "[4b] baking model ${BAKE_MODEL} (offline-first boot)..."
  sudo chroot "$ROOTFS" env HOME=/home/monster /usr/share/1bit/model-download.sh "$BAKE_MODEL" 2>&1 | tail -1
fi

# chroot bind mounts must be gone before the image rsync — otherwise the
# live /dev /proc /sys (device nodes, pty's, procfs) get copied into the
# image and rsync dies on special files ("error in file IO (code 11)").
sudo umount "$ROOTFS/dev/pts" "$ROOTFS/dev" "$ROOTFS/proc" "$ROOTFS/sys" 2>/dev/null || true
CLEANUP=""  # rootfs binds already released; EXIT trap has nothing left to do

echo "[5/7] assembling ${SIZE} GPT image..."
mkdir -p "$OUT_DIR" "$MOUNT"
sudo rm -f "$IMG"
truncate -s "$SIZE" "$IMG"
sudo parted -s "$IMG" mklabel gpt
sudo parted -s "$IMG" mkpart ESP fat32 1MiB 513MiB
sudo parted -s "$IMG" set 1 esp on
sudo parted -s "$IMG" mkpart root ext4 513MiB 100%
L="$(sudo losetup -fP --show "$IMG")"
CLEANUP="sudo umount -q "$L"p2 2>/dev/null || true; sudo umount -q "$L"p1 2>/dev/null || true; sudo losetup -d "$L" 2>/dev/null || true; $CLEANUP"
sudo mkfs.vfat -n 1BIT-ESP "${L}p1" > /dev/null
sudo mkfs.ext4 -q -L 1bit-live "${L}p2"
sudo mount "${L}p2" "$MOUNT"
sudo mkdir -p "$MOUNT/boot/efi"
sudo mount "${L}p1" "$MOUNT/boot/efi"
sudo rsync -a "$ROOTFS/" "$MOUNT/"
# fstab: systemd only remounts / rw when /etc/fstab says so — without a root
# entry the kernel's "ro" boot flag sticks and the whole image is read-only
# (model-fetch failed with "mkdir: Read-only file system"; found in QEMU).
sudo tee "$MOUNT/etc/fstab" > /dev/null <<EOF
UUID=$(sudo blkid -s UUID -o value "${L}p2") / ext4 defaults 0 1
UUID=$(sudo blkid -s UUID -o value "${L}p1") /boot/efi vfat umask=0077 0 1
EOF
# grub: --removable → EFI/BOOT/BOOTX64.EFI, boots from any UEFI USB
sudo mount --bind /dev "$MOUNT/dev"; sudo mount --bind /proc "$MOUNT/proc"; sudo mount --bind /sys "$MOUNT/sys"
sudo chroot "$MOUNT" grub-install --target=x86_64-efi --efi-directory=/boot/efi --boot-directory=/boot --removable --no-nvram 2>&1 | tail -1
sudo tee "$MOUNT/etc/default/grub.d/1bit.cfg" > /dev/null <<'EOF'
GRUB_CMDLINE_LINUX_DEFAULT="$GRUB_CMDLINE_LINUX_DEFAULT ttm.pages_limit=31457280 amdgpu.no_system_mem_limit=1 console=ttyS0,115200n8"
GRUB_SERIAL_COMMAND="serial --speed=115200 --unit=0 --word=8 --parity=no --stop=1"
GRUB_TERMINAL_OUTPUT="serial"
GRUB_TIMEOUT=3
EOF
sudo chroot "$MOUNT" update-grub 2>&1 | tail -1

echo "[6/7] unmount + cleanup..."
sudo umount "$MOUNT/dev" "$MOUNT/proc" "$MOUNT/sys"
sudo umount "$MOUNT/boot/efi"; sudo umount "$MOUNT"
sudo losetup -d "$L"
L=""  # loop already detached; EXIT cleanup only handles the rootfs binds now

echo "[7/7] done."
echo "  image: ${IMG}  ($(du -h "$IMG" | cut -f1), $(ls -l "$IMG" | awk '{print $5}') bytes)"
echo "  dd to USB: sudo dd if=${IMG} of=/dev/sdX bs=4M status=progress conv=fsync"
echo "  console recovery password: ${OUT_DIR}/console-recovery-password.txt"
