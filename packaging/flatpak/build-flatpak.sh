#!/usr/bin/env bash
set -euo pipefail
# packaging/flatpak/build-flatpak.sh — builds monster.onebit.Engine.flatpak
#
# Requires:
#   - flatpak + flatpak-builder installed (Fedora: dnf install flatpak
#     flatpak-builder; Ubuntu: apt install flatpak flatpak-builder)
#   - the built engine: build/1bit, build/librocm_cpp.so,
#     build/zinc_cpp_build/shaders (cmake --build build --target onebin)
#   - the TheRock payloads (run packaging/iso/fetch-payload.sh first, or they
#     are fetched below if missing)
#   - the org.freedesktop.Platform runtime (installed --user automatically)
#
# Output: packaging/flatpak/1bit-monster-<VERSION>.flatpak
FLATPAK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${FLATPAK_DIR}/../.." && pwd)"
VERSION="$(tr -d '[:space:]' < "${REPO_ROOT}/VERSION" || echo 0.0.0)"
BUILD_DIR="${REPO_ROOT}/build"
PAYLOAD="${REPO_ROOT}/packaging/iso/build/payload"
RUNTIME_VER="${FLATPAK_RUNTIME_VER:-25.08}"

cd "$FLATPAK_DIR"

# 1. Engine artifacts must exist
for f in 1bit librocm_cpp.so; do
  [ -f "${BUILD_DIR}/$f" ] || { echo "FATAL: ${BUILD_DIR}/$f missing — build the engine first" >&2; exit 1; }
done
[ -d "${BUILD_DIR}/zinc_cpp_build/shaders" ] || {
  echo "FATAL: ZINC shaders not built (${BUILD_DIR}/zinc_cpp_build/shaders)" >&2; exit 1
}

# 2. TheRock payloads — fetch if missing
if ! ls "${PAYLOAD}"/therock-10.1.0a20260822-*.tar.gz >/dev/null 2>&1; then
  echo "Fetching pinned TheRock payloads..."
  ( cd "${REPO_ROOT}/packaging/iso" && bash fetch-payload.sh )
fi
for f in therock-10.1.0a20260822-devel.tar.gz \
         therock-10.1.0a20260822-libraries.tar.gz \
         therock-10.1.0a20260822-core.tar.gz; do
  [ -f "${PAYLOAD}/$f" ] || { echo "FATAL: ${PAYLOAD}/$f missing" >&2; exit 1; }
done

# 3. Stage the manifest sources (paths are relative to this dir)
rm -f build
ln -sfn "$BUILD_DIR" build

# Glibc + friends from the HOST (Ubuntu 26.04, glibc 2.43) — the engine needs
# GLIBC_2.43 which the Freedesktop runtime's glibc doesn't provide. Bundled
# under /app/glibc and launched via its own ld-linux (see 1bit-wrapper.sh).
rm -rf glibc-libs
mkdir -p glibc-libs
for f in /lib/x86_64-linux-gnu/ld-linux-x86-64.so.2 \
         /usr/lib/x86_64-linux-gnu/libc.so.6 \
         /usr/lib/x86_64-linux-gnu/libm.so.6 \
         /usr/lib/x86_64-linux-gnu/libpthread.so.0 \
         /usr/lib/x86_64-linux-gnu/libdl.so.2 \
         /usr/lib/x86_64-linux-gnu/librt.so.1 \
         /usr/lib/x86_64-linux-gnu/libresolv.so.2 \
         /usr/lib/x86_64-linux-gnu/libgcc_s.so.1 \
         /usr/lib/x86_64-linux-gnu/libstdc++.so.6 \
         /usr/lib/x86_64-linux-gnu/libnss_dns.so.2 \
         /usr/lib/x86_64-linux-gnu/libnss_files.so.2; do
  [ -e "$f" ] && cp -L "$f" glibc-libs/
  [ -e "$f" ] || echo "WARN: $f missing on build host — glibc bundle incomplete" >&2
done
ls glibc-libs/ >/dev/null || { echo "FATAL: could not stage glibc bundle" >&2; exit 1; }

# XRT runtime libs (libxrt_core.so.2 etc.) — the engine links them at load
# time; extracted from the distro packages into xrt-libs/ (kept out of git).
# Scratch lives under this dir (NVMe), never /tmp — the tmpfs fills up.
rm -rf xrt-libs xrt-stage
mkdir -p xrt-libs xrt-stage
( cd xrt-stage && \
  apt-get download libxrt2 libxrt-npu2 >/dev/null 2>&1 && \
  for d in libxrt2_*.deb libxrt-npu2_*.deb; do dpkg-deb -x "$d" root; done && \
  cp root/usr/lib/x86_64-linux-gnu/libxrt_*.so.2 root/usr/lib/x86_64-linux-gnu/libxilinxopencl.so.2 "$FLATPAK_DIR"/xrt-libs/ )
ls xrt-libs/ >/dev/null || { echo "FATAL: could not fetch XRT libs — check apt availability of libxrt2/libxrt-npu2" >&2; exit 1; }
rm -rf xrt-stage

# 4. Runtime + SDK present? (user install — no root needed)
if ! flatpak list --user --runtime 2>/dev/null | grep -q "org.freedesktop.Platform.*${RUNTIME_VER}" || \
   ! flatpak list --user --runtime 2>/dev/null | grep -q "org.freedesktop.Sdk.*${RUNTIME_VER}"; then
  echo "Installing org.freedesktop.{Platform,Sdk}//${RUNTIME_VER} (user)..."
  flatpak remote-add --user --if-not-exists flathub https://flathub.org/repo/flathub.flatpakrepo
  flatpak install --user -y flathub "org.freedesktop.Platform//${RUNTIME_VER}"
  flatpak install --user -y flathub "org.freedesktop.Sdk//${RUNTIME_VER}"
fi

# 5. Build + bundle
rm -rf repo build-dir
flatpak-builder --user --force-clean --repo=repo build-dir monster.onebit.Engine.yml
flatpak build-bundle repo "1bit-monster-${VERSION}.flatpak" monster.onebit.Engine

echo "Built: ${FLATPAK_DIR}/1bit-monster-${VERSION}.flatpak"
echo "Run:   flatpak --user install 1bit-monster-${VERSION}.flatpak && flatpak run monster.onebit.Engine unified --port 8088"
