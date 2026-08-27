#!/usr/bin/env bash
set -euo pipefail
# packaging/flatpak/build-flatpak.sh — builds monster.1bit.Engine.flatpak
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

# 4. Runtime present? (user install — no root needed)
if ! flatpak list --user --runtime 2>/dev/null | grep -q "org.freedesktop.Platform.*${RUNTIME_VER}"; then
  echo "Installing org.freedesktop.Platform//${RUNTIME_VER} (user)..."
  flatpak remote-add --user --if-not-exists flathub https://flathub.org/repo/flathub.flatpakrepo
  flatpak install --user -y flathub "org.freedesktop.Platform//${RUNTIME_VER}"
fi

# 5. Build + bundle
rm -rf repo build-dir
flatpak-builder --user --force-clean --repo=repo build-dir monster.1bit.Engine.yml
flatpak build-bundle repo "1bit-monster-${VERSION}.flatpak" monster.1bit.Engine

echo "Built: ${FLATPAK_DIR}/1bit-monster-${VERSION}.flatpak"
echo "Run:   flatpak --user install 1bit-monster-${VERSION}.flatpak && flatpak run monster.1bit.Engine unified --port 8088"
