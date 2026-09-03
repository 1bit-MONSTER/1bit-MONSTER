#!/usr/bin/env bash
set -euo pipefail
# packaging/iso/fetch-payload.sh — vendors the pinned driver stack into
# packaging/iso/build/payload/. This makes the ISO build deterministic:
# nothing is resolved from Ubuntu's live apt repo at install time, only
# the exact versions fetched here. See:
# docs/superpowers/specs/2026-08-16-ubuntu-iso-design.md
#
# Payload contents (2026-08-31):
#   - mesa-vulkan-drivers / libvulkan1 / bolt  (pinned .debs, unchanged)
#   - hrx-b66 self-contained llama-server bundle (AMD "Hip Runtime Extended";
#     ships its own libhrx/libggml-hrx/libhsa-runtime64/libvulkan — no ROCm
#     install needed on target). Pinned asset + sha256 from the engine's
#     build/resources/backend_versions.json (source of truth for the pin).
# The full TheRock pip-SDK wheel payload (rocm-sdk-core/devel/libraries,
# ~5 GB) is GONE: the engine's own HIP 1BP path was gate-verified against
# Ubuntu 26.04's ROCm 7.1 runtime packages (libamdhip64-7/libhipblas3/
# librocblas5/librocsolver0/libhsa-runtime64-1, installed from the archive
# by autoinstall.yaml's packages: list), and the NPU/fused path runs on the
# HRX bundle + amdxdna kernel driver + Ubuntu libxrt2 — neither needs TheRock.

ISO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAYLOAD="${ISO_DIR}/build/payload"
mkdir -p "$PAYLOAD"
# All scratch space lives under the payload dir, which is on the big disk —
# /tmp is often a small tmpfs.
export TMPDIR="${PAYLOAD}/.tmp"
mkdir -p "$TMPDIR"

MESA_VER="26.0.3-1ubuntu1"
VULKAN1_VER="1.4.341.0-1"
BOLT_VER="0.9.10-1"

echo "-- Vulkan + Thunderbolt (bolt): apt-get download pinned versions --"
( cd "$PAYLOAD" && \
  apt-get download "mesa-vulkan-drivers=${MESA_VER}" "libvulkan1=${VULKAN1_VER}" "bolt=${BOLT_VER}" )
test -f "${PAYLOAD}/mesa-vulkan-drivers_${MESA_VER}_amd64.deb" || {
  echo "FATAL: mesa-vulkan-drivers ${MESA_VER} not available via apt-get download." >&2
  echo "       Try: sudo apt-get update, or check 'apt-cache policy mesa-vulkan-drivers'." >&2
  exit 1
}
test -f "${PAYLOAD}/libvulkan1_${VULKAN1_VER}_amd64.deb" || {
  echo "FATAL: libvulkan1 ${VULKAN1_VER} not available via apt-get download." >&2
  exit 1
}
test -f "${PAYLOAD}/bolt_${BOLT_VER}_amd64.deb" || {
  echo "FATAL: bolt ${BOLT_VER} not available via apt-get download." >&2
  exit 1
}

# ── HRX bundle (self-contained llama-server runtime) ──
# Pin source of truth: build/resources/backend_versions.json in the repo
# (checksums/github/ROCm/ggml-staging-automation/hrx-b66/...).
# The bundle ships bin/llama-server + lib/ (libhrx, libggml-hrx, its own
# libhsa-runtime64 + libvulkan), so the target needs no ROCm install.
HRX_RELEASE_REPO="ROCm/ggml-staging-automation"
HRX_RELEASE_TAG="hrx-b66"
HRX_ASSET="llama-hrx-b66-bin-manylinux-hrx-x64.tar.gz"
HRX_SHA256="b54df34bf7a94ea05445b9058920f7da179ecb8789b9a1930c05d815781c7e2c"
HRX_TARBALL="${PAYLOAD}/hrx-b66.tar.gz"

echo "-- HRX bundle ${HRX_RELEASE_TAG} (pinned, sha256-verified) --"
if [ -f "$HRX_TARBALL" ]; then
  echo "   cached: $(du -h "$HRX_TARBALL" | cut -f1)"
else
  curl -fL --retry 3 --progress-bar \
    "https://github.com/${HRX_RELEASE_REPO}/releases/download/${HRX_RELEASE_TAG}/${HRX_ASSET}" \
    -o "${HRX_TARBALL}.part"
  mv "${HRX_TARBALL}.part" "$HRX_TARBALL"
fi
echo "${HRX_SHA256}  ${HRX_TARBALL}" | sha256sum -c - > /dev/null || {
  echo "FATAL: sha256 mismatch on ${HRX_TARBALL}" >&2
  echo "       expected ${HRX_SHA256} — re-download or update the pin" >&2
  exit 1
}
# Sanity-check the bundle shape before it goes on an ISO: it must contain a
# runnable llama-server and the HRX runtime libs the engine's backend spawns.
# NOTE: capture the listing first — `grep -q` on a live `tar tzf` pipe exits at
# the first match and SIGPIPEs tar, which trips `set -o pipefail` and reports a
# false failure.
BUNDLE_LISTING="$(tar tzf "$HRX_TARBALL")"
echo "$BUNDLE_LISTING" | grep -qE 'bin/llama-server$' || {
  echo "FATAL: ${HRX_ASSET} has no bin/llama-server — bundle layout changed?" >&2
  exit 1
}
echo "$BUNDLE_LISTING" | grep -qE 'lib/libhrx\.so' || {
  echo "FATAL: ${HRX_ASSET} has no libhrx — bundle layout changed?" >&2
  exit 1
}
echo "$BUNDLE_LISTING" | grep -qE 'lib/libggml-hrx\.so' || {
  echo "FATAL: ${HRX_ASSET} has no libggml-hrx — bundle layout changed?" >&2
  exit 1
}
echo "   verified sha256 + bundle shape (llama-server, libhrx, libggml-hrx present)"

echo ""
echo "Payload ready in ${PAYLOAD}:"
ls -la "$PAYLOAD"
