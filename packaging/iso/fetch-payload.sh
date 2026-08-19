#!/usr/bin/env bash
set -euo pipefail
# packaging/iso/fetch-payload.sh — vendors the pinned driver stack into
# packaging/iso/build/payload/. This makes the ISO build deterministic:
# nothing is resolved from Ubuntu's live apt repo at install time, only
# the exact versions fetched here. See:
# docs/superpowers/specs/2026-08-16-ubuntu-iso-design.md
#
# Pin versions come from packaging/rocm-lane-pin.env — the single source
# shared with scripts/setup-therock.sh. Keep them in lockstep.
#
# 2026-08-19 fix (recorded per the ISO plan's vendoring-fallback note):
#   The plan's original package name `rocm-sdk-libraries-gfx1151` does NOT
#   exist on the nightlies index (verified: "from versions: none"), and the
#   original local-vendoring glob only matched *.dist-info metadata (13 KB
#   tarball — unusable). Per ROCm/TheRock RELEASES.md the correct wheels are
#   rocm-sdk-core / rocm-sdk-devel / rocm-sdk-libraries (host code) +
#   rocm-sdk-device-gfx1151 (GPU device code) — the same set the live setup
#   installs. fetch-payload now downloads those four wheels (≈1.1 GB); the
#   fallback vendors the site-packages CODE trees (not dist-info) from the
#   box's matching install. packaging/iso/therock-install.sh installs either
#   layout on the appliance.

ISO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ISO_DIR}/../.." && pwd)"
[ -f "$REPO_ROOT/packaging/rocm-lane-pin.env" ] && . "$REPO_ROOT/packaging/rocm-lane-pin.env"

PAYLOAD="${ISO_DIR}/build/payload"
mkdir -p "$PAYLOAD"

echo "-- Vulkan: apt-get download pinned versions --"
( cd "$PAYLOAD" && \
  apt-get download "mesa-vulkan-drivers=${MESA_VER}" "libvulkan1=${VULKAN1_VER}" )
test -f "${PAYLOAD}/mesa-vulkan-drivers_${MESA_VER}_amd64.deb" || {
  echo "FATAL: mesa-vulkan-drivers ${MESA_VER} not available via apt-get download." >&2
  echo "       Try: sudo apt-get update, or check 'apt-cache policy mesa-vulkan-drivers'." >&2
  exit 1
}
test -f "${PAYLOAD}/libvulkan1_${VULKAN1_VER}_amd64.deb" || {
  echo "FATAL: libvulkan1 ${VULKAN1_VER} not available via apt-get download." >&2
  exit 1
}

echo "-- TheRock gfx1151 ${THEROCK_VERSION}: downloading the exact-version wheel set --"
TMP_PIP="$(mktemp -d)"
# `rocm` is the meta sdist that provides the rocm-sdk CLI (needed for
# `rocm-sdk init`, which expands the devel tree the engine runs from). The
# other four are the SDK wheels per ROCm/TheRock RELEASES.md.
WHEELS=( "rocm" "rocm-sdk-core" "rocm-sdk-devel" "rocm-sdk-libraries" "rocm-sdk-device-gfx1151" )
OK=1
for w in "${WHEELS[@]}"; do
  if ! pip download --no-deps -d "$TMP_PIP" "${w}==${THEROCK_VERSION}" \
      --index-url "${NIGHTLY_INDEX}" >> /tmp/therock-pip.log 2>&1; then
    echo "   ${w}==${THEROCK_VERSION} not on the index (log: /tmp/therock-pip.log)"
    OK=0
    break
  fi
done
if [ "$OK" = 1 ] && ls "$TMP_PIP"/*.whl >/dev/null 2>&1; then
  # Build the rocm meta sdist to a pure wheel now so the appliance install
  # stays fully offline (no build backend needed at install time).
  if ls "$TMP_PIP"/*.tar.gz >/dev/null 2>&1; then
    echo "   building rocm meta sdist -> wheel"
    pip wheel --no-deps --no-build-isolation -w "$TMP_PIP" "$TMP_PIP"/*.tar.gz >> /tmp/therock-pip.log 2>&1
    rm -f "$TMP_PIP"/*.tar.gz
  fi
  tar czf "${PAYLOAD}/therock-${THEROCK_VERSION}-gfx1151.tar.gz" -C "$TMP_PIP" .
  echo "   fetched ${#WHEELS[@]} packages from the nightlies index:"
  ls -la "$TMP_PIP"
else
  echo "   falling back to vendoring the matching build already on this box"
  LOCAL="${ROCK_ROOT}/lib/python3.14/site-packages"
  if ! ls -d "$LOCAL"/rocm_sdk_device_gfx1151-${THEROCK_VERSION}.dist-info \
            "$LOCAL"/rocm_sdk_libraries-${THEROCK_VERSION}.dist-info >/dev/null 2>&1; then
    echo "FATAL: ${THEROCK_VERSION} not on the index and not installed locally at ${LOCAL} — cannot vendor a payload for this pinned version." >&2
    exit 1
  fi
  tar czf "${PAYLOAD}/therock-${THEROCK_VERSION}-gfx1151.tar.gz" -C "$LOCAL" \
    _rocm_sdk_core _rocm_sdk_devel _rocm_sdk_libraries rocm_sdk_libraries_None \
    rocm_sdk_core rocm_sdk_devel \
    rocm_sdk_core-${THEROCK_VERSION}.dist-info rocm_sdk_devel-${THEROCK_VERSION}.dist-info \
    rocm_sdk_libraries-${THEROCK_VERSION}.dist-info \
    rocm_sdk_device_gfx1151-${THEROCK_VERSION}.dist-info
  echo "   vendored the site-packages code trees from ${LOCAL}"
  echo "   WARNING: verify this vendored copy actually loads (HIP smoke test) before trusting it."
fi
rm -rf "$TMP_PIP"

echo ""
echo "Payload ready in ${PAYLOAD}:"
ls -la "$PAYLOAD"
