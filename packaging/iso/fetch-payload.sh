#!/usr/bin/env bash
set -euo pipefail
# packaging/iso/fetch-payload.sh — vendors the pinned driver stack into
# packaging/iso/build/payload/. This makes the ISO build deterministic:
# nothing is resolved from Ubuntu's live apt repo at install time, only
# the exact versions fetched here. See:
# docs/superpowers/specs/2026-08-16-ubuntu-iso-design.md

ISO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAYLOAD="${ISO_DIR}/build/payload"
mkdir -p "$PAYLOAD"

THEROCK_VER="7.14.0a20260612"
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

echo "-- TheRock gfx1151 ${THEROCK_VER}: attempting exact-version pip download --"
TMP_PIP="$(mktemp -d)"
if pip download "rocm-sdk-libraries-gfx1151==${THEROCK_VER}" \
    --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ \
    --no-deps -d "$TMP_PIP" > /tmp/therock-pip.log 2>&1; then
  tar czf "${PAYLOAD}/therock-${THEROCK_VER}-gfx1151.tar.gz" -C "$TMP_PIP" .
  echo "   fetched from nightlies index"
else
  echo "   not available in nightlies index (log: /tmp/therock-pip.log)"
  echo "   falling back to vendoring the matching build already on this box"
  LOCAL="/opt/rocm-therock/lib/python3.14/site-packages"
  DIST_INFO="${LOCAL}/rocm_sdk_libraries_gfx1151-${THEROCK_VER}.dist-info"
  CONTENT_DIR="${LOCAL}/_rocm_sdk_libraries_gfx1151"
  # The real installed files live under the underscore-prefixed content dir, NOT under
  # the versioned *.dist-info dir (which is only a pip metadata manifest — RECORD/METADATA/WHEEL,
  # a few KB, no .so/.hsaco files). A glob anchored on the dist-info's own versioned name
  # can never match the content dir, since the content dir's name carries no version at all.
  if [ ! -d "$DIST_INFO" ] || [ ! -d "$CONTENT_DIR" ]; then
    echo "FATAL: ${THEROCK_VER} not found in the nightlies index, and the local" >&2
    echo "       dist-info/content pair for rocm_sdk_libraries_gfx1151 is incomplete" >&2
    echo "       (dist-info: $([ -d "$DIST_INFO" ] && echo present || echo MISSING)," >&2
    echo "        content dir: $([ -d "$CONTENT_DIR" ] && echo present || echo MISSING))" >&2
    echo "       — cannot vendor a payload for this pinned version." >&2
    exit 1
  fi
  # Correlate the content dir to this exact dist-info/version before trusting it:
  # the dist-info's RECORD manifest lists every file it installed, content-dir-relative.
  # If the content dir's own basename doesn't appear as a path prefix in RECORD, this
  # dist-info does not describe what's currently sitting in the content dir (e.g. a later
  # install overwrote the content dir without updating this dist-info) — refuse rather than
  # silently vendor a possibly-mismatched payload.
  if ! grep -q "^_rocm_sdk_libraries_gfx1151/" "${DIST_INFO}/RECORD"; then
    echo "FATAL: ${DIST_INFO}/RECORD does not reference _rocm_sdk_libraries_gfx1151/" >&2
    echo "       — version correlation failed, refusing to vendor a possibly-stale" >&2
    echo "       or mismatched payload." >&2
    exit 1
  fi
  tar czf "${PAYLOAD}/therock-${THEROCK_VER}-gfx1151.tar.gz" -C "$LOCAL" "_rocm_sdk_libraries_gfx1151"
  echo "   vendored $(du -sh "$CONTENT_DIR" | cut -f1) from ${CONTENT_DIR}"
  echo "   (correlated against ${DIST_INFO}/RECORD)"
fi
rm -rf "$TMP_PIP"

echo "-- TheRock core runtime ${THEROCK_VER}: attempting exact-version pip download --"
# unified_server dynamically links against libamdhip64/libamd_comgr/libroctx64
# (HIP runtime + comgr) and libomp — these ship in rocm-sdk-core, a sibling
# package to rocm-sdk-libraries-gfx1151 fetched above, NOT inside it. Without
# this, the appliance's API service fails to start at all (dynamic linker
# can't resolve these at exec time) — found by actually booting a built ISO
# in QEMU and inspecting the failing systemd unit.
TMP_PIP="$(mktemp -d)"
if pip download "rocm-sdk-core==${THEROCK_VER}" \
    --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ \
    --no-deps -d "$TMP_PIP" > /tmp/therock-core-pip.log 2>&1; then
  tar czf "${PAYLOAD}/therock-${THEROCK_VER}-core.tar.gz" -C "$TMP_PIP" .
  echo "   fetched from nightlies index"
else
  echo "   not available in nightlies index (log: /tmp/therock-core-pip.log)"
  echo "   falling back to vendoring the matching build already on this box"
  LOCAL="/opt/rocm-therock/lib/python3.14/site-packages"
  DIST_INFO="${LOCAL}/rocm_sdk_core-${THEROCK_VER}.dist-info"
  CONTENT_DIR="${LOCAL}/_rocm_sdk_core"
  if [ ! -d "$DIST_INFO" ] || [ ! -d "$CONTENT_DIR" ]; then
    echo "FATAL: ${THEROCK_VER} not found in the nightlies index, and the local" >&2
    echo "       dist-info/content pair for rocm_sdk_core is incomplete" >&2
    echo "       (dist-info: $([ -d "$DIST_INFO" ] && echo present || echo MISSING)," >&2
    echo "        content dir: $([ -d "$CONTENT_DIR" ] && echo present || echo MISSING))" >&2
    echo "       — cannot vendor a payload for this pinned version. NOTE: if the box's" >&2
    echo "       local ROCm install has since moved on to a newer nightly, do NOT" >&2
    echo "       silently vendor a version-mismatched core against the already-pinned" >&2
    echo "       gfx1151 libraries above — re-pin THEROCK_VER for both instead." >&2
    exit 1
  fi
  if ! grep -q "^_rocm_sdk_core/" "${DIST_INFO}/RECORD"; then
    echo "FATAL: ${DIST_INFO}/RECORD does not reference _rocm_sdk_core/" >&2
    echo "       — version correlation failed, refusing to vendor a possibly-stale" >&2
    echo "       or mismatched payload." >&2
    exit 1
  fi
  tar czf "${PAYLOAD}/therock-${THEROCK_VER}-core.tar.gz" -C "$LOCAL" "_rocm_sdk_core"
  echo "   vendored $(du -sh "$CONTENT_DIR" | cut -f1) from ${CONTENT_DIR}"
  echo "   (correlated against ${DIST_INFO}/RECORD)"
fi
rm -rf "$TMP_PIP"

echo ""
echo "Payload ready in ${PAYLOAD}:"
ls -la "$PAYLOAD"
