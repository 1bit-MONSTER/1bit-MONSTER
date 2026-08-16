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
  mapfile -t FOUND < <(find "$LOCAL" -maxdepth 1 \( -iname "rocm_sdk_libraries_gfx1151-${THEROCK_VER}*" -o -iname "rocm_sdk_device_gfx1151-${THEROCK_VER}*" \) 2>/dev/null)
  if [ "${#FOUND[@]}" -eq 0 ]; then
    echo "FATAL: ${THEROCK_VER} not found in the nightlies index or locally at ${LOCAL} — cannot vendor a payload for this pinned version." >&2
    exit 1
  fi
  tar czf "${PAYLOAD}/therock-${THEROCK_VER}-gfx1151.tar.gz" -C "$LOCAL" $(for f in "${FOUND[@]}"; do basename "$f"; done)
  echo "   vendored ${#FOUND[@]} local package dir(s) from ${LOCAL}"
  echo "   WARNING: verify this vendored copy actually loads (import rocm / run a HIP smoke test)"
  echo "   before trusting it as a clean install payload — it may be leftover dist-info"
  echo "   from a superseded install rather than a currently-functional copy."
fi
rm -rf "$TMP_PIP"

echo ""
echo "Payload ready in ${PAYLOAD}:"
ls -la "$PAYLOAD"
