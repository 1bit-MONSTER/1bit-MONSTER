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

echo "-- TheRock gfx1151 ${THEROCK_VERSION}: attempting exact-version pip download --"
TMP_PIP="$(mktemp -d)"
if pip download "rocm-sdk-libraries-gfx1151==${THEROCK_VERSION}" \
    --index-url "${NIGHTLY_INDEX}" \
    --no-deps -d "$TMP_PIP" > /tmp/therock-pip.log 2>&1; then
  tar czf "${PAYLOAD}/therock-${THEROCK_VERSION}-gfx1151.tar.gz" -C "$TMP_PIP" .
  echo "   fetched from nightlies index"
else
  echo "   not available in nightlies index (log: /tmp/therock-pip.log)"
  echo "   falling back to vendoring the matching build already on this box"
  LOCAL="${ROCK_ROOT}/lib/python3.14/site-packages"
  mapfile -t FOUND < <(find "$LOCAL" -maxdepth 1 \( -iname "rocm_sdk_libraries_gfx1151-${THEROCK_VERSION}*" -o -iname "rocm_sdk_device_gfx1151-${THEROCK_VERSION}*" \) 2>/dev/null)
  if [ "${#FOUND[@]}" -eq 0 ]; then
    echo "FATAL: ${THEROCK_VERSION} not found in the nightlies index or locally at ${LOCAL} — cannot vendor a payload for this pinned version." >&2
    exit 1
  fi
  tar czf "${PAYLOAD}/therock-${THEROCK_VERSION}-gfx1151.tar.gz" -C "$LOCAL" $(for f in "${FOUND[@]}"; do basename "$f"; done)
  echo "   vendored ${#FOUND[@]} local package dir(s) from ${LOCAL}"
  echo "   WARNING: verify this vendored copy actually loads (import rocm / run a HIP smoke test)"
  echo "   before trusting it as a clean install payload — it may be leftover dist-info"
  echo "   from a superseded install rather than a currently-functional copy."
fi
rm -rf "$TMP_PIP"

echo ""
echo "Payload ready in ${PAYLOAD}:"
ls -la "$PAYLOAD"
