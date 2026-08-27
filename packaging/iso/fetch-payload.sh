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
# All scratch space (pip staging, wheel unzip, nested-tar extraction) lives
# under the payload dir, which is on the big disk — /tmp is often a small
# tmpfs and the extracted TheRock wheel alone is ~5 GB.
export TMPDIR="${PAYLOAD}/.tmp"
mkdir -p "$TMPDIR"

THEROCK_VER="10.1.0a20260822"
# Where to look for a local TheRock pip-SDK install when the exact pinned
# version is no longer in the nightlies index. Default: the first candidate
# that exists — the reference box's /opt/rocm-therock, else ~/.cache/pip/therock
# (see docs/superpowers/plans/...); override explicitly with
# THEROCK_PIP_ROOT=<venv-root>.
if [ -n "${THEROCK_PIP_ROOT:-}" ]; then
  :
elif [ -d /opt/rocm-therock ]; then
  THEROCK_PIP_ROOT="/opt/rocm-therock"
elif [ -d "${HOME}/.cache/pip/therock" ]; then
  THEROCK_PIP_ROOT="${HOME}/.cache/pip/therock"
else
  THEROCK_PIP_ROOT="/opt/rocm-therock"  # let the fallback's checks report the miss
fi
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

# TheRock wheel → payload tarball. `pip download` only fetches the raw .whl
# (itself a zip archive) — it does NOT unpack it the way `pip install` would.
# Tarring the download dir directly produces a tarball containing just the
# .whl file, not the _rocm_sdk_* tree the rest of the pipeline expects
# (caught by booting a built ISO in QEMU: /opt/rocm-therock ended up with
# two .whl files and unified_server died on missing libhipblas.so.3).
vendor_therock() {
  local pip_pkg="$1"   # e.g. rocm-sdk-devel
  local content_dir="$2"  # e.g. _rocm_sdk_devel
  local out_tar="$3"   # e.g. therock-...-devel.tar.gz
  local tmp_pip tmp_unzip whl

  tmp_pip="$(mktemp -d)"
  if TMPDIR="${PAYLOAD}/.pip-tmp" pip download "${pip_pkg}==${THEROCK_VER}" \
      --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ \
      --no-deps -d "$tmp_pip" > /tmp/therock-pip.log 2>&1; then
    whl="$(ls "${tmp_pip}"/*.whl | head -1)"
    tmp_unzip="$(mktemp -d)"
    python3 -m zipfile -e "$whl" "$tmp_unzip"
    if [ -d "${tmp_unzip}/${content_dir}" ]; then
      # Normal wheel: the content tree is in the wheel directly.
      tar czf "${PAYLOAD}/${out_tar}" -C "$tmp_unzip" "$content_dir"
    else
      # Shim wheel (10.x nightly scheme): the wheel only carries a small
      # python package whose payload is a nested tarball (e.g.
      # rocm_sdk_devel/_devel.tar) holding the real _rocm_sdk_* tree.
      # `pip install` extracts it at install time; we must do the same.
      # Without this, the payload would be just the .whl (or a stub dir)
      # and the appliance's unified_server dies on missing libhipblas.so.3
      # — caught by booting a built ISO in QEMU.
      local nested_tar
      nested_tar="$(find "$tmp_unzip" -name '*.tar' -o -name '*.tar.gz' | head -1)"
      test -n "$nested_tar" || {
        echo "FATAL: ${whl} has no ${content_dir}/ and no nested payload tar." >&2
        exit 1
      }
      tmp_nested="$(mktemp -d)"
      tar xf "$nested_tar" -C "$tmp_nested"
      test -d "${tmp_nested}/${content_dir}" || {
        echo "FATAL: ${nested_tar} did not contain ${content_dir}/ once extracted." >&2
        exit 1
      }
      tar czf "${PAYLOAD}/${out_tar}" -C "$tmp_nested" "$content_dir"
      rm -rf "$tmp_nested"
    fi
    rm -rf "$tmp_unzip"
    echo "   fetched ${pip_pkg}==${THEROCK_VER} from nightlies index"
  else
    echo "   ${pip_pkg}==${THEROCK_VER} not in nightlies index (log: /tmp/therock-pip.log)"
    echo "   falling back to vendoring the matching build already on this box"
    local local_root dist_info content_dir_abs
    local_root="${THEROCK_PIP_ROOT}/lib/python3.14/site-packages"
    dist_info="${local_root}/${pip_pkg//-/_}-${THEROCK_VER}.dist-info"
    content_dir_abs="${local_root}/${content_dir}"
    # The real installed files live under the underscore-prefixed content dir, NOT under
    # the versioned *.dist-info dir (which is only a pip metadata manifest — RECORD/METADATA/WHEEL,
    # a few KB, no .so/.hsaco files). A glob anchored on the dist-info's own versioned name
    # can never match the content dir, since the content dir's name carries no version at all.
    if [ ! -d "$dist_info" ] || [ ! -d "$content_dir_abs" ]; then
      echo "FATAL: ${THEROCK_VER} not found in the nightlies index, and the local" >&2
      echo "       dist-info/content pair for ${pip_pkg} is incomplete" >&2
      echo "       (dist-info: $([ -d "$dist_info" ] && echo present || echo MISSING)," >&2
      echo "        content dir: $([ -d "$content_dir_abs" ] && echo present || echo MISSING))" >&2
      echo "       — cannot vendor a payload for this pinned version." >&2
      exit 1
    fi
    # Correlate the content dir to this exact dist-info/version before trusting it:
    # the dist-info's RECORD manifest lists every file it installed, content-dir-relative.
    # If the content dir's own basename doesn't appear as a path prefix in RECORD, this
    # dist-info does not describe what's currently sitting in the content dir (e.g. a later
    # install overwrote the content dir without updating this dist-info) — refuse rather than
    # silently vendor a possibly-mismatched payload.
    if ! grep -q "^${content_dir}/" "${dist_info}/RECORD"; then
      echo "FATAL: ${dist_info}/RECORD does not reference ${content_dir}/" >&2
      echo "       — version correlation failed, refusing to vendor a possibly-stale" >&2
      echo "       or mismatched payload." >&2
      exit 1
    fi
    tar czf "${PAYLOAD}/${out_tar}" -C "$local_root" "$content_dir"
    echo "   vendored $(du -sh "$content_dir_abs" | cut -f1) from ${content_dir_abs}"
    echo "   (correlated against ${dist_info}/RECORD)"
  fi
  rm -rf "$tmp_pip"
}

echo "-- TheRock runtime libs ${THEROCK_VER}: rocm-sdk-devel --"
# unified_server dynamically links against libhipblas/libamdhip64/libamd_comgr/
# libroctx64 (HIP runtime + comgr) and libomp — these ship in rocm-sdk-devel,
# rocm-sdk-libraries and rocm-sdk-core, the three payloads fetched below.
# Without them the appliance's API service fails to start at all (dynamic
# linker can't resolve these at exec time) — found by actually booting a
# built ISO in QEMU and inspecting the failing systemd unit. NOTE: in the
# 10.x nightly scheme rocm-sdk-devel's lib/*.so.N entries are RELATIVE
# SYMLINKS into ../../_rocm_sdk_libraries/lib/ — the real files live in
# rocm-sdk-libraries, so all three packages must ship together or the
# symlinks dangle and the API service dies on missing libhipblas.so.3.
vendor_therock "rocm-sdk-devel" "_rocm_sdk_devel" "therock-${THEROCK_VER}-devel.tar.gz"

vendor_therock "rocm-sdk-libraries" "_rocm_sdk_libraries" "therock-${THEROCK_VER}-libraries.tar.gz"

echo "-- TheRock core runtime ${THEROCK_VER}: rocm-sdk-core --"
vendor_therock "rocm-sdk-core" "_rocm_sdk_core" "therock-${THEROCK_VER}-core.tar.gz"

echo ""
echo "Payload ready in ${PAYLOAD}:"
ls -la "$PAYLOAD"
