#!/usr/bin/env bash
set -euo pipefail
# packaging/iso/therock-install.sh — installs the pinned TheRock payload into
# /opt/rocm-therock on the appliance. Runs chrooted via autoinstall
# late-commands (see packaging/iso/autoinstall.yaml.tmpl). Handles both
# payload layouts produced by fetch-payload.sh:
#
#   wheels layout:   *.whl files → create venv, pip install --no-deps (fully
#                    offline — the ISO ships the wheels), rocm-sdk init to
#                    expand the devel tree and link device files.
#   vendored layout: site-packages code trees → copy into the venv's
#                    site-packages, then rocm-sdk init.
#
# Usage: therock-install.sh /path/to/therock-<ver>-gfx1151.tar.gz
TARBALL="${1:?usage: therock-install.sh /path/to/therock-*.tar.gz}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
tar xzf "$TARBALL" -C "$TMP"

if ls "$TMP"/*.whl >/dev/null 2>&1; then
  echo "-- wheels layout: venv + pip install --no-deps --"
  python3 -m venv /opt/rocm-therock
  /opt/rocm-therock/bin/pip install --no-deps "$TMP"/*.whl
  /opt/rocm-therock/bin/rocm-sdk init
else
  echo "-- vendored layout: copy code trees into site-packages --"
  [ -x /opt/rocm-therock/bin/python ] || python3 -m venv /opt/rocm-therock
  SP="$("/opt/rocm-therock/bin/python" -c 'import site; print(site.getsitepackages()[0])')"
  cp -a "$TMP"/. "$SP"/
  /opt/rocm-therock/bin/rocm-sdk init 2>/dev/null || true
fi

echo "-- TheRock install summary --"
/opt/rocm-therock/bin/pip show rocm-sdk-core rocm-sdk-libraries rocm-sdk-device-gfx1151 2>/dev/null \
  | grep -E "^(Name|Version)" || true
