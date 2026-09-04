#!/usr/bin/env bash
# download.sh — fetch 1bit packages from GHCR (no GitHub Releases, no git tags)
#
#   ./scripts/download.sh              # latest build
#   ./scripts/download.sh sha-abc1234  # a specific immutable build
#
# Pulls from ghcr.io/1bit-MONSTER/1bit-MONSTER:<tag> and unpacks whatever
# artifacts are in the package (tarball, .deb, AppImage).
#
# Requires: oras (auto-installed to ~/.local/bin if missing)
set -euo pipefail

REF="${1:-latest}"
REGISTRY="ghcr.io/1bit-MONSTER/1bit-MONSTER"
PKG_PAGE="https://github.com/orgs/1bit-MONSTER/packages?repo_name=1bit-MONSTER"
ORAS_VERSION="v1.2.2"

# ── oras bootstrap ────────────────────────────────────────────────────────
if ! command -v oras >/dev/null 2>&1; then
  echo "→ oras not found, installing ${ORAS_VERSION} to ~/.local/bin ..."
  mkdir -p "$HOME/.local/bin"
  curl -sL -o /tmp/oras.tgz \
    "https://github.com/oras-project/oras/releases/download/${ORAS_VERSION}/oras_${ORAS_VERSION#v}_linux_amd64.tar.gz"
  tar -xzf /tmp/oras.tgz -C "$HOME/.local/bin" oras
  chmod +x "$HOME/.local/bin/oras"
  export PATH="$HOME/.local/bin:$PATH"
fi

echo "== Pulling ${REGISTRY}:${REF} =="
oras pull "${REGISTRY}:${REF}"

# ── unpack what we got ────────────────────────────────────────────────────
if [ -f 1bit-linux-x86_64.tar.gz ]; then
  echo "→ extracting 1bit-linux-x86_64.tar.gz"
  tar xzf 1bit-linux-x86_64.tar.gz
fi

DEB="$(ls 1bit-monster_*_amd64.deb 2>/dev/null | head -1 || true)"
if [ -n "$DEB" ]; then
  echo "→ .deb: ${DEB}  (sudo dpkg -i ${DEB})"
fi

APP="$(ls 1bit-monster-*-x86_64.AppImage 2>/dev/null | head -1 || true)"
if [ -n "$APP" ]; then
  chmod +x "$APP"
  echo "→ AppImage: ${APP}  (./${APP})"
fi

# ── usage hints ───────────────────────────────────────────────────────────
echo ""
if [ -x run.sh ]; then
  cat <<'EOF'
Quick start:
  export HSA_OVERRIDE_GFX_VERSION=11.5.1
  export HSA_ENABLE_SDMA=0
  ./run.sh chat
  # any OpenAI client → http://127.0.0.1:13305/v1
EOF
fi
echo ""
echo "Package page: ${PKG_PAGE}"
echo "Other builds: oras pull ${REGISTRY}:<latest | sha-<short>>"
