#!/usr/bin/env bash
# 1bit.systems — one-binary install
# curl -sL https://1bit.systems/install.sh | bash
# Or: tar xzf 1bit-systems-*.tar.gz && bash install.sh
set -euo pipefail

say() { printf "✓ %s\n" "$*"; }
die() { printf "✗ %s\n" "$*" >&2; exit 1; }

INSTALL_DIR="${HOME}/.local/1bit-systems"
BIN_DIR="${HOME}/.local/bin"

# 1. Binary already staged next to us (tarball layout: usr/bin/1bit)?
if [ -x "$(dirname "$0")/usr/bin/1bit" ]; then
  SRC="$(cd "$(dirname "$0")" && pwd)/usr/bin/1bit"
  NPU_SRC="$(dirname "$SRC")/1bit-npu"
  say "Using staged binary at ${SRC}"
else
  # 2. Fetch the latest release tarball from GitHub.
  say "Fetching latest release metadata…"
  URL=$(curl -fsSL https://api.github.com/repos/1bit-systems/1bit-systems/releases/latest \
        | grep -o 'https://[^"]*linux-amd64.tar.gz' | head -1)
  [ -n "$URL" ] || die "No release tarball found on GitHub."
  TMP=$(mktemp -d)
  trap 'rm -rf "$TMP"' EXIT
  say "Downloading ${URL}"
  curl -fL "$URL" -o "$TMP/1bit.tar.gz"
  tar xzf "$TMP/1bit.tar.gz" -C "$TMP"
  SRC=$(find "$TMP" -name 1bit -path '*/usr/bin/1bit' | head -1)
  [ -n "$SRC" ] || die "Release tarball has no usr/bin/1bit."
  NPU_SRC="$(dirname "$SRC")/1bit-npu"
fi

mkdir -p "${INSTALL_DIR}" "${BIN_DIR}"
cp "$SRC" "${INSTALL_DIR}/1bit"
chmod +x "${INSTALL_DIR}/1bit"
ln -sf "${INSTALL_DIR}/1bit" "${BIN_DIR}/1bit"
# legacy server names → the same binary (argv[0] dispatch)
for name in zaya_server unified_server unified_router vision_server onebitd; do
  ln -sf "${INSTALL_DIR}/1bit" "${BIN_DIR}/$name"
done
if [ -f "$NPU_SRC" ]; then
  cp "$NPU_SRC" "${INSTALL_DIR}/1bit-npu" && chmod +x "${INSTALL_DIR}/1bit-npu"
fi

if [[ ":$PATH:" != *":${BIN_DIR}:"* ]]; then
  echo "export PATH=\"${BIN_DIR}:\$PATH\"" >> "${HOME}/.bashrc"
  say "Added ${BIN_DIR} to PATH in .bashrc"
fi

echo ""
say "Installed one binary: ${INSTALL_DIR}/1bit ($(stat -c%s "${INSTALL_DIR}/1bit" 2>/dev/null || echo "?") bytes)"
echo ""
echo "  Quick start:"
echo "    1bit status              # check hardware/stack"
echo "    1bit pull qwen3-0.6b     # download a model"
echo "    1bit zaya -m model.1bp -p 'Hello world'   # serve/infer"
echo "    1bit jarvis              # TTS/voice server"
echo "    1bit vision --mmproj …   # vision-language server"
echo ""
echo "  Docs: https://1bit.systems · Repo: https://github.com/1bit-systems/1bit-systems"
