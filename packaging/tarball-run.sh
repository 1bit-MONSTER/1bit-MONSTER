#!/usr/bin/env bash
# 1bit.MONSTER — launcher for the binary tarball.
#
# This file is a TEMPLATE: packaging/Makefile's `stage` target installs it as
# `run.sh` at the ROOT of the staged tree, which is what `package-tarball` and
# `package-site` (the website's .tar.xz) tar up. The documented quick start is
# therefore literally true:
#
#     tar xJf 1bit-monster-<version>-linux-amd64.tar.xz
#     cd 1bit-monster-<version>
#     ./run.sh chat
#
# Why a launcher at all: `librocm_cpp.so` ships inside the tarball next to the
# ELF, and a tree extracted anywhere has no ld.so entry pointing at it. The .deb
# and .rpm solve the same problem with /etc/ld.so.conf.d + ldconfig; a tarball
# has no install step, so the search path is wired up here.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"

# Two tarball layouts exist in the wild: `make stage` publishes an FHS tree
# (usr/bin + the shared lib under usr/lib/x86_64-linux-gnu), while the GHCR
# tarball built by .github/workflows/release.yml is flat (bin/ + lib/).
# Support both so this stays the single definition of the launcher.
if [ -x "$HERE/usr/bin/1bit" ]; then
    BIN_DIR="$HERE/usr/bin"
    LIB_DIRS="$HERE/usr/lib/x86_64-linux-gnu:$HERE/usr/lib/1bit"
elif [ -x "$HERE/bin/1bit" ]; then
    BIN_DIR="$HERE/bin"
    LIB_DIRS="$HERE/lib"
else
    echo "run.sh: no 1bit binary under $HERE (expected usr/bin/1bit or bin/1bit)" >&2
    echo "run.sh: extract the whole tarball first, then re-run" >&2
    exit 127
fi

export LD_LIBRARY_PATH="$LIB_DIRS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# The legacy entry points (zaya_server, onebit, onebitd, …) are symlinks beside
# the ELF and are resolved by argv[0] dispatch; keep them reachable for scripts
# the binary itself may spawn.
export PATH="$BIN_DIR:$PATH"

exec "$BIN_DIR/1bit" "$@"
