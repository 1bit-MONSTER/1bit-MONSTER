#!/bin/sh
# Wrapper installed as /app/bin/1bit in the Flatpak sandbox. Sets
# LD_LIBRARY_PATH to the bundled TheRock tree and /app/lib, then execs the
# real binary (argv[0] is preserved, so legacy-name symlinks still dispatch
# by subcommand).
HERE="$(dirname "$(readlink -f "$0")")"
export LD_LIBRARY_PATH="/app/therock/_rocm_sdk_devel/lib:/app/therock/_rocm_sdk_libraries/lib:/app/therock/_rocm_sdk_core/lib:/app/therock/_rocm_sdk_core/lib/llvm/lib:/app/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$HERE/1bit.bin" "$@"
