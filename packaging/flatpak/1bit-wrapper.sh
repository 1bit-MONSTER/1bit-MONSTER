#!/bin/sh
# Wrapper installed as /app/bin/1bit in the Flatpak sandbox. The engine is
# built on Ubuntu 26.04 (glibc 2.43) — newer than the Freedesktop runtime's
# glibc — so this launches it with the bundled glibc's own dynamic linker
# and --library-path, same trick AppImage runtimes use. argv[0] is preserved,
# so legacy-name symlinks still dispatch by subcommand.
HERE="$(dirname "$(readlink -f "$0")")"
LIBPATH="/app/glibc:/app/therock/_rocm_sdk_devel/lib:/app/therock/_rocm_sdk_libraries/lib:/app/therock/_rocm_sdk_core/lib:/app/therock/_rocm_sdk_core/lib/llvm/lib:/app/lib"
exec "/app/glibc/ld-linux-x86-64.so.2" --library-path "$LIBPATH" "$HERE/1bit.bin" "$@"
