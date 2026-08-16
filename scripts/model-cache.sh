#!/usr/bin/env bash
# model-cache.sh — symlink-backed model cache.
#
# The benchmark/test cycle (docs/wiki/models.md) downloads a model, uses it,
# deletes it — every re-run re-downloads, and any manual copy duplicates disk.
# This keeps ONE canonical copy per model in the cache and hands out symlinks,
# so N consumers of the same model cost 1 download + N symlinks.
#
# Usage:
#   model-cache.sh path <name> <url>           # ensure canonical copy, print cache path
#   model-cache.sh link <name> <url> <dest>    # canonical copy + symlink at <dest>
#   model-cache.sh cache                       # print the cache dir
#
# Env: ONEBIT_MODELS_CACHE overrides the cache dir
#       (default: $HOME/.cache/1bit-systems/models)
set -euo pipefail

CACHE="${ONEBIT_MODELS_CACHE:-$HOME/.cache/1bit-systems/models}"
mkdir -p "$CACHE"

ensure() {  # ensure <name> <url> -> prints canonical path
    local name="$1" url="$2"
    local dst="$CACHE/$name"
    if [ ! -f "$dst" ]; then
        echo "  [cache] downloading $name ..." >&2
        curl -L --fail --silent --show-error "$url" -o "$dst.part"
        mv "$dst.part" "$dst"
    fi
    echo "$dst"
}

case "${1:-}" in
    path)  [ $# -ge 3 ] || { echo "usage: $0 path <name> <url>" >&2; exit 1; }; ensure "$2" "$3" ;;
    link)  [ $# -ge 4 ] || { echo "usage: $0 link <name> <url> <dest>" >&2; exit 1; }; ln -sf "$(ensure "$2" "$3")" "$4" ;;
    cache) echo "$CACHE" ;;
    *) echo "usage: $0 {path|link|cache} ..." >&2; exit 1 ;;
esac
