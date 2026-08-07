#!/usr/bin/env bash
# check-stale-build.sh — guard against the "no work to do" stale-binary trap.
#
# `git commit` does not touch file mtimes, so `cmake --build` can report
# "no work to do" while build/1bit predates HEAD — every source file is
# older than the object files even though the binary is from a previous
# commit. After any git pull/reset/checkout, run this (or `--clean-first`).
#
# Usage: scripts/check-stale-build.sh [binary]
set -u
BIN="${1:-build/1bit}"
if [ ! -f "$BIN" ]; then
    echo "stale-check: $BIN missing — a build is needed"
    exit 2
fi
HEAD_TS=$(git log -1 --format=%ct 2>/dev/null || echo 0)
BIN_TS=$(stat -c %Y "$BIN" 2>/dev/null || echo 0)
if [ "$BIN_TS" -lt "$HEAD_TS" ]; then
    echo "STALE BUILD: $BIN ($(date -d @$BIN_TS '+%F %T')) is older than HEAD ($(git log -1 --format=%ci))"
    echo "  fix: cmake --build build --target 1bit --clean-first -j8"
    exit 1
fi
echo "ok: $BIN is newer than HEAD ($(git log -1 --format=%h))"
exit 0
