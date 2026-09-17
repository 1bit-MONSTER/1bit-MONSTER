#!/usr/bin/env bash
# xclbin_toolchain_gate_selfcheck.sh — the provenance writer must not re-record a
# blank build.toolchain in silence (issue #2262).
#
# Why: engine/npu/tests/check_xclbin_provenance.py regenerates PROVENANCE.json from
# the artifacts, and the artifacts carry no compiler marker — so build.toolchain can
# only come from the build that produced them. The writer preserved an existing
# value, but when there was none it wrote null silently. The committed manifest is
# null precisely because the last rebuild ran `--write-manifest` with no
# `--toolchain`, and its own `intent` told the next person to do exactly that. The
# field cannot be filled for the CURRENT artifact either — the artifacts record no
# arm and no recipe reproduces their bytes — so the durable fix is to make the
# omission impossible to repeat rather than to guess a value.
#
# The guard runs before the tree is observed, so these cases need only a manifest,
# not a real xclbin set. Assertions are on the message, not the exit code: past the
# guard the tool goes on to observe an empty tree and may legitimately fail.
#
# Run: bash Testing/xclbin_toolchain_gate_selfcheck.sh
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOL="$REPO/engine/npu/tests/check_xclbin_provenance.py"
REFUSAL='REFUSING to write a manifest with build.toolchain = null'
[ -f "$TOOL" ] || { echo "FAIL: no $TOOL"; exit 1; }

T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
fail=0

mk_tree() { # mk_tree <dir> <toolchain-json-literal>
    mkdir -p "$1/engine/npu/xclbins"
    printf '{"schema": 1, "build": {"toolchain": %s, "generating_script_revision": null}, "artifacts": {}}\n' \
        "$2" > "$1/engine/npu/xclbins/PROVENANCE.json"
}

check() { # check <name> <expect-refusal:yes|no> <dir> <args...>
    local name="$1" expect="$2" dir="$3"; shift 3
    local out rc
    out="$(python3 "$TOOL" --root "$dir" "$@" 2>&1)"; rc=$?
    if printf '%s' "$out" | grep -q "$REFUSAL"; then got=yes; else got=no; fi
    if [ "$got" = "$expect" ]; then
        printf '  ok   %-52s refusal=%s rc=%s\n' "$name" "$got" "$rc"
    else
        printf '  FAIL %-52s refusal=%s (wanted %s) rc=%s\n' "$name" "$got" "$expect" "$rc"
        printf '%s\n' "$out" | tail -4 | sed 's/^/         /'
        fail=1
    fi
}

echo "provenance writer, build.toolchain guard:"

A="$T/null"; mk_tree "$A" null
check "null + --write-manifest, no flag"        yes "$A" --write-manifest
check "null + --allow-null-toolchain"           no  "$A" --write-manifest --allow-null-toolchain
check "null + --toolchain given"                no  "$A" --write-manifest --toolchain "aiecc, LLVM 23"
check "null + read-only (no --write-manifest)"  no  "$A"

B="$T/set"; mk_tree "$B" '"recorded-arm"'
check "existing value + --write-manifest"       no  "$B" --write-manifest

# The point of the guard: the refusal must happen BEFORE anything is written.
C="$T/untouched"; mk_tree "$C" null
before="$(cat "$C/engine/npu/xclbins/PROVENANCE.json")"
python3 "$TOOL" --root "$C" --write-manifest >/dev/null 2>&1
if [ "$(cat "$C/engine/npu/xclbins/PROVENANCE.json")" = "$before" ]; then
    echo "  ok   refusal left the manifest untouched"
else
    echo "  FAIL refusal still wrote the manifest"; fail=1
fi

exit "$fail"
