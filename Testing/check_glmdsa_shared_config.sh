#!/usr/bin/env bash
# check_glmdsa_shared_config.sh — a "shared" DSA layer must never silently reuse
# a stale top-k row (issue #2423).
#
# A "shared" indexer layer reuses the top-k row of the last full layer *in the
# same forward pass*. When the first layer is "shared" there is no such row:
# prev_topk_row is cleared only at pos 0, so at pos > 0 the layer consumed the
# PREVIOUS POSITION's indices, and at pos 0 it fell through to "select every
# position" — neither is a selection, and neither is visible in the output.
#
# The engine cannot fall back to selecting for itself, because the indexer
# weights are loaded only for full layers (glm_moe_dsa.cpp:201). So it refuses
# the config at load, which is what HF does
# (modeling_glm_moe_dsa.py:444-447, ValueError). This gate asserts the refusal
# and, as its control, that a valid shared-after-full config still loads — so a
# pass cannot be an artefact of the harness failing for every input.
#
# usage: check_glmdsa_shared_config.sh [fixture_dir]
#   fixture_dir defaults to /tmp/onebit-glmdsa, the path run_all.sh documents.
set -u

FIX="${1:-/tmp/onebit-glmdsa}"
BIN="${BIN:-/tmp/onebit_tests}"
mkdir -p "$BIN"
CXX="${CXX:-g++}"

if [ ! -f "$FIX/config.json" ]; then
    echo "  - glmdsa shared-config: fixture absent, skipped (python3 Testing/make_mini_glm_moe_dsa.py $FIX)"
    exit 0
fi

"$CXX" -std=c++17 -Iinclude -Isrc -O2 Testing/dump_glmdsa_logits.cpp src/glm_moe_dsa.cpp \
    src/safetensors_reader.cpp src/q4nx_reader.cpp -o "$BIN/dump_glmdsa" 2>/dev/null || {
    echo "✗ glmdsa shared-config: COMPILE FAILED"; exit 1; }

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
echo "5 7 9 11 3" > "$WORK/ids.txt"

# Same weights, same ids; only indexer_types differs, and it is replaced wholesale
# so a variant cannot inherit the fixture's own list by accident.
variant() {  # variant <name> <json-list>
    local name="$1"
    local list="$2"
    local d="$WORK/$name"
    mkdir -p "$d"
    cp -a "$FIX"/. "$d"/
    python3 - "$d/config.json" "$list" <<'PY'
import json, sys
p, lst = sys.argv[1], json.loads(sys.argv[2])
c = json.load(open(p))
c["indexer_types"] = lst
json.dump(c, open(p, "w"), indent=1)
PY
}

fail=0

# ── The fix: a leading "shared" layer has no row to reuse, so refuse it ──
variant leading_shared '["shared","shared","shared","full"]'
if "$BIN/dump_glmdsa" "$WORK/leading_shared" "$WORK/ids.txt" \
        >"$WORK/ls.out" 2>"$WORK/ls.err"; then
    echo "✗ glmdsa shared-config: engine ACCEPTED a config whose first indexer layer is \"shared\""
    echo "    (it has no top-k row to reuse; HF raises ValueError for this config)"
    fail=1
elif grep -q 'no "full" indexer layer precedes it' "$WORK/ls.err"; then
    echo "✓ glmdsa shared-config (leading \"shared\" refused by name, not by accident)"
else
    echo "✗ glmdsa shared-config: refused, but not for the stated reason — the gate is not"
    echo "    proving anything until the message matches:"
    tail -3 "$WORK/ls.err" | sed 's/^/      /'
    fail=1
fi

# ── Control: a valid shared-after-full config must still load and run ──
variant shared_after_full '["full","shared","shared","full"]'
if "$BIN/dump_glmdsa" "$WORK/shared_after_full" "$WORK/ids.txt" \
        >"$WORK/sf.out" 2>"$WORK/sf.err"; then
    : # expected
else
    echo "✗ glmdsa shared-config: CONTROL FAILED — a valid shared-after-full config no longer loads,"
    echo "    so the refusal above is not evidence of a working guard"
    tail -3 "$WORK/sf.err" | sed 's/^/      /'
    fail=1
fi

# And that control must reproduce the fixture's own output exactly: the fix may
# not perturb the valid path.
if [ -f "$WORK/sf.out" ] && "$BIN/dump_glmdsa" "$FIX" "$WORK/ids.txt" >"$WORK/base.out" 2>/dev/null; then
    if cmp -s "$WORK/sf.out" "$WORK/base.out"; then
        echo "✓ glmdsa shared-config control (valid config unchanged, bit-identical to the fixture)"
    else
        echo "✗ glmdsa shared-config: a shared-after-full config diverges from the fixture's own output"
        fail=1
    fi
fi

exit "$fail"
