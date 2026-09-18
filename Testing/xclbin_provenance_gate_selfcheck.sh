#!/usr/bin/env bash
# xclbin_provenance_gate_selfcheck.sh — the provenance gate's comparisons must be able to
# fail, including after the same commit regenerates the manifest (issue #2513).
#
# Why: engine/npu/xclbins/PROVENANCE.json records nine population keys, a whole census block
# and an observed block, but compare() read four population keys plus the symlink/artifact
# maps. population.tracked_paths_under_dir therefore said 519 against a tree of 520 from
# 2026-09-16 (4aa5fa1ca regenerated the manifest with 12 of the 13 paths it added in the
# index) and this gate stayed green. hygiene()'s structural assertions could not fire
# either: observe() classifies each top-level path as a symlink XOR a regular file, and it
# only ever records *regular* names in a UUID group, so "a name is both" and "an alias left
# its group" were unreachable - on the real tree, 1068 guard iterations and 0 assertions.
#
# The cases below assert the exit code AND a marker in the message, so a FAIL cannot be
# mistaken for the tool failing to start. The escape cases run AFTER --write-manifest: a
# manifest diff is defeated by regenerating it, which the tool's own failure text tells you
# to do, so an invariant that matters has to hold without a baseline.
#
# The instruction streams are covered the same way: `streams` records a sha256 and byte
# count for every tracked top-level *.txt (and the target of the 12 aliases), which is
# 116 files and 60,962,904 bytes - 6x census.payload_bytes, all of it hashed nowhere on
# 2026-09-18, when appending TAMPER to insts_i8_G_qwen3_4b.txt still left the gate at
# "OK: matches PROVENANCE.json". The writer's own section literal is checked too: a
# section observe() produces and the writer forgets is dropped in silence.
#
# Run: bash Testing/xclbin_provenance_gate_selfcheck.sh
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOOL="$REPO/engine/npu/tests/check_xclbin_provenance.py"
[ -f "$TOOL" ] || { echo "FAIL: no $TOOL"; exit 1; }

T="$(mktemp -d)"; trap 'rm -rf "$T"' EXIT
FIX="$T/fixture"; D="$FIX/engine/npu/xclbins"
fail=0

uuid() { python3 -c 'import sys;print(sys.argv[1]*32)' "$1"; }

make_fixture() { # make_fixture <alias-target>
    rm -rf "$FIX"; mkdir -p "$D/sub"
    _xclbin() { printf 'AXLF{"XclBinUUID":"%s","TimeStamp":"%s"}' "$(uuid "$1")" "$2" > "$3"; }
    _xclbin a 1755000000 "$D/final_i8_D_K1_N1.xclbin"
    _xclbin b 1755000001 "$D/final_i8_G_K1_N1.xclbin"
    _xclbin c 1755000002 "$FIX/evil.xclbin"                 # outside the artifact dir
    _xclbin d 1755000003 "$D/sub/other.xclbin"              # tracked, but not a sibling
    echo "insts" > "$D/insts_i8_D_K1_N1.txt"
    ln -s insts_i8_D_K1_N1.txt "$D/insts_i8_D_alias.txt"   # an alias of a stream, not of an xclbin
    ln -s "$1" "$D/final_i8_D_alias.xclbin"
    git -C "$FIX" init -q
    git -C "$FIX" add -A >/dev/null 2>&1
    git -C "$FIX" -c user.email=n@e -c user.name=n commit -qm fixture >/dev/null 2>&1
    # written after the commit, so it is in the worktree but not in the commit
    _xclbin e 1755000004 "$D/untracked.xclbin"
}

regenerate() { python3 "$TOOL" --root "$FIX" --write-manifest --toolchain "selfcheck fixture" >/dev/null 2>&1; }
gate()       { python3 "$TOOL" --root "$FIX" 2>&1; }

expect() { # expect <name> <pass|fail> [marker-in-message]
    local name="$1" want="$2" marker="${3:-}" out rc verdict=ok
    out="$(gate)"; rc=$?
    local got=pass; [ "$rc" -eq 0 ] || got=fail
    [ "$got" = "$want" ] || verdict=BAD
    if [ "$want" = fail ] && ! printf '%s' "$out" | grep -qF "$marker"; then verdict=BAD; fi
    if [ "$verdict" = ok ]; then
        printf '  ok   %-54s rc=%s\n' "$name" "$rc"
    else
        printf '  FAIL %-54s rc=%s (wanted %s matching %s)\n' "$name" "$rc" "$want" "$marker"
        printf '%s\n' "$out" | grep -E 'VIOLATION|^  - |FAILURE|^OK' | head -4 | sed 's/^/         /'
        fail=1
    fi
}

# persists — the sections observe() produces that the written manifest does not carry.
persists() {
    python3 - "$TOOL" "$FIX" <<'PY'
import importlib.util, json, sys
from pathlib import Path
tool, root = sys.argv[1], Path(sys.argv[2])
spec = importlib.util.spec_from_file_location("prov", tool)
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
obs = mod.observe(root)
man = json.load(open(root / "engine/npu/xclbins/PROVENANCE.json"))
print(" ".join(sorted(set(obs) - set(man))))
PY
}

perturb() { # perturb <python statement(s) over `m`>, from a freshly regenerated manifest
    regenerate
    python3 - "$FIX/engine/npu/xclbins/PROVENANCE.json" "$1" <<'PY'
import json, sys
path, expr = sys.argv[1], sys.argv[2]
m = json.load(open(path))
exec(expr)
json.dump(m, open(path, "w"), indent=1, sort_keys=True)
PY
}

echo "xclbin provenance gate: the recorded set must be compared, not just recorded"

# --- floor: the fixture must exercise the comparison, or nothing below means anything ---
make_fixture final_i8_D_K1_N1.xclbin
regenerate
reg=$(git -C "$FIX" ls-files -s -- engine/npu/xclbins | awk '$1=="100644"' | wc -l)
lnk=$(git -C "$FIX" ls-files -s -- engine/npu/xclbins | awk '$1=="120000"' | wc -l)
tracked=$(git -C "$FIX" ls-files -- engine/npu/xclbins | wc -l)
recorded=$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1]))["population"]["tracked_paths_under_dir"])' \
    "$FIX/engine/npu/xclbins/PROVENANCE.json")
nsha=$(python3 -c 'import json,sys;s=json.load(open(sys.argv[1]))["streams"];print(sum(1 for v in s.values() if "sha256" in v))' \
    "$FIX/engine/npu/xclbins/PROVENANCE.json" 2>/dev/null || echo 0)
nalias=$(python3 -c 'import json,sys;s=json.load(open(sys.argv[1]))["streams"];print(sum(1 for v in s.values() if "symlink" in v))' \
    "$FIX/engine/npu/xclbins/PROVENANCE.json" 2>/dev/null || echo 0)
if [ "$reg" -ge 3 ] && [ "$lnk" -ge 1 ] && [ "$recorded" = "$tracked" ] && [ "$nsha" -ge 1 ] && [ "$nalias" -ge 1 ]; then
    printf '  ok   %-54s %s regular, %s symlink, %s stream(s), %s stream alias, manifest=%s\n' \
        "fixture floor" "$reg" "$lnk" "$nsha" "$nalias" "$recorded"
else
    printf '  FAIL %-54s %s regular, %s symlink, %s stream(s), %s stream alias, manifest=%s tracked=%s\n' \
        "fixture floor" "$reg" "$lnk" "$nsha" "$nalias" "$recorded" "$tracked"
    fail=1
fi

# --- the invariant needs no baseline: it must hold with the manifest regenerated ---
expect "baseline: alias -> tracked sibling artifact"        pass
make_fixture ../../../evil.xclbin;              regenerate
expect "alias escapes the artifact directory"               fail "leaves engine/npu/xclbins/"
make_fixture sub/other.xclbin;                  regenerate
expect "alias -> tracked path below the directory"          fail "is not a tracked *.xclbin"
make_fixture untracked.xclbin;                  regenerate
expect "alias -> worktree file the commit does not carry"   fail "is not a tracked *.xclbin"

# --- fields the manifest records about the commit, each one previously unread ---
make_fixture final_i8_D_K1_N1.xclbin;           regenerate
perturb 'm["population"]["tracked_paths_under_dir"] += 1'
expect "population.tracked_paths_under_dir drifted"         fail "population.tracked_paths_under_dir"
perturb 'm["population"]["txt_symlinks"] += 1'
expect "population.txt_symlinks drifted"                    fail "population.txt_symlinks"
perturb 'm["census"]["payload_bytes"] += 1'
expect "census.payload_bytes drifted"                       fail "census.payload_bytes"
perturb 'm["observed"]["pairs_with_insts"] = []'
expect "observed.pairs_with_insts drifted"                  fail "observed.pairs_with_insts"

# --- the instruction streams: recorded, compared, and persisted ------------------
perturb 'm["streams"]["insts_i8_D_K1_N1.txt"]["sha256"] = "0" * 64'
expect "recorded stream hash drifted"                       fail "streams.insts_i8_D_K1_N1.txt"
make_fixture final_i8_D_K1_N1.xclbin;           regenerate
printf 'x' >> "$D/insts_i8_D_K1_N1.txt"
expect "stream CONTENT changed under a green manifest"      fail "streams.insts_i8_D_K1_N1.txt"
make_fixture final_i8_D_K1_N1.xclbin;           regenerate
ln -sfn insts_i8_D_K1_N1.txt.new "$D/insts_i8_D_alias.txt"
expect "stream alias repointed"                             fail "streams.insts_i8_D_alias.txt"

# The writer's own section literal: a section observe() produces and this literal forgets
# is dropped in silence, which is how the streams stayed unrecorded in the first place.
make_fixture final_i8_D_K1_N1.xclbin;           regenerate
dropped="$(persists)"
if [ -z "$dropped" ]; then
    printf '  ok   %-54s %s\n' "writer persists every observed section" "none dropped"
else
    printf '  FAIL %-54s dropped: %s\n' "writer persists every observed section" "$dropped"
    fail=1
fi
perturb 'del m["streams"]'
dropped="$(persists)"
if [ "$dropped" = "streams" ]; then
    printf '  ok   %-54s %s\n' "control: a dropped section is reported" "streams"
else
    printf '  FAIL %-54s got %s\n' "control: a dropped section is reported" "${dropped:-<nothing>}"
    fail=1
fi

# --- and the two exclusions stay excluded: tightening must not resurrect host state (#2218) ---
perturb 'm["population"]["dangling_symlinks"] += 1'
expect "host-dependent dangling_symlinks is NOT compared"   pass
perturb 'm["population"]["alias_symlinks_in_tree"] += 1'
expect "host-dependent alias_symlinks_in_tree is NOT cmp"   pass

exit "$fail"
