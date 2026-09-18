#!/usr/bin/env bash
# lint_new_diagnostics_selfcheck.sh — the no-new-diagnostics predicate (#2486) must
# fail on a regression and pass on a pre-existing violation.
#
# The predicate lives in Testing/lint-new-diagnostics.sh and is exercised here against a
# throwaway git repo and a stub clang-format (CLANG_FORMAT is the seam for that; this
# check runs in the Self-checks job, which does not install clang-format).
#
# Two floors, because a gate whose counter matches nothing passes every case:
#   * the stub's own counts are asserted first (a stub that emits nothing fails here);
#   * the gate's output must show a NONZERO head count for the fixture that has one.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$REPO/Testing/lint-new-diagnostics.sh"
[ -f "$SRC" ] || { echo "FAIL: no $SRC"; exit 1; }

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
fail=0
ok()  { printf '  ok   %-52s %s\n' "$1" "$2"; }
bad() { printf '  FAIL %-52s %s\n' "$1" "$2"; fail=1; }
expect() { if [ "$2" = "$3" ]; then ok "$1" "$2"; else bad "$1" "$2 != $3"; fi; }

# ── the stub: one diagnostic line per line containing BADFMT ────────────────────
mkdir -p "$TMP/bin"
cat > "$TMP/bin/clang-format" <<'STUB'
#!/bin/bash
f="${@: -1}"
n=$(grep -c 'BADFMT' "$f" 2>/dev/null || true)
for i in $(seq 1 "$n"); do
  echo "$f:$i:1: error: code should be clang-formatted [-Wclang-format-violations]"
done
[ "$n" -gt 0 ] && exit 1
exit 0
STUB
chmod +x "$TMP/bin/clang-format"
export CLANG_FORMAT="$TMP/bin/clang-format"

# ── control 1: the checker sees what is there ───────────────────────────────────
mkdir -p "$TMP/probe"
printf 'int a;\nBADFMT\nBADFMT\nBADFMT\nint b;\n' > "$TMP/probe/three.cpp"
printf 'int clean;\n' > "$TMP/probe/zero.cpp"
probe() { "$CLANG_FORMAT" --dry-run --Werror "$1" 2>&1 | grep -cE '^[^:]+:[0-9]+:[0-9]+: (warning|error):' || true; }
expect "control: stub reports 3 for a 3-marker file" "$(probe "$TMP/probe/three.cpp")" "3"
expect "control: stub reports 0 for a clean file"    "$(probe "$TMP/probe/zero.cpp")"  "0"

# ── fixture repo: the predicate is the real script, copied verbatim ─────────────
mkdir -p "$TMP/repo/src" "$TMP/repo/Testing"
cp "$SRC" "$TMP/repo/Testing/lint-new-diagnostics.sh"
G() { git -C "$TMP/repo" "$@"; }
G init -q; G config user.email s@t; G config user.name t
printf 'int a;\nBADFMT\nBADFMT\nBADFMT\nint b;\n' > "$TMP/repo/src/unformatted.cpp"
printf 'int clean;\n'                            > "$TMP/repo/src/clean.cpp"
G add -A; G commit -qm base
base=$(G rev-parse HEAD)

gate() { # gate <base> -> prints the predicate's output, returns its status
  ( cd "$TMP/repo" && bash Testing/lint-new-diagnostics.sh "$1" )
}
step() { G add -A; G commit -qm "$1"; G rev-parse HEAD; }

# A: comment-only edit of a file that already has 3 violations — the #2554 shape
printf 'int a;  // touched\nBADFMT\nBADFMT\nBADFMT\nint b;\n' > "$TMP/repo/src/unformatted.cpp"
c1=$(step "comment-only edit")
outA=$(gate "$base"); rcA=$?
[ "$rcA" -eq 0 ] && ok "A comment-only edit passes" "exit 0" || bad "A comment-only edit passes" "exit $rcA"
case "$outA" in *"3 at head vs 3 at base"*) ok "A saw the pre-existing 3 (not a vacuous pass)" "3 vs 3";; *) bad "A saw the pre-existing 3" "$(printf '%s' "$outA" | tail -2 | tr '\n' ' ')";; esac

# B: one NEW violation
printf 'int a;  // touched\nBADFMT\nBADFMT\nBADFMT\nBADFMT\nint b;\n' > "$TMP/repo/src/unformatted.cpp"
c2=$(step "add a violation")
outB=$(gate "$c1"); rcB=$?
[ "$rcB" -ne 0 ] && ok "B a new violation fails" "exit $rcB" || bad "B a new violation fails" "exit 0"
case "$outB" in *"delta +1"*) ok "B names the delta" "+1";; *) bad "B names the delta" "$(printf '%s' "$outB" | grep -c delta) delta line(s)";; esac

# C: the change FIXES a violation
printf 'int a;  // touched\nBADFMT\nBADFMT\nint b;\n' > "$TMP/repo/src/unformatted.cpp"
c3=$(step "fix a violation")
gate "$c2" >/dev/null; rcC=$?
[ "$rcC" -eq 0 ] && ok "C fixing a violation passes" "exit 0" || bad "C fixing a violation passes" "exit $rcC"

# D: a NEW unformatted file
printf 'BADFMT\n' > "$TMP/repo/src/newfile.cpp"
c4=$(step "new unformatted file")
gate "$c3" >/dev/null; rcD=$?
[ "$rcD" -ne 0 ] && ok "D a new unformatted file fails" "exit $rcD" || bad "D a new unformatted file fails" "exit 0"

# E: a deleted file is skipped, not a failure
rm -f "$TMP/repo/src/clean.cpp"
c5=$(step "delete a file")
outE=$(gate "$c4"); rcE=$?
[ "$rcE" -eq 0 ] && ok "E a deletion passes" "exit 0" || bad "E a deletion passes" "exit $rcE"
case "$outE" in *clean.cpp*) bad "E a deletion is not compared" "it was listed";; *) ok "E a deletion is not compared" "absent from the diff";; esac

# E2: the "not present at head" branch is reachable only through FILES, so cover it there
outE2=$( cd "$TMP/repo" && FILES="src/clean.cpp" bash Testing/lint-new-diagnostics.sh "$c4" ); rcE2=$?
[ "$rcE2" -eq 0 ] && ok "E2 a named-but-absent file passes" "exit 0" || bad "E2 a named-but-absent file passes" "exit $rcE2"
case "$outE2" in *"skip (not present at head)"*) ok "E2 reports the skip" "skip";; *) bad "E2 reports the skip" "no skip line";; esac

# F: an edit to an already-clean file
printf 'int clean;  // still clean\n' > "$TMP/repo/src/ok.cpp"
printf 'int ok2;\n' >> "$TMP/repo/src/ok.cpp"
c6=$(step "clean file")
gate "$c5" >/dev/null; rcF=$?
[ "$rcF" -eq 0 ] && ok "F a clean file passes" "exit 0" || bad "F a clean file passes" "exit $rcF"

# G: the environment seam is honest — a missing binary is exit 2, not a pass
( cd "$TMP/repo" && CLANG_FORMAT="$TMP/bin/does-not-exist" bash Testing/lint-new-diagnostics.sh "$c6" >/dev/null 2>&1 ); rcG=$?
[ "$rcG" -eq 2 ] && ok "G a missing clang-format is exit 2" "exit 2" || bad "G a missing clang-format is exit 2" "exit $rcG"

# H: a path containing a SPACE must be compared, not split and skipped. The lint job's own
# comment records that this bit it once (-print0/-0), and a split here fails silently: both
# halves read as "not present at head", so the file would be skipped and a new violation
# would pass. The first version of this predicate had that bug; this case is why.
printf 'BADFMT\n' > "$TMP/repo/src/with space.cpp"
step "path with a space" >/dev/null   # commits the fixture; the gate's base stays $c6
outH=$(gate "$c6"); rcH=$?
[ "$rcH" -ne 0 ] && ok "H a spaced path is linted (fails on +1)" "exit $rcH" || bad "H a spaced path is linted (fails on +1)" "exit 0 — it was skipped"
case "$outH" in *"with space.cpp"*) ok "H names the spaced path" "listed";; *) bad "H names the spaced path" "absent from the output";; esac

if [ "$fail" -eq 0 ]; then echo "OK: no-new-diagnostics predicate (7 cases + 2 controls + env seam)"; fi
exit "$fail"
