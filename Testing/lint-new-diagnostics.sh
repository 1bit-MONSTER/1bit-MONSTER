#!/usr/bin/env bash
# lint-new-diagnostics.sh — fail a change that ADDS clang-format diagnostics.
#
# Why (#2486). The required check "Lint (clang-format)" cannot fail: the tree carries
# 184,118 diagnostics in 740 of the 744 files the job selects, so "fail on any
# diagnostic" is either a tree-wide reformat or nothing at all — and plain diff-gating
# fails any touched file that already had violations, which would have blocked #2554,
# a comment-only change.
#
# The predicate here is "no NEW diagnostics": for every file this change touches,
# compare the diagnostic count at HEAD with the count for the same file at BASE and
# fail only on an INCREASE. Touching an unformatted file stays free; making it worse
# does not. That keeps the gate able to fail without making the existing backlog
# every author's problem on every edit.
#
# Usage:
#   Testing/lint-new-diagnostics.sh [BASE]
#     BASE defaults to the merge base with origin/main.
#   FILES="a.cpp b.cpp"   override the changed-file detection (used by the self-check)
#   CLANG_FORMAT=...      override the binary. The self-check cannot install
#                         clang-format, so it points this at a stub — the same
#                         override-for-testability seam as NPU_FW_DIR in npu-reset.yml.
#
# Exit: 0 no new diagnostics, 1 at least one file got worse, 2 environment problem.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 2
CF="${CLANG_FORMAT:-clang-format}"
command -v "$CF" >/dev/null 2>&1 || { echo "ENVIRONMENT: no $CF on PATH" >&2; exit 2; }

BASE="${1:-}"
if [ -z "$BASE" ]; then
    BASE="$(git merge-base origin/main HEAD 2>/dev/null || git merge-base main HEAD 2>/dev/null || true)"
fi
if [ -z "$BASE" ] || ! git rev-parse --verify -q "$BASE^{commit}" >/dev/null; then
    echo "ENVIRONMENT: cannot resolve a base commit (got '${BASE:-<empty>}')" >&2
    exit 2
fi

# Same selection the lint job makes: these extensions, minus the vendored/build trees.
# The git side is NUL-delimited and read into an array: the lint job's own comment records
# that a path containing a space used to split into two arguments, and a split here would
# not fail — both halves would be "not present at head" and the file would be skipped in
# silence, which is the failure this whole predicate exists to avoid.
changed=()
if [ -n "${FILES:-}" ]; then
    read -r -a changed <<< "$FILES"  # documented override; space-separated, for the self-check
else
    while IFS= read -r -d '' f; do
        changed+=("$f")
    done < <(git diff -z --name-only --diff-filter=d "$BASE" -- '*.cpp' '*.h' '*.hip' '*.hpp' 2>/dev/null \
             | grep -zvE '/(build|_deps|third_party|\.git)/' || true)
fi

count() { # count <path> -> diagnostics on stdout
    local out
    out="$("$CF" --dry-run --Werror "$1" 2>&1)"
    printf '%s\n' "$out" | grep -cE '^[^:]+:[0-9]+:[0-9]+: (warning|error):' || true
}

# The base side must be linted AS IF it were the same file. Materialising it to a temp
# path instead does not work, and fails in the one direction a gate must never fail:
# clang-format discovers .clang-format by walking up from the file's directory, so a
# /tmp copy gets LLVM defaults, and a name without a .cpp/.h extension does not even
# select the language. Measured on src/model_router.cpp with clang-format 23.1.1:
#
#   the repo file itself                     38 diagnostics
#   the same bytes as /tmp/base (no ext)    269      <- base inflated, gate passes everything
#   the same bytes as /tmp/base.cpp         269
#   stdin with --assume-filename=<repo path> 34      <- matches the repo file (pre-edit)
#
# So the base content goes through stdin with --assume-filename pointing at the real path.
count_base() { # <repo-relative path>; base content on stdin
    local out
    out="$("$CF" --dry-run --Werror --assume-filename="$1" - 2>&1)"
    printf '%s\n' "$out" | grep -cE '^[^:]+:[0-9]+:[0-9]+: (warning|error):' || true
}

checked=0; worse=0; total_head=0
for f in "${changed[@]}"; do
    [ -e "$f" ] || { echo "  skip (not present at head): $f"; continue; }
    head_n="$(count "$f")"
    base_n=0
    if git show "$BASE:$f" >/dev/null 2>&1; then
        base_n="$(git show "$BASE:$f" | count_base "$f")"
    fi
    checked=$((checked + 1)); total_head=$((total_head + head_n))
    if [ "$head_n" -gt "$base_n" ]; then
        echo "  FAIL $f: $head_n diagnostic(s) at head vs $base_n at base (delta +$((head_n - base_n)))"
        worse=$((worse + 1))
    else
        echo "  ok   $f: $head_n at head vs $base_n at base"
    fi
done

if [ "$checked" -eq 0 ]; then
    echo "clang-format: no changed C/C++ file in $(git rev-parse --short "$BASE")..HEAD — nothing to compare"
    exit 0
fi
if [ "$worse" -gt 0 ]; then
    echo "clang-format: $worse of $checked changed file(s) gained diagnostics"
    exit 1
fi
echo "clang-format: no new diagnostics ($checked changed file(s), $total_head diagnostic(s) at head — pre-existing ones are not this change's problem)"
exit 0
