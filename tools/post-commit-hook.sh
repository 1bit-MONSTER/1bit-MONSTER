#!/usr/bin/env bash
#
# INSTALL (git hooks are NOT versioned — per-clone, and this repo has several clones). This file is the
# fixed form of the installed hook. SEMANTICALLY one change and nothing else — `set -u` ->
# `set -uo pipefail`; verified by diffing both files with comments and blank lines stripped, which
# yields that single line. Everything else this file adds is the INSTALL block below.
# (An earlier claim of mine said "diff is exactly one expression + comment"; that was true before the
# INSTALL block was added, and a diff of the raw files now reads 16 lines because of it. The semantic
# diff is the one that means anything, which is the same distinction as everywhere else in this repo.)
# The install is a SCRIPT, not an instruction: `tools/hooks/install.sh` (PR #2186, reopened by
# @agent-44437c), which GATES ON THIS FILE'S OWN CONTROL and refuses to install the broken variant even if
# someone copies the wrong file. This block documents the manual equivalent for anyone working without it.
# (WHY A SCRIPT: a comment cannot refuse a copy — and this block previously claimed a gate that had been
# deleted with the duplicate PR, so the file advertised machinery one level above what it had. That is the
# same shape as concluding "machinery beats documentation" and then shipping documentation.)
#
#   bash -n tools/post-commit-hook.sh
#   opts="$(grep -E '^set ' tools/post-commit-hook.sh | head -1)"
#   bash -c "set +e +u +o pipefail; $opts; ! false 2>&1 | sed 's/^/x/' >/dev/null" ||
#       { echo "control FAILED under '$opts' — refusing to install"; exit 1; }
# TWO THINGS THE CONTROL MUST DO, each learned by mutation after the previous form failed:
#   1. apply THIS FILE's options, not a literal copy of them — a control that sets the options itself
#      passes for any file, including a buggy one;
#   2. clear the CALLER's options first (`set +e +u +o pipefail` in a fresh shell) — a subshell INHERITS
#      the caller's, so an installer running under `set -euo pipefail` left pipefail in force and the buggy
#      file passed anyway. The caller's options were enough to hide the file's.
# Verified by mutation in the robust form: this file passes; a `set -u` copy is REFUSED; a file with no
# `set` line is REFUSED.
#   cp tools/post-commit-hook.sh <repo>/.git/hooks/post-commit
#   chmod +x <repo>/.git/hooks/post-commit
# The two copies cover every worktree on both boxes (worktrees inherit the COMMON hooks directory):
#     ryzen     /home/bcloud/projects/1bit-MONSTER/.git/hooks/post-commit
#     strixhalo /home/bcloud/1bit-MONSTER/.git/hooks/post-commit
# `cp` rather than the `ln -sf` that tools/commit-msg-hook.sh uses: that form is right for a NEW hook, and
# this one REPLACES an existing regular file — so an explicit copy is clearer and does not depend on
# relative-path depth from .git/hooks/. Without installing it, the defect returns at the next clone, which
# is the same argument as the symlink in commit-msg-hook.sh.
# post-commit — hands-off PR loop for the 1bit-MONSTER repo
#
# After every commit:
#   - on a feature branch (anything but main): auto-push to origin so the
#     associated PR updates itself. This is the "automatic" step that used to
#     be done by hand.
#   - on main: do NOT push (main is owned by the GitHub merge queue; a direct
#     push would bypass it or be rejected as non-fast-forward) — instead print
#     a reminder that PRs only update from feature branches.
#
# Opt out per-command:  GIT_AUTO_PUSH=0 git commit ...
# Disable entirely:     chmod -x .git/hooks/post-commit
set -uo pipefail   # `set -u` plus the pipe-status fix: without pipefail the
                   # `if ! git push ... | sed ...` below tests SED's status, so its
                   # FAILED branch is unreachable and the success line prints either way.

# Allow opting out (e.g. for WIP commits you do not want on the PR yet).
if [ "${GIT_AUTO_PUSH:-1}" = "0" ]; then
    exit 0
fi

# Only run inside a work tree on a named branch (skip detached HEAD, bare repos).
branch="$(git symbolic-ref --short HEAD 2>/dev/null)" || exit 0
[ -n "$branch" ] || exit 0

# Never push main — the merge queue owns it. Point the user at the right flow.
if [ "$branch" = "main" ]; then
    echo "[auto-push] committed to main — NOT pushed. main is merge-queue owned."
    echo "[auto-push] to get this into a PR: git switch -c <topic> && git push -u origin <topic>"
    echo "[auto-push] (then open the PR with: gh pr create --fill)"
    exit 0
fi

# Only auto-push branches that look like PR branches (topic/*, fix/*, etc.).
# Safety net: refuse to auto-push if the remote branch has diverged (non-FF),
# so we never clobber someone else's pushes to the same branch.
case "$branch" in
    main|master|dev|develop|release/*) exit 0 ;;
esac

echo "[auto-push] pushing $branch → origin/$branch ..."
if ! git push -u origin "$branch" 2>&1 | sed 's/^/[auto-push] /'; then
    echo "[auto-push] push FAILED (non-fast-forward or auth?)."
    echo "[auto-push] fix manually: git pull --rebase origin $branch && git push origin $branch"
    exit 0   # never break the commit itself
fi
echo "[auto-push] done — PR for $branch is up to date"
exit 0
