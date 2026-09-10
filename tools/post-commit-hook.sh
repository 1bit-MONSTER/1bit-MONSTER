#!/usr/bin/env bash
#
# INSTALL (git hooks are NOT versioned — per-clone, and this repo has several clones). This file is the
# fixed form of the installed hook. SEMANTICALLY one change and nothing else — `set -u` ->
# `set -uo pipefail`; verified by diffing both files with comments and blank lines stripped, which
# yields that single line. Everything else this file adds is the INSTALL block below.
# (An earlier claim of mine said "diff is exactly one expression + comment"; that was true before the
# INSTALL block was added, and a diff of the raw files now reads 16 lines because of it. The semantic
# diff is the one that means anything, which is the same distinction as everywhere else in this repo.)
# The install GATES ON ITS OWN CONTROL, so a fresh clone cannot install the broken variant even if someone
# copies the wrong file. The middle line is the property this defect violated — under `pipefail`, a failing
# pipeline must reach the failure branch — so the copy is refused if the file under it is the buggy shape.
# (@agent-44437c's snippet, offered additively to this file when their own PR was closed in its favour.)
#   bash -n tools/post-commit-hook.sh
#   ( . <(grep '^set ' tools/post-commit-hook.sh); ! false 2>&1 | sed 's/^/x/' >/dev/null ) || {
#       echo "control FAILED: failure branch unreachable — refusing to install" >&2; exit 1; }
# NOTE the `grep '^set '` inside the subshell rather than a literal `set -uo pipefail`: the offered form set
# the options ITSELF, so it passed for ANY file including a buggy copy and would have installed the defect it
# exists to refuse. Sourcing the file's OWN options is what makes the control a check on the artifact.
# Verified by mutation: with a buggy `set -u` copy the gate refuses; with this file it passes.
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
