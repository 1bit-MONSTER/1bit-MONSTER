#!/usr/bin/env bash
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
