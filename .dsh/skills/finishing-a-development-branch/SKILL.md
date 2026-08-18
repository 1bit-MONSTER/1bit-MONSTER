---
name: finishing-a-development-branch
description: Use when a branch is ready to land — final self-review, pre-push checks, PR creation, merge-queue submission, post-merge verification and cleanup. Covers this repo's GitHub merge-queue flow (gh-readonly-queue).
whenToUse: Before opening a PR, before adding a PR to the merge queue, and right after a merge lands.
---

# Finishing a Development Branch

## Before opening the PR

1. **Final self-review** (see `code-review`): diff vs stated problem, correctness, no stray files/artifacts, `.gitignore` covers any one-off build dirs (precedent: `build-1715/`).
2. **Pre-push checks** (see `pre-push-checks`): smallest covering build + tests + lint.
3. **PR body discipline**: state the problem (link the issue), the mechanism, and verification evidence with numbers. This repo's good PRs read like `#1723` (diagnosis → fix → verified-on-hardware).
4. If the PR closes an issue, use `Closes #NNNN` so it auto-closes on merge.

## The merge queue flow (this repo)

- Merges happen through GitHub's **merge queue** (branches `gh-readonly-queue/main/pr-<n>-<base>`), not direct pushes to `main`.
- Add the PR with `gh pr merge --auto` (or the queue button). The queue integrates the PR onto the current `main`, re-runs CI on the integration commit, then lands it in order.
- **Watch the integration commit's CI**, not just the PR head's CI: `gh api repos/1bit-MONSTER/1bit-MONSTER/commits/<queue-sha>/check-runs`. The long pole is "C++ (cmake configure + build)".
- If the queue removes the PR (conflict or failed check), fix and re-add — check the timeline for `added_to_merge_queue` / `removed_from_merge_queue` events.

## After the merge

1. Confirm: PR state MERGED, issue auto-closed, `main` advanced (`git fetch origin && git rev-parse origin/main`).
2. Confirm the merged tree equals what you verified: `git diff <verified-sha> origin/main` — empty means the landed code is exactly the validated code.
3. Clean up: delete the local branch (`git branch -d`; `-D` when the merge was squash — the tree check above is the safety argument), update local `main` (`git reset --hard origin/main` only after confirming no local-only commits; if there is one, check it's a duplicate of an upstream commit first).
4. Leave a verification comment on the PR if it adds value (evidence: command, output, numbers).

## If the repo's flow changes

Re-check the actual merge mechanism before assuming the queue flow — CI workflows in `.github/workflows/` and the PR timeline are the source of truth.
