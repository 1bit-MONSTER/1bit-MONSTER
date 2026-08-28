# RULES — lemonade is LOCAL-ONLY (no phone-home to upstream)

Effective: 2026-08-28

## Why

We previously contributed to upstream `lemonade-sdk/lemonade` (PR #3425,
issue #1363, etc.). Maintainers pushed back ("please let us assess whether a
new config is actually needed", "it probably can be closed") and our CI runs
on their repo kept failing (12 jobs, Aug 28). **We no longer interact with
the upstream project at all.** We keep lemonade for ourselves, locally.

## The rule

1. **Never phone home to `lemonade-sdk/lemonade`:**
   - no opening PRs, issues, or comments on the upstream repo (or any fork of it);
   - no `git fetch` / `pull` / `clone` from `https://github.com/lemonade-sdk/lemonade`;
   - no CI runs against their repo / their runners / their artifact servers.
2. **This worktree is local-only.** The branch `chore/lemonade-v11.7.0` has no
   upstream (`origin/chore/lemonade-v11.7.0` is gone) and must not be pushed
   anywhere under the `lemonade-sdk` org. Pushing within our own
   `1bit-MONSTER` org is fine — that is still "us".
3. **No telemetry / update checkers.** Do not add or run scripts that
   download from, check for updates against, or upload usage data to
   upstream lemonade servers.
4. **Vendored copies** under `third_party/lemonade` in our projects
   (1bit-MONSTER, 1bit-MONSTER-pi, 1bit-MONSTER-zaya, 1bit-fused-verify, …)
   are snapshots of our own lemonade tree. Refresh them only from our local
   source — never from upstream.
5. `.github/` in this repo is **our own** CI (1bit-MONSTER). It stays intact.
   It must not be pointed at, or used to run against, lemonade-sdk infra.

## Enforcement

- Never push to `github.com/lemonade-sdk/*` (no remotes, no PRs, no comments).
- If an agent or script is about to touch `github.com/lemonade-sdk`, stop and
  read this file first.
