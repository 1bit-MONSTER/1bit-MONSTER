---
name: pre-push-checks
description: Use before pushing, force-pushing, or marking a branch ready for review — run the smallest relevant local checks that cover the outgoing diff (build the touched targets, run the touched tests, lint the touched files) instead of reflexively running the full suite.
whenToUse: Immediately before every push / PR-ready transition in this repo.
---

# Pre-Push Checks

CI owns exhaustive coverage. The goal here is fast, relevant local evidence for the outgoing diff.

## Inspect the outgoing change

1. `git status --short` + `git diff --stat origin/main...HEAD` — confirm only intended files changed (AGENTS.md: `detect_changes()` equivalent).
2. Identify the affected components: engine (`src/`), kernels (`kernels/*.hip`), loader (`engine/npu/src/`), server (`src/server/`), tests, CMake.

## Smallest covering checks

| Changed surface | Minimum local check |
|---|---|
| Any `.cpp`/`.hip`/`.h` | Build the touched target(s) in `build/` (`cmake --build build --target <t> -j`) — full rebuild only if the change is broad |
| Engine / kernels | The A/B harness for the affected engine (e.g. `zaya_chain_check` on the small model `models/ZAYA1-8B.1bp`) + `ctest` GPU subset |
| Loader / formats | The format gate path (Q8_0 → Q4NX 1BP) and a small-model load |
| Server / REST | The affected handler's test; smoke start/stop |
| CMakeLists.txt | Configure a fresh build dir (`cmake -B build-check`) — cache staleness hides target mistakes |
| Docs only | No build needed; verify links/lint if CI has a docs job |

## Lint & hygiene

- `clang-format` on touched files (CI enforces it; check the repo's `.clang-format`).
- Version consistency: if the change bumps `VERSION`, check the consistency gate in CI.
- No stray artifacts: watch for new build dirs / logs that need `.gitignore` entries (precedent: `build-1715/`).

## When to run the full suite

Only when the diff touches CI itself, CMake structure broadly, or cross-cutting headers included by many TUs. Otherwise trust CI for the rest — and never claim "all checks pass" from a local partial run; say exactly what ran.

## Before pushing to a PR

- The merge queue re-runs CI on the integration commit — push, then verify the queue's CI starts and turns green (check the `gh-readonly-queue` check runs for the PR).
