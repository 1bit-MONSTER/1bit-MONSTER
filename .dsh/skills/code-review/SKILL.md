---
name: code-review
description: Use when reviewing a pull request or self-reviewing a branch before pushing — verify the diff against the repo's standards (AGENTS.md, CI gates, merge-queue rules), check correctness/lifecycle/security over style, and substantiate blockers.
whenToUse: Every PR review in this repo, and every self-review before marking a branch ready or adding it to the merge queue.
---

# Code Review

**This skill is guidance, not a complete checklist.** For GitHub PRs, fetch the live base and exact head, review the actual diff (not the description), and read enough surrounding code to understand the design.

## Sources of truth

- `AGENTS.md` — standing repo rules (GitNexus impact analysis before edits, detect_changes before commit).
- `CI` workflows in `.github/workflows/` — the gates: C++ build, clang-format lint, Version consistency, Metal backend, Inference smoke test, PPL format gate (Q8_0 → Q4NX 1BP), Scope Guard, CodeQL.
- `CHANGELOG.md` — dated bulk sections; a PR does not normally edit it (do not demand per-PR changelog entries).
- `AUDIT_ISSUES.md`, `TODO_TRACKING.md` — known-issue context; stale TODOs are routinely resolved, check before flagging.

## Review order

1. **Intent**: does the PR's diff actually match its stated problem? Extra unrelated changes are a blocker-worthy finding.
2. **Correctness first**: kernel indexing, buffer sizes (watch signed 32-bit `hipMemcpyAsync` byte counts > 2 GiB), quantization layout/orientation, expert indexing, stream/async error surfacing, memory lifetime (hipMalloc/Free pairing), OOB writes.
3. **Lifecycle & cleanup**: allocations freed on failure paths; no dangling `*_q` pointers when a fallback path is taken; partial-load abort leaves no half-initialized state used later.
4. **Security**: no new trust boundaries (paths from model files, server handlers, tool call parsing); bounds checks on untrusted input (GGUF/1bp headers, tokenizer input).
5. **Style last**: clang-format is CI-enforced; comment on style only where it signals a real problem.
6. **Verification claims**: if the PR claims hardware verification, check the claim is plausible (numbers, method). Flag "compiles" presented as "verified".

## Reporting

- A short review with one substantiated blocker beats a list of nits.
- State required fixes as blockers; everything else as suggestions.
- For this repo's flow: after review passes, the PR goes to the GitHub merge queue (`gh pr merge --auto` / queue) — confirm the merge state is CLEAN and CI is green before approving.
