---
name: prose-standard
description: Use when writing or reviewing comments, docs, PR descriptions, READMEs, or visible strings in this repo — preserve contracts and invariants in prose, remove decoration and repetition, keep comments at the level of non-obvious rationale.
whenToUse: Any prose in code, docs, PRs, or issues; "improve the comments", "clean up docs", "this comment is confusing".
---

# Prose Standard

Write enough to preserve the contract, then remove reasoning transcripts, repetition, and decoration. A contract is an obligation, invariant, precondition, postcondition, or compatibility promise that callers, kernels, loaders, or users rely on.

## Comments describe what code cannot express

- Non-obvious contracts and rationale: why a kernel is split-K, why a copy is chunked at 1 GiB (signed 32-bit byte count), why a fallback exists, why an orientation is accepted both ways.
- They do **not** restate the code: "// multiply by scale" next to `x *= scale` is decoration.
- Keep the why at the site that would otherwise confuse; link to the issue number for the full story (`issue #1715` style), do not paste the issue into the comment.

## Required coverage

Every public-ish surface gets enough prose to be used without reading the implementation:
- New kernels: header comment with orientation (row-major vs transposed), dims, and which engine path launches it (see `kernels/zaya_gemv_bf16.hip` as the house style).
- New format/loader fields: layout, byte sizes, endianness, expert packing.
- New env vars / config: default, units, effect — in docs, not only in code.
- PR bodies: problem → mechanism → evidence (see `finishing-a-development-branch`).

## Editorial judgment

- One home per fact: if `AGENTS.md`, `ROADMAP.md`, or a docs page owns it, link — don't duplicate.
- Historical narration ("used to", "now", "was previously") belongs in CHANGELOG/issues, not in comments — comments describe the state at HEAD.
- If a passage reads like a session transcript or review argument, apply `trim-cot-leakage`.
- Length is not the defect: a 10-line rationale for a subtle kernel is right; a 10-line walkthrough of a loop is wrong.

## Docs placement

- User-facing behavior → `docs/` (this repo keeps engine docs and research notes there).
- Process/decision rationale → issue/PR/CHANGELOG, not code.
- When in doubt, ask where the reader will look, and put it there once.
