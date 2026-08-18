---
name: verification-before-completion
description: Use before declaring any fix, port, or feature done — run the original repro and the project's real checks on the real hardware/model, record evidence, and only then report completion. Especially for NPU/GPU engine work where "it compiles" is not "it works".
whenToUse: Every fix or feature completion report; before saying "verified", "done", "works", or "fixed".
---

# Verification Before Completion

A fix is done when the original failure is reproduced-and-eliminated with recorded evidence — not when the code looks right.

## Minimum evidence bar (this repo)

1. **Original repro, end-to-end**: the exact command that failed before must now pass. E.g. `zaya_chain_check models/ZAYA1-74B.1bp` — full load, not a stub.
2. **Numbers recorded**: load time, tok/s, token IDs, exit code — in the report and ideally in the PR (a comment with the command and output is the norm here; e.g. the #1723 verification comment).
3. **Correctness oracle**: tokens/argmax agree with a reference path (A/B masks, legacy path, smaller known model). "Runs without crashing" is not correctness.
4. **No silent fallbacks**: confirm the intended path is engaged (e.g. grep the load log for "falling back to bf16" — zero occurrences means the packed path is active).
5. **Regression check**: the changed area's existing tests pass (`ctest` subset), and the pre-change behavior is unchanged where it must be (e.g. 8B model tokens unchanged when touching the shared engine).

## Hardware honesty

- This box (Strix Halo, iGPU+NPU, 122 GB unified) is the source of truth for GPU/NPU claims. If you could not run on real hardware, say so explicitly — never imply "verified" from a compile.
- Note the environment in evidence: model file size, `free -g` at load, any other heavy jobs running that could skew timing.

## When hardware verification is impossible

- Say "compiled and unit-tested only, NOT hardware-verified" — and mark the claim as unverified in the PR. The repo already has precedent for documenting untested-hardware caveats (#1704).
- CI (GitHub Actions) covers build, lint, smoke, and PPL format gates; that is a floor, not the verification.

## Completion checklist

- [ ] Original repro passes end-to-end
- [ ] Numbers/evidence recorded and attached to the report
- [ ] Reference comparison (tokens/argmax) matches
- [ ] No silent fallback path engaged
- [ ] Existing tests still green; regression stated
- [ ] Hardware/CI limitations stated honestly
