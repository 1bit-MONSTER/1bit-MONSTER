---
name: find-simplifications
description: Use when asked to find simplifications, reduce complexity, remove dead/duplicated code, or audit YAGNI violations in this repo. Turn findings into evidence-backed proposals (or TODOs), not guesses.
whenToUse: Requests to simplify, cut dead code, reduce complexity, or clean up — e.g. "simplify X", "what can we cut", "dead code", "reduce complexity".
---

# Finding Simplifications

Goal: remove or collapse real surface with evidence that the current cost exceeds the benefit. Prefer a few well-proven candidates over a pile of thin guesses.

## Start with repo context

- `ROADMAP.md` "What was cut" section — this repo deliberately guts scope; a proposal that re-adds cut surface (voice cloning, SaaS, agent stack) is rejected by design, not by accident.
- `TODO_TRACKING.md` / `AUDIT_ISSUES.md` — many entries are already resolved; check before re-proposing.
- Read the owning code before judging anything: files under `engine/`, `kernels/`, `src/` have intentional cross-backend (NPU/GPU/CPU) seams; simplifications that fight those need extra evidence.

## What counts as a strong candidate

- A function, kernel, config knob, env var, or harness with no production caller (check with grep across `src/ kernels/ engine/ tools/ tests/`; tests-only consumers need justification).
- Two code paths that mirror the same computation — e.g. legacy scalar GEMV kernels kept only for A/B comparison: keep while the A/B harness is active, propose removal only when the new path is proven on all models.
- Dead fallback branches that can no longer trigger (e.g. "falling back to bf16" paths for tensors that always exist in the current format).
- Duplicated loader/orientation logic across `onebp_loader.cpp` and the GGUF importers.
- Hand-rolled code where a vendored dependency already provides it — but check `third_party/` licensing and build weight first.
- Build/config surface: unused CMake options, dead `Makefile*`/scripts.

## What is NOT a candidate (by design)

- The dual-backend/twin paths (NPU vs GPU vs CPU dispatch) — intentional, protected.
- The 1BP canonical format and its importers — core roadmap.
- A/B comparison harnesses while correctness work is ongoing.
- Anything whose removal would silently change model output (tokens are the oracle: 8B reference tokens must stay stable).

## Deliverable

- A short ranked list: candidate → evidence (grep results, caller counts) → proposed action → risk.
- For each accepted candidate: implement in a minimal diff with its own verification, or file a TODO with the evidence. Do not batch-remove speculative candidates.
