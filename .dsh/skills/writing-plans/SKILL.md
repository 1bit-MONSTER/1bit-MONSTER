---
name: writing-plans
description: Use before starting any non-trivial implementation — extract the actual goal through questions, write a short spec with completion criteria, then an implementation plan decomposed into verifiable units. Do not jump to code first.
whenToUse: Any feature, port, refactor, or multi-step fix; anything that will take more than one focused edit.
---

# Writing Plans (Spec-First)

Superpowers methodology: step back, tease out the spec, get sign-off, then plan — before writing code.

## Step 1 — Extract the real goal

Ask what the user is actually trying to achieve. Distinguish:
- **Behavior** (what must change for the user) vs **mechanism** (how it's built) — write the behavior.
- In scope / out of scope (explicitly list cuts; this repo has a strong "everything else was cut" culture — see ROADMAP).
- Constraints: hardware targets (NPU/GPU/CPU), model formats (1BP canonical; GGUF/ONNX/Q4NX/H1B importers), performance budgets (tok/s, VRAM), compat (8B reference tokens must not change when touching shared engine code).

## Step 2 — Spec with completion criteria

A spec is a few paragraphs + a checklist, each item **observable and verifiable**:

- "Loads ZAYA1-74B.1bp: all 120 layers, < N minutes, no bf16 fallbacks" (not "fixes loading")
- "Tokens for ZAYA1-8B.1bp unchanged: 25772 70505"
- "ctest <names> pass; CI gates green"

Write the completion criteria before any code exists. The user signs off on these.

## Step 3 — Implementation plan

- Decompose into **agent-sized units**: each independently verifiable, one dominant risk, a clear done condition (agentic-engineering 15-minute unit rule).
- Order: risky/unknown first (format questions, kernel feasibility) before polish.
- Note which units can run in parallel (independent files/tests) vs sequential (dependency chains).
- Prefer the smallest diff that meets the criteria (YAGNI). If the plan needs new files, name them; if it needs model files on the box, check they exist first.

## Step 4 — Execute against the plan

- One unit at a time, verify each done condition (see `verification-before-completion`).
- If reality diverges from the plan (measurement shows a different bottleneck), stop and revise the plan — do not silently drift.
- Record deviations and results in the final report.
