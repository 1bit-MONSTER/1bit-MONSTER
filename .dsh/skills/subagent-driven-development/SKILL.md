---
name: subagent-driven-development
description: Use for multi-step implementation work — fan tasks out to subagents from a written plan, then review each result twice (spec compliance first, then code quality) before integrating. Also use when asked to run a task with subagents or "SDD".
whenToUse: Multi-unit implementations where units are independently verifiable; batch execution with checkpoints.
---

# Subagent-Driven Development (SDD)

## Prerequisites

A written plan with verifiable units (see `writing-plans`). SDD without completion criteria just parallelizes chaos.

## Dispatch

- Give each subagent a **complete, self-contained prompt**: it cannot see this conversation. Include: files to touch (exact paths), the unit's done condition, the repo conventions it must respect (AGENTS.md: impact analysis before edits), how to verify (which tests/harness), and what to return (a structured report with evidence).
- Run independent units concurrently (background subagents); sequence dependent ones.
- Do not duplicate work: track which unit each subagent owns.

## Two-stage review (before integrating any result)

1. **Spec compliance**: does the result meet the unit's done condition? Run its verification yourself or inspect its evidence — never take "done" on faith.
2. **Code quality**: correctness (see `code-review` order), minimal diff, no regressions in the shared paths, tests added where behavior changed.

Fail the unit back to its subagent with the specific gap, or fix it yourself if the gap is small — do not accumulate unreviewed agent output.

## Integration

- Integrate units in plan order, keeping the tree green after each integration (build the touched target, run the touched tests).
- Only after all units pass review do you run the broader checks (see `pre-push-checks`).
- Final report: units, evidence per unit, deviations from plan, verification results.

## Guardrails

- Never let a subagent commit to `main` or push without your review (PR/merge-queue flow).
- Never let a subagent skip verification because the model file is big — small fixtures exist or can be made.
- If a subagent reports a blocker, verify it yourself before changing the plan.
