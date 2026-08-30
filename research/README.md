# research/ — Research-to-Implementation Workspace

This directory turns the 65-paper research archive (`~/research-papers/`) into workstreams against the 1bit-monster codebase.

```
research/
├── PLAN.md          ← THE PLAN: phases, workstreams, tasks, dependencies, metrics
├── IDEAS.md         ← parking lot for adopted-adjacent ideas
├── TRACKING.md      ← status tables (single source of truth for progress)
├── new-workstream.sh ← scaffold a new workstream from the template
├── templates/workstream/
└── ws00..ws12/      ← one folder per workstream (README = goal/tasks/validation, FINDINGS = results)

⚠ THIS DIRECTORY WAS WIPED ONCE (2026-07-31) BY AN UNKNOWN PROCESS and recreated.
  It is now git-tracked. Mirror: ~/research-papers/backup/ — keep it in sync.
```

## How to use

1. Read `PLAN.md` first — it maps papers → workstreams → tasks with dependencies.
2. Phase 0 items (P0.1-P0.5) are the floor. They're engineering, not research.
3. Each workstream folder has a `README.md` (goal/papers/tasks/validation) and `FINDINGS.md` (results of completed probes).
4. Update `TRACKING.md` when a task moves. Every completed task must have a validation number with an honesty tag.
5. New idea without a home? → `IDEAS.md`. New workstream? → `./new-workstream.sh wsNN-name`.

## Paper access

All PDFs: `~/research-papers/<Name>-<arXivID>.pdf` (65 papers). Deep reads: `SYNTHESIS.md`, `RESEARCH-BRIEF-2026-07-31.md`. BibTeX: `bibliography.bib`.

## Conventions

- **Honesty tags everywhere**: every number carries `validated/optimized/broken/corrected`.
- **No task is done without a benchmark line.**
- **P0.2 (one router) and P0.3 (40-column decision)** are decisions with code attached — make the call in writing first.
- Respect `AGENTS.md` (GitNexus): impact analysis before editing engine symbols; `detect_changes()` before committing.
