# 1bit-MONSTER Agent Skills

Skills loaded by the harness skill registry from `<projectRoot>/.dsh/skills/<name>/SKILL.md`.
Format requirements, naming rules, and audit guidance: see [`writing-skills/SKILL.md`](writing-skills/SKILL.md).

## The set

**Correctness & process**
- `impact-analysis` — blast radius before editing any symbol (AGENTS.md mandate)
- `test-driven-development` — red-green-refactor with ctest/HIP harnesses
- `systematic-debugging` — 4-phase root cause (OOM-at-layer-N, stale HIP errors, >2 GiB copies)
- `verification-before-completion` — prove fixes on real hardware with recorded evidence
- `code-review` — PR/self-review against this repo's standards and CI gates

**Planning & execution**
- `writing-plans` — spec-first with observable completion criteria
- `subagent-driven-development` — fanned-out execution with two-stage review
- `find-simplifications` — evidence-backed YAGNI/dead-code reduction
- `pre-push-checks` — smallest covering local checks before pushing

**Landing & hygiene**
- `finishing-a-development-branch` — PR + GitHub merge-queue flow, post-merge cleanup
- `prose-standard` — contract-preserving comments/docs
- `trim-cot-leakage` — remove session-vantage narration from prose
- `writing-skills` — author/audit skills themselves

## Notes

- Adapted from the deepseek-harness MIT-licensed `dsh-*` skills, obra/superpowers methodology,
  and this repo's own conventions (AGENTS.md, merge queue, hardware verification norms).
- Skills are versionable: `.dsh/` is not gitignored — review changes like code.
