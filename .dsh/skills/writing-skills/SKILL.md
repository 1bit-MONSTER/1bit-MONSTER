---
name: writing-skills
description: Use when authoring, adapting, or auditing skills in this repo's .dsh/skills/ — format requirements, naming, descriptions that trigger correctly, and testing a new skill actually registers.
whenToUse: Creating a new skill, fixing a skill that doesn't load, or auditing the skill set.
---

# Writing Skills

This repo's skills live in `.dsh/skills/<name>/SKILL.md` and are loaded by the harness skill registry (provider rank 100 for project `.dsh/skills`).

## Format (required to register)

- Path: `.dsh/skills/<kebab-case-name>/SKILL.md` — exactly two segments under the root.
- YAML frontmatter with at least:
  - `name`: must match `/^[a-z0-9]+(?:-[a-z0-9]+)*$/` (kebab-case) and equal the directory name.
  - `description`: when to use this skill. Written for the model, first person, actionable: "Use when …" with concrete triggers and example phrasings.
- Optional: `whenToUse` (additional trigger detail).
- Body: Markdown instructions. Frontmatter parse failures, missing `name`/`description`, or invalid names silently drop the skill (warned in logs) — verify after creating.

## Content guidance

- **Repo-specific, not generic**: name the actual files, harnesses, commands, and conventions of this repo (e.g. `zaya_chain_check`, merge queue, `models/ZAYA1-8B.1bp` tokens 25772 70505, `build-1715/` precedent). A skill that could have been written for any repo is a blog post, not a skill.
- **Guidance, not scripts**: state the principle and the judgment call, give the concrete steps that are actually mechanical.
- **Chain skills**: when a skill depends on another (e.g. `finishing-a-development-branch` → `pre-push-checks` → `code-review`), reference them by name.
- **Trigger precision**: the description must fire on the user's phrasing ("simplify X", "is it safe to change X", "before I push") and stay quiet otherwise.
- Keep one skill per concern. If a skill would have two "when to use" stories, split it.

## Testing a new skill

1. Create the file; the registry picks up project-root changes (catalog updates appear in-session).
2. Verify the skill appears in the available-skills catalog with the intended description.
3. Optionally invoke it once (`skill` tool) to confirm the body loads cleanly.
4. If it does not appear: check name regex, directory layout (`.dsh/skills/<name>/SKILL.md`), frontmatter validity.

## Auditing the set

- Every skill earns its place: if a skill has never fired, either its triggers are too narrow (fix the description) or it's redundant (remove it).
- Keep the set small enough that the model can actually hold the catalog in context — prefer 8-15 sharp skills over 40 overlapping ones.
- Skills are versionable (`.dsh/` is not gitignored): review them like code — changes to triggers or steps affect future behavior.
