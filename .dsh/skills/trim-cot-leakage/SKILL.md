---
name: trim-cot-leakage
description: Use when auditing or fixing prose that reads like a leaked reasoning transcript — session-vantage narration, dead citations, change stories ("used to", "no longer", "this fix"), reviewer-addressed justifications, or planning residue in comments, docs, or PR text.
whenToUse: Cleaning up comments/docs, reviewing PR prose, or when text reads like an agent's stream of thought rather than repo state.
---

# Trimming Chain-of-Thought Leakage

Chain-of-thought leakage is prose whose vantage is the authoring session rather than the repository: it cites artifacts only that session could see, narrates the change instead of the state, or argues with a reviewer who has left.

## The one test

For every suspect passage ask: **could a reader at HEAD, with no access to any session transcript, PR thread, or uncommitted draft, resolve every reference and verify every claim?**

- **Yes** → not leakage (however historical it sounds).
- **No** → restate the surviving facts from the repository's vantage, then delete the rest.

## Common classes (this repo)

1. **Dead citations**: "(decision N)", audit item codes, "§N of the draft", issue numbers with no linkable content at HEAD. Keep the issue number when it names the durable rationale (e.g. `issue #1715`), delete the ephemeral reference.
2. **Change narration**: "used to", "no longer", "this cut", "was previously lost uncommitted work". Restate the current state; move the story to CHANGELOG/PR body if it matters.
3. **Reviewer-addressed justifications**: "the reviewer asked why…", "rejected in review…", "this is intentional because…". If the constraint is real, state it as a fact at HEAD; delete the argument transcript.
4. **Control-flow narration**: "first we load, then we upload" walking the code line by line. Delete; the code shows the flow.
5. **Planning residue**: "TODO: decide later", hedged claims ("should work", "probably fine") where verification is possible. Verify or remove; a hard "NOT hardware-verified" is honest, a hedge is leakage.

## The fix is never deletion alone

If a passage carries factual clauses, restate each so it stands at HEAD, then delete the transcript around it. A passage carrying none (an audit code, control-flow narration) is deleted outright.

## After trimming

- The comment should read as if written by someone who knows the final state and nothing else.
- If the trimmed fact belongs in the PR body or CHANGELOG, move it there — don't drop durable information.
