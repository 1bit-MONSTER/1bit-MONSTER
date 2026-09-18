#!/usr/bin/env python3
"""auto_enqueue_selfcheck.py — the auto-enqueue workflow must not report success without enabling auto-merge.

Why this exists, measured 2026-09-18:

  * `gh pr merge "$PR" --auto`, with no method, exits 1 when the PR's base is not
    the default branch: gh cannot derive a merge method from a ruleset that does
    not apply to that base, and a workflow cannot answer its prompt. The repo's
    first stacked PR (#2534) is the single failure in this workflow's history,
    for exactly that reason.
  * An exit code is not evidence. The step's own comment says a silent enqueue
    failure is the failure mode this workflow exists to prevent, so the outcome
    (`autoMergeRequest` non-null afterwards) is read back rather than assumed.
  * A retarget arrives as `pull_request: edited`. Without that trigger, a stacked
    PR whose base merges becomes a main-based PR that nothing ever enqueues.

Seven properties are checked. Each is also checked against a mutated copy of the
file, because a check that cannot fail is not a check. Each mutation removes the
signal its matcher claims to test, located BY CONTENT: the first version deleted
the first `::warning::` line and relied on it being the stacked-base one, so the
guard added in the conflict-skip change (a second warning, earlier in the file)
silently moved the mutation onto the wrong line and the control stopped
controlling anything - it passed while deleting a warning the matcher never
looked at.

  1. `edited` is among the trigger types.
  2. exactly one `gh pr merge` invocation, and it names a method explicitly.
  3. the step reads the outcome back after enabling - `autoMergeRequest` OR a
     `mergeQueueEntry`, because GitHub clears the request once the PR is queued
     (measured on #2529: `mergeQueueEntry { state QUEUED, position 31 }` with
     `autoMergeRequest: null`) - and can fail when neither is present.
  4. a non-default base is reported as a warning that names it, rather than either
     passing silently or being confused with the default-branch failure.
  5. the `gh pr merge` call cannot ABORT the step, because the readback in 3 and
     the warning in 4 come after it and the step runs under `set -e`. A guard that
     is present but unreachable passes every check that only asks whether its text
     is there - which is what 4 did until #2548's run 35297583998 died at the
     merge call with the warning never printed.
  6. a PR whose head conflicts with the default branch is skipped rather than
     enqueued: the queue builds an entry before it can discover the conflict, and
     under grouping_strategy ALLGREEN one ejected entry costs every PR it was
     grouped with a rebuild cycle.
  7. the non-default-base decision in 4 is made BEFORE the merge call, so a stacked
     PR never reaches that call at all. Its behaviour there is not deterministic -
     on the same PR it exited 1 once and exited 0 having enabled nothing the next
     time - and a call that sometimes silently works would auto-merge a stacked PR
     into its base FEATURE branch, with the warning below it never reached to say
     so.

Comments are stripped before matching: a `gh pr merge` spelled out in the step's
own explanation is not an invocation. That is the same mistake
Testing/npu_build_script_selfcheck.sh was fixed for.

Run: python3 Testing/auto_enqueue_selfcheck.py [--workflow PATH]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_WORKFLOW = REPO / ".github" / "workflows" / "auto-enqueue.yml"

failures: list[str] = []
checks = 0


def check(ok: bool, label: str, detail: str = "") -> None:
    global checks
    checks += 1
    if not ok:
        failures.append(f"{label}{(': ' + detail) if detail else ''}")


def code(text: str) -> str:
    """The workflow without comments. In YAML a `#` opens a comment at line start
    or after whitespace; stripping them keeps prose from satisfying these checks."""
    return "\n".join(re.sub(r"(^|\s)#.*$", "", l) for l in text.splitlines())


def trigger_types(text: str) -> list[str]:
    m = re.search(r"types:\s*\[([^\]]*)\]", code(text))
    return [t.strip() for t in m.group(1).split(",")] if m else []


def merge_invocations(text: str) -> list[str]:
    """Lines that INVOKE it, not lines that name it in output. Stripping comments
    is not enough: the step also narrates the command in an `echo`, and counting
    that as an invocation made this matcher report two of them the moment a notice
    quoted the call it was reporting on."""
    return [l.strip() for l in code(text).splitlines()
            if re.search(r"\bgh pr merge\b", l) and not re.match(r"\s*echo\b", l)]


def methods_named(text: str) -> bool:
    invs = merge_invocations(text)
    return bool(invs) and all(re.search(r"--(squash|merge|rebase)\b", i) for i in invs)


def has_outcome_readback(text: str) -> bool:
    """Both signals must be read AND combined in the decision. Reading only
    autoMergeRequest fails the PRs this job just enqueued: the request is cleared
    when the queue entry is created, so the query has to ask for mergeQueueEntry
    and the filter has to accept either one. (The first version of this matcher
    only looked for the two words, so dropping the signal from the filter did not
    trip its own control.)"""
    body = code(text)
    return bool(
        re.search(r"mergeQueueEntry\{state\}", body)
        and re.search(r"mergeQueueEntry != null", body)
        and re.search(r"autoMergeRequest != null", body)
        and re.search(r"::error::", body)
    )


def names_stacked_base(text: str) -> bool:
    """The warning, on the line that emits it, names the non-default base.

    It is not enough for `default_branch` and `::warning::` to appear anywhere in
    the file: the `if [ "$base" != ... default_branch ]` test above it supplies
    the first token, and any other warning supplies the second, so the guard this
    claims to check could be deleted and the matcher would still pass."""
    body = code(text)
    return bool(re.search(r'(?m)^\s*echo "::warning::[^"]*not the default branch', body))


def refuses_conflicts(text: str) -> bool:
    """The conflict skip: read `mergeable`, act on CONFLICTING, and say which
    conflict in a warning. Reading mergeability and then enqueuing anyway would
    satisfy a matcher that only looked for the word."""
    body = code(text)
    return bool(
        re.search(r"--json mergeable", body)
        and re.search(r'"CONFLICTING"', body)
        and re.search(r'(?m)^\s*echo "::warning::[^"]*conflicts with', body)
    )


def merge_failure_is_tolerated(text: str) -> bool:
    """The `gh pr merge` call must not be able to abort the step.

    Everything that interprets its result - the readback and the stacked-base
    warning - comes after it, and the step runs under `set -euo pipefail`. On a
    base whose ruleset does not apply the call exits 1, so a bare invocation kills
    the step at that line: the warning is present in the file and never printed.
    Asking only whether the warning's text is there is what let that through
    (measured on #2548's run 35297583998)."""
    invs = merge_invocations(text)
    return bool(invs) and all(
        re.match(r"^\s*if\s+!", l) or re.search(r"\|\|\s*(?:true|:)\s*$", l) for l in invs
    )


def stacked_base_is_decided_first(text: str) -> bool:
    """The non-default-base branch must come BEFORE the merge call.

    Deciding after the call means the call runs on a stacked PR, and its behaviour
    there is not deterministic: `gh pr merge --squash --auto` on #2548 exited 1 with
    "Protected branch rules not configured for this branch" in run 35297583998 and
    exited 0 having enabled nothing in run 35297895113. If it ever does stick, the
    PR is auto-merged into its base FEATURE branch and this warning - the one that
    is supposed to describe exactly that - is unreachable."""
    body = code(text).splitlines()
    warn = next((i for i, l in enumerate(body)
                 if re.search(r'^\s*echo "::warning::[^"]*not the default branch', l)), None)
    merge = next((i for i, l in enumerate(body)
                  if re.search(r"\bgh pr merge\b", l) and not re.match(r"\s*echo\b", l)), None)
    return warn is not None and merge is not None and warn < merge


def warn_after_enqueue(text: str) -> str:
    """Mutation: the same warning, moved after the merge call."""
    keep = [l for l in text.split("\n")
            if not re.search(r'^\s*echo "::warning::[^"]*not the default branch', l)]
    return "\n".join(keep + ['          echo "::warning::$base is not the default branch"'])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--workflow", default=None, help=f"default: {DEFAULT_WORKFLOW}")
    args = ap.parse_args(argv)

    path = Path(args.workflow) if args.workflow else DEFAULT_WORKFLOW
    if not path.is_file():
        print(f"FAIL: no {path}", file=sys.stderr)
        return 1
    text = path.read_text(encoding="utf-8")

    types = trigger_types(text)
    check("edited" in types, "trigger list re-runs on a retarget (edited)", f"types = {types}")

    invocations = merge_invocations(text)
    check(len(invocations) == 1, "exactly one gh pr merge invocation", f"found {len(invocations)}")
    check(methods_named(text), "the invocation names a merge method explicitly",
          "; ".join(invocations))

    check(has_outcome_readback(text),
          "the step reads the outcome back (auto-merge request OR queue entry) and can fail")
    check(names_stacked_base(text), "a non-default base is reported as a warning naming it")
    check(merge_failure_is_tolerated(text),
          "a failing `gh pr merge` cannot abort the step before the readback")
    check(refuses_conflicts(text), "a conflicting PR is skipped rather than enqueued")
    check(stacked_base_is_decided_first(text),
          "a non-default base is decided before the merge call, not after it")

    # ---- controls: each matcher must be able to fail ---------------------------
    mutations = [
        ("edited dropped from the trigger list",
         re.sub(r"types:\s*\[[^\]]*\]", "types: [opened, synchronize]", text, count=1),
         lambda t: "edited" in trigger_types(t)),
        ("method flag dropped from the invocation",
         text.replace("--squash --auto", "--auto"),
         methods_named),
        ("merge-queue-entry signal dropped from the readback",
         text.replace("((.mergeQueueEntry != null) or (.autoMergeRequest != null))",
                      "(.autoMergeRequest != null)"),
         has_outcome_readback),
        ("stacked-base warning deleted",
         re.sub(r'(?m)^\s*echo "::warning::[^\n]*not the default branch[^\n]*$\n', "", text),
         names_stacked_base),
        ("merge call allowed to abort the step",
         re.sub(r'(?m)^(\s*)if ! (gh pr merge\b)', r"\1if \2", text),
         merge_failure_is_tolerated),
        ("conflict warning deleted",
         re.sub(r'(?m)^\s*echo "::warning::[^\n]*conflicts with[^\n]*$\n', "", text),
         refuses_conflicts),
        ("mergeable read dropped from the conflict guard",
         re.sub(r'--json mergeable', "--json state", text),
         refuses_conflicts),
        ("stacked-base warning moved after the enqueue",
         warn_after_enqueue(text),
         stacked_base_is_decided_first),
    ]
    for label, mutated, probe in mutations:
        check(mutated != text, f"control setup: '{label}' changes the file")
        check(not probe(mutated), f"control: '{label}' is caught")

    if failures:
        for m in failures:
            print(f"  FAIL {m}")
        print(f"\n{len(failures)} FAILURE(S) — the auto-enqueue contract is not met")
        return 1
    print(f"auto-enqueue contract ok ({checks} checks, {len(mutations)} with negative controls)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
