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

Four properties are checked. Each is also checked against a mutated copy of the
file, because a check that cannot fail is not a check:

  1. `edited` is among the trigger types.
  2. exactly one `gh pr merge` invocation, and it names a method explicitly.
  3. the step reads `autoMergeRequest` back after enabling, and can fail on null.
  4. a non-default base is reported as a warning that names it, rather than either
     passing silently or being confused with the default-branch failure.

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
    return [l.strip() for l in code(text).splitlines() if re.search(r"\bgh pr merge\b", l)]


def methods_named(text: str) -> bool:
    invs = merge_invocations(text)
    return bool(invs) and all(re.search(r"--(squash|merge|rebase)\b", i) for i in invs)


def has_outcome_readback(text: str) -> bool:
    body = code(text)
    return bool(re.search(r"autoMergeRequest", body) and re.search(r"::error::", body))


def names_stacked_base(text: str) -> bool:
    body = code(text)
    return bool(re.search(r"default_branch", body) and re.search(r"::warning::", body))


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

    check(has_outcome_readback(text), "the step reads autoMergeRequest back and fails on null")
    check(names_stacked_base(text), "a non-default base is reported as a warning naming it")

    # ---- controls: each matcher must be able to fail ---------------------------
    mutations = [
        ("edited dropped from the trigger list",
         re.sub(r"types:\s*\[[^\]]*\]", "types: [opened, synchronize]", text, count=1),
         lambda t: "edited" in trigger_types(t)),
        ("method flag dropped from the invocation",
         re.sub(r'(?m)^(\s*)gh pr merge "\$PR" --squash --auto',
                r'\1gh pr merge "$PR" --auto', text),
         methods_named),
        ("autoMergeRequest readback deleted",
         re.sub(r'(?m)^\s*if \[ "\$\(gh pr view "\$PR".*autoMergeRequest.*$\n', "", text, count=1),
         has_outcome_readback),
        ("stacked-base warning deleted",
         re.sub(r'(?m)^\s*echo "::warning::.*$\n', "", text, count=1),
         names_stacked_base),
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
