#!/usr/bin/env python3
"""census_autopr_selfcheck.py — the alias autopr must base its branch on main, not on the run's HEAD.

Why (#2498): `_open_draft_pr` built its branch with `git switch -C <branch>` from
whatever HEAD the run happened to have. In a checkout 51 commits behind main that
produced an **18-file draft PR for a one-line alias**, of which 16 were byte-identical
to main and appeared only because the merge base was old:

    census/auto-map-picolm     18 files changed, 16 byte-identical to main
    census/auto-map-language   18 files changed, 16 byte-identical to main

Rebasing after the commit does not fix it — the stale base's own commits are replayed
and they do conflict with main (`git rebase origin/main` on census/auto-map-picolm stops
on engine/npu/xclbins/PROVENANCE.json). So the branch has to be cut from origin/main
*before* the alias is applied, which is what this checks: fetch `origin main`, switch to
`origin/main`, and never rebase.

The rule landed on main as #2499, written slightly differently from the branch this guard
was first written against. That is why the matchers below search an invocation's
arguments instead of matching exact argv prefixes: the first version asserted
`calls[i][4] == "origin/main"`, and main's `["git", "switch", "--quiet", "-C", branch,
"origin/main"]` put the start-point one slot later, so all three order assertions failed
against correct code. A guard that recognises only one spelling of the fix cries wolf —
and would have gone red on the change it exists to protect.

Run: python3 Testing/census_autopr_selfcheck.py [--tool PATH]
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TOOL = REPO / "Testing" / "census_autopr.py"

failures: list[str] = []
checks = 0


def check(ok: bool, label: str, detail: str = "") -> None:
    global checks
    checks += 1
    if not ok:
        failures.append(f"{label}{(': ' + detail) if detail else ''}")


class _Result:
    def __init__(self, out: str = "") -> None:
        self.returncode, self.stdout, self.stderr = 0, out, ""


def record_sequence(tool: Path) -> list[list[str]]:
    """Drive _open_draft_pr with subprocess.run recorded; return the commands issued."""
    calls: list[list[str]] = []

    def fake_run(cmd, **kw):
        calls.append(list(cmd))
        if "symbolic-ref" in cmd:
            return _Result("fix/some-branch\n")
        if "rev-parse" in cmd:
            return _Result("deadbeef\n")
        if cmd[:2] == ["gh", "pr"]:
            return _Result("https://github.com/1bit-MONSTER/1bit-MONSTER/pull/9999\n")
        return _Result("")

    spec = importlib.util.spec_from_file_location("census_autopr", tool)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    mod.subprocess.run = fake_run
    mod._apply_alias = lambda arch, target: True          # do not touch the tree
    url = mod._open_draft_pr("selfcheckarch", "RCPP_ARCH_LLAMA", ["org/model"])
    check(bool(url and url.startswith("https://github.com/")),
          "the tool still returns the PR url", str(url))
    return calls


def _find(calls: list[list[str]], *needles: str, prog: str = "git") -> int | None:
    """Index of the first `<prog>` invocation whose arguments contain every needle.

    Flags are ignored on purpose, and the program is a parameter because the sequence
    has two of them: everything here is `git` except the final `gh pr create`, and a
    matcher that hard-coded `git` silently returned None for that one — which failed
    "pr create last" against correct code.
    """
    for i, c in enumerate(calls):
        if prog and c[:1] != [prog]:
            continue
        if all(n in c for n in needles):
            return i
    return None


def _switch_startpoint(calls: list[list[str]], i: int | None) -> str | None:
    """Start-point of `git switch -C <branch> [<start>]`: the argument after the branch."""
    if i is None:
        return None
    argv = calls[i]
    j = argv.index("-C")
    return argv[j + 2] if len(argv) > j + 2 else None


def verdict(calls: list[list[str]]) -> dict[str, bool]:
    i_fetch, i_sw = _find(calls, "fetch", "origin", "main"), _find(calls, "switch", "-C")
    i_add, i_com = _find(calls, "add"), _find(calls, "commit")
    i_push, i_pr = _find(calls, "push"), _find(calls, "pr", "create", prog="gh")
    return {
        "fetch origin main before the switch":
            i_fetch is not None and i_sw is not None and i_fetch < i_sw,
        "branch cut from origin/main": _switch_startpoint(calls, i_sw) == "origin/main",
        "the alias commit follows the switch":
            i_add is not None and i_sw is not None and i_add > i_sw,
        "commit, then push": i_com is not None and i_push is not None and i_com < i_push,
        "pr create last": i_pr is not None and i_push is not None and i_push < i_pr,
        "no rebase of a stale base": _find(calls, "rebase") is None,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tool", default=None, help=f"autopr to check (default: {TOOL})")
    args = ap.parse_args(argv)
    tool = Path(args.tool) if args.tool else TOOL
    if not tool.is_file():
        print(f"FAIL: no {tool}", file=sys.stderr)
        return 1

    calls = record_sequence(tool)
    got = verdict(calls)
    for label, ok in got.items():
        check(ok, label)
    if not all(got.values()):
        print("  recorded order:",
              [c[1] if c[:1] == ["git"] else " ".join(c[:2]) for c in calls])

    # ---- controls: each matcher must be able to fail ---------------------------
    def drop(*needles):
        return [c for c in calls
                if not (c[:1] == ["git"] and all(n in c[1:] for n in needles))]

    def head_instead():
        return [["HEAD" if a == "origin/main" else a for a in c] for c in calls]

    controls = [
        ("no fetch at all",
         verdict(drop("fetch", "origin", "main"))["fetch origin main before the switch"]),
        ("branch cut from the run's HEAD",
         verdict(head_instead())["branch cut from origin/main"]),
        ("a rebase replayed the stale base",
         verdict(calls + [["git", "rebase", "origin/main"]])["no rebase of a stale base"]),
    ]
    for label, result in controls:
        check(not result, f"control: '{label}' is caught")

    if failures:
        for f in failures:
            print(f"  FAIL {f}")
        print(f"\n{len(failures)} FAILURE(S) — the alias autopr no longer bases on main")
        return 1
    print(f"census autopr branch base ok ({checks} checks, {len(controls)} negative controls)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
