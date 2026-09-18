#!/usr/bin/env python3
"""census_autopr_selfcheck.py — the alias autopr must base its branch on main, not on the run's HEAD.

Why (#2498): `_open_draft_pr` built its branch with `git switch -C <branch>` from whatever
HEAD the run happened to have. In a checkout 51 commits behind main that produced an
**18-file draft PR for a one-line alias**, of which 16-17 files were byte-identical to
main and appeared only because the merge base was old. That is not a correctness bug in
the tree, but it is a review bug: the line that needs judgement is buried, and the PR
reads as if it rewrites NPU artifacts and PROVENANCE.json. Measured on the real branches:

    census/auto-map-picolm    18 files changed, 16 byte-identical to main
    census/auto-map-language  18 files changed, 16 byte-identical to main

Rebasing after the commit does not fix it either - the stale base's own commits are
replayed and they DO conflict with main (measured: `git rebase origin/main` on
census/auto-map-picolm stops on engine/npu/xclbins/PROVENANCE.json). So the branch has to
be cut from origin/main *before* the alias is applied, which is what this checks: the
command sequence must fetch, switch to `origin/main`, and never rebase.

The check drives the real function with `subprocess.run` recorded, so it asserts the
production code path rather than a paraphrase of it, and each assertion is paired with a
mutation of the recorded sequence that must fail it.

Run: python3 Testing/census_autopr_selfcheck.py
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
    """Run _open_draft_pr with subprocess.run recorded; return the commands it issued."""
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
    check(bool(url and url.startswith("https://github.com/")), "the tool still returns the PR url", str(url))
    return calls


def verdict(calls: list[list[str]]) -> dict[str, bool]:
    def idx(*prefix):
        return next((i for i, c in enumerate(calls) if c[:len(prefix)] == list(prefix)), None)
    i_fetch, i_sw = idx("git", "fetch", "origin"), idx("git", "switch", "-C")
    i_add, i_com = idx("git", "add"), idx("git", "commit")
    i_push, i_pr = idx("git", "push"), idx("gh", "pr", "create")
    startpoint = calls[i_sw][4] if i_sw is not None and len(calls[i_sw]) > 4 else None
    return {
        "fetch before switch": i_fetch is not None and i_sw is not None and i_fetch < i_sw,
        "branch cut from origin/main": startpoint == "origin/main",
        "the alias commit follows the switch": i_add is not None and i_sw is not None and i_add > i_sw,
        "commit, then push": i_com is not None and i_push is not None and i_com < i_push,
        "pr create last": i_pr is not None and i_push is not None and i_push < i_pr,
        "no rebase of a stale base": idx("git", "rebase") is None,
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
        print("  recorded order:", [c[1] if c[0] == "git" else " ".join(c[:2]) for c in calls])

    # ---- controls: each matcher must be able to fail ---------------------------
    def drop(*prefix):
        return [c for c in calls if c[:len(prefix)] != list(prefix)]

    def flip_startpoint():
        return [c[:4] + ["HEAD"] if c[:3] == ["git", "switch", "-C"] else c for c in calls]

    controls = [
        ("no fetch at all", verdict(drop("git", "fetch", "origin"))["fetch before switch"]),
        ("branch cut from the run's HEAD", verdict(flip_startpoint())["branch cut from origin/main"]),
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
