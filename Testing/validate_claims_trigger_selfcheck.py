#!/usr/bin/env python3
"""validate_claims_trigger_selfcheck.py — the claims workflow must run on the files it scans.

WHY
---
`.github/workflows/validate-claims.yml` runs three checks, and two of them are content scanners:
the quarantine gate scans the rendered text of `site/index.html` plus the JSON payloads the page
fetches, and the badge gate re-runs the offline badge logic against `site/benchmarks.json` before
comparing `site/badge_*.json`. None of those files were in the workflow's `pull_request.paths`, so
the gate never ran on a PR that edited exactly what it exists to catch — the same defect that
`#2507` fixed for `bench.yml`'s compiled inputs, one workflow over (#2525).

This pins the surfaces those checks read, so narrowing `paths:` again is a failure here rather
than a silent loss of coverage. It cannot know about a *new* scanner added to the workflow — the
dynamic version of that belongs with the input-coverage machinery in
Testing/repo_docs_selfcheck.py (#2507) — so the list below is a floor to keep, updated by hand
when a check gains an input.

EXIT CODES
----------
0  every scanned surface is covered          1  a surface is not covered          2  environment
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "validate-claims.yml"

# (path as written in `paths:`, what reads or scans it)
REQUIRED = (
    ("benchmarks/latest.json", "the published table and _verify.commands (coverage + drift)"),
    ("site/numbers.json", "the generated payload index.html fetches"),
    ("site/index.html", "rendered text scanned for quarantined claims (#185/#190)"),
    ("site/benchmarks.json", "the source the badge logic is re-run against"),
    ("site/*-badge.json", "the badges that regeneration is compared against"),
    ("README.md", "the --check-readme target (#185/#190)"),
    ("docs/wiki/performance.md", "where those numbers live now, and the target after #2476"),
    ("engine/fusion/**", "what the re-measurement driver builds against"),
)


def main() -> int:
    try:
        text = WORKFLOW.read_text()
    except OSError as exc:
        print(f"ENVIRONMENT: cannot read {WORKFLOW}: {exc}", file=sys.stderr)
        return 2

    m = re.search(r"^  pull_request:\s*$(.*?)(?=^\S|\Z)", text, re.S | re.M)
    if not m:
        print("ENVIRONMENT: no pull_request trigger in the workflow", file=sys.stderr)
        return 2
    listed = set(re.findall(r'^\s*-\s*"([^"]+)"\s*$', m.group(1), re.M))
    if not listed:
        print("ENVIRONMENT: the pull_request trigger lists no paths", file=sys.stderr)
        return 2

    missing = [(p, why) for p, why in REQUIRED if p not in listed]
    print(f"validate-claims trigger - {len(listed)} path(s) listed, {len(REQUIRED)} scanned surface(s) required")
    for p, why in REQUIRED:
        print(f"  {'ok  ' if p in listed else 'MISS'}  {p:26s} {why}")
    if missing:
        print("\nVIOLATION: a file these checks scan is not in the workflow's trigger, so a PR "
              "editing it does not run the check (#2525):", file=sys.stderr)
        for p, why in missing:
            print(f"  - {p}: {why}", file=sys.stderr)
        return 1
    print("OK: every scanned surface is in pull_request.paths")
    return 0


if __name__ == "__main__":
    sys.exit(main())
