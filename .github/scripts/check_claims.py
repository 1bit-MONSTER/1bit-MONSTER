#!/usr/bin/env python3
"""Fail if the site publishes a figure that cannot be re-measured.

Two invariants, neither of which needs GPU hardware:

1. No value listed in benchmarks/latest.json `_unverified` appears as a literal
   in site/index.html. These are falsified figures, unit errors, or claims with
   no source anywhere in the repo.

2. Every numbers.json key that site/index.html reads actually exists, so the
   page can never render `undefined` into a title, meta tag, or stat tile.

Run locally:  python3 .github/scripts/check_claims.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
INDEX = REPO / "site/index.html"
NUMBERS = REPO / "site/numbers.json"
LATEST = REPO / "benchmarks/latest.json"

# Figures retired for good. They must never come back, in any formatting.
RETIRED = ["291 tok/s", "94 tok/s", "55.7", "437 KB", "113 tok/s", "28.4 TFlops"]


def literals_for(value) -> list[str]:
    """Plausible renderings of an unverified number, e.g. 19.3 -> '19.3'."""
    if value is None:
        return []
    s = str(value)
    out = {s}
    if isinstance(value, float) and value.is_integer():
        out.add(str(int(value)))
    return sorted(out)


CSS_UNIT = r"(?:px|r?em|vw|vh|%|s|deg|fr|ch)\b"


def strip_css(html: str) -> str:
    """Drop <style> blocks and inline style="..." — CSS lengths are not claims."""
    html = re.sub(r"<style\b[^>]*>.*?</style>", " ", html, flags=re.S | re.I)
    return re.sub(r'style="[^"]*"', " ", html)


def main() -> int:
    html = INDEX.read_text(encoding="utf-8")
    prose = strip_css(html)
    numbers = json.loads(NUMBERS.read_text())
    latest = json.loads(LATEST.read_text())

    failures: list[str] = []

    # --- invariant 1: no unverified figure in the page -----------------------
    unverified = latest.get("_unverified", {})
    for key, entry in unverified.items():
        if key.startswith("_"):
            continue
        claimed = entry.get("claimed") if isinstance(entry, dict) else entry
        for lit in literals_for(claimed):
            # bare integers like "82" or "6" are far too common to grep for;
            # only flag figures distinctive enough to be a real claim.
            if len(lit) < 3:
                continue
            # not preceded/followed by another digit, and not a CSS length
            if re.search(rf"(?<![\d.]){re.escape(lit)}(?![\d.])(?!\s*{CSS_UNIT})", prose):
                failures.append(
                    f"unverified claim {key}={claimed!r} appears in site/index.html "
                    f"(blocker: {entry.get('blocker', '?')[:80]})"
                )
                break

    for lit in RETIRED:
        if lit in prose:
            failures.append(f"retired claim {lit!r} reappeared in site/index.html")

    # --- invariant 2: no numbers.json key the page reads is missing ----------
    scopes = {
        "b": numbers.get("binary", {}),
        "B": numbers.get("benchmarks", {}),
        "R": numbers.get("repo", {}),
        "S": numbers.get("site", {}),
    }
    for var, key in sorted(set(re.findall(r"\b([bBRS])\.([a-z_0-9]+)", html))):
        if key not in scopes[var]:
            failures.append(f"site/index.html reads {var}.{key}, absent from numbers.json")

    for path in sorted(set(re.findall(r'data-num="([^"]+)"', html))):
        node = numbers
        for part in path.split("."):
            node = node.get(part) if isinstance(node, dict) else None
        if node is None:
            failures.append(f"data-num=\"{path}\" does not resolve in numbers.json")

    if failures:
        print("Claim validation FAILED:\n", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1

    print("Claim validation passed: no unverifiable figure published, no missing keys.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
