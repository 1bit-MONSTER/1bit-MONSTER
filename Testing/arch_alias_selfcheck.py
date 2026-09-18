#!/usr/bin/env python3
"""arch_alias_selfcheck.py — no architecture alias may be defined twice in rcpp_arch_from_string.

WHY
---
`rcpp_arch_from_string` (include/rocm_cpp/bitnet_model.h) is a linear if-chain: the FIRST
`strcmp(s, "...") == 0` wins, so a later definition of the same string is unreachable dead code.
That is not a style question. A tool that builds a table by reading the file in order and
assigning into a dict gets LAST-match-wins — the opposite answer from the engine — which is how
Testing/census_autopr.py reported RCPP_ARCH_QWEN3 for `qwen3_5moe` while the engine answered
RCPP_ARCH_QWEN35 (#2501).

Eight such duplicates existed on `main`. Seven returned the same token as the first definition
(dead, harmless); the eighth — `qwen3_5moe` -> RCPP_ARCH_QWEN3 at the foot of the Qwen3 block —
disagreed with the Qwen3.5 block that matches first. They are gone; this is what keeps the file
to one answer per alias.

Floors: a regex that stopped matching would otherwise pass this check by finding nothing, so the
alias count is asserted against a literal minimum and the function's bounds must be found.

EXIT CODES
----------
0  ok          1  a duplicate (or a floor breach)          2  environment (header missing)
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "include" / "rocm_cpp" / "bitnet_model.h"

# Today: 2,036 distinct aliases, each defined once. Raise the floor when aliases are added; never
# lower it — a parser that quietly stopped matching is the failure this guards against.
MIN_ALIASES = 2000

DEF = re.compile(r'strcmp\(s,\s*"([^"]+)"\)\s*==\s*0\)\s*return\s+(RCPP_ARCH_[A-Z0-9_]+)')
FN = re.compile(r"rcpp_arch_from_string\s*\(")


def main() -> int:
    try:
        text = HEADER.read_text()
    except OSError as exc:
        print(f"ENVIRONMENT: cannot read {HEADER}: {exc}", file=sys.stderr)
        return 2
    lines = text.splitlines()

    starts = [i for i, l in enumerate(lines) if FN.search(l) and "{" in l]
    if not starts:
        print(f"ENVIRONMENT: rcpp_arch_from_string not found in {HEADER}", file=sys.stderr)
        return 2
    start = starts[0]
    ends = [i for i in range(start, len(lines)) if lines[i].strip() == "}"]
    if not ends:
        print(f"ENVIRONMENT: no closing brace after line {start + 1}", file=sys.stderr)
        return 2
    end = ends[0]

    defs: dict[str, list[tuple[int, str]]] = {}
    for i in range(start, end):
        m = DEF.search(lines[i])
        if m:
            defs.setdefault(m.group(1), []).append((i + 1, m.group(2)))

    problems = []
    for alias, seen in sorted(defs.items()):
        if len(seen) > 1:
            toks = {t for _, t in seen}
            kind = ("the same token - dead code" if len(toks) == 1
                    else "DIFFERENT tokens - the engine uses the first, a last-match-wins parser the last")
            where = ", ".join(f"line {ln} -> {t}" for ln, t in seen)
            problems.append(f"{alias}: defined {len(seen)} times ({where}) — {kind}")
    if len(defs) < MIN_ALIASES:
        problems.append(
            f"only {len(defs)} distinct aliases parsed from lines {start + 1}-{end + 1}, floor is "
            f"{MIN_ALIASES}: the scan stopped matching, so a clean result here would mean nothing")

    print(f"arch alias table - {HEADER.relative_to(ROOT)}:{start + 1}-{end + 1}, "
          f"{len(defs)} distinct aliases, {sum(len(v) for v in defs.values())} definitions")
    if problems:
        print("VIOLATION:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        print("  The chain is first-match-wins: keep the FIRST definition and delete the later one, "
              "or change the first if the later token was right (#2501).", file=sys.stderr)
        return 1
    print("OK: every alias is defined exactly once")
    return 0


if __name__ == "__main__":
    sys.exit(main())
