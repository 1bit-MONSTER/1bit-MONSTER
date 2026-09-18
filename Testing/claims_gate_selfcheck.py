#!/usr/bin/env python3
"""The published-claim gate must be able to fail.

Why this exists: tools/validate_claims.py --check-readme is the CI gate that stops
a quarantined tok/s figure from being published as validated. It scanned README.md
line by line for five engine names, and README became a 70-line landing page — so
the loop body never ran, the function returned [] unconditionally, and the check
printed "consistent" while the page that had inherited the numbers carried a live
violation (issue #2476). Nothing failed, because a scan of zero rows is not an
error.

A test that only runs the gate over the current tree cannot catch that: today it
passes for the right reason, and it would also have passed for the wrong one. So
this asserts the property directly — for every page the gate claims to read, an
injected bad row must produce a violation, and the pages must still be shaped the
way the parser expects.

Stdlib only, no build, no device — safe to run in the host-only suite.
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FAILED: list[str] = []


def check(what: str, ok: bool, detail: str = "") -> None:
    if ok:
        print(f"  ok   {what}")
    else:
        print(f"  FAIL {what}" + (f"\n         {detail}" if detail else ""))
        FAILED.append(what)


def load_gate():
    spec = importlib.util.spec_from_file_location(
        "validate_claims", ROOT / "tools" / "validate_claims.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    vc = load_gate()
    real_repo, real_bench = vc.REPO, vc.BENCH

    # A quarantined engine, taken from the live data rather than hardcoded.
    import json
    unverified = set(json.loads(real_bench.read_text()).get("_unverified", {})) - {"_comment"}
    engine = next((e for e, k in vc.README_ENGINES_WITHOUT_SOURCE.items() if k in unverified), None)
    if engine is None:
        print("  FAIL no quarantined engine in benchmarks/latest.json to test with")
        return 1
    print(f"  note using quarantined engine {engine!r} from benchmarks/latest.json")

    # 1/2. The pages must exist, and the claim set as a whole must still yield
    # rows. That second half is the #2476 failure mode exactly: a target that moved
    # silently, leaving the scan reading nothing. It is asserted over the whole set
    # rather than per file, because README legitimately has no benchmark table today
    # — that is *why* a README-only scan returned [] forever — while still being
    # worth scanning in case the table comes back.
    row_counts: dict[str, int] = {}
    for rel in vc.CLAIM_FILES:
        page = real_repo / rel
        check(f"{rel} exists", page.is_file())
        if page.is_file():
            row_counts[rel] = sum(
                1 for l in page.read_text().splitlines() if vc.claim_row(l))
    print(f"  note parseable rows per page: {row_counts}")
    check("at least one claim page yields rows the parser can read",
          any(row_counts.values()),
          "no page matched claim_row — every target has moved and the gate is "
          "scanning nothing")

    # 3. The real tree is clean (what CI runs).
    check("the live tree has no violations", vc.check_readme_consistency() == [],
          str(vc.check_readme_consistency()[:2]))

    # Rows in the two shapes the tree uses, keyed by the page that carries them.
    # Named for what they DO to the gate, not "good"/"bad": the first attempt at
    # this section called the violating row GOOD and then wrote it into the other
    # page as if it were clean, which made the check fail for the wrong reason.
    VIOLATING = {
        "README.md":
            "| Engine | tok/s | status |\n|---|---|---|\n"
            "| %s | **123.4** | measured |\n",
        "docs/wiki/performance.md":
            "| Name | tok/s | Engine | status |\n|---|---|---|---|\n"
            "| %s (Vulkan) | **318 tok/s** | Vulkan ZINC | ✅ validated |\n",
    }
    HEDGED = {
        "README.md":
            "| Engine | tok/s | status |\n|---|---|---|\n"
            "| %s | **123.4** | ❓ unsourced |\n",
        "docs/wiki/performance.md":
            "| Name | tok/s | Engine | status |\n|---|---|---|---|\n"
            "| %s (Vulkan) | **318 tok/s** | Vulkan ZINC | ⚙️ raw |\n",
    }

    # Asserted as a set rather than by index, so a gate narrowed back to one page
    # reports "this page is not scanned" instead of dying on an IndexError.
    for rel in VIOLATING:
        check(f"{rel} is scanned by the gate", rel in vc.CLAIM_FILES)

    # 4. Inject the violating row into ONE page at a time; every other page gets
    # the hedged form, so exactly one violation is expected and it must name the
    # page it came from. BENCH stays on the real quarantine data; only pages move.
    for rel in VIOLATING:
        if rel not in vc.CLAIM_FILES:
            continue  # already reported above
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td)
            for other in vc.CLAIM_FILES:
                p = tmp / other
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(HEDGED.get(other, HEDGED[rel]) % engine)
            (tmp / rel).write_text(VIOLATING[rel] % engine)
            vc.REPO = tmp
            try:
                found = vc.check_readme_consistency()
            finally:
                vc.REPO = real_repo
            check(f"a {engine!r} row marked 'measured/validated' in {rel} is caught",
                  len(found) == 1 and rel in found[0],
                  f"got {found!r}")

    # 5. ...and the hedged form of the same row is not a violation, in both shapes.
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        for rel in HEDGED:
            p = tmp / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(HEDGED[rel] % engine)
        vc.REPO = tmp
        try:
            found = vc.check_readme_consistency()
        finally:
            vc.REPO = real_repo
        check("hedged statuses are accepted in both row shapes", found == [], f"got {found!r}")

    if FAILED:
        print(f"claims_gate_selfcheck: {len(FAILED)} failed")
        return 1
    print("claims_gate_selfcheck: ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
