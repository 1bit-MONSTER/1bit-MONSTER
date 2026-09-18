#!/usr/bin/env python3
"""Suite-log integrity check: a skipped gate is not a green gate.

Defect class this closes, reported by the peer lane and confirmed here: run_prism_tests.sh builds several
optional device gates, and when a build failed it printed "(hipcc present but X failed to build — skipping)"
and carried on. Nothing incremented any counter, so the final line still said ALL PRISM GATES PASSED. A
whole device gate (P3.3, the 64-layer forward vs the fork oracle) was skipped through several green reports,
with only a parenthetical in the middle of a long log as evidence. A summary that cannot distinguish PASS
from SKIPPED is not evidence, and the fix has to be structural rather than a promise to read carefully.

Rules:
  R1  "ALL PRISM GATES PASSED" may not co-occur with any SKIPPED/skipping line.
  R2  Any FAIL line invalidates a green summary.
  R3  A skip must be declared: the log must carry a 'gates:' count line naming skipped>0, and if the run is
      declared (PRISM_NO_DEVICE), the summary must say so rather than claiming an unqualified pass.
  R4  A log with no summary line at all is not a pass (the run did not finish).

Usage: check_suite_log.py [logfile]      (reads stdin if no file)
       check_suite_log.py --self-test    (fixtures; asserts this checker can fail)
"""
import re
import sys
import pathlib

GREEN = "ALL PRISM GATES PASSED"
SKIP_RE = re.compile(r"\bskip(ped|ping)?\b", re.I)
FAIL_RE = re.compile(r"\bFAIL\b")
COUNTS_RE = re.compile(r"^gates:\s*passed=(\d+)\s+failed=(\d+)\s+skipped=(\d+)", re.M)


def check(text):
    """Return a list of violations; empty list means admissible."""
    fails = []
    green = GREEN in text
    skips = []
    for ln in text.splitlines():
        t = ln.strip()
        if not t or t.startswith("#") or COUNTS_RE.match(t):
            continue
        if SKIP_RE.search(t) and "skipped=0" not in t:
            skips.append(ln)
    fails_lines = [ln for ln in text.splitlines() if FAIL_RE.search(ln)]
    counts = COUNTS_RE.search(text)

    if green and skips:
        fails.append(
            f"claims '{GREEN}' while {len(skips)} skipping line(s) are present - first: {skips[0].strip()[:90]}"
        )
    if green and fails_lines:
        fails.append(f"claims '{GREEN}' with {len(fails_lines)} FAIL line(s) - first: {fails_lines[0].strip()[:90]}")
    if not (green or "GATES" in text or counts):
        fails.append("no summary line found - a run that does not report its result is not a pass")
    if counts is None and (green or skips):
        fails.append("no 'gates: passed=.. failed=.. skipped=..' count line - the summary does not carry its counts")
    if counts:
        p, f, s = (int(x) for x in counts.groups())
        if green and s != 0:
            fails.append(f"claims a green run with skipped={s}")
        if f != 0 and green:
            fails.append(f"claims a green run with failed={f}")
        if s != 0 and "DECLARED" not in text.upper():
            fails.append(f"skipped={s} without a declaration (e.g. PRISM_NO_DEVICE) - an undeclared skip is a defect")
    return fails


def self_test():
    """Prove the checker can fail and can pass; run on every invocation."""
    clean = f"gates: passed=42 failed=0 skipped=0\n{GREEN}\n"
    sneaky = f"gates: passed=41 failed=0 skipped=1\n{GREEN}\n  (hipcc present but the full device forward failed to build — skipping)\n"
    undeclared = "gates: passed=41 failed=0 skipped=1\nGATES PASSED WITH 1 SKIPPED\n  (build failed — skipping)\n"
    declared = "PRISM_NO_DEVICE declared\ngates: passed=30 failed=0 skipped=12\nGATES PASSED WITH 12 DECLARED SKIPS\n  (skipping)\n"
    honest_fail = "gates: passed=40 failed=2 skipped=0\n2 GATE(S) FAILED\nsome FAIL line\n"
    lying_green = f"gates: passed=42 failed=0 skipped=0\nGREEN_PLACEHOLDER\nsome FAIL line\n".replace("GREEN_PLACEHOLDER", GREEN)
    cases = [("clean green", clean, 0), ("green with a skip", sneaky, 1), ("greener with skipped=1", clean.replace("skipped=0", "skipped=1"), 1),
             ("undeclared skip", undeclared, 1), ("declared skips", declared, 0),
             ("honest failure", honest_fail, 0), ("green over a FAIL line", lying_green, 1)]
    bad = []
    for name, text, want in cases:
        got = 1 if check(text) else 0
        if got != want:
            bad.append(f"self-test: '{name}' expected {'reject' if want else 'accept'}, got the opposite")
    return bad


def main(argv):
    if "--self-test" in argv:
        bad = self_test()
        for b in bad:
            print(f"  FAIL {b}")
        print("check_suite_log self-test: " + ("PASS" if not bad else "FAIL"))
        return 1 if bad else 0
    text = pathlib.Path(argv[1]).read_text() if len(argv) > 1 else sys.stdin.read()
    selftest = self_test()
    fails = selftest + check(text)
    for f in fails:
        print(f"  FAIL {f}")
    print("SUITE LOG: " + ("ADMISSIBLE" if not fails else "NOT ADMISSIBLE"))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
