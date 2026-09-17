#!/usr/bin/env python3
"""census_scripts_selfcheck.py — the census diagnostics must agree on two things.

1. Where the repo is. A pinned `/home/bcloud/1bit-MONSTER` makes a run from a
   worktree read and write the shared checkout — which the worktree rules
   forbid (AGENTS-WORKTREES.md rules 2 and 5) — and it hides itself, because
   the run looks successful against the wrong inputs (#2387).
2. What is out of scope. NON_TEXT_GEN is a policy with ONE definition,
   `Testing/census_coverage.py`. Copies drifted once already (#2387: two stale
   14-entry copies against the 276-entry set), so the invariant is checked here
   instead of in review.

Exits non-zero with the offending file:line, so the failure names itself.
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

# Every script in the family, including the ones that only read the policy set.
# census_autopr.py is here for the ROOT rule: it wrote to the engine header, so
# a pinned ROOT would edit the shared checkout from a worktree run.
SCRIPTS = ["census_coverage.py", "census_tail_verify.py", "census_classify.py",
           "census_batch_verify.py", "retrieve_pass.py", "census_autopr.py"]

# The one file allowed to define the policy set; everyone else imports it.
OWNER = "census_coverage.py"
PINNED_ROOT = re.compile(r'^\s*ROOT\s*=\s*["\']/')
DEFINES_SET = re.compile(r'^\s*NON_TEXT_GEN\s*=')

problems = []
for name in SCRIPTS:
    path = os.path.join(HERE, name)
    if not os.path.exists(path):
        problems.append("%s: missing" % name)
        continue
    with open(path) as fh:
        for lineno, line in enumerate(fh, 1):
            if PINNED_ROOT.match(line):
                problems.append(
                    "%s:%d: ROOT pinned to an absolute path — derive it from __file__"
                    % (name, lineno))
            if DEFINES_SET.match(line) and name != OWNER:
                problems.append(
                    "%s:%d: re-defines NON_TEXT_GEN — import it from %s"
                    % (name, lineno, OWNER))

sys.path.insert(0, HERE)
import census_coverage  # noqa: E402

policy = census_coverage.NON_TEXT_GEN
if not isinstance(policy, set) or len(policy) < 10:
    problems.append("%s: NON_TEXT_GEN is not a populated set" % OWNER)

# The consumers must hold that object, not an equal-looking copy: an equal copy
# can drift again, and identity is what the fix actually establishes.
_argv = sys.argv
sys.argv = [sys.argv[0]]  # census_tail_verify reads sys.argv[1:2] at import
try:
    import census_tail_verify  # noqa: E402
    import census_classify  # noqa: E402
finally:
    sys.argv = _argv
for mod in (census_tail_verify, census_classify):
    if mod.NON_TEXT_GEN is not policy:
        problems.append("%s.NON_TEXT_GEN is not %s.NON_TEXT_GEN"
                        % (mod.__name__, OWNER))

if problems:
    print("census_scripts_selfcheck: FAIL")
    for p in problems:
        print("  - " + p)
    sys.exit(1)

print("census_scripts_selfcheck: PASS — %d scripts share one policy set "
      "(%d entries, defined only in %s)" % (len(SCRIPTS), len(policy), OWNER))
