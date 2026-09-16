#!/usr/bin/env python3
"""Flag audit for `1bit registry` — the class check, not another patch.

WHY THIS EXISTS
Three separate instances of "a flag that exists but is invisible or inert" were
hand-patched before it became obvious the design invites the class:
  1. usage() omitted --catalog / --catalog-default      (docs)   -> patched
  2. --at-context silently ignored without --capability (inert)  -> patched
  3. --prefer without --route was a silent no-op        (inert)  -> patched
  4. usage() omitted --route / --prefer                 (docs)   -> patched
Raised by @agent-ec855d, who then handed over the CHECK rather than a fifth patch.

TWO CLASSES, TWO CHECKS, AND EACH ONE'S BLIND SPOT NAMED

  A. DOCUMENTATION PARITY (static). Every flag the parser accepts must appear in
     usage(). Catches 1 and 4. Cannot see anything about behaviour.

  B. SILENT NO-OP (behavioural). Run each flag ALONE against a fixture and require
     that it either exits non-zero or changes the output. This is the check that
     catches 2 and 3.

     @agent-ec855d proposed instead a textual "every parsed flag is consumed"
     count, quoting a trap: per-file counting yields two FALSE orphans
     (opt.probe_headers / opt.max_depth appear once in tools/registry_scan.cpp and
     are read in src/model_registry.cpp). The trap is real and handled here — but
     that check CANNOT catch instance 3, because prefer_arg *was* referenced; it was
     read inside a branch guarded by route_arg. A reference count sees one
     reference and passes. Only execution distinguishes "read on some path" from
     "read on the path the user actually took", so B is behavioural.
     The cross-TU reference count is still run, as a cheap third check, with the
     trap handled — it catches a genuinely unreferenced variable, which is a real
     but different defect.

USAGE
  tools/registry_flag_audit.py --binary /tmp/registry_scan --fixture /tmp/regtest
  tools/registry_flag_audit.py --static-only
Exit 0 = clean. Exit 1 = a check failed (with the specific flag named).
"""
import argparse
import os
import re
import subprocess
import sys

SCAN = "tools/registry_scan.cpp"
CROSS_TU = ["tools/registry_scan.cpp", "src/model_registry.cpp", "include/model_registry.h"]

# Flags that legitimately do not appear in usage() as their own token.
DOC_EXEMPT = {"help"}          # -h / --help advertises itself


def read(path):
    with open(path, "r", encoding="utf-8") as fh:
        return fh.read()


def usage_body(text):
    """Body of the usage() function only — not the file-top comment, which
    documents flags too and would mask a usage() omission."""
    m = re.search(r"void usage\(.*?\)\s*\{(.*?)\n\}", text, re.S)
    if not m:
        sys.exit("audit: could not find usage() in %s" % SCAN)
    return m.group(1)


def parsed_flags(text):
    return set(re.findall(r'a == "(--[a-z0-9-]+)"', text))


def documented_flags(text):
    body = usage_body(text)
    return set(re.findall(r"(--[a-z0-9-]+)", body))


def assigned_vars(text):
    """Variables assigned inside the flag-parsing chain."""
    body = text.split("for (int i = 1; i < argc; i++)", 1)[-1].split("if (roots.empty())", 1)[0]
    out = set()
    for line in body.splitlines():
        if '"--' not in line and '"-h"' not in line:
            continue
        out |= set(re.findall(r"\b([A-Za-z_][A-Za-z0-9_.]*)\s*=\s*(?:true|false|argv|\(|std::)", line))
    return {v for v in out if not v.startswith("std")}


def check_a_docs(text):
    p = {f[2:] for f in parsed_flags(text)}
    d = {f[2:] for f in documented_flags(text)}
    missing = sorted(p - d - DOC_EXEMPT)
    extra = sorted(d - p - {"h"})
    return missing, extra


def check_c_references(root):
    """Cross-TU reference count. THE TRAP: must span all three files, because
    opt.probe_headers / opt.max_depth are assigned in the scanner and read in the
    registry. A per-file check reports both as orphans."""
    text = read(os.path.join(root, SCAN))
    corpus = "\n".join(read(os.path.join(root, f)) for f in CROSS_TU)
    orphans = []
    for var in sorted(assigned_vars(text)):
        # one assignment + at least one other mention, anywhere in the TU set
        if len(re.findall(r"\b%s\b" % re.escape(var), corpus)) < 2:
            orphans.append(var)
    return orphans


def run(binary, args):
    p = subprocess.run([binary] + args, capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def check_b_behaviour(binary, fixture):
    """Each flag alone must not look like it did nothing.

    Effectiveness is judged against BOTH output modes. A table-only comparison
    produced a false positive on --max-depth: on a shallow fixture a smaller depth
    changes nothing, even though the flag is consumed. Some flags only show up in
    JSON (--digest adds sha256 and does not touch the table at all), so a flag
    counts as effective if it changes the table, changes the JSON, or exits
    non-zero. Values are chosen to make the effect observable (depth 1 prunes).
    """
    base_rc, base_tbl, _ = run(binary, [fixture])
    _, base_js, _ = run(binary, ["--json", fixture])
    if base_rc != 0:
        sys.exit("audit: baseline invocation failed (rc=%d) — fixture problem" % base_rc)
    # Guard the OTHER degenerate baseline, found by @agent-ec855d: a fixture with
    # ZERO recognized artifacts makes --digest/--max-depth/--no-probe look inert
    # because there is nothing for them to affect, so B reports three phantom
    # failures. On a fresh machine that reads as "the check is broken", which is the
    # direction that gets a check weakened instead of fixed.
    if '"artifacts": 0' in base_js or '"artifacts":0' in base_js:
        sys.exit("audit: fixture problem — baseline sees 0 artifacts, so flags that act ON "
                 "artifacts cannot show an effect. Point --fixture at a directory with content.")

    text = read(SCAN)
    problems = []
    for flag in sorted(parsed_flags(text)):
        if flag == "--help":
            continue
        value = {"--max-depth": "1", "--capability": "CPU", "--resolve": "x",
                 "--route": "x", "--prefer": "CPU", "--at-context": "4096",
                 "--catalog": "x"}.get(flag)
        if flag == "--catalog-default":
            extra = ["--catalog-default"]
        else:
            extra = [flag] + ([value] if value else [])

        rc_t, out_t, err_t = run(binary, extra + [fixture])
        _, out_j, err_j = run(binary, ["--json"] + extra + [fixture])
        changed = (out_t != base_tbl) or (out_j != base_js) or bool(err_t.strip()) or bool(err_j.strip())
        if rc_t == 0 and not changed:
            problems.append(flag)
    return problems


# ── D. REPEAT SAFETY ───────────────────────────────────────────────────────────
# Sixth instance of "a flag that looks accepted and is silently ignored"
# (@agent-ec855d), and now the THIRD time my check for that class had the class's
# own disease in it.
#
# v1 compared "output differs when repeated" -> passed on the known-bad binary
#    (last-wins differs from the single run too; "differs" is not evidence).
# v2 required refusals to be attributable -> flagged a correct flag, because
#    informational stderr was read as a refusal.
# v3 (this one) fixes a FALSE POSITIVE found on a store I could not test:
#    repeat(A,B) == once(A) was read as "B discarded", but equality is ALSO produced
#    by a CORRECT implementation whose second value is serveable and simply does not
#    win, so it leaves no trace. The check could not separate "silently discarded"
#    from "silently invisible" and resolved toward the defect. "Identical is not
#    evidence of discard" is the mirror of "differs is not evidence of accumulation".
#
# TWO PART FIX:
#   1. OBSERVABILITY PRECONDITION — a check must first establish that what it
#      measures is measurable HERE. For list-valued flags, once(v1), once(v2) and
#      once(v1,v2) must be pairwise distinct; otherwise the list form is not
#      observable on this fixture and the verdict is INCONCLUSIVE — never PASS,
#      never FAIL. Verdicts must not depend on which fixture someone runs.
#   2. POSITIVE ASSERTIONS instead of "matches neither":
#      - list-valued: repeat(v1,v2) must EQUAL once("v1,v2") exactly. The comma form
#        is a valid oracle and it catches ORDER REVERSAL directly, which the old rule
#        only caught by luck of which element happened to win.
#      - single-value: the repeat must be REFUSED. Fixture-independent, because it
#        asserts on the exit status rather than on output equality.
# A list flag's oracle depends on whether it accepts CSV. --prefer parses commas, so
# the comma form is a valid oracle; --engine-limit does NOT (each occurrence is one
# CAP=TOKENS[:bundle] spec), so it needs a different positive assertion.
CSV_LIST_FLAGS = {"--prefer"}
NO_CSV_LIST_FLAGS = {"--engine-limit"}
SINGLE_VALUE_FLAGS = ["--max-depth", "--capability", "--at-context", "--resolve", "--route"]
REPEAT_VALUES = {
    "--max-depth": ("1", "4"),
    "--capability": ("CPU", "NPU-Q4NX"),
    "--at-context": ("1024", "4096"),
    "--prefer": ("RADV-GGUF", "CPU"),
    # Two DIFFERENT capabilities on purpose: with the same capability twice,
    # overwrite-in-place and accumulation are identical by design, so the values would
    # prove nothing about either (my first version made exactly that mistake).
    "--engine-limit": ("HRX-GGUF=4096:b1", "CPU=100:c1"),
    "--resolve": (None, None),
    "--route": (None, None),
}


def first_artifact_id(binary, fixture):
    """Pick a target where the flag is EXERCISABLE, not merely present.

    Three failure modes, each found by running this against a fixture that was fine:
      1. taking the first table row let `--route x` (a target that never existed) be the
         only case ever exercised — the flag was never read;
      2. preferring the highest capability COUNT picked an artifact whose four lanes were
         all from one container (the ONEBP chain), so neither `--prefer` value could
         resolve: count is not richness;
      3. a single-capability artifact made every value collapse to the same lane and
         reported INCONCLUSIVE for the wrong reason.
    So the question is not "how many capabilities" but "which ones RESOLVE", and the
    selector asks it directly: accept the first artifact with >= 2 capabilities that both
    actually resolve.
    """
    _rc, out, _err = run(binary, [fixture])
    cands = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 6 and parts[1] in ("gguf", "onebp", "mlx", "safetensors", "raw_bin"):
            caps = []
            for c in ",".join(parts[5:]).split(","):
                c = c.strip()
                if c and "(" not in c and c not in caps:
                    caps.append(c)
            cands.append((parts[0], caps))
    for aid, caps in cands:
        # The --prefer run IS the validation: an unknown name exits 2 and simply is not
        # counted, so no separate name check is needed (and capability_from_string is a
        # C++ symbol, not a Python one — reaching for it here was my mistake).
        working = [c for c in caps
                   if run(binary, ["--route", aid, "--prefer", c, fixture])[0] == 0]
        if len(working) >= 2:
            return aid, working
    for aid, _c in cands:
        if run(binary, ["--route", aid, fixture])[0] == 0:
            return aid, []
    return (cands[0][0], []) if cands else (None, [])


def check_d_repeats(binary, fixture):
    text = read(SCAN)
    parsed = {f[2:] for f in parsed_flags(text)}
    # The --prefer values must come from the TARGET'S OWN capabilities. Hardcoding them
    # was the fourth defect in this function: the selector can legitimately return an
    # artifact whose working lanes are (FUSED-GPU-NPU, HIP-1BP), and a hardcoded
    # (RADV-GGUF, CPU) pair then fails on both values and reports INCONCLUSIVE — the test
    # measuring its own assumption rather than the flag. A test's inputs must be drawn
    # from the state of the thing under test.
    target, target_caps = first_artifact_id(binary, fixture)
    problems, inconclusive = [], []

    for flag, (v1, v2) in REPEAT_VALUES.items():
        if flag[2:] not in parsed:
            continue
        extra = []
        if flag in ("--resolve", "--route"):
            if not target:
                continue
            v1, v2 = target, "no-such-artifact-xyz"
        elif flag == "--prefer":
            if not target or len(target_caps) < 2:
                continue
            extra = ["--route", target]
            v1, v2 = target_caps[0], target_caps[1]
        elif flag == "--at-context":
            extra = ["--capability", "HRX-GGUF"]

        rc_ab, out_ab, err_ab = run(binary, extra + [flag, v1, flag, v2, fixture])

        if flag in SINGLE_VALUE_FLAGS:
            # POSITIVE ASSERTION, fixture-independent: a repeat must be refused.
            if rc_ab == 0 and not (("more than once" in err_ab) or (flag in err_ab)):
                problems.append("%s (repeat accepted silently; single-value flags must refuse)"
                                % flag)
            continue

        # List-valued: establish observability BEFORE interpreting anything.
        _r1, out1, e1 = run(binary, extra + [flag, v1, fixture])
        _r2, out2, e2 = run(binary, extra + [flag, v2, fixture])
        if (out1, e1) == (out2, e2):
            inconclusive.append(
                "%s (list form NOT OBSERVABLE here: once(v1) == once(v2), so no "
                "output-comparing check can see a two-element list)" % flag)
            continue

        if flag in CSV_LIST_FLAGS:
            # THE COMMA ORACLE IS DEGENERATE FOR AN ORDERED FIRST-WINS LIST, and I only
            # found that by running it: `--prefer A,B` resolves A, so once(v1,v2) ALWAYS
            # equals once(v1), and an observability guard built on comparing them can
            # never be satisfied. The oracle was measuring a tautology.
            #
            # The discriminating test for "accumulated with order preserved" versus
            # "last value replaced the first" needs no oracle at all:
            #     once(v1) != once(v2)      the two values must lead to different lanes
            #     repeat(v1,v2) == once(v1) the FIRST value survived
            #     repeat(v1,v2) != once(v2) and it was not simply the last value
            if (out1, e1) == (out2, e2):
                inconclusive.append(
                    "%s (not observable here: once(v1) == once(v2), so dropped-vs-kept "
                    "cannot be distinguished)" % flag)
                continue
            if rc_ab != 0:
                if ("more than once" in err_ab) or (flag in err_ab):
                    continue
                inconclusive.append("%s (run failed for another reason)" % flag)
                continue
            if (out_ab, err_ab) == (out2, e2):
                problems.append("%s (second value %r won: the first was discarded)"
                                % (flag, v2))
            elif (out_ab, err_ab) != (out1, e1):
                problems.append("%s (repeat matches neither single-value run: not "
                                "accumulation with order preserved)" % flag)
        else:
            # No CSV oracle. Both values must still leave a trace, so assert positively
            # that the repeat differs from EACH single-value run.
            if rc_ab != 0:
                if ("more than once" in err_ab) or (flag in err_ab):
                    continue
                inconclusive.append("%s (run failed for another reason)" % flag)
                continue
            if (out_ab, err_ab) == (out1, e1):
                problems.append("%s (first value silently discarded)" % flag)
            elif (out_ab, err_ab) == (out2, e2):
                problems.append("%s (second value silently discarded)" % flag)
    return problems, inconclusive


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--binary")
    ap.add_argument("--fixture")
    ap.add_argument("--static-only", action="store_true")
    a = ap.parse_args()

    text = read(os.path.join(a.root, SCAN))
    failed = False

    missing, extra = check_a_docs(text)
    print("A. usage parity: %d parsed, %d documented" % (len(parsed_flags(text)), len(documented_flags(text))))
    if missing:
        failed = True
        print("   FAIL — accepted by the parser, undocumented in usage(): %s" % ", ".join("--" + f for f in missing))
    elif extra:
        print("   note — documented but not parsed: %s" % ", ".join("--" + f for f in extra))
    else:
        print("   ok")

    orphans = check_c_references(a.root)
    print("C. cross-TU references (spans %s):" % ", ".join(CROSS_TU))
    if orphans:
        failed = True
        print("   FAIL — assigned and never read anywhere: %s" % ", ".join(orphans))
    else:
        print("   ok")

    if not a.static_only and a.binary and a.fixture:
        print("D. repeat safety (a repeated flag must differ or refuse):")
        rep, inconc = check_d_repeats(a.binary, a.fixture)
        if rep:
            failed = True
            print("   FAIL — repeated value silently discarded: %s" % ", ".join(rep))
        if inconc:
            failed = True
            print("   INCONCLUSIVE — cannot attribute the failure to the repeat: %s"
                  % "; ".join(inconc))
        if not rep and not inconc:
            print("   ok")

    if not a.static_only:
        if not (a.binary and a.fixture):
            print("B. behavioural: skipped (pass --binary and --fixture)")
        else:
            print("B. behavioural no-op check:")
            problems = check_b_behaviour(a.binary, a.fixture)
            if problems:
                failed = True
                print("   FAIL — exits 0 and changes nothing (silent no-op): %s"
                      % ", ".join(problems))
            else:
                print("   ok")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
