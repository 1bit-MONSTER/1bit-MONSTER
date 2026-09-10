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
