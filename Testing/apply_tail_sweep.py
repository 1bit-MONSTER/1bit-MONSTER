#!/usr/bin/env python3
"""Insert census_tail_aliases.json into the arch registries + selfcheck.
Idempotent: replaces the auto-generated block between markers if present.
Usage: python3 Testing/apply_tail_sweep.py [--repo PATH]..."""
import json, os, sys

ALIASES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "census_tail_aliases.json")

HEADER_MARK = "// ── 2026-08-15 census tail sweep"
CHECK_MARK = "// ── census tail sweep checks ──"


def patch(path, block, marker):
    src = open(path).read()
    # remove previous auto block if present (idempotent re-runs)
    if marker in src:
        lines = src.split("\n")
        out, skipping = [], False
        for ln in lines:
            if marker in ln:
                skipping = True
                continue
            if skipping and (ln.strip().startswith("//") or ln.strip().startswith("check(")):
                continue
            skipping = False
            out.append(ln)
        src = "\n".join(out)
    anchor = '    if (strcmp(s, "lfm2")      == 0) return RCPP_ARCH_LFM2;'
    assert src.count(anchor) == 1
    src = src.replace(anchor, anchor + "\n" + block.rstrip("\n"), 1)
    open(path, "w").write(src)
    print("patched:", path)


def main():
    a = json.load(open(ALIASES))
    lines = ["    %s (auto-generated, model_type-verified) ──" % "// ── 2026-08-15 census tail sweep"]
    for s, v in sorted(a.items()):
        lines.append('    if (strcmp(s, "%s") == 0) return RCPP_ARCH_%s;  // %s' %
                     (s, v["token"], v["model_type"]))
    block = "\n".join(lines) + "\n"

    checks = ["    %s" % "// ── census tail sweep checks ──"]
    for s, v in sorted(a.items()):
        checks.append('    check("%s", RCPP_ARCH_%s, "%s");' % (s, v["token"], s))
    cblock = "\n".join(checks) + "\n"

    repos = []
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--repo":
            repos.append(sys.argv[1 + i + 1])
    if not repos:
        repos = ["."]
    for r in repos:
        h = os.path.join(r, "include", "rocm_cpp", "bitnet_model.h")
        if os.path.exists(h):
            patch(h, block, HEADER_MARK)
        sc = os.path.join(r, "Testing", "arch_mapping_selfcheck.cpp")
        if os.path.exists(sc):
            src = open(sc).read()
            if CHECK_MARK in src:
                src = src[:src.index(CHECK_MARK)].rstrip("\n") + "\n"
            anchor = "    if (fails) {"
            assert src.count(anchor) == 1, sc
            src = src.replace(anchor, cblock.rstrip("\n") + "\n" + anchor, 1)
            open(sc, "w").write(src)
            print("patched:", sc)
    print("aliases applied: %d" % len(a))


if __name__ == "__main__":
    main()
