#!/usr/bin/env python3
"""check_artifact_families.py — every artifact family the NPU engine can construct
must have tracked members, unless it is declared build-only.

Issue #2601: NPU_BF16=1 constructs

    final_bf16_<T>_K<K>_N<N>.xclbin
    insts_bf16_<T>_K<K>_N<N>.txt

A docs commit deleted all 54 of them (a81662ab8) and nothing noticed, because no
check mapped the families the engine *constructs* to the artifacts the commit
*carries*. A family dropping to zero is invisible until a user selects the mode
and gets "FAIL bf16 QKV" with no in-tree pointer to the fix
(engine/npu/generators/build_bf16_xclbins.sh).

DEFINITIONS
-----------
* Family: (prefix, tag) where the engine has a construction site that can emit
  `prefix + tag + "_..."`. Prefixes and tags are discovered from the engine
  source (the `final_bf16_` / `insts_bf16_` name templates and the `init_bf16`
  call tags); extra families may be added in the declaration file.
* Tracked member: a path under engine/npu/xclbins/ in `git ls-files` whose
  basename starts with `prefix + tag + "_"`.
* Build-only: a family named in `build_only` in
  engine/npu/xclbins/ARTIFACT_FAMILIES.json. Per the owning lane's disposition
  (see #2601) a build output is a valid answer — but it must be *declared*, so a
  family dropping to zero is a decision, not an accident.

Exit 0 = every discovered family is tracked or declared build-only.
Exit 1 = at least one family has zero tracked members and is not declared.
Exit 2 = the check itself could not run (missing source/manifest, no families).
"""
import argparse
import json
import os
import re
import subprocess
import sys

DEFAULT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_SOURCE = "engine/npu/src/npu_engine_universal.cpp"
DEFAULT_XCLBIN_DIR = "engine/npu/xclbins"
DEFAULT_DECL = "engine/npu/xclbins/ARTIFACT_FAMILIES.json"


def discover_families(source_text, extra):
    """(prefix, tag) pairs the engine can construct."""
    # Prefix templates: the string literals that build the artifact name.
    prefixes = set()
    for m in re.finditer(r'"(?:[^"]*/)?(final_bf16_|insts_bf16_)"', source_text):
        prefixes.add(m.group(1))
    if not prefixes:
        prefixes = {"final_bf16_", "insts_bf16_"}
    # Type tags: the string literal argument of each init_bf16(<ctx>, "<TAG>", ...).
    tags = set(re.findall(r'init_bf16\(\s*[A-Za-z_][A-Za-z0-9_]*\s*,\s*"([A-Z][A-Z0-9]*)"', source_text))
    families = {(p, t) for p in prefixes for t in tags}
    for fam in extra:
        p = fam.get("prefix")
        t = fam.get("tag")
        if p and t:
            families.add((p, t))
    return sorted(families, key=lambda x: (x[0], x[1])), sorted(tags)


def tracked_paths(root, tracked_file):
    if tracked_file:
        with open(tracked_file) as f:
            return [ln.strip() for ln in f if ln.strip()]
    out = subprocess.run(["git", "-C", root, "ls-files", "--", DEFAULT_XCLBIN_DIR],
                         capture_output=True, text=True, check=True)
    return [ln for ln in out.stdout.splitlines() if ln.strip()]


def count_members(paths, prefix, tag):
    want = os.path.join(DEFAULT_XCLBIN_DIR, prefix + tag + "_")
    return sum(1 for p in paths if p.startswith(want))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--source", default=None, help="engine source (default: <root>/" + DEFAULT_SOURCE + ")")
    ap.add_argument("--decl", default=None, help="family declaration JSON")
    ap.add_argument("--tracked-file", default=None, help="newline-separated paths (tests); default: git ls-files")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    source = os.path.abspath(args.source) if args.source else os.path.join(root, DEFAULT_SOURCE)
    decl_path = os.path.abspath(args.decl) if args.decl else os.path.join(root, DEFAULT_DECL)
    if not os.path.isfile(source):
        print(f"artifact-families: source not found: {source}", file=sys.stderr)
        return 2
    try:
        decl = json.load(open(decl_path)) if os.path.isfile(decl_path) else {}
    except json.JSONDecodeError as e:
        print(f"artifact-families: bad declaration {decl_path}: {e}", file=sys.stderr)
        return 2
    build_only = {tuple(x) if isinstance(x, list) else (x.get("prefix"), x.get("tag"))
                  for x in decl.get("build_only", [])}
    extra = decl.get("families", [])

    with open(source) as f:
        families, tags = discover_families(f.read(), extra)
    if not families:
        print("artifact-families: discovered 0 families — the source scan is blind", file=sys.stderr)
        return 2
    try:
        paths = tracked_paths(root, args.tracked_file)
    except subprocess.CalledProcessError as e:
        print(f"artifact-families: git ls-files failed: {e}", file=sys.stderr)
        return 2

    bad = []
    for prefix, tag in families:
        n = count_members(paths, prefix, tag)
        if n:
            state = "tracked"
        elif (prefix, tag) in build_only or tag in build_only:
            state = "build-only (declared)"
        else:
            state = "ZERO TRACKED MEMBERS — undeclared"
            bad.append((prefix, tag))
        print(f"  {prefix}{tag:<4} {n:>3}  {state}")

    print(f"artifact-families: {len(families)} families from {os.path.relpath(source, root)}"
          f" (tags: {' '.join(tags) if tags else 'none'})")
    if bad:
        print("artifact-families: FAIL — engine-constructible families with no tracked members:", file=sys.stderr)
        for prefix, tag in bad:
            print(f"    {prefix}{tag} — build them (see engine/npu/generators/) or declare them"
                  f" build-only in {os.path.relpath(decl_path, root)}", file=sys.stderr)
        return 1
    print("artifact-families: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
