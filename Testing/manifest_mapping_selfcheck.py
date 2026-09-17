#!/usr/bin/env python3
"""manifest_mapping_selfcheck.py — the family manifest's declared mappings must be real.

`Testing/models_manifest.json` is the project's family registry: one entry per
architecture family, each declaring `hf_arch_strings` and the `mapping_target`
token the family resolves to. The file's own `_comment` defines that field as
"the rcpp_arch_t token the census RESOLVES this family to". Nothing checked it.

`Testing/bringup_runner.sh` does exactly this comparison as its step 1 — and
nothing invokes bringup_runner.sh, because its step 3 needs fixture dirs and
torch. Step 1 needs neither, so the whole check was idle. That is how two
families came to declare `MIMO` and `GLM`, tokens that do not exist in the enum
(issue #2511): the ids in the plan table were checked against the header and the
names were not.

Resolution mirrors the census, not `rcpp_arch_from_string` alone. An HF
`model_type` is not the same string as a family tag, so both
`Testing/census_coverage.py` and `safetensors_reader.cpp` fall back to the
underscore/dash-stripped form (`mimo_v2` -> `mimov2` -> QWEN2, `glm_moe_dsa` ->
`glmmoedsa` -> LLAMA). Probing only the raw string reports those as unmapped,
which is what a first version of this check did and got wrong.

What fails, and what is only reported:

  FAIL  a `mapping_target` that is not a token at all — never a judgement call
  FAIL  an `hf_arch_string` that resolves to nothing — a coverage gap
  FAIL  a `status` outside the set its own `_comment` documents
  note  a string whose resolved token differs from the declared one — that can be
        either side being stale, and a human decides (issue #2501 is the live one)

Usage: python3 Testing/manifest_mapping_selfcheck.py
Exit: 0 ok · 1 findings · 2 cannot determine (an input is missing)
"""
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MANIFEST = os.path.join(ROOT, "Testing", "models_manifest.json")
HEADER = os.path.join(ROOT, "include", "rocm_cpp", "bitnet_model.h")

# The `-` is load-bearing: eleven aliases in the header contain one, and the
# narrower class cannot see them (see census_autopr._known_mappings).
ALIAS = re.compile(r'strcmp\(s,\s*"([a-z0-9_-]+)"\)\s*==\s*0\)\s*return\s+(RCPP_ARCH_[A-Z0-9_]+)')
ENUM = re.compile(r'(RCPP_ARCH_[A-Z0-9_]+)\s*=\s*\d+')

# The meta-target a family may declare instead of one token.
MIXED = "MIXED"

STATUSES = {"validated", "mapped-unvalidated", "documented-limitation",
            "in_progress", "pending"}

# Floors. A parse that finds nothing must not pass: these are the numbers the
# file carries today (32 families, 95 arch strings), set low enough that adding
# or retiring a family does not need this file edited.
MIN_FAMILIES = 20
MIN_STRINGS = 50


def load_header():
    """{alias: token} first-match-wins, plus the set of tokens that exist."""
    aliases, tokens = {}, set()
    with open(HEADER, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = ALIAS.search(line)
            if m:
                aliases.setdefault(m.group(1), m.group(2))
            e = ENUM.search(line)
            if e:
                tokens.add(e.group(1))
    return aliases, tokens


def resolve(arch, aliases):
    """The census path: the raw string, then the stripped form."""
    if arch in aliases:
        return aliases[arch], arch
    stripped = arch.replace("_", "").replace("-", "")
    if stripped in aliases:
        return aliases[stripped], stripped
    return "RCPP_ARCH_UNKNOWN", None


def main():
    for path in (MANIFEST, HEADER):
        if not os.path.exists(path):
            print(f"manifest_mapping_selfcheck: cannot determine — missing {path}")
            return 2

    aliases, tokens = load_header()
    with open(MANIFEST, encoding="utf-8") as f:
        families = json.load(f)["families"]

    n_strings = sum(len(f.get("hf_arch_strings", [])) for f in families)
    bad, notes = [], []
    if len(families) < MIN_FAMILIES or n_strings < MIN_STRINGS:
        bad.append(f"parsed {len(families)} families / {n_strings} arch strings — "
                   f"below the floor ({MIN_FAMILIES}/{MIN_STRINGS}), so this check "
                   f"is not looking at the manifest it thinks it is")

    for fam in families:
        name = fam.get("family", "?")
        status = fam.get("status", "")
        target = fam.get("mapping_target", "")

        if status not in STATUSES:
            bad.append(f"{name}: status {status!r} is not one of "
                       f"{sorted(STATUSES)}")

        # The meta-target is not a token and is not meant to be.
        declared = None if target == MIXED else target
        if declared is not None:
            want = "RCPP_ARCH_" + declared.replace("-", "_")
            if want not in tokens:
                bad.append(f"{name}: mapping_target {declared!r} is not a token — "
                           f"no {want} in the enum")

        for arch in fam.get("hf_arch_strings", []):
            got, via = resolve(arch, aliases)
            if got == "RCPP_ARCH_UNKNOWN":
                bad.append(f"{name}: {arch!r} resolves to nothing — it is counted "
                           f"uncovered while this manifest calls it a family")
            elif declared is not None and got != "RCPP_ARCH_" + declared.replace("-", "_"):
                notes.append(f"{name}: {arch!r} resolves to "
                             f"{got.split('_', 2)[2]} (via {via}), manifest says {declared}")

    for n in notes:
        print(f"  note  {n}")
    if notes:
        print(f"  ({len(notes)} declared mapping(s) differ from what the header "
              f"resolves — a judgement per row, not a failure; see #2501)")

    if bad:
        print(f"manifest_mapping_selfcheck: FAIL — {len(bad)} finding(s)")
        for b in bad:
            print(f"  - {b}")
        return 1

    print(f"manifest_mapping_selfcheck: PASS — {len(families)} families, "
          f"{n_strings} arch strings, every declared target is a token")
    return 0


if __name__ == "__main__":
    sys.exit(main())
