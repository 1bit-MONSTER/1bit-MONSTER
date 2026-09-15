#!/usr/bin/env python3
"""seo_claim_selfcheck.py — published coverage claims must equal the census.

`scripts/seo_sync.py` rewrites these numbers in the daily apply workflows, but it
is not a gate: nothing checked the CONTENT of the pages, and four separate
false-claim shapes survived for months because a wording was not in its pattern
list (#2389, #2392, #2394, #2397). This checks the published surface —
`site/*.html` and `README.md` — against the same measured facts seo_sync uses, so
that class of bug fails in review instead of drifting.

Scope is deliberate. `docs/` carries dated research notes and the engineering
journal, whose numbers record what was true then; they are not claims about now.
And seo_sync's own probe tests whether its *patterns* match; this tests whether
the *pages* do.

Usage:
    python3 Testing/seo_claim_selfcheck.py [--site DIR] [--readme FILE]
"""
import argparse
import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import seo_sync  # noqa: E402

# (fact, regex): group 1 is the published number and must be one of that fact's
# measured spellings. `(?<![\d,/])` keeps a ratio's denominator
# ("323,996/324,126 checkpoints mapped") from being read as a bare count — the
# mistake that once published "323,579/323,579 checkpoints mapped" on every page.
CLAIMS = [
    ("tokens",  r"(?<![\d,/])(\d[\d,]*) architecture tokens"),
    ("arch",    r"(?<![\d,/])(\d[\d,]*) (?:HF |HuggingFace )?arch strings"),
    ("covered", r"(?<![\d,/])(\d[\d,]*) arch-bearing checkpoints"),
    ("covered", r"(?<![\d,/])(\d[\d,]*) checkpoints mapped"),
    ("covered", r"(?<![\d,/])(\d[\d,]*) checkpoints map to"),
    ("covered", r"(?<![\d,/])(\d[\d,]*) checkpoints \u00b7"),
    ("tokens",  r"(?<![\d,/])(\d[\d,]*) tokens resolve to one engine"),
    ("either",  r"(?<![\d,/])(\d[\d,]*) of them"),
    ("pct",     r"(\d+(?:\.\d+)?)% HuggingFace[ a-zA-Z]*coverage"),
    ("pct",     r"(\d+(?:\.\d+)?)% of HuggingFace's arch-bearing checkpoints"),
    ("pct",     r"(\d+(?:\.\d+)?)% coverage"),
    # Canonical docs (#2408): docs/wiki/models.md, docs/model-families/README.md
    # and docs/CODEBASE.md state the same facts in their own words.
    ("arch",    r"(?<![\d,/])(\d[\d,]*) HF `architectures` strings"),
    ("tokens",  r"(?<![\d,/])(\d[\d,]*) engine arch tokens"),
    ("tokens",  r"(?<![\d,/])(\d[\d,]*) engine tokens"),
    ("total",   r"(?<![\d,/])(\d[\d,]*) text-gen checkpoints"),
    ("total",   r"(?<![\d,/])(\d[\d,]*) text-generation checkpoints\*\*, of which"),
    ("with_arch", r"(?<![\d,/])(\d[\d,]*) declare an `architectures` field"),
    ("with_arch", r"(?<![\d,/])(\d[\d,]*) arch-bearing text-gen checkpoints\*\* remain"),
    ("pct",     r"(\d+(?:\.\d+)?)%\) map to an engine token"),
]

# Ratios: groups 1 and 2 are the numerator and denominator of the census ratio.
RATIOS = [
    r"(?<![\d,/])(\d[\d,]*) / (\d[\d,]*) text-generation checkpoints",
    r"(?<![\d,/])(\d[\d,]*) / (\d[\d,]*) checkpoints mapped",
    r"(?<![\d,/])(\d[\d,]*) / (\d[\d,]*) arch-bearing text-gen checkpoints",
    r"(?<![\d,/])(\d[\d,]*) / (\d[\d,]*) \(\d+(?:\.\d+)?%\) map to an engine token",
]

# The front-page hero pairs three facts in one sentence (#2399). A *global*
# "N backends" rule would flag the Lemonade posts' version-scoped "14/15
# backends" -- statements about the SDK, not the engine -- so the pairing is
# matched exactly instead.
PAIRS = [
    ("families_backends", r"(\d[\d,]*) families, (\d[\d,]*) backends"),
]


def _line(text, pos):
    return text.count("\n", 0, pos) + 1


def _ctx(text, m):
    return " ".join(text[max(0, m.start() - 55):m.end() + 25].split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default=os.path.join(ROOT, "site"))
    ap.add_argument("--readme", default=os.path.join(ROOT, "README.md"))
    args = ap.parse_args()

    tokens = seo_sync.count_tokens()
    arch = seo_sync.count_arch_strings()
    covered, with_arch = seo_sync.census_coverage()
    total = seo_sync.census_total()
    pct = seo_sync._pct(covered, with_arch)

    allowed = {
        "tokens": {str(tokens), f"{tokens:,}"},
        "arch": {str(arch), f"{arch:,}"},
        "covered": {str(covered), f"{covered:,}"},
        "with_arch": {str(with_arch), f"{with_arch:,}"},
        "either": {str(tokens), f"{tokens:,}", str(arch), f"{arch:,}"},
        "pct": {pct, pct.rstrip("%")},
        "total": {str(total), f"{total:,}"},
    }

    families = seo_sync.count_families()
    backends = seo_sync.count_backends()
    pair_facts = {}
    if families is not None and backends is not None:
        pair_facts["families_backends"] = (f"{families:,}", f"{backends:,}")

    files = sorted(glob.glob(os.path.join(args.site, "*.html")))
    if os.path.exists(args.readme):
        files.append(args.readme)
    # The canonical documents that state the same facts; seo_sync rewrites them
    # in the daily sweep, and this makes a stale one fail at PR time (#2408).
    # site/search-index.json embeds chunks of the pages' text, so it carries the
    # same claims; it is generated off-CI and went stale for two days (#2411).
    for rel in ("docs/wiki/models.md", "docs/model-families/README.md",
                "docs/CODEBASE.md", "site/search-index.json"):
        p = os.path.join(ROOT, rel)
        if os.path.exists(p):
            files.append(p)

    checked = 0
    bad = []
    for path in files:
        text = open(path, encoding="utf-8", errors="replace").read()
        rel = os.path.relpath(path, ROOT)
        for fact, pat in CLAIMS:
            for m in re.finditer(pat, text):
                checked += 1
                if m.group(1) in allowed[fact]:
                    continue
                bad.append("%s:%d: %r — expected one of %s\n      …%s…"
                           % (rel, _line(text, m.start()), m.group(0),
                              sorted(allowed[fact]), _ctx(text, m)))
        for pat in RATIOS:
            for m in re.finditer(pat, text):
                checked += 1
                if (m.group(1) in allowed["covered"]
                        and m.group(2) in allowed["with_arch"]):
                    continue
                bad.append("%s:%d: %r — expected %s / %s\n      …%s…"
                           % (rel, _line(text, m.start()), m.group(0),
                              f"{covered:,}", f"{with_arch:,}", _ctx(text, m)))
        for fact, pat in PAIRS:
            want = pair_facts.get(fact)
            if want is None:
                continue
            for m in re.finditer(pat, text):
                checked += 1
                if ([g.replace(",", "") for g in m.groups()]
                        == [w.replace(",", "") for w in want]):
                    continue
                bad.append("%s:%d: %r — expected %s\n      …%s…"
                           % (rel, _line(text, m.start()), m.group(0),
                              " families, ".join(want) + " backends", _ctx(text, m)))

    if bad:
        print("seo_claim_selfcheck: FAIL — %d published claim(s) do not match the "
              "census (checked %d in %d file(s))" % (len(bad), checked, len(files)))
        for b in bad:
            print("  - " + b)
        return 1

    print("seo_claim_selfcheck: PASS — %d claim(s) in %d file(s) match the census "
          "(tokens=%d arch_strings=%d coverage=%d/%d = %s)"
          % (checked, len(files), tokens, arch, covered, with_arch, pct))
    return 0


if __name__ == "__main__":
    sys.exit(main())
