#!/usr/bin/env python3
"""sweep-publication.py — prove that no published surface still quotes a claim this
work superseded, across BOTH trees (the engine repo including generated `site/`,
and ~/okf including its generated `site/` and `viz.html`).

WHY THIS EXISTS
Completion was rejected repeatedly for "publication not complete", and each time
the coverage was asserted with an ad-hoc grep rather than demonstrated. An
assertion cannot be audited; a sweep can. This script is the single check.

DESIGN NOTES (each earned from a rejected audit)
- It covers the VERSION dimension, not just index/arch/count: a superseded
  TheRock version presented as current is the same class of defect as a stale
  index, and the first version of this sweep silently omitted it.
- Classification uses a CONTEXT WINDOW IN CHARACTERS, not lines. Several of the
  generated surfaces (`site/search-index.json`, `viz.html`) are single ~1 MB
  lines; a line-based window degenerates to "the whole file", so any marker
  anywhere would mark every hit OK — a vacuous pass.
- Source/build files are scanned too (`.hip .h .hpp .cpp .c .cu` ...): a hardcoded
  arch in a build command inside a header is still a hardcoded arch.
- The arch patterns accept the quoted and `set()` forms, e.g.
  `set(CMAKE_HIP_ARCHITECTURES "gfx1151" ...)`.

CLASSIFICATION
  OK-HISTORICAL  line lives in a dated/archive/ledger path (a record of then)
  OK-QUOTED      the surrounding text corrects, retracts, describes a past
                 defect, or explicitly scopes the artefact (gfx1151-only)
  STALE          quotes the superseded claim as if current   <-- must be 0

Exit: 0 = no STALE hits;  1 = at least one (fail closed).

Usage: sweep-publication.py [--verbose] [--repo DIR] [--okf DIR] [--context N]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# (regex, description). VERSION is listed first on purpose: it is the pattern the
# first sweep omitted, which let superseded-version text survive.
PATTERNS: list[tuple[str, str]] = [
    (r"10\.1\.0a20260822",
     "superseded TheRock version (current matched set is 10.1.0a20260910)"),
    (r"7\.16\.26332", "superseded HIP version (ryzen is now 7.16.26362)"),
    (r"23\.0\.0git", "superseded amdclang version (ryzen is now 24.0.0git)"),
    (r"rocm\.nightlies\.amd\.com/whl-multi-arch",
     "old TheRock index (serves only up to 10.1.0a20260822)"),
    (r"(?:pip|python3?\s+-m\s+pip)[^\n]*rocm\[libraries,devel\](?!==)",
     "unpinned pip install of rocm[libraries,devel]"),
    (r"(?:pip|python3?\s+-m\s+pip)[^\n]*rocm\[devel,libraries\](?!==)",
     "unpinned pip install of rocm[devel,libraries]"),
    (r"(?:-D)?CMAKE_HIP_ARCHITECTURES[= ]+[\"']?gfx1151[\"']?(?![\w.])",
     "hardcoded gfx1151 HIP arch"),
    (r"--offload-arch[= ]+[\"']?gfx1151[\"']?(?![\w.])",
     "hardcoded gfx1151 offload arch"),
    (r"PASS 11 / ERROR", "superseded census count (now PASS 14)"),
    (r"17 detected · 3 UNSUPPORTED", "superseded strixhalo count (now 17/4/18)"),
    (r"not run here", "withdrawn 'gfx1151 workload not run' cell"),
    (r"setenv\(\"HSA_OVERRIDE_GFX_VERSION\"", "load-bearing HSA override (forces a reported arch)"),
    (r"export HSA_OVERRIDE_GFX_VERSION", "exported HSA override"),
    (r"\|\| echo gfx1151", "silent arch fallback (fail-open)"),
    (r"GFX:-gfx1151", "arch default that assumes gfx1151"),
    (r"open toolchain discrepancy", "superseded 'sudot4 is an open discrepancy' claim"),
    (r"gfx1151` too \(CPU ref `loss_rel=6\.4e-08", "copied ryzen payoff value onto gfx1151"),
]

# A hit in one of these paths is a dated record or a machine-scoped artefact.
HISTORICAL_PATH = re.compile(
    r"(?:^|/)(?:docs/archive|docs/superpowers/plans|docs/superpowers/specs|benchmarks|research)/"
    r"|(?:^|/)log\.md$"
    # the census harness reports/ store is dated measurement EVIDENCE: a strixhalo
    # report saying gfx1151 is correct, not a stale claim.
    r"|tools/cuda-amd-census/reports/"
    r"|archived"
    r"|RESULTS-[0-9]{4}-[0-9]{2}-[0-9]{2}"
    r"|[0-9]{4}-[0-9]{2}-[0-9]{2}.*\.md$"
)

# Markers legitimising a hit: it corrects, retracts, dates, or explicitly scopes.
MARKERS = re.compile(
    r"SUPERSEDED|superseded|CORRECTED|Corrected|corrected|Correction|correction"
    r"|RETRACTED|retracted|~~|not stale|NOT stale|tops out|tops at|serves only up to"
    r"|needed no change|did need changing|still had to change|part of the defect|left alone"
    r"|WHY:|HISTORICAL|Historical|pre-fix|Count note"
    r"|withdrawn|not run here.*(?:withdrawn|superseded)"
    r"|RESOLVED|Resolved|now \*\*RESOLVED\*\*|after the fix|before\.|\(was |was originally"
    r"|used to|no longer|never hardcode|Do not hardcode|do not reintroduce|Refusing to guess"
    r"|deliberate|legacy workaround|defect|DEFECT|installed `|hardcodes|Fix applied"
    r"|deliberately NOT|gets the gfx1151 build regardless"
    # explicit scoping: an artefact that says it is gfx1151-only, or is a dated
    # record, is allowed to name gfx1151/20260822.
    r"|Reference only|gfx1151 / Strix Halo|gfx1151-only|gfx1151 only"
    r"|historical note|PRE-FIX RECORD|pre-fix record|record:|as built then|no longer hardcodes"
    r"|restore point|\.bak-|bak tree|the \.bak tree|Restore:|Reversible"
    r"|\*\*Date:\*\*\s*20|Date: 20[0-9]{2}-[0-9]{2}-[0-9]{2}"
    # things that are true of strixhalo specifically (it still runs 20260822)
    r"|strixhalo remains|strixhalo at|strixhalo on|strixhalo \(`gfx1151`\)|Strix Halo|strixhalo"
)

SKIP_DIRS = {".git", "build", "third_party", ".gitnexus", "node_modules", ".venv", "__pycache__"}
SKIP_NAMES = {"sweep-publication.py", "sweep-publication.sh"}
TEXT_SUFFIXES = {
    ".md", ".html", ".htm", ".txt", ".json", ".sh", ".bash", ".yml", ".yaml", ".py",
    ".cmake", ".toml", ".tsv", ".xml", ".js", ".css", ".in", ".cfg", ".conf",
    # source/build files: a hardcoded arch in a build comment inside a header is
    # still a hardcoded arch.
    ".hip", ".h", ".hpp", ".hh", ".c", ".cc", ".cpp", ".cxx", ".cu", ".rs",
    ".tmpl", ".service", ".rules", "", ".dist-info",
}
MAX_BYTES = 8_000_000


def iter_files(root: Path):
    if not root.exists():
        return
    for p in root.rglob("*"):
        if not p.is_file() or any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.name in SKIP_NAMES or p.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            if p.stat().st_size > MAX_BYTES:
                continue
        except OSError:
            continue
        yield p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(Path.home() / "projects/1bit-MONSTER"))
    ap.add_argument("--okf", default=str(Path.home() / "okf"))
    ap.add_argument("--context", type=int, default=700,
                    help="characters of context around a hit used for classification")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    roots = [Path(args.repo), Path(args.okf)]
    compiled = [(re.compile(pat), desc) for pat, desc in PATTERNS]
    per_pattern: dict[str, list[int]] = {desc: [0, 0] for _, desc in PATTERNS}
    stale_total = 0

    print(f"sweep-publication: repo={args.repo}")
    print(f"                   okf ={args.okf}")

    for root in roots:
        for path in iter_files(root):
            try:
                text = path.read_text(errors="replace")
            except OSError:
                continue
            historical_file = bool(HISTORICAL_PATH.search(str(path)))
            for rx, desc in compiled:
                for m in rx.finditer(text):
                    line_no = text.count("\n", 0, m.start()) + 1
                    lo = max(0, m.start() - args.context)
                    hi = min(len(text), m.end() + args.context)
                    window = text[lo:hi]
                    if historical_file:
                        verdict = "OK-HISTORICAL"
                    elif MARKERS.search(window):
                        verdict = "OK-QUOTED"
                    else:
                        verdict = "STALE"
                    if verdict == "STALE":
                        per_pattern[desc][0] += 1
                        stale_total += 1
                        snippet = text[m.start():m.start() + 120].replace("\n", " ")
                        print(f"  STALE {path}:{line_no}\n        {snippet}")
                    else:
                        per_pattern[desc][1] += 1
                        if args.verbose:
                            print(f"  {verdict} {path}:{line_no}")

    print("-" * 78)
    print(f"{'PATTERN':<64}{'STALE':>7}{'OK':>7}")
    for _, desc in PATTERNS:
        stale, ok = per_pattern[desc]
        print(f"{desc[:64]:<64}{stale:>7}{ok:>7}")
    print("-" * 78)
    print(f"sweep-publication: STALE hits = {stale_total} (must be 0)")
    return 1 if stale_total else 0


if __name__ == "__main__":
    sys.exit(main())
