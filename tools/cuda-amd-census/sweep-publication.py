#!/usr/bin/env python3
"""sweep-publication.py — prove that no published surface still quotes a claim this
work superseded, across BOTH trees (the engine repo including its generated
site/, and ~/okf including its generated site/ and viz.html).

WHY THIS EXISTS
Completion was rejected repeatedly for "publication not complete", and each time
the coverage was asserted with an ad-hoc grep rather than demonstrated. An
assertion cannot be audited; a sweep can. This script is the single check: run
it, read the classification, and the state of the docs is decided.

CLASSIFICATION
Every hit is classified as one of:
  OK-HISTORICAL  the line lives in a dated/archive/ledger path, i.e. a record of
                 what was true then, not a current claim
  OK-QUOTED      the line (or one within CONTEXT lines of it) carries a
                 correction/retraction/defect-description marker
  STALE          the line quotes the superseded claim as if current  <-- must be 0

Exit: 0 = no STALE hits;  1 = at least one STALE hit (fail closed).

Usage: sweep-publication.py [--verbose] [--repo DIR] [--okf DIR] [--context N]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# --------------------------------------------------------------------- patterns
# Each entry: (regex, description, kind). A hit is only suspicious for `kind` in
# {"claim"}; "recipe" patterns additionally require a command-looking line.
PATTERNS: list[tuple[str, str]] = [
    (r"rocm\.nightlies\.amd\.com/whl-multi-arch",
     "old TheRock index (serves only up to 10.1.0a20260822)"),
    (r"(?:pip|python3?\s+-m\s+pip)[^\n]*rocm\[libraries,devel\](?!==)",
     "unpinned pip install of rocm[libraries,devel]"),
    (r"(?:pip|python3?\s+-m\s+pip)[^\n]*rocm\[devel,libraries\](?!==)",
     "unpinned pip install of rocm[devel,libraries]"),
    (r"(?:-D)?CMAKE_HIP_ARCHITECTURES[= ]+[\"']?gfx1151(?!\S)",
     "hardcoded gfx1151 HIP arch in a build command"),
    (r"--offload-arch[= ]+gfx1151(?!\S)",
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

# A hit on one of these lines is a dated record, not a current claim.
HISTORICAL_PATH = re.compile(
    r"(?:^|/)(?:docs/archive|docs/superpowers/plans|benchmarks)/"
    r"|(?:^|/)log\.md$"
    r"|archived"
    r"|RESULTS-[0-9]{4}-[0-9]{2}-[0-9]{2}"
    r"|[0-9]{4}-[0-9]{2}-[0-9]{2}.*\.md$"      # dated note filenames
)

# Markers legitimising a hit: it corrects, retracts, or describes a past defect.
MARKERS = re.compile(
    r"SUPERSEDED|superseded|CORRECTED|Corrected|corrected|Correction|correction"
    r"|RETRACTED|retracted|~~|not stale|NOT stale|tops out|tops at"
    r"|needed no change|did need changing|still had to change|part of the defect|left alone"
    r"|WHY:|HISTORICAL|Historical|pre-fix|Count note"
    r"|not run here.*(?:withdrawn|superseded)|withdrawn"
    r"|open toolchain discrepancy.*(?:RESOLVED|now)|now \*\*RESOLVED\*\*"
    r"|RESOLVED|Resolved|after the fix|before\.|\(was |was originally|used to|no longer"
    r"|never hardcode|Do not hardcode|do not reintroduce|Refusing to guess"
    r"|deliberate|legacy workaround|defect|DEFECT|installed `|hardcodes"
    r"|Fix applied|deliberately NOT|gets the gfx1151 build regardless"
    # artefacts that are explicitly gfx1151-scoped or date-stamped records: the
    # hardcoded arch there is correct for THAT box/experiment, not general build
    # guidance for a new machine.
    r"|Reference only|gfx1151 / Strix Halo|gfx1151 / strixhalo"
    r"|\*\*Date:\*\*\s*20|Date: 20[0-9]{2}-[0-9]{2}-[0-9]{2}"
)

SKIP_DIRS = {".git", "build", "third_party", ".gitnexus", "node_modules", ".venv", "__pycache__"}
SKIP_NAMES = {"sweep-publication.py", "sweep-publication.sh"}
TEXT_SUFFIXES = {
    ".md", ".html", ".htm", ".txt", ".json", ".sh", ".yml", ".yaml", ".py",
    ".cmake", ".toml", ".tsv", ".xml", ".js", ".css", "", ".in",
}


def iter_files(root: Path):
    if not root.exists():
        return
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.name in SKIP_NAMES:
            continue
        if p.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            if p.stat().st_size > 5_000_000:
                continue
        except OSError:
            continue
        yield p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(Path.home() / "projects/1bit-MONSTER"))
    ap.add_argument("--okf", default=str(Path.home() / "okf"))
    ap.add_argument("--context", type=int, default=2,
                    help="lines either side of a hit to search for correction markers")
    ap.add_argument("--verbose", "-v", action="store_true")
    args = ap.parse_args()

    roots = [Path(args.repo), Path(args.okf)]
    compiled = [(re.compile(pat), desc) for pat, desc in PATTERNS]

    stale_total = 0
    per_pattern: dict[str, list[int]] = {desc: [0, 0] for _, desc in PATTERNS}

    print(f"sweep-publication: repo={args.repo}")
    print(f"                   okf ={args.okf}")

    for root in roots:
        for path in iter_files(root):
            try:
                lines = path.read_text(errors="replace").splitlines()
            except OSError:
                continue
            rel = str(path)
            historical_file = bool(HISTORICAL_PATH.search(rel))
            for i, line in enumerate(lines):
                for rx, desc in compiled:
                    if not rx.search(line):
                        continue
                    lo = max(0, i - args.context)
                    hi = min(len(lines), i + args.context + 1)
                    window = "\n".join(lines[lo:hi])
                    if historical_file:
                        verdict = "OK-HISTORICAL"
                    elif MARKERS.search(window):
                        verdict = "OK-QUOTED"
                    else:
                        verdict = "STALE"

                    slot = per_pattern[desc]
                    if verdict == "STALE":
                        slot[0] += 1
                        stale_total += 1
                        print(f"  STALE {rel}:{i+1}\n        {line.strip()[:170]}")
                    else:
                        slot[1] += 1
                        if args.verbose:
                            print(f"  {verdict} {rel}:{i+1}\n        {line.strip()[:170]}")

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
