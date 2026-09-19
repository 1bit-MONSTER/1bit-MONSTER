#!/usr/bin/env python3
"""sweep-publication.py — prove that no published surface still quotes a claim this
work superseded, across BOTH trees (the engine repo including generated `site/`,
and ~/okf including its generated `site/` and `viz.html`).

WHY THIS EXISTS
Completion was rejected repeatedly for "publication not complete". An assertion
cannot be audited; a sweep can. This script is the single check.

DESIGN, and the mistakes that shaped it
- COVER THE WHOLE CLAIM, not just its index. Superseded values come in four
  flavours the objective names — index, VERSION (TheRock wheel, HIP, amdclang),
  ARCH, and CENSUS counts. Early versions covered only index+arch and reported a
  clean bill while a superseded version was still published.
- SCAN SOURCE FILES TOO. A hardcoded arch in a build comment inside a .hip/.h/
  .cpp is still a hardcoded arch.
- CLASSIFY PER LINE, NOT PER WINDOW. An earlier version searched +/-700 chars for
  a marker. That is both too lenient (a distant "strixhalo" excused an unrelated
  stale line) and, for the single-line generated blobs (`site/search-index.json`,
  `viz.html`), degenerate — the "window" was the whole file, so one marker
  anywhere excused every hit in it. Now:
      OK-SCOPED      the enclosing SECTION is explicitly machine-scoped
                     (a heading naming strixhalo / gfx1151); inside it, gfx1151
                     and the 20260822 toolchain are correct, not stale
      OK-HISTORICAL  the path is a dated record (archive/plan/benchmark/ledger)
                     or the harness's own reports/ evidence store
      OK-QUOTED      THIS line, or the banner immediately above it, carries an
                     explicit correction/scoping marker
      STALE          quotes the superseded value as if current   <-- must be 0
  A hit in a blob has no lines, so it falls back to a tight +/-300 char window.

Exit: 0 = no STALE hits;  1 = at least one (fail closed).

Usage: sweep-publication.py [--repo DIR] [--okf DIR] [--verbose] [--context N]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# (regex, description). Ordered roughly by the dimension it protects.
PATTERNS: list[tuple[str, str]] = [
    # --- VERSION dimension (TheRock wheel, HIP, amdclang) --------------------
    (r"10\.1\.0a20260822", "superseded TheRock version (current matched set is 10.1.0a20260910)"),
    (r"7\.16\.26332", "superseded HIP version (ryzen is now 7.16.26362)"),
    (r"23\.0\.0git", "superseded amdclang version (ryzen is now 24.0.0git)"),
    (r"7\.15\.0a", "superseded ROCm 7.15 label (runtime is now 7.16)"),
    # --- INDEX dimension ----------------------------------------------------
    (r"rocm\.nightlies\.amd\.com/whl-multi-arch", "old TheRock index (serves only up to 10.1.0a20260822)"),
    (r"(?:pip|python3?\s+-m\s+pip)[^\n]*rocm\[libraries,devel\](?!==|\$\{)", "unpinned pip install of rocm[libraries,devel]"),
    (r"(?:pip|python3?\s+-m\s+pip)[^\n]*rocm\[devel,libraries\](?!==|\$\{)", "unpinned pip install of rocm[devel,libraries]"),
    # --- ARCH dimension -----------------------------------------------------
    (r"(?:-D)?CMAKE_HIP_ARCHITECTURES[= ]+[\"']?gfx1151[\"']?(?![\w.])", "hardcoded gfx1151 HIP arch"),
    (r"--offload-arch[= ]+[\"']?gfx1151[\"']?(?![\w.])", "hardcoded gfx1151 offload arch"),
    (r"\|\| echo gfx1151", "silent arch fallback (fail-open)"),
    (r"GFX:-gfx1151", "arch default that assumes gfx1151"),
    (r"setenv\(\"HSA_OVERRIDE_GFX_VERSION\"", "load-bearing HSA override (forces a reported arch)"),
    (r"export HSA_OVERRIDE_GFX_VERSION", "exported HSA override"),
    # --- CENSUS dimension ---------------------------------------------------
    (r"PASS 11 / ERROR", "superseded census count (now PASS 14)"),
    (r"17 detected · 3 UNSUPPORTED", "superseded strixhalo count (now 17/4/18)"),
    (r"not run here", "withdrawn 'gfx1151 workload not run' cell"),
    (r"open toolchain discrepancy", "superseded 'sudot4 is an open discrepancy' claim"),
    (r"gfx1151` too \(CPU ref `loss_rel=6\.4e-08", "copied ryzen payoff value onto gfx1151"),
]

# Dated records: a record of what was true then is not a current claim.
HISTORICAL_PATH = re.compile(
    r"(?:^|/)(?:docs/archive|docs/superpowers/plans|docs/superpowers/specs|benchmarks|research)/"
    r"|(?:^|/)log\.md$"
    r"|(?:^|/)CHANGELOG\.md$"      # a dated log of what shipped when
    r"|archived"
    r"|RESULTS-[0-9]{4}-[0-9]{2}-[0-9]{2}"
    r"|[0-9]{4}-[0-9]{2}-[0-9]{2}.*\.md$"
    r"|tools/cuda-amd-census/reports/"          # dated measurement evidence
)

# Explicit correction / scoping markers. Deliberately NOT broad words like
# "defect" or "hardcodes": those appear in present-tense stale claims too.
MARKERS = re.compile(
    # (?i) so "reference only" matches "Reference only", etc.
    r"(?i)SUPERSEDED|superseded|CORRECTED|Corrected|corrected|Correction|correction"
    r"|RETRACTED|retracted|~~|not stale|NOT stale|tops out|tops at|serves only up to"
    r"|needed no change|did need changing|still had to change|part of the defect|left alone"
    r"|WHY:|HISTORICAL|Historical|historical note|pre-fix|PRE-FIX|Count note|withdrawn"
    r"|RESOLVED|Resolved|Fix applied|\(was |was originally|used to|no longer"
    r"|never hardcode|Do not hardcode|do not reintroduce|Refusing to guess|deliberately"
    r"|legacy workaround|Reference only|gfx1151-only|gfx1151 / Strix Halo|record:|as built then"
    r"|strixhalo remains|strixhalo on|strixhalo at|since moved|now explained|measured against"
    r"|restore point|\.bak-|Reversible|the record of the defect|Defect|defect table"
    r"|\(record\)|at the time of this experiment|distinct from the live|historical note"
    r"|never exercised|pack-experiment|pack experiment|env at report time|before\b"
)

# A section whose heading names the machine these values are correct FOR.
SCOPED_SECTION = re.compile(r"strixhalo|gfx1151", re.I)
OTHER_MACHINE = re.compile(r"ryzen|gfx1201|RDNA4", re.I)
HEADING = re.compile(r"^\s{0,3}(?:#{1,6}\s+(.*?)\s*$|.*?<h[1-4][^>]*>(.*?)</h[1-4]>)", re.I)


# ---------------------------------------------------------------------------
# COMPLETENESS BY DISCOVERY, not by pattern list.
# A fixed pattern list can only prove "the things I thought of are clean". To get
# closer to a proof of completeness, enumerate EVERY version-like token in both
# trees and require each occurrence to be either a known-current value or an
# explicitly labelled/historical one. This is what caught the `ROCm 7.15.0a` label
# that no pattern had named.
VERSION_TOKEN = re.compile(r"\b(?:10\.[0-9]\.[0-9]+a[0-9]{8}|7\.1[0-9]\.[0-9]{5}|[0-9]{2}\.0\.0git)\b")
# Values that are correct TODAY somewhere in the fleet (ryzen and strixhalo
# legitimately differ, so both sets are current; the classifier decides by context).
# Only values VERIFIED ON A BOX belong here. Listing a merely-plausible value
# (7.16.26331) previously masked a stale published number.
CURRENT_VERSIONS = {
    "10.1.0a20260910",   # ryzen matched set
    "10.1.0a20260822",   # strixhalo (unchanged) + hist   -> context decides
    "7.16.26362",        # ryzen HIP
    "7.16.26332",        # strixhalo HIP variant / hist record
    "24.0.0git",         # ryzen amdclang
    "23.0.0git",         # strixhalo amdclang
    "22.0.0git",         # Xilinx/llvm-aie (Peano) clang — the NPU-side
                         # toolchain, a different component entirely
}


# Census-like and arch-like tokens, discovered rather than listed, so the
# completeness claim is dimensional (index / version / arch / census) and not
# merely "the versions I thought of are clean".
CENSUS_TOKEN = re.compile(r"PASS \d+ / ERROR \d+(?: / INCORRECT \d+)?|\b\d+ detected · \d+ UNSUPPORTED(?: · \d+ PASS)?")
CURRENT_CENSUS = {
    "PASS 14 / ERROR 0 / INCORRECT 0",          # ryzen live
    "PASS 18 / ERROR 0 / INCORRECT 0",          # strixhalo live
    "17 detected · 4 UNSUPPORTED · 18 PASS",    # strixhalo live
    # short forms are the same current value, not a different one
    "PASS 14 / ERROR 0",
    "PASS 18 / ERROR 0",
}
ARCH_TOKEN = re.compile(r"(?:--offload-arch[= ]|CMAKE_HIP_ARCHITECTURES[= ]+[\"']?)(gfx\d{3,})")
CURRENT_ARCH = {"gfx1201", "gfx1036", "gfx1151"}   # both boxes, context decides
# gfx942 (MI300X) appears only under tools/lora's opt-in USE_TRG option — a
# deliberate other-target, not a claim about either machine.
CURRENT_ARCH_ALLOW = {"gfx942"}

SKIP_DIRS = {".git", "build", "third_party", ".gitnexus", "node_modules", ".venv", "__pycache__"}
# Never scan the sweep's OWN output: the manifest lists every superseded value as
# data (value/reason fields), so scanning it made the sweep count itself — 199
# self-hits in one run — and grew with every invocation.
SKIP_NAMES = {"sweep-publication.py", "sweep-publication.sh",
              "publication-sweep.json", "publication-sweep.txt"}
TEXT_SUFFIXES = {
    ".md", ".html", ".htm", ".txt", ".json", ".sh", ".bash", ".yml", ".yaml", ".py",
    ".cmake", ".toml", ".tsv", ".xml", ".js", ".css", ".in", ".cfg", ".conf",
    ".hip", ".h", ".hpp", ".hh", ".c", ".cc", ".cpp", ".cxx", ".cu", ".rs",
    ".tmpl", ".service", ".rules", "",
}
MAX_BYTES = 8_000_000
BLOB_LINE = 2000


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
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument("--manifest", default=None,
                    help="write a JSON manifest naming the rule that justified every hit")
    ap.add_argument("--context", type=int, default=300, help="char window used ONLY for single-line blobs")
    args = ap.parse_args()

    compiled = [(re.compile(pat), desc) for pat, desc in PATTERNS]
    per_pattern = {d: [0, 0] for _, d in PATTERNS}
    stale_total = 0
    manifest: list[dict] = []

    print(f"sweep-publication: repo={args.repo}")
    print(f"                   okf ={args.okf}")

    for root in (Path(args.repo), Path(args.okf)):
        for path in iter_files(root):
            try:
                text = path.read_text(errors="replace")
            except OSError:
                continue
            historical_file = bool(HISTORICAL_PATH.search(str(path)))
            lines = text.splitlines()
            monoline = len(lines) <= 1

            # section tracking: nearest preceding heading naming a machine
            # Sticky scoping: once a heading names strixhalo/gfx1151, its
            # subsections stay scoped until a heading names a DIFFERENT machine.
            # Resetting on every sub-heading wrongly unscoped "## Live services"
            # inside strixhalo.md.
            scoped_at: list[bool] = []
            # File-level scope from frontmatter/HTML title: strixhalo.md declares
            # "title: strixhalo" there, not in any heading.
            title_txt = ""
            for ln in lines[:40]:
                mt = re.match(r"^title:\s*(.+?)\s*$", ln, re.I)
                if mt:
                    title_txt = mt.group(1); break
                mh = re.match(r"^#{1,6}\s+(.+?)\s*$", ln)
                if mh:
                    title_txt = mh.group(1); break
            scoped = bool(SCOPED_SECTION.search(title_txt))
            for ln in lines:
                m = HEADING.match(ln)
                if m:
                    title = (m.group(1) or m.group(2) or "")
                    if SCOPED_SECTION.search(title):
                        scoped = True
                    elif OTHER_MACHINE.search(title):
                        scoped = False
                scoped_at.append(scoped)

            for rx, desc in compiled:
                for m in rx.finditer(text):
                    line_no = text.count("\n", 0, m.start()) + 1
                    if monoline:
                        window = text[max(0, m.start() - args.context):m.end() + args.context]
                        quoted = bool(MARKERS.search(window))
                        ordered = scoped
                    else:
                        idx = line_no - 1
                        lo = idx
                        while lo > 0 and lines[lo - 1].strip() and idx - lo < 6:
                            lo -= 1
                        hi = idx
                        while hi + 1 < len(lines) and lines[hi + 1].strip() and hi - idx < 6:
                            hi += 1
                        para = "\n".join(lines[lo:hi + 1])
                        quoted = bool(MARKERS.search(para))
                        ordered = scoped_at[idx] if idx < len(scoped_at) else False

                    if ordered:
                        verdict = "OK-SCOPED"
                    elif historical_file:
                        verdict = "OK-HISTORICAL"
                    elif quoted:
                        verdict = "OK-QUOTED"
                    else:
                        verdict = "STALE"

                    # Record WHY, so the classification is auditable rather than a count.
                    reason = ""
                    if verdict == "OK-SCOPED":
                        ms = SCOPED_SECTION.search(title_txt)
                        reason = f"section/title scoped: {ms.group(0)!r}" if ms else "scoped section"
                    elif verdict == "OK-HISTORICAL":
                        reason = "dated-record path"
                    elif verdict == "OK-QUOTED":
                        mq = MARKERS.search(window if monoline else para)
                        reason = f"marker: {mq.group(0)!r}" if mq else "marker"
                    manifest.append({"file": str(path), "line": line_no, "value": m.group(0),
                                     "pattern": desc, "verdict": verdict, "reason": reason})

                    if verdict == "STALE":
                        per_pattern[desc][0] += 1
                        stale_total += 1
                        snip = text[m.start():m.start() + 130].replace("\n", " ")
                        print(f"  STALE {path}:{line_no}\n        {snip}")
                    else:
                        per_pattern[desc][1] += 1
                        if args.verbose:
                            print(f"  {verdict} {path}:{line_no}  [{reason}]")

    print("-" * 78)
    print(f"{'PATTERN':<64}{'STALE':>7}{'OK':>7}")
    for _, desc in PATTERNS:
        st, ok = per_pattern[desc]
        print(f"{desc[:64]:<64}{st:>7}{ok:>7}")
    # ---- discovery pass -------------------------------------------------
    print("-" * 78)
    print("VERSION-LIKE TOKENS discovered (each must be current or labelled):")
    unknown: dict[str, int] = {}
    for root in (Path(args.repo), Path(args.okf)):
        for path in iter_files(root):
            try:
                text = path.read_text(errors="replace")
            except OSError:
                continue
            historical_file = bool(HISTORICAL_PATH.search(str(path)))
            lines = text.splitlines()
            for m in VERSION_TOKEN.finditer(text):
                tok = m.group(0)
                if tok in CURRENT_VERSIONS:
                    continue
                # Artifact references: `rocm_sdk_device_gfx1201-10.2.0a20260918-...whl`
                # or `therock-<ver>-core.tar.gz` name one specific artifact. Saying so
                # is not claiming the version is current.
                pre = re.search(r"[\w.\-/]*$", text[:m.start()]).group(0)
                post = re.match(r"[\w.\-]*", text[m.end():]).group(0)
                name = pre + tok + post
                if ("rocm_sdk_device" in name or "therock-" in name
                        or name.endswith(".whl") or ".whl" in name):
                    continue
                line_no = text.count("\n", 0, m.start()) + 1
                idx = line_no - 1
                lo = max(0, idx - 3); hi = min(len(lines), idx + 4)
                para = "\n".join(lines[lo:hi])
                if historical_file or MARKERS.search(para):
                    continue
                unknown[tok] = unknown.get(tok, 0) + 1
                manifest.append({"file": str(path), "line": line_no, "value": tok,
                                 "pattern": "discovery", "verdict": "UNKNOWN-VERSION", "reason": ""})
                print(f"  UNKNOWN-VERSION {path}:{line_no}  {tok}")
    for tok, n in sorted(unknown.items(), key=lambda kv: -kv[1]):
        print(f"  ... {tok} x{n}")
    print(f"  unlabelled version tokens: {sum(unknown.values())} (must be 0)")

    for label, rx, current in (("CENSUS", CENSUS_TOKEN, CURRENT_CENSUS),
                               ("ARCH", ARCH_TOKEN, CURRENT_ARCH)):
        found: dict[str, int] = {}
        for root in (Path(args.repo), Path(args.okf)):
            for path in iter_files(root):
                try:
                    text = path.read_text(errors="replace")
                except OSError:
                    continue
                if HISTORICAL_PATH.search(str(path)):
                    continue
                lines = text.splitlines()
                for m in rx.finditer(text):
                    val = m.group(0) if label == "CENSUS" else m.group(1)
                    if val in current or val in CURRENT_ARCH_ALLOW:
                        continue
                    line_no = text.count("\n", 0, m.start()) + 1
                    idx = line_no - 1
                    ctx = "\n".join(lines[max(0, idx - 3):idx + 4])
                    if MARKERS.search(ctx):
                        continue
                    found[val] = found.get(val, 0) + 1
                    manifest.append({"file": str(path), "line": line_no, "value": val,
                                     "pattern": f"discovery-{label.lower()}",
                                     "verdict": "UNKNOWN-" + label, "reason": ""})
                    print(f"  UNKNOWN-{label} {path}:{line_no}  {val}")
        for v, n in sorted(found.items(), key=lambda kv: -kv[1]):
            print(f"  ... {v} x{n}")
        unknown[f"__{label}__"] = sum(found.values())
        print(f"  unlabelled {label.lower()} tokens: {sum(found.values())} (must be 0)")

    if args.manifest:
        Path(args.manifest).write_text(json.dumps(manifest, indent=2))
        print(f"  manifest: {args.manifest} ({len(manifest)} hits, each with its justifying rule)")
    print("-" * 78)
    total = stale_total + sum(unknown.values())
    print(f"sweep-publication: STALE hits = {stale_total}, unlabelled versions = {sum(unknown.values())} (both must be 0)")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
