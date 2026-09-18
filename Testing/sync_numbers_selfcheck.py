#!/usr/bin/env python3
"""sync_numbers_selfcheck.py — the daily numbers writer must survive a page that is
not in the tree, and the daily driver must not name a file that is not there.

Why this exists: tools/sync_numbers.py read site/benchmarks.html unconditionally, and
the 2026-08-22 site redesign (#1780) deleted that file when it renamed the pages to
site/1bit-*.html. packaging/services/daily-benchmark-validate.sh runs the writer
under `set -euo pipefail`, so the FileNotFoundError aborted the daily pipeline at the
step *before* its commit: no PR, no drift issue, nothing published — and site/numbers
.json still carried its 2026-08-18 re-measure stamp (#2587). Nothing caught it
because no selfcheck exercised this script at all.

Every case below copies the real tools/sync_numbers.py into a synthetic tree, so the
assertions are about the script's own behaviour and not about a re-implementation of
it. No NPU, no network, no repo data — a fixture, a temp dir and the stdlib.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WRITER = ROOT / "tools" / "sync_numbers.py"
DRIVER = ROOT / "packaging" / "services" / "daily-benchmark-validate.sh"

KEYS = ("halo_gemv_gbps", "prefill_tflops_4h", "prefill_tflops_i8apre",
        "sherry_gemv_gbps", "tq1_gemv_gbps")
PREFILL = 38.84              # the fixture's measured value
DISPLAY = f"{PREFILL:.1f}"   # what the writer formats into the pages: 38.8

failures: list[str] = []
checks = 0


def check(ok: bool, label: str, detail: str = "") -> None:
    global checks
    checks += 1
    if not ok:
        failures.append(f"{label}{(': ' + detail) if detail else ''}")


def strip_shell_comments(text: str) -> str:
    """The driver's code, without its comments.

    Its own comment explains why site/benchmarks.html is NOT listed, so a presence
    check over the raw text fails on the explanation — measured, that is what the
    first version of case D did. Same trap Testing/run_all.sh's selfcheck tripwire
    documents ("a comment is not an invocation"): assert on the code, not the prose.
    """
    keep = []
    for line in text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        keep.append(line.split(" #")[0] if " #" in line else line)
    return "\n".join(keep)


def make_tree(tmp: Path, name: str, *, index: bool, benchmarks: bool,
              index_anchor: bool = True, bench_anchor: bool = True) -> Path:
    """A throwaway tree with the real writer, a fixture latest.json and numbers.json.

    `name` is explicit rather than derived from the flags: two cases with the same
    shape would otherwise share a directory, and the second would build on the
    first's already-synced numbers.json (measured — case E did).
    """
    tree = tmp / name
    (tree / "tools").mkdir(parents=True)
    (tree / "site").mkdir()
    shutil.copy2(WRITER, tree / "tools" / "sync_numbers.py")

    latest = {
        "benchmarks": {k: 100.0 + i for i, k in enumerate(KEYS)},
        "_sources": {"fixture": "sync_numbers_selfcheck.py"},
    }
    latest["benchmarks"]["prefill_tflops_i8apre"] = PREFILL
    (tree / "benchmarks").mkdir()
    (tree / "benchmarks" / "latest.json").write_text(json.dumps(latest))

    numbers = {"benchmarks": {k: 1.0 for k in KEYS}, "engines": {"keep": {"tok_s": 1}}}
    (tree / "site" / "numbers.json").write_text(json.dumps(numbers))

    if index:
        sentence = ("measured 2026-08-01. Plus 30.0 TFLOPS INT8 prefill."
                    if index_anchor else "no anchor sentence here")
        (tree / "site" / "index.html").write_text(f"<p>{sentence}</p>\n")
    if benchmarks:
        markup = ("<b>30.0</b><span>TFLOPS int8 prefill (WMMA)</span>"
                  if bench_anchor else "<b>30.0</b><span>something else</span>")
        (tree / "site" / "benchmarks.html").write_text(f"<div>{markup}</div>\n")
    return tree


def run(tree: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(tree / "tools" / "sync_numbers.py")],
                          capture_output=True, text=True)


def numbers_of(tree: Path) -> dict:
    return json.loads((tree / "site" / "numbers.json").read_text())


tmp = Path(tempfile.mkdtemp())
try:
    # ── A: the defect — neither page is in the tree ────────────────────────────
    # Before the fix this was exit 1 with FileNotFoundError, and the driver's `set -e`
    # turned that into a dead pipeline. The writer's real job must still happen.
    a = make_tree(tmp, "a", index=False, benchmarks=False)
    ra = run(a)
    check(ra.returncode == 0, "A absent pages: exit 0", f"exit {ra.returncode}: {ra.stderr.strip()[-160:]}")
    check("not in this tree" in ra.stderr, "A absent pages: names the skip", ra.stderr.strip()[-120:])
    check("Traceback" not in ra.stderr, "A absent pages: no traceback", ra.stderr.strip()[-120:])
    an = numbers_of(a)
    check(an["benchmarks"]["prefill_tflops_i8apre"] == PREFILL,
          "A absent pages: numbers.json still synced", str(an["benchmarks"].get("prefill_tflops_i8apre")))
    check(an["benchmarks"]["tflops"] == PREFILL, "A absent pages: tflops mirrors prefill",
          str(an["benchmarks"].get("tflops")))
    check(an["_sources"] == {"fixture": "sync_numbers_selfcheck.py"},
          "A absent pages: _sources copied", str(an.get("_sources")))
    check(an["engines"] == {"keep": {"tok_s": 1}}, "A absent pages: untouched keys survive",
          str(an.get("engines")))

    # ── B: the pages ARE there, with the anchors this script patches ───────────
    # The positive path must keep working: a fix that only stops the crash would
    # otherwise be indistinguishable from "this script no longer patches anything".
    b = make_tree(tmp, "b", index=True, benchmarks=True)
    rb = run(b)
    check(rb.returncode == 0, "B anchored pages: exit 0", f"exit {rb.returncode}")
    idx = (b / "site" / "index.html").read_text()
    check(f"Plus {DISPLAY} TFLOPS INT8 prefill." in idx, "B index.html figure updated", idx.strip())
    check("Plus 30.0 TFLOPS" not in idx, "B index.html old figure gone", idx.strip())
    ben = (b / "site" / "benchmarks.html").read_text()
    check(f"<b>{DISPLAY}</b>" in ben, "B benchmarks.html figure updated", ben.strip())
    check("<b>30.0</b>" not in ben, "B benchmarks.html old figure gone", ben.strip())

    # ── C: the page is there but its anchor is gone (today's real state) ───────
    c = make_tree(tmp, "c", index=True, benchmarks=True, index_anchor=False, bench_anchor=False)
    rc = run(c)
    check(rc.returncode == 0, "C anchorless pages: exit 0", f"exit {rc.returncode}")
    check("anchor sentence not found" in rc.stderr, "C index anchor reported", rc.stderr.strip()[-110:])
    check("anchor not found" in rc.stderr, "C benchmarks anchor reported", rc.stderr.strip()[-110:])
    check((c / "site" / "index.html").read_text().startswith("<p>no anchor"),
          "C anchorless page left alone", (c / "site" / "index.html").read_text().strip())

    # ── D: the daily driver must not name a file the tree does not carry ───────
    # `git add` on a missing pathspec exits 128, and the driver is `set -e`: naming
    # the deleted page there would abort the publish even with the writer fixed.
    driver = strip_shell_comments(DRIVER.read_text())
    check("site/benchmarks.html" not in driver,
          "D driver no longer names the deleted page",
          "; ".join(ln.strip() for ln in driver.splitlines() if "site/benchmarks.html" in ln)[:110])
    add_line = next((ln for ln in driver.splitlines() if ln.strip().startswith("git add ")), "")
    args = [a.lstrip("\"'").rstrip("\"'") for a in add_line.split()[2:]]
    missing = [a for a in args if not a.startswith("$") and not (ROOT / a).exists()]
    check(not missing, "D every path the driver git-adds exists", ", ".join(missing) or add_line.strip()[:80])
    check("{MANAGED_FILES" in add_line or "MANAGED_FILES" in add_line,
          "D driver still adds its managed files", add_line.strip()[:80])

    # ── E: floor — the fixture must actually differ from what the writer produces ─
    # A no-op writer would satisfy "exit 0" above while syncing nothing.
    e = make_tree(tmp, "e", index=False, benchmarks=False)
    before = numbers_of(e)["benchmarks"]["halo_gemv_gbps"]
    run(e)
    after = numbers_of(e)["benchmarks"]["halo_gemv_gbps"]
    check(before != after, "E floor: the writer changed a value", f"{before} -> {after}")
    check(after == 100.0, "E floor: the value came from latest.json", str(after))
finally:
    shutil.rmtree(tmp, ignore_errors=True)

if failures:
    print("sync_numbers FAILED:")
    for f in failures:
        print(f"  ✗ {f}")
    sys.exit(1)

print(f"sync_numbers OK ({checks} checks: absent/anchored/anchorless pages, numbers.json sync, driver paths)")
