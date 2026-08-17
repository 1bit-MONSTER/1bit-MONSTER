#!/usr/bin/env python3
"""Sync the 5 hardware-verified benchmark keys from benchmarks/latest.json
into site/numbers.json and the one static HTML figure derived from them.

Narrow by design: only touches what tools/validate_claims.py --update just
refreshed --  benchmarks/_sources/_benchmarks_remeasured in numbers.json,
plus the single "prefill_tflops_i8apre" figure printed on the live site
(site/index.html and site/benchmarks.html). Everything else in numbers.json
(binary/repo/site/engines/blackmamba_*/...) is left byte-for-byte alone --
those come from a different, human-governed pipeline (site/benchmarks.json),
out of scope here.

Usage:
    python3 tools/sync_numbers.py
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
LATEST = REPO / "benchmarks/latest.json"
NUMBERS = REPO / "site/numbers.json"
INDEX_HTML = REPO / "site/index.html"
BENCHMARKS_HTML = REPO / "site/benchmarks.html"

KEYS = (
    "halo_gemv_gbps",
    "prefill_tflops_4h",
    "prefill_tflops_i8apre",
    "sherry_gemv_gbps",
    "tq1_gemv_gbps",
)


def sync_numbers_json(today: str) -> tuple[bool, float]:
    latest = json.loads(LATEST.read_text())
    numbers = json.loads(NUMBERS.read_text())

    bm = dict(numbers.get("benchmarks", {}))
    changed = any(bm.get(k) != latest["benchmarks"][k] for k in KEYS)

    for k in KEYS:
        bm[k] = latest["benchmarks"][k]
    bm["tflops"] = bm["prefill_tflops_i8apre"]
    bm["note"] = (
        f"Auto re-measured {today} by tools/validate_claims.py "
        f"(bench_sherry, bench_prefill_variants, median of 3 runs, "
        f"1 warmup discarded). See benchmarks/latest.json for full history."
    )
    numbers["benchmarks"] = bm
    numbers["_sources"] = dict(latest["_sources"])
    numbers["_benchmarks_remeasured"] = (
        f"{today}, median of 3 runs on real gfx1151 hardware "
        f"(bench_sherry, bench_prefill_variants) — see _sources."
    )
    numbers["prefill_tflops_i8apre"] = bm["prefill_tflops_i8apre"]
    numbers["tflops"] = bm["prefill_tflops_i8apre"]

    NUMBERS.write_text(json.dumps(numbers, indent=2) + "\n")
    return changed, bm["prefill_tflops_i8apre"]


def patch_index_html(new_value: str, today: str) -> bool:
    text = INDEX_HTML.read_text(encoding="utf-8")
    m = re.search(r"measured (\d{4}-\d{2}-\d{2})\. Plus ([\d.]+) TFLOPS INT8 prefill\.", text)
    if not m:
        print("WARN: site/index.html anchor sentence not found -- skipping", file=sys.stderr)
        return False
    old_date, old_value = m.group(1), m.group(2)
    if old_value == new_value:
        return False
    new_text = text[: m.start()] + f"measured {today}. Plus {new_value} TFLOPS INT8 prefill." + text[m.end() :]
    INDEX_HTML.write_text(new_text, encoding="utf-8")
    print(f"site/index.html: {old_value} -> {new_value} (measured {old_date} -> {today})")
    return True


def patch_benchmarks_html(new_value: str) -> bool:
    text = BENCHMARKS_HTML.read_text(encoding="utf-8")
    m = re.search(r"<b>([\d.]+)</b><span>TFLOPS int8 prefill \(WMMA\)</span>", text)
    if not m:
        print("WARN: site/benchmarks.html anchor not found -- skipping", file=sys.stderr)
        return False
    old_value = m.group(1)
    if old_value == new_value:
        return False
    new_text = text[: m.start(1)] + new_value + text[m.end(1) :]
    BENCHMARKS_HTML.write_text(new_text, encoding="utf-8")
    print(f"site/benchmarks.html: {old_value} -> {new_value}")
    return True


def main() -> int:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    changed, i8apre = sync_numbers_json(today)
    print(f"site/numbers.json: {'updated' if changed else 'unchanged'} (prefill_tflops_i8apre={i8apre})")

    display_value = f"{i8apre:.1f}"
    patch_index_html(display_value, today)
    patch_benchmarks_html(display_value)
    return 0


if __name__ == "__main__":
    sys.exit(main())
