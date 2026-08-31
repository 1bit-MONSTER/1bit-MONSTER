#!/usr/bin/env python3
"""lint_docs_markdown.py — parse-check the curated docs with markdown-it-py.

Context7 (and docs.1bit.monster's generator) index the repo's curated docs/
Markdown; pages that fail to parse are silently missing from the index
(issue #1963: 37 -> 56 parse failures as snippets grew). This script runs the
same parser family over the curated set and FAILS on any file that does not
produce a clean token stream, so indexing failures are caught at CI time
instead of on the Context7 dashboard.

Usage:
    python3 scripts/lint_docs_markdown.py            # lint the curated docs
    python3 scripts/lint_docs_markdown.py --all      # lint every docs/**/*.md

Exit 0 = all files parse clean; exit 1 = at least one file failed (printed).
"""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = os.path.join(REPO, "docs")


def curated_files() -> list[str]:
    """The docs indexed by Context7 / the docs site (context7.json rules)."""
    cfg_path = os.path.join(REPO, "context7.json")
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    excl_dirs = set(cfg.get("excludeFolders", []))
    excl_files = set(cfg.get("excludeFiles", []))

    out: list[str] = []
    for root, dirs, files in os.walk(DOCS):
        rel_root = os.path.relpath(root, REPO)
        dirs[:] = [d for d in dirs if f"{rel_root}/{d}" not in excl_dirs
                   and d not in ("superpowers",)]
        for f in files:
            if not f.endswith(".md") or f in excl_files:
                continue
            out.append(os.path.join(root, f))
    return sorted(out)


def all_files() -> list[str]:
    out: list[str] = []
    for root, _dirs, files in os.walk(DOCS):
        for f in files:
            if f.endswith(".md"):
                out.append(os.path.join(root, f))
    return sorted(out)


def lint(paths: list[str]) -> int:
    try:
        from markdown_it import MarkdownIt
    except ImportError:
        print("markdown-it-py not installed — run: pip install markdown-it-py",
              file=sys.stderr)
        return 2

    md = MarkdownIt("commonmark", {"html": True}).enable("table")
    failures: list[tuple[str, str]] = []
    ok = 0
    for p in paths:
        try:
            with open(p, encoding="utf-8") as fh:
                src = fh.read()
            if "\x00" in src:
                failures.append((p, "null bytes in file"))
                continue
            tokens = md.parse(src)
            # A clean parse always yields at least the final inline token.
            if not tokens:
                failures.append((p, "empty token stream (parser returned nothing)"))
                continue
            ok += 1
        except Exception as exc:  # noqa: BLE001
            failures.append((p, f"parse raised {type(exc).__name__}: {exc}"))

    for p, why in failures:
        print(f"FAIL  {os.path.relpath(p, REPO):55s} {why}")
    print(f"{ok} files parse clean, {len(failures)} failures")
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all", action="store_true",
                    help="lint every docs/**/*.md (not just the curated set)")
    args = ap.parse_args()
    paths = all_files() if args.all else curated_files()
    return lint(paths)


if __name__ == "__main__":
    sys.exit(main())
