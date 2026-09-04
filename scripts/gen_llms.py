#!/usr/bin/env python3
"""gen_llms.py — generate llms.txt + llms-full.txt for AI crawlers.

The llmstxt.org standard: a concise index (llms.txt) plus a full-text
bundle (llms-full.txt) that ChatGPT, Perplexity, Bing AI, Copilot and
other LLM-based search engines read. This is the "AI SEO takeover" file:
it turns the site into a citation source for AI answers.

Usage: python3 scripts/gen_llms.py [--site-dir site]
"""
import html
import os
import re
import sys
from pathlib import Path

# Reuse seo_sync's live number extraction (arch tokens / HF arch strings /
# coverage) so llms.txt can't drift from the site's SEO tags.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import seo_sync  # noqa: E402

SITE = "https://1bit.monster"
HEADING = re.compile(r"<h([12])[^>]*>(.*?)</h\1>", re.S)
PARA = re.compile(r"<p[^>]*>(.*?)</p>", re.S)
TAG = re.compile(r"<[^>]+>")
WHITESPACE = re.compile(r"\s+")


def clean(s: str) -> str:
    s = TAG.sub(" ", s)
    s = html.unescape(s)
    return WHITESPACE.sub(" ", s).strip()


def extract(path: Path) -> list[str]:
    """Extract (heading, text) sections from a page's <main> content."""
    raw = path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"<main[^>]*>(.*?)</main>", raw, re.S)
    body = m.group(1) if m else raw
    sections: list[str] = []
    buf: list[str] = []

    def flush():
        txt = " ".join(clean(p) for p in buf)
        if txt:
            sections.append(txt)
        buf.clear()

    for hm in HEADING.finditer(body):
        if buf:
            flush()
        sections.append(clean(hm.group(2)))
    for pm in PARA.finditer(body):
        buf.append(pm.group(1))
    flush()
    return sections


def main() -> int:
    site_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "site")
    pages = sorted(p for p in site_dir.glob("*.html")
                   if p.name not in ("sticker-gallery.html",))

    tokens = seo_sync.count_tokens()
    arch = seo_sync.count_arch_strings()
    covered, with_arch = seo_sync.census_coverage()
    pct = seo_sync._pct(covered, with_arch)  # same formatter as the site HTML
    blurb = ("One engine, any model. A model-agnostic, hardware-agnostic pure-C++ "
             f"inference engine (MIT): {tokens:,} architecture tokens, "
             f"{arch:,} HF arch strings, {pct} HuggingFace coverage, "
             f"{covered:,} checkpoints mapped, running on Ryzen AI NPUs and ROCm "
             "with a GGUF-native 1-bit pipeline. Zero Python at runtime.")

    # llms.txt — the index
    out = ["# 1bit.MONSTER", "", f"> {blurb}", "", "## Pages", ""]
    for p in pages:
        m = re.search(r"<title>(.*?)</title>", p.read_text(encoding="utf-8"), re.S)
        title = clean(m.group(1)) if m else p.name
        name = "" if p.name == "index.html" else p.name
        out.append(f"- [{title}]({SITE}/{name})")
    (site_dir / "llms.txt").write_text("\n".join(out) + "\n", encoding="utf-8")

    # llms-full.txt — full content bundle
    full = ["# 1bit.MONSTER — full content", "", f"> {blurb}", ""]
    for p in pages:
        m = re.search(r"<title>(.*?)</title>", p.read_text(encoding="utf-8"), re.S)
        title = clean(m.group(1)) if m else p.name
        full.append(f"---\n\n# {title}\n")
        for sec in extract(p):
            full.append(sec + "\n")
    (site_dir / "llms-full.txt").write_text("\n".join(full) + "\n", encoding="utf-8")

    print(f"llms.txt: {len(pages)} pages indexed")
    print(f"llms-full.txt: {(site_dir/'llms-full.txt').stat().st_size//1024} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
