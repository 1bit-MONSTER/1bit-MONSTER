#!/usr/bin/env python3
"""gap_analysis.py — content-gap analysis for the 1bit.MONSTER SEO takeover.

Embeds the site's pages (engine /v1/embeddings) and scores coverage against
a curated list of high-intent topics in the local-AI / NPU / 1-bit space.
Output: covered topics, semantic gaps, and concrete post ideas.

Usage: python3 scripts/gap_analysis.py --site-dir site [--url http://127.0.0.1:8099]
"""
import html
import json
import math
import re
import sys
from pathlib import Path
from urllib import request

MODEL = "nomic-embed-text-v1-GGUF"
TAG = re.compile(r"<[^>]+>")
WS = re.compile(r"\s+")

TOPICS = [
    # search-demand topics in the engine's space
    "Ryzen AI NPU inference",
    "ROCm vs CUDA",
    "run LLM on AMD GPU",
    "1-bit quantization",
    "GGUF format explained",
    "speculative decoding",
    "local LLM privacy",
    "offline AI assistant",
    "run LLM on laptop",
    "LLM without GPU",
    "vLLM alternatives",
    "AMD Strix Halo AI performance",
    "int4 vs int8 quantization",
    "MoE sparse inference",
    "CPU-only LLM inference",
    "llama.cpp alternatives",
    "FastFlowLM",
    "voice assistant local",
    "Zamba2 model",
    "AI on Windows NPU",
    "Kraken Point NPU",
    "model quantization benchmarks",
    "edge AI deployment",
    "LLM inference cost",
    "local RAG embeddings",
    "AI coding agent local",
    "large context window inference",
    "model compression techniques",
    "LLM token throughput optimization",
    "HuggingFace model zoo",
]


def clean(s: str) -> str:
    return WS.sub(" ", TAG.sub(" ", s)).strip()


def embed_one(url: str, text: str):
    body = json.dumps({"input": text[:4000], "model": MODEL}).encode()
    req = request.Request(f"{url}/v1/embeddings", data=body,
                          headers={"Content-Type": "application/json"})
    with request.urlopen(req, timeout=120) as r:
        d = json.loads(r.read())
    return d["data"][0]["embedding"]


def norm(v):
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def main():
    args = sys.argv[1:]
    site_dir = Path("site")
    url = "http://127.0.0.1:8099"
    it = iter(args)
    for a in it:
        if a == "--site-dir":
            site_dir = Path(next(it))
        elif a == "--url":
            url = next(it)

    # page fingerprints: title + lead paragraph
    pages = {}
    for p in sorted(site_dir.glob("*.html")):
        raw = p.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"<title>(.*?)</title>", raw, re.S)
        title = clean(m.group(1)) if m else p.name
        lead = ""
        lm = re.search(r'<meta name="description" content="([^"]*)"', raw)
        if lm:
            lead = clean(lm.group(1))
        pages[p.name] = norm(embed_one(url, f"{title}. {lead}"))

    # topic demand vectors
    topics = []
    for t in TOPICS:
        topics.append((t, norm(embed_one(url, t))))

    print(f"{'TOPIC':<42} {'BEST COVERAGE':>12}  page")
    print("-" * 80)
    rows = []
    for t, tv in topics:
        best = max(pages.items(), key=lambda kv: dot(kv[1], tv))
        rows.append((dot(best[1], tv), t, best[0]))
    rows.sort()
    print("\n== STRONG COVERAGE (similarity > 0.55) ==\n")
    for s, t, pg in reversed(rows):
        if s > 0.55:
            print(f"  {t:<40} {s:6.3f}  {pg}")
    print("\n== SEMANTIC GAPS (similarity < 0.42) — content to write ==\n")
    for s, t, pg in rows:
        if s < 0.42:
            print(f"  {t:<40} {s:6.3f}  (closest: {pg})")


if __name__ == "__main__":
    main()
