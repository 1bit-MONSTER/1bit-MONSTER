#!/usr/bin/env python3
"""build_embed_index.py — embeddings-powered SEO index for the 1bit.MONSTER site.

Uses the engine's own /v1/embeddings endpoint (nomic-embed-text-v1-GGUF)
to build:
  * site/search-index.json  — full-text chunks for client-side search
  * site/related.json       — semantic related-pages (cosine similarity)
  * injects a "Related" block into every post (internal linking = SEO)

Run against the engine's lemonade server:
  ssh -f -N -L 8099:127.0.0.1:8099 strixhalo   # tunnel to the embed server
  python3 scripts/build_embed_index.py

Usage: python3 scripts/build_embed_index.py [--site-dir site] [--url http://127.0.0.1:8099] [--write]
"""
import html
import json
import math
import re
import sys
from pathlib import Path
from urllib import request

SITE = "https://1bit.monster"
MODEL = "nomic-embed-text-v1-GGUF"
TAG = re.compile(r"<[^>]+>")
WHITESPACE = re.compile(r"\s+")
HEADING = re.compile(r"<h([12])[^>]*>(.*?)</h\1>", re.S)
PARA = re.compile(r"<p[^>]*>(.*?)</p>", re.S)
CHUNK_SIZE = 200   # words — keeps each /v1/embeddings request under the
                   # backend's 512-token physical batch (real prose ≈ 1.5-2
                   # tok/word; 300-word chunks exceeded it and 500'd)
CHUNK_OVERLAP = 40
BATCH = 8


def clean(s: str) -> str:
    s = TAG.sub(" ", s)
    s = html.unescape(s)
    return WHITESPACE.sub(" ", s).strip()


def extract_text(path: Path) -> str:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"<main[^>]*>(.*?)</main>", raw, re.S)
    body = m.group(1) if m else raw
    parts = []
    for hm in HEADING.finditer(body):
        parts.append(clean(hm.group(2)))
    for pm in PARA.finditer(body):
        t = clean(pm.group(1))
        if t:
            parts.append(t)
    return "\n".join(parts)


def chunk_text(text: str) -> list[str]:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + CHUNK_SIZE]))
        i += max(1, CHUNK_SIZE - CHUNK_OVERLAP)
    return chunks or [""]


def embed(url: str, inputs: list[str]) -> list[list[float]]:
    body = json.dumps({"input": inputs, "model": MODEL}).encode()
    req = request.Request(f"{url}/v1/embeddings", data=body,
                          headers={"Content-Type": "application/json"})
    try:
        with request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read())
    except Exception:
        if len(inputs) == 1:
            raise
        half = (len(inputs) + 1) // 2
        return embed(url, inputs[:half]) + embed(url, inputs[half:])
    if "data" not in data:
        raise RuntimeError(f"embedding API error: {data}")
    rows = sorted(data["data"], key=lambda r: r["index"])
    return [r["embedding"] for r in rows]


def normalize(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / n for x in v]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def main() -> int:
    args = sys.argv[1:]
    site_dir = Path("site")
    url = "http://127.0.0.1:8099"
    write = False
    it = iter(args)
    for a in it:
        if a == "--site-dir":
            site_dir = Path(next(it))
        elif a == "--url":
            url = next(it)
        elif a == "--write":
            write = True

    pages = sorted(p for p in site_dir.glob("*.html"))
    chunks = []   # {id, page, title, text}
    chunk_texts = []
    for p in pages:
        m = re.search(r"<title>(.*?)</title>", p.read_text(encoding="utf-8"), re.S)
        title = clean(m.group(1)) if m else p.name
        for ci, ct in enumerate(chunk_text(extract_text(p))):
            chunks.append({
                "id": f"{p.stem}:{ci}",
                "page": p.name,
                "title": title,
                "text": ct,
            })
            chunk_texts.append(ct)
    print(f"pages={len(pages)} chunks={len(chunks)}")

    # embed in batches
    vecs = []
    for i in range(0, len(chunk_texts), BATCH):
        batch = chunk_texts[i:i + BATCH]
        if i % 64 == 0:
            print(f"  embedding {i}/{len(chunk_texts)}...")
        vecs.extend(normalize(v) for v in embed(url, batch))
    print(f"embedded {len(vecs)} chunks")

    # page-level vectors (mean of chunks) + related
    page_meta = {}
    page_vecs = {}
    for c, v in zip(chunks, vecs):
        page_meta.setdefault(c["page"], c["title"])
        page_vecs.setdefault(c["page"], []).append(v)

    related = {}
    keys = sorted(page_vecs)
    for p in keys:
        pv = [sum(x) for x in zip(*page_vecs[p])]
        pv = normalize(pv)
        scores = []
        for q in keys:
            if q == p:
                continue
            qv = [sum(x) for x in zip(*page_vecs[q])]
            qv = normalize(qv)
            scores.append((q, dot(pv, qv)))
        scores.sort(key=lambda t: -t[1])
        related[p] = [{"url": f"{SITE}/{q}", "title": page_meta[q],
                       "score": round(s, 4)} for q, s in scores[:3] if s > 0.3]

    if not write:
        print("dry run — pass --write to write site/search-index.json, related.json, search.html")
        for p, rel in list(related.items())[:5]:
            print(f"  {p} -> {[r['title'][:30] for r in rel]}")
        return 0

    # search index (text only — query embedding needs the server; keyword + related covers it)
    index = {
        "model": MODEL,
        "generated": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
        "pages": [{"url": f"{SITE}/" if p == "index.html" else f"{SITE}/{p}",
                   "title": page_meta[p]} for p in keys],
        "chunks": [{"id": c["id"], "page": c["page"], "title": c["title"], "text": c["text"]}
                   for c in chunks],
    }
    (site_dir / "search-index.json").write_text(
        json.dumps(index), encoding="utf-8")
    (site_dir / "related.json").write_text(
        json.dumps({"generated": index["generated"], "related": related}, indent=1),
        encoding="utf-8")
    print(f"wrote search-index.json ({len(chunks)} chunks), related.json")

    # inject semantic "Related" blocks into post pages (internal linking)
    injected = 0
    for p in keys:
        if not related.get(p) or not p.startswith("1bit-post-"):
            continue
        fp = site_dir / p
        html = fp.read_text(encoding="utf-8")
        if "class=\"related\"" in html:
            continue
        links = "\n".join(
            f'      <a href="{r["url"]}">{r["title"].replace(" | 1bit.MONSTER", "")}</a>'
            for r in related[p])
        block = (
            "\n    <section class=\"related\">\n"
            "      <h2>Related</h2>\n"
            f"{links}\n"
            "    </section>"
        )
        if "</article>" in html:
            html = html.replace("</article>", block + "\n  </article>", 1)
        elif "</main>" in html:
            html = html.replace("</main>", block + "\n  </main>", 1)
        else:
            continue
        fp.write_text(html, encoding="utf-8")
        injected += 1
    print(f"injected related blocks into {injected} posts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
