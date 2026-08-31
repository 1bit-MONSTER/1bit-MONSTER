#!/usr/bin/env python3
"""seo_meta.py — inject social/structured-data tags into every site page.

Phase 1 of the 1bit.MONSTER SEO takeover:
  * Open Graph + Twitter card tags (share previews — the virality enabler)
  * canonical URL
  * JSON-LD: WebSite+Organization on the index, BlogPosting on posts,
    WebPage on hub pages
  * sitemap.xml with lastmod (from git history) + priority

Idempotent: re-running is a no-op on already-tagged pages.
Usage: python3 scripts/seo_meta.py [--site-dir site]
"""
import datetime
import json
import re
import subprocess
import sys
from pathlib import Path

SITE = "https://1bit.monster"
OG_IMG = f"{SITE}/assets/og-card.png"
AUTHOR = {"@type": "Organization", "name": "1bit.MONSTER", "url": SITE}


def git_lastmod(rel: str) -> str:
    """Date the file last changed, from git history (dateModified)."""
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%cI", "origin/main", "--", rel],
            capture_output=True, text=True, cwd=".",
        ).stdout.strip()
        if out:
            return out[:10]  # YYYY-MM-DD
    except Exception:
        pass
    return ""


def git_firstmod(rel: str) -> str:
    """Date the file was first published, from git history (stable datePublished).

    Uses the commit that ADDED the file; falls back to the earliest commit
    touching it. Never moves when later syncs touch the file.
    """
    try:
        for args in (["--diff-filter=A"], ["--reverse"]):
            out = subprocess.run(
                ["git", "log", *args, "--format=%cI", "origin/main", "--", rel],
                capture_output=True, text=True, cwd=".",
            ).stdout.strip().splitlines()
            if out:
                return out[0][:10]  # YYYY-MM-DD
    except Exception:
        pass
    return ""


def read_page(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def page_meta(html: str) -> dict:
    """Pull title + description from an existing page."""
    title = ""
    m = re.search(r"<title>(.*?)</title>", html, re.S)
    if m:
        title = re.sub(r"\s+", " ", m.group(1)).strip()
    desc = ""
    m = re.search(r'<meta name="description" content="(.*?)"', html, re.S)
    if m:
        desc = re.sub(r"\s+", " ", m.group(1)).strip()
    return {"title": title, "description": desc}


def inject_head(html: str, name: str) -> str:
    """Inject OG/twitter/canonical after the meta description, if missing."""
    if 'property="og:site_name"' in html:
        return html
    meta = page_meta(html)
    title, desc = meta["title"], meta["description"]
    og_type = "article" if name.startswith("1bit-post-") else "website"
    page_url = SITE + "/" if name == "index.html" else f"{SITE}/{name}"
    tags = (
        f'<meta property="og:site_name" content="1bit.MONSTER" />\n'
        f'  <meta property="og:type" content="{og_type}" />\n'
        f'  <meta property="og:title" content="{title}" />\n'
        f'  <meta property="og:description" content="{desc}" />\n'
        f'  <meta property="og:url" content="{page_url}" />\n'
        f'  <meta property="og:image" content="{OG_IMG}" />\n'
        f'  <meta name="twitter:card" content="summary_large_image" />\n'
        f'  <meta name="twitter:title" content="{title}" />\n'
        f'  <meta name="twitter:description" content="{desc}" />\n'
        f'  <meta name="twitter:image" content="{OG_IMG}" />\n'
        f'  <link rel="canonical" href="{page_url}" />'
    )
    # insert right after the meta description line
    return re.sub(
        r'(<meta name="description"[^>]*/?>)',
        r"\1\n  " + tags,
        html,
        count=1,
    )


def json_ld(html: str, name: str) -> str:
    """Build JSON-LD for the page."""
    meta = page_meta(html)
    url = SITE + ("/" if name == "index.html" else f"/{name}")
    if name == "index.html":
        return json.dumps([
            {
                "@context": "https://schema.org",
                "@type": "WebSite",
                "name": "1bit.MONSTER",
                "url": SITE,
                "description": "The 1-bit inference engine. One engine, any model — 552 architecture tokens, 1,775 HF arch strings, 100% HuggingFace coverage, running on Ryzen AI NPUs and ROCm.",
                "potentialAction": {
                    "@type": "SearchAction",
                    "target": {"@type": "EntryPoint", "urlTemplate": f"{SITE}/search.html?q={{search_term_string}}"},
                    "query-input": "required name=search_term_string",
                },
            },
            {
                "@context": "https://schema.org",
                "@type": "Organization",
                "name": "1bit.MONSTER",
                "url": SITE,
                "logo": f"{SITE}/assets/apple-touch-icon.png",
                "sameAs": ["https://github.com/1bit-MONSTER"],
            },
        ], indent=None, ensure_ascii=False)
    if name.startswith("1bit-post-"):
        return json.dumps({
            "@context": "https://schema.org",
            "@type": "BlogPosting",
            "headline": meta["title"],
            "description": meta["description"],
            "url": url,
            "mainEntityOfPage": url,
            "image": OG_IMG,
            "datePublished": git_firstmod(f"site/{name}"),
            "dateModified": git_lastmod(f"site/{name}"),
            "author": AUTHOR,
            "publisher": AUTHOR,
        })
    return json.dumps({
        "@context": "https://schema.org",
        "@type": "WebPage",
        "name": meta["title"],
        "description": meta["description"],
        "url": url,
        "publisher": AUTHOR,
    })


def inject_json_ld(html: str, name: str) -> str:
    ld = json_ld(html, name)
    block = (
        f'  <script type="application/ld+json">\n'
        f'  {ld}\n'
        f"  </script>"
    )
    if "application/ld+json" in html:
        # Dates come from origin/main git history, so a page generated before
        # its file was merged carries empty datePublished/dateModified. Re-inject
        # (in place) only when a date is still missing — otherwise idempotent.
        if '"datePublished": ""' not in html:
            return html
        return re.sub(
            r'  <script type="application/ld\+json">.*?</script>',
            lambda m: block, html, count=1, flags=re.S)  # callable: skips \-escape parsing of the JSON
    return re.sub(r"(</head>)", lambda m: block + "\n" + m.group(1), html, count=1)


def gen_sitemap(site_dir: Path) -> None:
    """Regenerate sitemap.xml with lastmod (git) + priority."""
    prio = {
        "index.html": "1.0",
        "1bit-models.html": "0.9",
        "1bit-benchmarks.html": "0.9",
        "1bit-docs.html": "0.9",
        "1bit-blog.html": "0.9",
        "1bit-monster-v2.html": "0.9",
        "1bit-jarvis.html": "0.8",
        "1bit-store.html": "0.8",
    }
    freq = {"index.html": "daily"}
    urls = []
    for p in sorted(site_dir.glob("*.html")):
        name = p.name
        loc = (SITE + "/") if name == "index.html" else f"{SITE}/{name}"  # slash form matches canonical
        lm = git_lastmod(f"site/{name}")
        urls.append((loc, lm, prio.get(name, "0.7"), freq.get(name, "weekly")))
    lines = ["<?xml version=\"1.0\" encoding=\"UTF-8\"?>",
             "<urlset xmlns=\"http://www.sitemaps.org/schemas/sitemap/0.9\">"]
    for loc, lm, p, f in urls:
        lines.append("  <url>")
        lines.append(f"    <loc>{loc}</loc>")
        if lm:
            lines.append(f"    <lastmod>{lm}</lastmod>")
        lines.append(f"    <changefreq>{f}</changefreq>")
        lines.append(f"    <priority>{p}</priority>")
        lines.append("  </url>")
    lines.append("</urlset>")
    (site_dir / "sitemap.xml").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"sitemap: {len(urls)} urls")


def main() -> int:
    site_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "site")
    pages = sorted(site_dir.glob("*.html"))
    changed = []
    for p in pages:
        html = read_page(p)
        out = inject_head(html, p.name)
        out = inject_json_ld(out, p.name)
        if out != html:
            p.write_text(out, encoding="utf-8")
            changed.append(p.name)
    print(f"tagged {len(changed)}/{len(pages)} pages: {', '.join(changed)}")
    gen_sitemap(site_dir)
    try:
        from gen_rss import gen_blog_rss
        gen_blog_rss(site_dir)
    except Exception as e:  # never let RSS failure break the sync
        print(f"rss: skipped ({e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
