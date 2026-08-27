#!/usr/bin/env python3
"""gen_rss.py — build site/blog.xml (RSS 2.0) from the blog listing page.

Pulls the post list (href + title) from site/1bit-blog.html in the order the
page shows (newest first), attaches a stable pubDate derived from git history
(first commit touching the post file) and the meta description, and writes a
valid RSS 2.0 feed with an atom self-link.

Usage: python3 scripts/gen_rss.py [--site-dir site]
"""
import argparse
import datetime
import html
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from seo_meta import SITE, git_firstmod  # noqa: E402

RSS_NS = 'xmlns:atom="http://www.w3.org/2005/Atom"'
RFC2822 = "%a, %d %b %Y %H:%M:%S +0000"


def rfc2822(iso_date: str) -> str:
    try:
        d = datetime.date.fromisoformat(iso_date)
        return d.strftime(RFC2822)
    except ValueError:
        return ""


def collect_posts(blog_html: str) -> list:
    """Ordered (file, title) pairs from the blog listing page."""
    posts = []
    for m in re.finditer(
        r'<h3><a href="(1bit-post-[^"]+\.html)"[^>]*>([^<]+)</a></h3>',
        blog_html,
    ):
        posts.append((m.group(1), html.unescape(m.group(2)).strip()))
    return posts


def post_description(site_dir: Path, name: str) -> str:
    m = re.search(
        r'<meta name="description" content="(.*?)"',
        (site_dir / name).read_text(encoding="utf-8"),
        re.S,
    )
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else ""


def gen_blog_rss(site_dir: Path) -> None:
    blog = (site_dir / "1bit-blog.html").read_text(encoding="utf-8")
    posts = collect_posts(blog)
    items = []
    for name, title in posts:
        url = f"{SITE}/{name}"
        desc = post_description(site_dir, name)
        pub = rfc2822(git_firstmod(f"site/{name}"))
        items.append(
            "    <item>\n"
            f"      <title>{html.escape(title)}</title>\n"
            f"      <link>{url}</link>\n"
            f"      <guid isPermaLink=\"true\">{url}</guid>\n"
            f"      <pubDate>{pub}</pubDate>\n"
            f"      <description>{html.escape(desc)}</description>\n"
            "    </item>"
        )
    feed = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<rss version="2.0" {RSS_NS}>\n'
        "  <channel>\n"
        "    <title>1bit.MONSTER — Blog</title>\n"
        f"    <link>{SITE}/1bit-blog.html</link>\n"
        "    <description>How the 1-bit inference engine actually gets built — NPU, ROCm, kernels, formats.</description>\n"
        "    <language>en</language>\n"
        f"    <atom:link href=\"{SITE}/blog.xml\" rel=\"self\" type=\"application/rss+xml\" />\n"
        f"    <lastBuildDate>{datetime.datetime.now(datetime.timezone.utc).strftime(RFC2822)}</lastBuildDate>\n"
        + "\n".join(items)
        + "\n  </channel>\n</rss>\n"
    )
    (site_dir / "blog.xml").write_text(feed, encoding="utf-8")
    print(f"rss: {len(posts)} items -> site/blog.xml")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site-dir", default="site")
    args = ap.parse_args()
    gen_blog_rss(Path(args.site_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
