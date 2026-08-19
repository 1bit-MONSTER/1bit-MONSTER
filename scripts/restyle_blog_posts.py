#!/usr/bin/env python3
"""Restyle 1bit.monster blog posts to the shared monster.css design language.

Swaps: Google Fonts link (DM Serif Text -> DM Serif Display), the inline
<style> block -> monster.css link, the old minimal nav -> standard site nav,
and the old .foot/.footer div -> standard site footer.
"""
import glob
import re
import sys

NAV = '''<nav class="site-nav">
<a class="brand" href="/index.html"><b>1bit</b>.MONSTER</a>
<div class="links">
<a href="/models.html">Models</a>
<a href="/benchmarks.html">Benchmarks</a>
<a href="/docs.html">Docs</a>
<a href="/jarvis/">1bit JARVIS</a>
<a href="/blog/" class="current">Blog</a>
<a href="/store/">Store</a>
<a href="/demo/">Demos</a>
<a href="/analytics/">Analytics</a>
<a href="https://github.com/1bit-MONSTER/1bit-MONSTER">GitHub</a>
</div>
</nav>'''

FOOT = '''<footer class="site-foot">MIT License · <a href="https://github.com/1bit-MONSTER/1bit-MONSTER">github.com/1bit-MONSTER</a> · "Sorry but not sorry."</footer>'''

FONTS_OLD = 'https://fonts.googleapis.com/css2?family=DM+Serif+Text:ital@0;1&family=DM+Mono:wght@400;500&display=swap'
FONTS_NEW = 'https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Mono:wght@400;500;700&display=swap'

CSS_LINK = '<link rel="stylesheet" href="/assets/monster.css?v=1">'

WARN_OLD = '<div style="background:#ffd;border:2px solid #ea0;padding:12px 20px;margin:0;text-align:center;font-size:14px;">'
WARN_NEW = '<div class="warn-strip">'


def restyle(path):
    with open(path, encoding='utf-8') as f:
        html = f.read()
    orig = html

    # 1. fonts
    if FONTS_OLD in html:
        html = html.replace(FONTS_OLD, FONTS_NEW)
    else:
        print(f'  !! {path}: old fonts link not found')

    # 2. inline style block -> monster.css
    html, n = re.subn(r'(?s)<style>.*?</style>', CSS_LINK, html, count=1)
    if n != 1:
        print(f'  !! {path}: style block not replaced ({n})')

    # 3. nav -> standard site nav
    if re.search(r'<nav', html):
        html, n = re.subn(r'(?s)<nav>.*?</nav>', NAV, html, count=1)
        if n != 1:
            print(f'  !! {path}: nav not replaced ({n})')
    else:
        html, n = re.subn(r'(?s)(<body[^>]*>)', r'\1\n' + NAV, html, count=1)
        if n != 1:
            print(f'  !! {path}: nav not inserted ({n})')

    # 4. footers -> standard footer
    html, n1 = re.subn(r'(?s)<div class="foot[^"]*">.*?</div>', FOOT, html, count=1)
    html, n2 = re.subn(r'(?s)<div class="footer">.*?</div>', FOOT, html, count=1)
    if n1 + n2 != 1:
        print(f'  !! {path}: footer not replaced ({n1}+{n2})')

    # 5. legacy stale-warning banners -> styled strip
    html = html.replace(WARN_OLD, WARN_NEW)

    if html != orig:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(html)
        print(f'  ok {path}')
    else:
        print(f'  -- {path}: unchanged (check!)')


def main():
    paths = sorted(glob.glob('site/blog/*.html'))
    paths = [p for p in paths if not p.endswith('index.html')]
    for p in paths:
        restyle(p)


if __name__ == '__main__':
    sys.exit(main())
