#!/usr/bin/env python3
"""Roll the 1bm logo mark into every page nav.

1. .site-nav pages:  <a class="brand" href="/index.html"><b>1bit</b>.MONSTER</a>
                     -> brand + <img class="brand-mark" src="/assets/logo-mark.svg">
2. landing pages (inline-style nav): inject the mark img into the brand anchor.
3. Bump favicon cache-bust ?v=2 -> ?v=3.
"""
import glob
import re

NAV_BRAND_OLD = '<a class="brand" href="/index.html"><b>1bit</b>.MONSTER</a>'
NAV_BRAND_NEW = ('<a class="brand" href="/index.html">'
                 '<img class="brand-mark" src="/assets/logo-mark.svg" alt="1bit MONSTER">'
                 '<b>1bit</b>.MONSTER</a>')

LAND_OLD = ('<a href="index.html" style="font-family:\'DM Serif Display\',Georgia,serif;'
            'font-size:20px;color:#e7f6fd;text-decoration:none"><span style="color:#00e5ff">1bit</span>.MONSTER</a>')
LAND_NEW = ('<a href="index.html" style="display:inline-flex;align-items:center;gap:10px;'
            'font-family:\'DM Serif Display\',Georgia,serif;font-size:20px;color:#e7f6fd;text-decoration:none">'
            '<img src="/assets/logo-mark.svg" alt="1bit MONSTER" '
            'style="width:30px;height:30px;border-radius:999px;box-shadow:0 0 10px rgba(255,159,28,.35);flex:none">'
            '<span style="color:#00e5ff">1bit</span>.MONSTER</a>')

files = []
files += glob.glob('site/jarvis/*.html')
files += glob.glob('site/blog/*.html')
files += glob.glob('site/demo/*.html')
files += glob.glob('site/store/*.html')
files += ['site/index.html', 'site/benchmarks.html', 'site/docs.html', 'site/models.html', 'site/story.html']

nav_count = land_count = fav_count = 0
for p in sorted(set(files)):
    with open(p, encoding='utf-8') as f:
        html = f.read()
    orig = html
    if NAV_BRAND_OLD in html:
        html = html.replace(NAV_BRAND_OLD, NAV_BRAND_NEW)
        nav_count += 1
    if LAND_OLD in html:
        html = html.replace(LAND_OLD, LAND_NEW)
        land_count += 1
    if 'favicon.svg?v=2' in html:
        html = html.replace('favicon.svg?v=2', 'favicon.svg?v=3')
        fav_count += 1
    if html != orig:
        with open(p, 'w', encoding='utf-8') as f:
            f.write(html)
        print('ok', p)

print(f'\nbrand-nav replacements: {nav_count}, landing-nav: {land_count}, favicon bumps: {fav_count}')
