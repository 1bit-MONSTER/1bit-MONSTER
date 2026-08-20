#!/usr/bin/env python3
"""Restyle jarvis/index.html to the shared monster.css design language."""
import re

PATH = 'site/jarvis/index.html'

NAV = '''<nav class="site-nav">
<a class="brand" href="/index.html"><b>1bit</b>.MONSTER</a>
<div class="links">
<a href="/models.html">Models</a>
<a href="/benchmarks.html">Benchmarks</a>
<a href="/docs.html">Docs</a>
<a href="/jarvis/" class="current">1bit JARVIS</a>
<a href="/blog/">Blog</a>
<a href="/store/">Store</a>
<a href="/demo/">Demos</a>
<a href="/analytics/">Analytics</a>
<a href="https://github.com/1bit-MONSTER/1bit-MONSTER">GitHub</a>
</div>
</nav>'''

FOOT = '''<footer class="site-foot">MIT License · <a href="https://github.com/1bit-MONSTER/1bit-MONSTER">github.com/1bit-MONSTER</a> · "Sorry but not sorry."
<div class="fine">Built with DeepSeek v4 (99.9%) · Shipped with Claude (0.1%) · One human.<br>
NPU: XDNA2 · GPU: Radeon 8060S · CPU: Zen 5 · 94 tok/s on a consumer laptop · 50 TOPS INT8 · Open source.<br>
—bong-water-water-bong · "Sorry but not Sorry :)"</div>
</footer>'''

FONTS_OLD = 'https://fonts.googleapis.com/css2?family=DM+Serif+Text:ital@0;1&family=DM+Mono:wght@400;500&display=swap'
FONTS_NEW = 'https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Mono:wght@400;500;700&display=swap'

CSS_LINK = '<link rel="stylesheet" href="/assets/monster.css?v=1">'

with open(PATH, encoding='utf-8') as f:
    html = f.read()

# fonts
assert FONTS_OLD in html, 'fonts link not found'
html = html.replace(FONTS_OLD, FONTS_NEW)

# style block -> monster.css
html, n = re.subn(r'(?s)<style>.*?</style>', CSS_LINK, html, count=1)
assert n == 1, f'style block: {n}'

# nav -> standard site nav
html, n = re.subn(r'(?s)<nav>.*?</nav>', NAV, html, count=1)
assert n == 1, f'nav: {n}'

# footer -> standard site footer
html, n = re.subn(r'(?s)<footer>.*?</footer>', FOOT, html, count=1)
assert n == 1, f'footer: {n}'

# hero h2: drop the heavy weight so it matches the serif display look
html = html.replace(
    'font-size:clamp(20px,2.6vw,32px);font-weight:800;letter-spacing:-.025em;line-height:1.04;',
    'font-size:clamp(20px,2.6vw,32px);font-weight:400;letter-spacing:-.02em;line-height:1.1;')

with open(PATH, 'w', encoding='utf-8') as f:
    f.write(html)

print('jarvis restyled')
