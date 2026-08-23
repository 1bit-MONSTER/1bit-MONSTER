#!/usr/bin/env python3
"""generate_banner.py — README hero banner, plain black/white identity.

Matches the site's actual homepage hero (site/index.html) content and a
plain sans-serif wordmark — no serif font, no badge/meta clutter, no
color accent. Inverted: black background, white lettering.

Logo: the relay mark (2x2 pixel grid, one block lit, in white) +
"1bit.MONSTER" wordmark, tagline "One engine, any model, zero Python."

Run from repo root:  python3 scripts/merch/generate_banner.py
"""
import os

from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SITE = os.path.join(ROOT, 'site', 'assets')
W, H = 2560, 840

BG     = (1, 2, 2)
FG     = (245, 248, 250)
MUTED  = (168, 176, 184)
BORDER = (40, 44, 50)


def _font(name, size):
    from PIL import ImageFont
    return ImageFont.truetype(os.path.join(ROOT, 'scripts', 'merch', 'fonts', name), size)


SANS_BLACK = lambda s: _font('Inter-Black.ttf', s)
SANS       = lambda s: _font('Inter-Regular.ttf', s)


def relay_mark(d, ox, oy, size):
    """2x2 pixel grid — the site's relay mark. One block solid white,
    the rest white outlines (monochrome — no accent color)."""
    cell = size * 9 / 24
    step = size * 11 / 24
    rx   = size * 2 / 24
    lit  = 3  # r4
    for i, (gx, gy) in enumerate([(2, 2), (13, 2), (2, 13), (13, 13)]):
        x0 = ox + gx / 24 * size
        y0 = oy + gy / 24 * size
        if i == lit:
            d.rounded_rectangle([x0, y0, x0 + cell, y0 + cell], radius=rx, fill=FG)
        else:
            d.rounded_rectangle([x0, y0, x0 + cell, y0 + cell], radius=rx,
                                outline=FG, width=max(4, size // 30))


def build():
    img = Image.new('RGB', (W, H), BG)
    d = ImageDraw.Draw(img)

    # hairline frame
    d.rectangle([2, 2, W - 3, H - 3], outline=BORDER, width=3)

    # ── lockup: relay mark + wordmark (left, vertically centered) ──
    mark_size = 300
    ox, oy = 380, (H - mark_size) // 2 - 60
    relay_mark(d, ox, oy, mark_size)

    x_txt = ox + mark_size + 110
    y_txt = oy + mark_size / 2 - 105
    f_wm = SANS_BLACK(190)
    d.text((x_txt, y_txt), '1bit.MONSTER', font=f_wm, fill=FG)

    # ── tagline (two lines, left-aligned under the lockup) ──
    f_tag = SANS(56)
    line1 = 'One engine, any model, zero Python.'
    line2 = 'A model-agnostic, hardware-agnostic pure-C++ inference engine.'
    d.text((ox, oy + mark_size + 70), line1, font=f_tag, fill=MUTED)
    d.text((ox, oy + mark_size + 140), line2, font=f_tag, fill=MUTED)

    return img


def main():
    os.makedirs(SITE, exist_ok=True)
    out = os.path.join(SITE, 'banner.png')
    build().save(out, quality=95)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
