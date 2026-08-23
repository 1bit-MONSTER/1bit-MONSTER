#!/usr/bin/env python3
"""generate_banner.py — README hero banner in the new 1-bit pixel identity.

Matches the 2026-08-22 site redesign (light modern-minimal + relay mark):
  --bg      #f5f8fa   (oklch(99% 0.002 240))
  --fg      #010202   (oklch(18% 0.012 250))
  --muted   #5c666e   (oklch(54% 0.012 250))
  --border  #e5eaee   (oklch(92% 0.005 250))
  --accent  #0230c0   (oklch(58% 0.18 255))
  --status  #003711   (oklch(52% 0.14 158))

Logo: the relay mark (2x2 pixel grid, one block lit) + "1bit.MONSTER"
wordmark, tagline "One engine. Any model. Zero Python.".

Run from repo root:  python3 scripts/merch/generate_banner.py
"""
import os

from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SITE = os.path.join(ROOT, 'site', 'assets')
W, H = 2560, 840

BG     = (245, 248, 250)
FG     = (1, 2, 2)
MUTED  = (92, 102, 110)
BORDER = (229, 234, 238)
ACCENT = (2, 48, 192)
STATUS = (0, 55, 17)


def _font(name, size):
    from PIL import ImageFont
    return ImageFont.truetype(os.path.join(ROOT, 'scripts', 'merch', 'fonts', name), size)


SERIF   = lambda s: _font('DMSerifDisplay-Regular.ttf', s)
SERIF_I = lambda s: _font('DMSerifDisplay-Italic.ttf', s)
MONO    = lambda s: _font('DMMono-Regular.ttf', s)
MONO_M  = lambda s: _font('DMMono-Medium.ttf', s)


def relay_mark(d, ox, oy, size):
    """2x2 pixel grid — the site's relay mark. One block solid at a time;
    here bottom-right (r4) is lit in accent."""
    cell = size * 9 / 24
    step = size * 11 / 24
    rx   = size * 2 / 24
    lit  = 3  # r4
    for i, (gx, gy) in enumerate([(2, 2), (13, 2), (2, 13), (13, 13)]):
        x0 = ox + gx / 24 * size
        y0 = oy + gy / 24 * size
        if i == lit:
            d.rounded_rectangle([x0, y0, x0 + cell, y0 + cell], radius=rx, fill=ACCENT)
        else:
            d.rounded_rectangle([x0, y0, x0 + cell, y0 + cell], radius=rx,
                                outline=FG, width=max(4, size // 30))


def build():
    img = Image.new('RGB', (W, H), BG)
    d = ImageDraw.Draw(img)

    # hairline frame (site --border)
    d.rectangle([2, 2, W - 3, H - 3], outline=BORDER, width=3)

    # ── lockup: relay mark + wordmark (left, vertically centered) ──
    mark_size = 300
    ox, oy = 380, (H - mark_size) // 2
    relay_mark(d, ox, oy, mark_size)

    x_txt = ox + mark_size + 110
    y_txt = H // 2 - 130
    f_wm = SERIF(210)
    w1 = f_wm.getlength('1bit')
    d.text((x_txt, y_txt), '1bit', font=f_wm, fill=FG)
    d.text((x_txt + w1 + 24, y_txt), '.MONSTER', font=f_wm, fill=ACCENT)

    # ── tagline (centered, lower third) ──
    f_tag = SERIF_I(96)
    tag = 'One engine. Any model. Zero Python.'
    tw = f_tag.getlength(tag)
    d.text(((W - tw) / 2, H - 320), tag, font=f_tag, fill=MUTED)

    # ── meta line (centered, bottom) ──
    f_meta = MONO_M(44)
    meta = 'NPU + GPU + CPU  ·  MIT  ·  ZERO PYTHON  ·  100% HUGGINGFACE'
    mw = f_meta.getlength(meta)
    d.text(((W - mw) / 2, H - 200), meta, font=f_meta, fill=ACCENT)

    # ── status dot + line ──
    f_stat = MONO(40)
    stat = 'engine online'
    sw = f_stat.getlength(stat)
    dot_r = 12
    gap = 24
    line_x = (W - (sw + gap + 2 * dot_r)) / 2
    cy = H - 200 + 22
    d.ellipse([line_x, cy - dot_r, line_x + 2 * dot_r, cy + dot_r], fill=STATUS)
    d.text((line_x + 2 * dot_r + gap, cy - 28), stat, font=f_stat, fill=MUTED)

    return img


def main():
    os.makedirs(SITE, exist_ok=True)
    out = os.path.join(SITE, 'banner.png')
    build().save(out, quality=95)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
