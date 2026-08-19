#!/usr/bin/env python3
"""1bit MONSTER — logo & branding generator.

The mark: navy circle, amber ring, "1bm" in cyan DM Mono — the circular
badge from the blog-post nav. Emits the full brand kit:

  site/assets/logo-mark.svg|png       the mark (THE logo)
  site/assets/logo.svg|png|webp       lockup: mark + "1bit.MONSTER"
  site/assets/mark.png                mark @1024
  site/assets/favicon.svg|ico         + favicon-16/32/512.png
  site/assets/apple-touch-icon.png    (180)
  site/assets/og-card.png             social share card (1200x630)
  site/assets/banner.png              README hero banner (2560x840)
  assets/ (repo root)                 favicons, android-chrome, og cards
  favicon.svg|ico|apple-touch-icon.png, site.webmanifest

Run from repo root:  python3 scripts/merch/generate_logo.py
"""
import os

from PIL import Image

import generate_merch as gm

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SITE = os.path.join(ROOT, 'site', 'assets')
ROOT_ASSETS = os.path.join(ROOT, 'assets')
SERIF, SERIF_I = gm.SERIF, gm.SERIF_I
MONO, MONO_M = gm.MONO, gm.MONO_M


def mark_ops(cx, cy, r, glow_r=0):
    ops = []
    if glow_r:
        ops.append(('glow', cx, cy, int(glow_r), gm.AMBER, 70))
    ops += [
        ('circ', cx, cy, r, gm.NAVY, max(2, int(r * 0.105)), gm.AMBER),
        ('txt', cx, cy, '1bm', 'mono-m', int(r * 0.94), gm.CYAN, -int(max(1, r * 0.045))),
    ]
    return ops


def lockup_ops(cx_mark, cy, mark_r, font_size, tag=None, tag_i=None, bar=True):
    """mark + '1bit.MONSTER' wordmark (1bit cyan, .MONSTER white) + gradient bar."""
    f = gm.F(SERIF, font_size)
    w1 = f.getlength('1bit')
    w2 = f.getlength('.MONSTER')
    x_txt = cx_mark + mark_r + font_size * 0.55
    ops = mark_ops(cx_mark, cy, mark_r, glow_r=mark_r * 1.8)
    ops += [
        ('txtl', x_txt, cy - font_size * 0.02, '1bit', 'serif', font_size, gm.CYAN, 0),
        ('txtl', x_txt + w1 + font_size * 0.04, cy - font_size * 0.02, '.MONSTER', 'serif', font_size, gm.TEXT, 0),
    ]
    if bar:
        x_end = x_txt + w1 + font_size * 0.04 + w2
        y_bar = cy + font_size * 0.58
        ops.append(('gradh', int(x_txt), int(y_bar), int(x_end), int(y_bar + max(3, font_size * 0.11)),
                    gm.CYAN, gm.AMBER))
    if tag:
        ops.append(('txtl', x_txt, cy + font_size * 0.78, tag, 'mono-m', int(font_size * 0.3), gm.MUTED, 4))
    if tag_i:
        ops.append(('txtl', x_txt, cy + font_size * 0.78 + int(font_size * 0.3) * 1.6, tag_i, 'mono', int(font_size * 0.24), gm.DIM, 3))
    return ops


def bg_ops(w, h):
    return [
        ('grad', 0, 0, w, h, (8, 44, 60), gm.NAVY),
        ('glow', int(w * 0.16), int(h * 0.1), int(w * 0.42), gm.CYAN, 22),
        ('glow', int(w * 0.85), int(h * 0.12), int(w * 0.38), gm.AMBER, 18),
    ]


def og_ops(w, h):
    ops = bg_ops(w, h)
    s = w / 1200.0
    ops += lockup_ops(int(190 * s), int(300 * s), int(150 * s), int(96 * s))
    x = int(190 * s) + int(150 * s) + int(96 * s * 0.55)
    y = int(440 * s)
    ops += [
        ('txtl', x, y, 'One engine. Every model. Any chip.', 'serif-i', int(48 * s), gm.AMBER, 0),
        ('txtl', x, y + int(70 * s), '1bit.monster · zero python · npu + gpu + cpu', 'mono', int(26 * s), gm.MUTED, 3),
        ('txtl', x, y + int(112 * s), 'MIT · OPEN SOURCE', 'mono-m', int(24 * s), gm.DIM, 4),
    ]
    return ops


def banner_ops(w, h):
    ops = bg_ops(w, h)
    s = w / 2560.0
    mark_cx, cy, mark_r, fs = int(360 * s), int(420 * s), int(230 * s), int(170 * s)
    ops += lockup_ops(mark_cx, cy, mark_r, fs)
    x_txt = mark_cx + mark_r + fs * 0.55
    f = gm.F(gm.SERIF, fs)
    x_mid = x_txt + (f.getlength('1bit') + fs * 0.04 + f.getlength('.MONSTER')) / 2
    y = int(620 * s)
    ops += [
        ('txt', x_mid, y, 'One engine. Every model. Any chip.', 'serif-i', int(84 * s), gm.AMBER, 0),
        ('txt', x_mid, y + int(120 * s), 'NPU + GPU + CPU · ZERO PYTHON', 'mono-m', int(42 * s), gm.MUTED, 5),
        ('gradh', int(40 * s), h - int(16 * s), w - int(40 * s), h - int(2 * s), gm.CYAN, gm.AMBER),
    ]
    return ops


def save_png(ops, w, h, path):
    gm.render_png(ops, w, h).convert('RGB').save(path, quality=94)


def save_svg(ops, w, h, path):
    with open(path, 'w') as f:
        f.write(gm.svg_doc(ops, w, h))


def main():
    os.makedirs(SITE, exist_ok=True)
    os.makedirs(ROOT_ASSETS, exist_ok=True)

    # mark (THE logo)
    save_svg(mark_ops(32, 32, 28.5, glow_r=56), 64, 64, os.path.join(SITE, 'logo-mark.svg'))
    save_png(mark_ops(256, 256, 228, glow_r=470), 512, 512, os.path.join(SITE, 'logo-mark.png'))
    save_png(mark_ops(512, 512, 455, glow_r=940), 1024, 1024, os.path.join(SITE, 'mark.png'))

    # lockup
    save_svg(lockup_ops(48, 32, 27, 30), 340, 64, os.path.join(SITE, 'logo.svg'))
    save_png(lockup_ops(160, 128, 114, 110), 1024, 256, os.path.join(SITE, 'logo.png'))
    gm.render_png(lockup_ops(160, 128, 114, 110), 1024, 256).convert('RGB').save(
        os.path.join(SITE, 'logo.webp'), quality=92)

    # favicons
    save_svg(mark_ops(16, 16, 14.2, glow_r=28), 32, 32, os.path.join(SITE, 'favicon.svg'))
    for size in (16, 32, 512):
        save_png(mark_ops(size / 2, size / 2, size * 0.445), size, size,
                 os.path.join(SITE, f'favicon-{size}x{size}.png'))
    # ico (16/32/48)
    imgs = [gm.render_png(mark_ops(s / 2, s / 2, s * 0.445), s, s).convert('RGB') for s in (16, 32, 48)]
    imgs[0].save(os.path.join(SITE, 'favicon.ico'), format='ICO', sizes=[(16, 16), (32, 32), (48, 48)])

    # apple touch + android chrome
    save_png(mark_ops(90, 90, 80, glow_r=170), 180, 180, os.path.join(SITE, 'apple-touch-icon.png'))
    save_png(mark_ops(256, 256, 228, glow_r=470), 512, 512, os.path.join(ROOT_ASSETS, 'android-chrome-512x512.png'))
    for size in (16, 32):
        save_png(mark_ops(size / 2, size / 2, size * 0.445), size, size,
                 os.path.join(ROOT_ASSETS, f'favicon-{size}x{size}.png'))

    # social / README imagery
    save_png(og_ops(1200, 630), 1200, 630, os.path.join(SITE, 'og-card.png'))
    save_png(og_ops(2400, 1260), 2400, 1260, os.path.join(ROOT_ASSETS, 'og-card.png'))
    save_png(og_ops(1200, 630), 1200, 630, os.path.join(ROOT_ASSETS, 'og-image.png'))
    save_png(banner_ops(2560, 840), 2560, 840, os.path.join(SITE, 'banner.png'))

    # repo-root favicon/apple-touch
    save_svg(mark_ops(16, 16, 14.2, glow_r=28), 32, 32, os.path.join(ROOT, 'favicon.svg'))
    imgs = [gm.render_png(mark_ops(s / 2, s / 2, s * 0.445), s, s).convert('RGB') for s in (16, 32, 48)]
    imgs[0].save(os.path.join(ROOT, 'favicon.ico'), format='ICO', sizes=[(16, 16), (32, 32), (48, 48)])
    save_png(mark_ops(90, 90, 80, glow_r=170), 180, 180, os.path.join(ROOT, 'apple-touch-icon.png'))

    # webmanifest icon path
    manifest = os.path.join(ROOT, 'site.webmanifest')
    with open(manifest) as f:
        m = f.read()
    m = m.replace('"/android-chrome-512x512.png"', '"/assets/favicon-512.png"')
    m = m.replace('"theme_color": "#0B0B0B"', '"theme_color": "#021621"').replace('"background_color": "#0B0B0B"', '"background_color": "#021621"')
    with open(manifest, 'w') as f:
        f.write(m)

    print('logo kit generated into', SITE, 'and', ROOT_ASSETS)


if __name__ == '__main__':
    main()
