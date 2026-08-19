#!/usr/bin/env python3
"""1bit MONSTER — merch design generator.

Designs the full merch line in the home-page design language
(navy #021621 / cyan #00e5ff / amber #ff9f1c / blue #0e80be,
DM Serif Display + DM Mono) and emits three artifacts per product:

  site/assets/merch/art-<name>.png      print-ready transparent PNG (300dpi scale)
  site/assets/merch/art-<name>.svg      vector master (DM fonts by name)
  site/assets/merch/preview-<name>.png  web store preview (900x600, navy card)

Run from the repo root:  python3 scripts/merch/generate_merch.py
"""
import os
import xml.sax.saxutils as sax

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FONT_DIR = os.path.join(ROOT, 'scripts', 'merch', 'fonts')
OUT = os.path.join(ROOT, 'site', 'assets', 'merch')

SERIF = 'DMSerifDisplay-Regular.ttf'
SERIF_I = 'DMSerifDisplay-Italic.ttf'
MONO = 'DMMono-Regular.ttf'
MONO_M = 'DMMono-Medium.ttf'

# palette — home page
CYAN = (0, 229, 255)
AMBER = (255, 159, 28)
BLUE = (14, 128, 190)
TEXT = (231, 246, 253)
MUTED = (160, 217, 248)
DIM = (93, 158, 194)
NAVY = (2, 22, 33)
PANEL = (4, 32, 47)
GREEN = (0, 200, 83)
RED = (255, 92, 92)

C = lambda rgb: '#%02x%02x%02x' % rgb
FAM_SERIF = "'DM Serif Display', Georgia, serif"
FAM_MONO = "'DM Mono', ui-monospace, Menlo, monospace"

_fonts = {}


def F(name, size):
    key = (name, size)
    if key not in _fonts:
        _fonts[key] = ImageFont.truetype(os.path.join(FONT_DIR, name), size)
    return _fonts[key]


# ─────────────────────────── element model ───────────────────────────
# ops:
#  ('txt',  cx, cy, text, font, size, fill, tracking)      centered text
#  ('txtl', x,  y,  text, font, size, fill, tracking)      left text (y = baseline-ish middle)
#  ('line', x1,y1,x2,y2,w,fill)
#  ('rect', x,y,w,h,r,fill,ow,ofill)
#  ('circ', cx,cy,r,fill,ow,ofill)
#  ('poly', pts,fill,ow,ofill)
#  ('arc',  x1,y1,x2,y2,start,end,w,fill)
#  ('grad', x1,y1,x2,y2,c1,c2)       vertical gradient (full canvas)
#  ('glow', cx,cy,r,color,alpha)     radial glow blob


def _text_font(font):
    return {
        'serif': SERIF, 'serif-i': SERIF_I, 'mono': MONO, 'mono-m': MONO_M,
    }[font]


def _text_family(font, italic=False):
    return {'serif': FAM_SERIF, 'serif-i': FAM_SERIF + ';font-style:italic',
            'mono': FAM_MONO, 'mono-m': FAM_MONO}[font]


def render_png(ops, w, h):
    img = Image.new('RGBA', (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for op in ops:
        k = op[0]
        if k == 'grad':
            x1, y1, x2, y2, c1, c2 = op[1:]
            steps = max(1, y2 - y1)
            for i in range(steps + 1):
                t = i / steps
                col = tuple(int(c1[j] + (c2[j] - c1[j]) * t) for j in range(3))
                d.line([(x1, y1 + i), (x2, y1 + i)], fill=col + (255,))
        elif k == 'gradh':
            x1, y1, x2, y2, c1, c2 = op[1:]
            steps = max(1, x2 - x1)
            for i in range(steps + 1):
                t = i / steps
                col = tuple(int(c1[j] + (c2[j] - c1[j]) * t) for j in range(3))
                d.line([(x1 + i, y1), (x1 + i, y2)], fill=col + (255,))
        elif k == 'glow':
            cx, cy, r, color, alpha = op[1:]
            blob = Image.new('RGBA', (2 * r, 2 * r), (0, 0, 0, 0))
            bd = ImageDraw.Draw(blob)
            bd.ellipse([0, 0, 2 * r, 2 * r], fill=color + (alpha,))
            blob = blob.filter(ImageFilter.GaussianBlur(r * 0.45))
            img.alpha_composite(blob, (int(cx - r), int(cy - r)))
        elif k == 'txt':
            cx, cy, text, font, size, fill, tracking = op[1:]
            f = F(_text_font(font), size)
            if tracking:
                widths = [f.getlength(ch) for ch in text]
                total = sum(widths) + tracking * (len(text) - 1)
                x = cx - total / 2
                for ch, wdt in zip(text, widths):
                    d.text((x + wdt / 2, cy), ch, font=f, fill=fill + (255,), anchor='mm')
                    x += wdt + tracking
            else:
                d.text((cx, cy), text, font=f, fill=fill + (255,), anchor='mm')
        elif k == 'txtl':
            x, y, text, font, size, fill, tracking = op[1:]
            f = F(_text_font(font), size)
            if tracking:
                xc = x
                for ch in text:
                    d.text((xc, y), ch, font=f, fill=fill + (255,), anchor='lm')
                    xc += f.getlength(ch) + tracking
            else:
                d.text((x, y), text, font=f, fill=fill + (255,), anchor='lm')
        elif k == 'line':
            x1, y1, x2, y2, w, fill = op[1:]
            d.line([(x1, y1), (x2, y2)], fill=fill + (255,), width=w)
        elif k == 'rect':
            x, y, w, h, r, fill, ow, ofill = op[1:]
            if fill:
                d.rounded_rectangle([x, y, x + w, y + h], radius=r, fill=fill + (255,))
            if ow:
                d.rounded_rectangle([x, y, x + w, y + h], radius=r, outline=ofill + (255,), width=ow)
        elif k == 'circ':
            cx, cy, r, fill, ow, ofill = op[1:]
            if fill:
                d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fill + (255,))
            if ow:
                d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=ofill + (255,), width=ow)
        elif k == 'poly':
            pts, fill, ow, ofill = op[1:]
            if fill:
                d.polygon(pts, fill=fill + (255,))
            if ow:
                d.line(pts + [pts[0]], fill=ofill + (255,), width=ow, joint='curve')
        elif k == 'arc':
            x1, y1, x2, y2, start, end, w, fill = op[1:]
            d.arc([x1, y1, x2, y2], start, end, fill=fill + (255,), width=w)
    return img


def svg_doc(ops, w, h):
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">']
    for op in ops:
        k = op[0]
        if k == 'grad':
            x1, y1, x2, y2, c1, c2 = op[1:]
            parts.append(f'<defs><linearGradient id="g" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}">'
                         f'<stop offset="0" stop-color="{C(c1)}"/><stop offset="1" stop-color="{C(c2)}"/>'
                         f'</linearGradient></defs><rect x="{x1}" y="{y1}" width="{x2 - x1}" height="{y2 - y1}" fill="url(#g)"/>')
        elif k == 'gradh':
            x1, y1, x2, y2, c1, c2 = op[1:]
            parts.append(f'<defs><linearGradient id="gh" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}">'
                         f'<stop offset="0" stop-color="{C(c1)}"/><stop offset="1" stop-color="{C(c2)}"/>'
                         f'</linearGradient></defs><rect x="{x1}" y="{y1}" width="{x2 - x1}" height="{y2 - y1}" fill="url(#gh)"/>')
        elif k == 'glow':
            cx, cy, r, color, alpha = op[1:]
            parts.append(f'<defs><radialGradient id="gl" cx="0.5" cy="0.5" r="0.5">'
                         f'<stop offset="0" stop-color="{C(color)}" stop-opacity="{alpha / 255}"/>'
                         f'<stop offset="1" stop-color="{C(color)}" stop-opacity="0"/>'
                         f'</radialGradient></defs><circle cx="{cx}" cy="{cy}" r="{r}" fill="url(#gl)"/>')
        elif k == 'txt':
            cx, cy, text, font, size, fill, tracking = op[1:]
            parts.append(f'<text x="{cx}" y="{cy}" text-anchor="middle" dominant-baseline="central" '
                        f'font-family="{_text_family(font)}" font-size="{size}" fill="{C(fill)}" '
                        f'letter-spacing="{tracking}">{sax.escape(text)}</text>')
        elif k == 'txtl':
            x, y, text, font, size, fill, tracking = op[1:]
            parts.append(f'<text x="{x}" y="{y}" text-anchor="start" dominant-baseline="central" '
                        f'font-family="{_text_family(font)}" font-size="{size}" fill="{C(fill)}" '
                        f'letter-spacing="{tracking}">{sax.escape(text)}</text>')
        elif k == 'line':
            x1, y1, x2, y2, w, fill = op[1:]
            parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{C(fill)}" stroke-width="{w}"/>')
        elif k == 'rect':
            x, y, w, h, r, fill, ow, ofill = op[1:]
            stroke = f' stroke="{C(ofill)}" stroke-width="{ow}"' if ow else ''
            parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{C(fill) if fill else "none"}"{stroke}/>')
        elif k == 'circ':
            cx, cy, r, fill, ow, ofill = op[1:]
            stroke = f' stroke="{C(ofill)}" stroke-width="{ow}"' if ow else ''
            parts.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{C(fill) if fill else "none"}"{stroke}/>')
        elif k == 'poly':
            pts, fill, ow, ofill = op[1:]
            stroke = f' stroke="{C(ofill)}" stroke-width="{ow}"' if ow else ''
            pts_s = ' '.join(f'{x},{y}' for x, y in pts)
            parts.append(f'<polygon points="{pts_s}" fill="{C(fill) if fill else "none"}"{stroke}/>')
        elif k == 'arc':
            x1, y1, x2, y2, start, end, w, fill = op[1:]
            parts.append(f'<path d="M {(x1 + x2) / 2} {(y1 + y2) / 2}" fill="none"/>')
            # approximate: stroke arc via path — use circle stroke-dasharray trick
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            rx, ry = (x2 - x1) / 2, (y2 - y1) / 2
            parts.append(f'<ellipse cx="{cx}" cy="{cy}" rx="{rx}" ry="{ry}" fill="none" stroke="{C(fill)}" '
                        f'stroke-width="{w}" stroke-dasharray="{(end - start) / 360 * 6.283 * rx} {(6.283 * rx)}" '
                        f'transform="rotate(0 {cx} {cy})"/>')
    parts.append('</svg>')
    return '\n'.join(parts)


def emit(name, ops, art_size, preview=True, art=True):
    os.makedirs(OUT, exist_ok=True)
    w, h = art_size
    art_img = render_png(ops, w, h) if (art or preview) else None
    if art:
        art_img.save(os.path.join(OUT, f'art-{name}.png'))
        with open(os.path.join(OUT, f'art-{name}.svg'), 'w') as f:
            f.write(svg_doc(ops, w, h))
    if preview:
        pw, ph = 900, 600
        base = Image.new('RGBA', (pw, ph), (0, 0, 0, 0))
        # navy gradient + glows (home-page look)
        base = render_png([
            ('grad', 0, 0, pw, ph, (8, 44, 60), NAVY),
            ('glow', 120, 40, 420, CYAN, 26),
            ('glow', 800, 60, 380, AMBER, 22),
            ('glow', 450, 620, 460, BLUE, 18),
        ], pw, ph)
        bd = ImageDraw.Draw(base)
        bd.rounded_rectangle([14, 14, pw - 14, ph - 14], radius=20, outline=DIM + (80,), width=1)
        # crop tall/wide artwork to its content for a focused card image
        src = art_img
        if art_img:
            alpha = art_img.split()[3]
            bbox = alpha.getbbox()
            if bbox:
                pad = int(0.06 * max(art_img.size))
                x0, y0, x1, y1 = bbox
                x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
                x1 = min(art_img.size[0], x1 + pad); y1 = min(art_img.size[1], y1 + pad)
                src = art_img.crop((x0, y0, x1, y1))
            scale = min(0.72 * pw / src.size[0], 0.72 * ph / src.size[1])
            aw, ah = int(src.size[0] * scale), int(src.size[1] * scale)
            src = src.resize((aw, ah), Image.LANCZOS)
            alpha = src.split()[3]
            shadow = Image.new('RGBA', (aw + 40, ah + 40), (0, 0, 0, 0))
            sa = alpha.resize((aw + 40, ah + 40), Image.LANCZOS).point(lambda a: int(a * 0.5))
            shadow.putalpha(sa.filter(ImageFilter.GaussianBlur(14)))
            base.alpha_composite(shadow, (int((pw - aw) / 2) - 20 + 8, int((ph - ah) / 2) - 20 + 10))
            base.alpha_composite(src, (int((pw - aw) / 2), int((ph - ah) / 2)))
        base.convert('RGB').save(os.path.join(OUT, f'preview-{name}.png'), quality=92)
    print(f'  {name}: art={art} preview={preview}')


# ─────────────────────────── designs ───────────────────────────
S = 1200  # sticker canvas

def die_cut_round():
    """white-ish backing circle (die-cut) — transparent bg outside."""
    return [('circ', S / 2, S / 2, 545, (2, 22, 33), 4, (14, 128, 190))]


def design_sticker_1bit():
    """the logo mark itself — navy circle, amber ring, 1bm."""
    return [
        ('circ', 600, 400, 205, NAVY, 10, AMBER),
        ('txt', 600, 400, '1bm', 'mono-m', 175, CYAN, -8),
        ('txt', 600, 700, '1bit.MONSTER', 'serif', 140, TEXT, 0),
        ('line', 380, 800, 820, 800, 3, (14, 128, 190)),
        ('txt', 600, 870, 'ONE ENGINE · EVERY MODEL · ANY CHIP', 'mono-m', 44, MUTED, 4),
        ('txt', 600, 945, 'ZERO PYTHON · MIT · EST. 2026', 'mono', 34, DIM, 4),
    ]


def design_sticker_94toks():
    return die_cut_round() + [
        ('txt', 600, 440, '94', 'mono-m', 330, CYAN, 0),
        ('txt', 600, 660, 'tok/s', 'mono-m', 120, AMBER, 10),
        ('line', 380, 760, 820, 760, 3, (14, 128, 190)),
        ('txt', 600, 830, 'STRIX HALO · XDNA 2 · 50 TOPS', 'mono-m', 44, MUTED, 4),
        ('txt', 600, 905, 'MEASURED, NOT MARKETED', 'mono', 34, DIM, 4),
    ]


def design_sticker_npu():
    return die_cut_round() + [
        ('txt', 600, 470, 'NPU', 'serif', 280, TEXT, 0),
        ('txt', 600, 700, 'UNLOCKED', 'mono-m', 100, AMBER, 14),
        ('line', 380, 800, 820, 800, 3, (14, 128, 190)),
        ('txt', 600, 865, 'XDNA 2 · 32 TILES · INT8', 'mono-m', 44, MUTED, 4),
        ('txt', 600, 940, 'REVERSE-ENGINEERED · OPEN SOURCE', 'mono', 32, DIM, 4),
    ]


def design_sticker_void():
    pts = [(600, 210), (1040, 880), (160, 880)]
    return [('poly', pts, (2, 22, 33), 6, AMBER)] + [
        ('txt', 600, 470, 'I VOIDED MY', 'mono-m', 84, MUTED, 8),
        ('txt', 600, 620, 'NPU WARRANTY', 'mono-m', 130, AMBER, 8),
        ('line', 380, 760, 820, 760, 3, (14, 128, 190)),
        ('txt', 600, 830, 'PROCEED WITH CONFIDENCE', 'mono-m', 42, MUTED, 4),
        ('txt', 600, 900, '38 KB BINARY · NO REGRETS', 'mono', 34, DIM, 4),
    ]


def design_sticker_amd():
    cx, cy = 600, 420
    chip = [
        ('rect', cx - 220, cy - 220, 440, 440, 40, PANEL, 6, CYAN),
    ]
    for i in range(6):  # pins top/bottom
        x = cx - 150 + i * 60
        chip += [('rect', x - 10, cy - 290, 20, 60, 6, BLUE, 0, None),
                 ('rect', x - 10, cy + 230, 20, 60, 6, BLUE, 0, None)]
    for i in range(6):  # pins left/right
        y = cy - 150 + i * 60
        chip += [('rect', cx - 290, y - 10, 60, 20, 6, BLUE, 0, None),
                 ('rect', cx + 230, y - 10, 60, 20, 6, BLUE, 0, None)]
    chip += [
        ('txt', cx, cy - 40, 'AMD', 'serif', 120, TEXT, 0),
        ('txt', cx, cy + 90, 'UNLOCKED', 'mono-m', 62, CYAN, 10),
    ]
    return chip + [
        ('line', 380, 780, 820, 780, 3, (14, 128, 190)),
        ('txt', 600, 850, 'EDITION · REVERSE-ENGINEERED IN 4 DAYS', 'mono-m', 42, MUTED, 4),
        ('txt', 600, 925, 'HOMEBREWED DRIVER · MIT LICENSED', 'mono', 32, DIM, 4),
    ]


def design_sticker_jarvis():
    return die_cut_round() + [
        ('circ', 600, 360, 120, PANEL, 4, AMBER),
        ('txt', 600, 360, 'J', 'mono-m', 130, CYAN, 0),
        ('txt', 600, 590, '1bit', 'serif', 150, CYAN, 0),
        ('txt', 600, 760, 'JARVIS', 'serif', 150, AMBER, 0),
        ('line', 380, 850, 820, 850, 3, (14, 128, 190)),
        ('txt', 600, 915, 'PRIVATE AI · ON-DEVICE · ZERO CLOUD', 'mono-m', 42, MUTED, 4),
        ('txt', 600, 980, '94 tok/s · VOICE · VISION · RAG', 'mono', 32, DIM, 4),
    ]


def design_sticker_zero_python():
    return die_cut_round() + [
        ('txt', 600, 430, 'ZERO', 'serif', 200, CYAN, 0),
        ('txt', 600, 660, 'PYTHON', 'serif', 200, AMBER, 0),
        ('line', 380, 780, 820, 780, 3, (14, 128, 190)),
        ('txt', 600, 850, '100% C++23 · ONE 38 KB BINARY', 'mono-m', 44, MUTED, 4),
        ('txt', 600, 925, 'NPU + GPU + CPU · SAME BINARY', 'mono', 34, DIM, 4),
    ]


def design_sticker_sorry():
    return die_cut_round() + [
        ('txt', 600, 430, 'Sorry but', 'serif-i', 170, AMBER, 0),
        ('txt', 600, 650, 'not Sorry', 'serif-i', 170, AMBER, 0),
        ('line', 380, 780, 820, 780, 3, (14, 128, 190)),
        ('txt', 600, 860, '1bit.MONSTER', 'mono-m', 64, CYAN, 8),
        ('txt', 600, 950, 'WE UNLOCKED AMD\u2019S NPU', 'mono', 38, MUTED, 4),
    ]


def _sticker_collage(parts):
    """overlapping mini stickers composition for pack previews"""
    ops = []
    y = 0
    for label, cy, x0, x1, w0, w1 in parts:
        ops += [('line', x0, cy + 80, x1, cy + 80, 3, (14, 128, 190))]
    return ops


def design_sticker_3pack():
    """three mini round badges side by side"""
    ops = []
    r = 210
    for i, (t1, t2, col) in enumerate([
            ('1bit', '.MONSTER', CYAN),
            ('94', 'tok/s', AMBER),
            ('NPU', 'UNLOCKED', TEXT)]):
        cx = 600 + (i - 1) * 330
        cy = 470
        ops += [('circ', cx, cy, r, NAVY, 4, BLUE),
                ('txt', cx, cy - 55, t1, 'serif', 120, col, 0),
                ('txt', cx, cy + 75, t2, 'mono-m', 44, AMBER if col != AMBER else CYAN, 4)]
    ops += [
        ('line', 300, 810, 900, 810, 3, (14, 128, 190)),
        ('txt', 600, 885, 'STARTER STICKER PACK', 'mono-m', 52, MUTED, 6),
        ('txt', 600, 960, '3\u00d73" · DIE-CUT VINYL · WEATHERPROOF', 'mono', 36, DIM, 4),
    ]
    return ops


def design_sticker_mega():
    """10-up grid — the mega pack card"""
    ops = []
    labels = ['1bit', '94 tok/s', 'NPU', 'VOID', 'AMD',
              'JARVIS', 'PYTHON', 'SORRY', '50 TOPS', 'LOCAL AI']
    for i, lab in enumerate(labels):
        r = 120
        cx = 300 + (i % 5) * 150
        cy = 300 + (i // 5) * 340
        col = [CYAN, AMBER, TEXT, AMBER, BLUE, CYAN, AMBER, AMBER, CYAN, BLUE][i]
        ops += [('circ', cx, cy, r, NAVY, 3, BLUE),
                ('txt', cx, cy, lab, 'mono-m', 34, col, 0)]
    ops += [
        ('line', 300, 950, 900, 950, 3, (14, 128, 190)),
        ('txt', 600, 1010, 'LAPTOP STICKER MEGA PACK · 10', 'mono-m', 52, MUTED, 6),
        ('txt', 600, 1085, 'DIE-CUT · WEATHERPROOF · 3\u00d73"', 'mono', 36, DIM, 4),
    ]
    return ops


# ── apparel artwork (transparent print files) ──
AW, AH = 2400, 3000


def design_tshirt_sorry():
    return [
        ('txt', AW / 2, 1050, 'Sorry but not Sorry', 'serif-i', 200, AMBER, 0),
        ('line', 700, 1350, 1700, 1350, 5, (14, 128, 190)),
        ('txt', AW / 2, 1500, '1bit.MONSTER', 'mono-m', 90, CYAN, 12),
        ('txt', AW / 2, 1650, 'WE UNLOCKED THE NPU · YOU GET THE DRIP', 'mono', 46, MUTED, 6),
    ]


def design_tshirt_npu_front():
    return [
        ('txt', AW / 2, 1050, 'NPU', 'serif', 300, TEXT, 0),
        ('txt', AW / 2, 1420, 'UNLOCKED', 'mono-m', 120, AMBER, 20),
        ('line', 700, 1620, 1700, 1620, 5, (14, 128, 190)),
        ('txt', AW / 2, 1770, 'XDNA 2 · 32 TILES · 50 TOPS', 'mono-m', 62, MUTED, 8),
    ]


def _tile_grid(x, y, s, gap, cols, rows, colors):
    ops = []
    for r in range(rows):
        for c in range(cols):
            ops.append(('rect', x + c * (s + gap), y + r * (s + gap), s, s, 4, colors[(r * cols + c) % len(colors)], 0, None))
    return ops


def design_tshirt_npu_back():
    ops = [
        ('txt', AW / 2, 700, 'one engine.', 'serif', 210, CYAN, 0),
        ('txt', AW / 2, 980, 'every model.', 'serif', 210, TEXT, 0),
        ('txt', AW / 2, 1260, 'any chip.', 'serif-i', 210, AMBER, 0),
    ]
    ops += _tile_grid(760, 1560, 46, 18, 8, 4, [CYAN, AMBER, BLUE, TEXT])
    ops += [
        ('txt', AW / 2, 2060, 'NPU + GPU + CPU · SAME 38 KB BINARY', 'mono-m', 62, MUTED, 8),
        ('txt', AW / 2, 2200, 'ZERO PYTHON · MIT · OPEN SOURCE', 'mono', 46, DIM, 6),
    ]
    return ops


def design_hoodie_front():
    return [
        ('txt', AW / 2, 1150, '1bit.MONSTER', 'serif', 180, TEXT, 0),
        ('line', 700, 1400, 1700, 1400, 5, (14, 128, 190)),
        ('txt', AW / 2, 1560, 'ONE ENGINE · EVERY MODEL · ANY CHIP', 'mono-m', 62, CYAN, 8),
    ]


def design_hoodie_back():
    ops = [
        ('txt', AW / 2, 650, 'One engine.', 'serif', 220, CYAN, 0),
        ('txt', AW / 2, 940, 'Every model.', 'serif', 220, TEXT, 0),
        ('txt', AW / 2, 1230, 'Any chip.', 'serif-i', 220, AMBER, 0),
        ('txt', AW / 2, 1560, 'ZERO DEPENDENCIES · 38 KB · MIT', 'mono-m', 62, MUTED, 8),
        ('txt', AW / 2, 1700, 'THE RUNTIME HAS NO PYTHON. NEITHER DOES THE HOODIE.', 'mono', 44, DIM, 6),
    ]
    return ops


def design_jarvis_front():
    return [
        ('circ', AW / 2, 800, 170, PANEL, 6, AMBER),
        ('txt', AW / 2, 800, 'J', 'mono-m', 200, CYAN, 0),
        ('txt', AW / 2, 1150, '1bit JARVIS', 'serif', 170, TEXT, 0),
        ('line', 700, 1400, 1700, 1400, 5, (14, 128, 190)),
        ('txt', AW / 2, 1560, 'PRIVATE AI · ON-DEVICE · ZERO CLOUD', 'mono-m', 60, AMBER, 8),
        ('txt', AW / 2, 1700, 'CHAT · VOICE · VISION · RAG', 'mono', 44, MUTED, 6),
    ]


# ── mug / bottle / tattoo ──
def design_mug():
    return [
        ('txt', 1200, 560, '1bit', 'serif', 300, CYAN, 0),
        ('txt', 1200, 920, 'MONSTER', 'serif', 200, TEXT, 0),
        ('line', 640, 1150, 1760, 1150, 6, (14, 128, 190)),
        ('txt', 1200, 1300, '94 tok/s · ZERO CLOUD · MIT', 'mono-m', 76, AMBER, 10),
    ]


def design_bottle():
    return [
        ('txt', 600, 500, '100%', 'serif', 190, TEXT, 0),
        ('txt', 600, 760, 'LOCAL AI', 'serif', 190, CYAN, 0),
        ('txt', 600, 1040, '1bit.MONSTER', 'mono-m', 70, AMBER, 10),
        ('line', 260, 1180, 940, 1180, 5, (14, 128, 190)),
        ('txt', 600, 1300, '94 tok/s · STRIX HALO · XDNA 2', 'mono', 52, MUTED, 6),
        ('txt', 600, 1450, 'ZERO PYTHON IN THE RUNTIME', 'mono', 44, DIM, 6),
    ]


def design_tattoo():
    """6 mini badges on a sheet"""
    ops = []
    items = [
        ('1bit', '.MONSTER', CYAN), ('94', 'tok/s', AMBER),
        ('NPU', 'UNLOCKED', TEXT), ('ZERO', 'PYTHON', AMBER),
        ('J', 'JARVIS', CYAN), ('SORRY', 'NOT SORRY', AMBER),
    ]
    for i, (t1, t2, col) in enumerate(items):
        cx = 500 + (i % 3) * 1000
        cy = 600 + (i // 3) * 900
        ops += [('circ', cx, cy, 300, NAVY, 4, BLUE),
                ('txt', cx, cy - 80, t1, 'serif', 160, col, 0),
                ('txt', cx, cy + 110, t2, 'mono-m', 56, AMBER if col != AMBER else CYAN, 4)]
    ops += [
        ('line', 400, 1720, 2600, 1720, 5, (14, 128, 190)),
        ('txt', 1500, 1820, '1BIT MONSTER · TEMPORARY TATTOO SHEET', 'mono-m', 60, MUTED, 8),
        ('txt', 1500, 1895, 'SKIN-SAFE · 3\u20135 DAYS · WATERPROOF', 'mono', 42, DIM, 6),
    ]
    return ops


def main():
    os.makedirs(OUT, exist_ok=True)
    print('generating merch into', OUT)

    emit('sticker-1bit', design_sticker_1bit(), (S, S))
    emit('sticker-94toks', design_sticker_94toks(), (S, S))
    emit('sticker-npu', design_sticker_npu(), (S, S))
    emit('sticker-void', design_sticker_void(), (S, S))
    emit('sticker-amd', design_sticker_amd(), (S, S))
    emit('sticker-jarvis', design_sticker_jarvis(), (S, S))
    emit('sticker-zero-python', design_sticker_zero_python(), (S, S))
    emit('sticker-sorry', design_sticker_sorry(), (S, S))
    emit('sticker-3pack', design_sticker_3pack(), (S, S))
    emit('sticker-mega', design_sticker_mega(), (S, S))

    emit('tee-sorry', design_tshirt_sorry(), (AW, AH))
    emit('tee-npu-front', design_tshirt_npu_front(), (AW, AH))
    emit('tee-npu-back', design_tshirt_npu_back(), (AW, AH))
    emit('hoodie-front', design_hoodie_front(), (AW, AH))
    emit('hoodie-back', design_hoodie_back(), (AW, AH))
    emit('tee-jarvis', design_jarvis_front(), (AW, AH))

    emit('mug', design_mug(), (2400, 1200))
    emit('bottle', design_bottle(), (1200, 2400))
    emit('tattoo', design_tattoo(), (3000, 2000))

    print('done')


if __name__ == '__main__':
    main()
