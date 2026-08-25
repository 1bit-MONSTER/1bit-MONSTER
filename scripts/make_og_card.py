#!/usr/bin/env python3
"""make_og_card.py — generate the 1200x630 social share card.

Matches the locked 1-bit identity (black banner, white lettering, plain
sans-serif) and leads with the wow-numbers that make the project sharable.
Numbers pulled live from the site's own claims so the card can't drift.
"""
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

W, H = 1200, 630
BG = (10, 10, 12)          # near-black
FG = (245, 245, 248)       # near-white
ACCENT = (110, 150, 255)   # site accent-ish blue
MUTED = (150, 150, 160)

SITE_DIR = Path(__file__).resolve().parent.parent / "site"
OUT = SITE_DIR / "assets" / "og-card.png"

FONT_DIRS = [
    "/usr/share/fonts/truetype/dejavu",
    "/usr/share/fonts/truetype/liberation",
]


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    for d in FONT_DIRS:
        try:
            return ImageFont.truetype(f"{d}/{name}", size)
        except Exception:
            continue
    return ImageFont.load_default()


def pull_number(pattern, default):
    for f in SITE_DIR.glob("*.html"):
        m = re.search(pattern, f.read_text(encoding="utf-8", errors="ignore"))
        if m:
            return m.group(1)
    return default


def main():
    tokens = pull_number(r"(\d[\d,]*)\s*architecture tokens", "552")
    arch = pull_number(r"(\d[\d,]*)\s*HF arch strings", "1,775")
    coverage = pull_number(r"(\d+(?:\.\d+)?)%\s*HuggingFace coverage", "100") + "%"
    checkpoints = pull_number(r"(\d[\d,]*)\s*checkpoints mapped", "317,310")

    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)

    # header
    d.text((80, 52), "1bit.MONSTER", font=font(66, bold=True), fill=FG)
    d.text((80, 140), "One engine. Any model.", font=font(30), fill=MUTED)

    # stat band
    stats = [
        (tokens, "architecture tokens"),
        (arch, "HF arch strings"),
        (coverage, "HuggingFace coverage"),
    ]
    x = 80
    for big, label in stats:
        d.text((x, 230), big, font=font(92, bold=True), fill=ACCENT)
        d.text((x, 344), label, font=font(27), fill=MUTED)
        x += 370

    # footer
    d.text((80, 455), f"{checkpoints} arch-bearing checkpoints, one 1-bit engine",
           font=font(31), fill=FG)
    d.text((80, 515), "Ryzen AI NPU · ROCm · GGUF-native · FastFlowLM · MIT",
           font=font(26), fill=MUTED)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(OUT, "PNG")
    print(f"wrote {OUT} ({W}x{H})")
    print(f"  tokens={tokens} arch={arch} coverage={coverage} checkpoints={checkpoints}")


if __name__ == "__main__":
    main()
