#!/usr/bin/env python3
"""site_theme_modes.py — add light / auto / dark theme modes to the 1bit.MONSTER site.

Design: the site is a static, self-hosted set of pages where every page inlines
its own <style> and token block (oklch vars in :root). No build step exists, so
this script stamps three pieces into each page deterministically:

  1. <script src="theme.js"></script>  in <head>  (reads pref, sets data-theme
     before first paint; no preference => light).
  2. A compact 3-segment .theme-switch (light / auto / dark) in the top nav,
     right before the GitHub link.
  3. Appended theme CSS before </style>: color-scheme, dark token overrides,
     the same overrides under @media (prefers-color-scheme: dark) for auto,
     plus targeted fixes for components that invert the palette (primary
     buttons, ink-console sections, merch tiles, terminal code blocks) and
     the switch's own styles.

Idempotent: pages that already carry the marker comment are skipped.

Usage: python3 scripts/site_theme_modes.py            # site/*.html
"""
import re
import sys
from pathlib import Path

SITE = Path(__file__).resolve().parent.parent / "site"
MARK = "1bit theme modes"
HEAD_SCRIPT = '\n  <script src="theme.js"></script>'

# ── dark overrides (shared by forced dark and by auto-under-dark) ────────────
# Each entry: (scope_selector_suffix, body). Scope suffix is appended after
# `:root[data-theme="dark"]` / `html[data-theme="dark"]` for the forced block
# and after the equivalent `[data-theme="auto"]` selector inside the media
# query for the auto block.
DARK_RULES = [
    ("", """
    color-scheme: dark;
    --bg:      oklch(16.5% 0.014 250);
    --surface: oklch(21% 0.016 250);
    --fg:      oklch(94% 0.01 250);
    --muted:   oklch(68% 0.02 250);
    --border:  oklch(29% 0.018 250);
    --accent:  oklch(74% 0.15 258);
    --status:  oklch(79% 0.13 156);
    /* pinned: these carry the "ink console / paper" identity in both modes */
    --ink-deep:      oklch(26% 0.011 250);
    --on-dark:       oklch(97% 0.004 250);
    --on-dark-muted: color-mix(in oklch, white 68%, transparent);
    --border-dark:   color-mix(in oklch, white 18%, transparent);
"""),
    (" .btn-primary", """
    background: color-mix(in oklch, var(--accent) 62%, black);
    color: oklch(98% 0.002 250);
"""),
    (" .btn-primary:hover", """
    background: color-mix(in oklch, var(--accent) 74%, black);
"""),
    (" .section-dark .btn-primary", """
    background: var(--accent-bright);
    color: var(--ink-deep);
"""),
    (" .section-dark .btn-primary:hover", """
    background: color-mix(in oklch, var(--accent) 88%, white);
"""),
    (" .codeblock", """
    background: oklch(23% 0.015 250);
    color: oklch(95% 0.005 250);
"""),
    (" .codeblock .c", """
    color: color-mix(in oklch, oklch(95% 0.005 250) 55%, transparent);
"""),
    # merch tiles stay light "product photos" on the dark page
    (" .product-tile", """
    background: linear-gradient(180deg, oklch(99.4% 0.002 240), oklch(96% 0.004 250));
"""),
    (" .sticker-tile", """
    background-color: oklch(96.8% 0.004 250);
    background-image: radial-gradient(color-mix(in oklch, oklch(18% 0.012 250) 16%, transparent) 1px, transparent 1px);
    color: oklch(26% 0.011 250);
"""),
    (" .product-tile .sheet", """
    fill: oklch(99.4% 0.002 240);
    stroke: oklch(26% 0.011 250);
"""),
    (" .product-tile .ink, .product-tile .p-ink, .product-tile .ink-stroke", """
    fill: oklch(26% 0.011 250);
"""),
    (" .product-tile .ink-stroke", """
    stroke: oklch(26% 0.011 250);
"""),
    (" .product-tile .ink2", "    fill: oklch(39% 0.009 250);\n"),
    (" .product-tile .sub", "    fill: oklch(52% 0.008 250);\n"),
    (" .product-tile .shade", "    fill: color-mix(in oklch, oklch(18% 0.012 250) 13%, transparent);\n"),
    (" .product-tile .p-mid", "    fill: oklch(45% 0.009 250);\n"),
    (" .product-tile .p-soft", "    fill: oklch(75% 0.005 250);\n"),
    (" .product-tile .p-foot", "    fill: color-mix(in oklch, oklch(18% 0.012 250) 16%, transparent);\n"),
    (" .product-tile .print, .product-tile .p-print, .product-tile .p-line, .product-tile .relay", """
    fill: oklch(99% 0.001 240);
"""),
    (" .product-tile .print-stroke", """
    stroke: oklch(99% 0.001 240);
"""),
]

SWITCH_CSS = """
    /* ─── theme switch (light / auto / dark) ───────────────────────── */
    .theme-switch {
      display: inline-flex; align-items: center; gap: 2px;
      padding: 3px; border: 1px solid var(--border); border-radius: 999px;
      background: color-mix(in oklch, var(--fg) 4%, transparent);
      flex: 0 0 auto;
    }
    .theme-switch .ts-btn {
      display: inline-flex; align-items: center; justify-content: center;
      width: 26px; height: 24px; padding: 0; border: 0; border-radius: 999px;
      background: transparent; color: var(--muted); cursor: pointer;
      transition: background 0.15s ease, color 0.15s ease;
    }
    .theme-switch .ts-btn:hover { color: var(--fg); }
    .theme-switch .ts-btn:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
    html[data-theme="light"] .theme-switch [data-choice="light"],
    html[data-theme="auto"]  .theme-switch [data-choice="auto"],
    html[data-theme="dark"]  .theme-switch [data-choice="dark"] {
      background: var(--surface); color: var(--fg);
      box-shadow: 0 1px 3px color-mix(in oklch, var(--fg) 18%, transparent);
    }
    @media (max-width: 720px) {
      .topnav-inner { flex-wrap: wrap; row-gap: 12px; }
      .topnav nav { order: 3; width: 100%; gap: var(--gap-md); justify-content: space-between; }
      .theme-switch { order: 4; }
    }
"""


def build_css() -> str:
    forced = []
    auto = []
    for suffix, body in DARK_RULES:
        body = "\n".join("    " + ln if ln.strip() else ln for ln in body.split("\n"))
        forced.append(f'    :root[data-theme="dark"]{suffix} {{{body}    }}\n')
    forced = "".join(forced)
    for suffix, body in DARK_RULES:
        auto.append(f'    :root[data-theme="auto"]{suffix} {{{body}    }}\n')
    auto = "".join(auto)
    return f"""    /* ─── {MARK} ────────────────────────────────────────────────
       light mode (default) | auto (follows OS) | dark mode.
       Appended by scripts/site_theme_modes.py — rerun to regenerate. */
    :root {{
      color-scheme: light;
    }}
    :root[data-theme="dark"] {{
      color-scheme: dark;
    }}
{forced}
    @media (prefers-color-scheme: dark) {{
{auto}    }}
    /* auto + light: nothing to override — the base tokens are already light. */
{SWITCH_CSS}
"""

SWITCH_HTML = """
      <span class="theme-switch" role="group" aria-label="Theme: light, auto, or dark">
        <button type="button" class="ts-btn" data-choice="light" title="Light mode" aria-label="Light mode" aria-pressed="false">
          <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" aria-hidden="true"><circle cx="12" cy="12" r="4.2"/><path d="M12 2.4v2.6M12 19v2.6M2.4 12H5M19 12h2.6M4.9 4.9l1.9 1.9M17.2 17.2l1.9 1.9M19.1 4.9l-1.9 1.9M6.8 17.2l-1.9 1.9"/></svg>
        </button>
        <button type="button" class="ts-btn" data-choice="auto" title="Auto (follow system)" aria-label="Auto (follow system)" aria-pressed="false">
          <svg viewBox="0 0 24 24" width="15" height="15" aria-hidden="true"><path d="M12 3a9 9 0 1 0 0 18V3Z" fill="currentColor"/><circle cx="12" cy="12" r="8.6" fill="none" stroke="currentColor" stroke-width="1.7"/></svg>
        </button>
        <button type="button" class="ts-btn" data-choice="dark" title="Dark mode" aria-label="Dark mode" aria-pressed="false">
          <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M20.2 14.6A8.4 8.4 0 1 1 9.4 3.8 6.9 6.9 0 0 0 20.2 14.6Z"/></svg>
        </button>
      </span>
"""


def strip_theme(html: str) -> str:
    """Reverse stamp(): remove the theme script tag, the switch markup, and the
    appended css block. Returns None if the page does not carry the theme."""
    if MARK not in html:
        return None
    # css block: from marker line start to the closing brace of SWITCH_CSS
    start = html.find("/* ─── " + MARK)
    end_marker = "    /* theme switch (light / auto / dark) ─"
    # the switch css block ends with the active-button rule + '}' — cut at the
    # line right before the page's original closing of <style>
    i_style = html.find("</style>", start)
    # find the start of our appended text: it was inserted right after the
    # original content that preceded '</style>'; locate by re-finding marker
    end = html.rfind("}\n", start, i_style)
    if start < 0 or end < 0:
        raise SystemExit(f"cannot strip theme css region")
    html = html[:start] + html[end + 2:]
    html = html.replace("\n  <script src=\"theme.js\"></script>\n</head>", "\n</head>", 1)
    html = html.replace(SWITCH_HTML + "\n", "", 1)
    return html


def stamp(html: str, with_switch: bool) -> str:
    if MARK in html:
        raise SystemExit(f"already themed — rerun would double-apply: skipped")
    # 1) head script
    html = html.replace("</head>", HEAD_SCRIPT + "\n</head>", 1)
    # 2) theme switch in the topnav, right before the GitHub link
    if with_switch:
        h_start = html.find("<header")
        h_end = html.find("</header>", h_start)
        if h_start < 0 or h_end < 0:
            raise SystemExit("no <header> found")
        head_slice = html[h_start:h_end]
        gh = head_slice.find('href="https://github.com/1bit-MONSTER/1bit-MONSTER"')
        if gh < 0:
            raise SystemExit("no github link in header")
        a_start = head_slice.rfind("<a", 0, gh)
        html = html[: h_start + a_start] + SWITCH_HTML + "\n" + html[h_start + a_start:]
    # 3) append theme css to the first style block
    css = build_css()
    i = html.find("</style>")
    if i < 0:
        raise SystemExit("no </style>")
    html = html[:i] + css + "\n" + html[i:]
    return html


def main() -> int:
    targets = sorted(SITE.glob("*.html"))
    done, skipped = 0, []
    if len(sys.argv) > 1 and sys.argv[1] == "--strip":
        for f in targets:
            text = f.read_text(encoding="utf-8")
            out = strip_theme(text)
            if out is None:
                continue
            f.write_text(out, encoding="utf-8")
            print(f"stripped {f.name}")
            done += 1
        print(f"\n{done} pages stripped")
        return 0
    for f in targets:
        text = f.read_text(encoding="utf-8")
        if f.name in ("sticker-gallery.html", "search.html"):
            skipped.append(f.name + " (special-case page)")
            continue
        if ":root" not in text:
            skipped.append(f.name + " (no token system)")
            continue
        if "<header" not in text:
            skipped.append(f.name + " (no topnav)")
            continue
        if MARK in text:
            skipped.append(f.name + " (already themed)")
            continue
        f.write_text(stamp(text, with_switch=True), encoding="utf-8")
        print(f"themed {f.name}")
        done += 1
    # search.html: token-based but headerless-nav (minimal header, no GitHub
    # link) — theme it and drop the switch on its own header row.
    sf = SITE / "search.html"
    if sf.exists():
        text = sf.read_text(encoding="utf-8")
        if ":root" in text and MARK not in text:
            text = text.replace("</head>", HEAD_SCRIPT + "\n</head>", 1)
            css = build_css() + """
    /* search page: theme switch rides the header row */
    .wrap > header { display: flex; flex-wrap: wrap; align-items: center; gap: 4px 14px; }
    .wrap > header .theme-switch { margin-left: auto; }
"""
            i = text.find("</style>")
            text = text[:i] + css + "\n" + text[i:]
            text = text.replace("</header>", SWITCH_HTML + "\n    </header>", 1)
            sf.write_text(text, encoding="utf-8")
            print(f"themed {sf.name} (custom header)")
            done += 1
    print(f"\n{done} pages themed; skipped {len(skipped)}:")
    for s in skipped:
        print("  -", s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
