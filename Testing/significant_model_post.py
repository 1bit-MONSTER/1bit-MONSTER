#!/usr/bin/env python3
"""significant_model_post.py — scaffold a DRAFT blog post for a significant
model-architecture arrival (e.g. a DeepSeek V4-flash that adds vision).

The Lemonade loop auto-publishes a post per SDK release. This mirrors that for
HF models: when the census/escalation flags a SIGNIFICANT architecture arrival
(major family, or a vision/multimodal variant of one), generate a new post so
the blog "stays current" the way it does for Lemonade.

The post is a DISCOVERABLE SCAFFOLD, not final prose: it carries the model
identity (arch class, example model id, family), the engine's coverage framing,
and full SEO tags (title/description/og/twitter/JSON-LD/canonical) so it is
findable the moment it is published. A human or PR-agent pass fills the real
details (what the new architecture does, decode validation status) before
merge. This mirrors the Lemonade `seo_sync` "NEW:" draft-PR pattern.

Usage:
    python3 Testing/significant_model_post.py \
        --arch <class> --model <hf_model_id> [--family <name>] [--date YYYY-MM-DD]

Idempotent: tracks generated posts in Testing/significant_posts.json so the
same arrival never double-posts. Env:
  CENSUS_DRY_RUN=1  print what would be written, don't write.
"""
import json, os, sys, datetime, re, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
TEMPLATE = SITE / "1bit-post-one-engine.html"
STATE = ROOT / "Testing" / "significant_posts.json"
SITE_URL = "https://1bit.monster"
OG_IMG = f"{SITE_URL}/assets/og-card.png"
AUTHOR = {"@type": "Organization", "name": "1bit.MONSTER", "url": SITE_URL}

STYLE_START, STYLE_END = "<style>", "</style>"
NAV_START, NAV_END = "<nav>", "</nav>"
FOOT, BODY_END = '  <footer class="pagefoot"', "</body>"


def _carve(tpl: str):
    style = tpl[tpl.index(STYLE_START): tpl.index(STYLE_END) + len(STYLE_END)]
    nav = tpl[tpl.index(NAV_START): tpl.index(NAV_END) + len(NAV_END)]
    foot = tpl[tpl.index(FOOT): tpl.index(BODY_END)]
    return style, nav, foot


def _slug(v: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", v.lower()).strip("-")
    return s or "model"


def _meta(model_id, arch, family, desc, mode="covered"):
    fam = family or arch
    if mode == "announcement":
        title = f"A new significant architecture arrived: {fam}"
        slug = f"1bit-post-announcement-{_slug(model_id)}"
    else:
        title = f"The engine now runs {fam}"
        slug = f"1bit-post-significant-{_slug(model_id)}"
    jsonld = {
        "@context": "https://schema.org", "@type": "BlogPosting",
        "headline": f"{title} | 1bit.MONSTER",
        "description": desc,
        "url": f"{SITE_URL}/{slug}.html",
        "mainEntityOfPage": f"{SITE_URL}/{slug}.html",
        "image": OG_IMG,
        "datePublished": "2026-09-02", "dateModified": "2026-09-02",
        "author": AUTHOR, "publisher": AUTHOR,
    }
    return title, slug, json.dumps(jsonld, separators=(",", ":"))


def render(model_id, arch, family, date, desc, body_paras, style, nav, foot, mode="covered"):
    title, slug, jsonld = _meta(model_id, arch, family, desc, mode=mode)
    fname = f"{slug}.html"
    body = "\n".join(f"        <p>{p}</p>" for p in body_paras)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{title} | 1bit.MONSTER</title>
  <meta name="description" content="{desc}" />
  <meta property="og:site_name" content="1bit.MONSTER" />
  <meta property="og:type" content="article" />
  <meta property="og:title" content="{title} | 1bit.MONSTER" />
  <meta property="og:description" content="{desc}" />
  <meta property="og:url" content="{SITE_URL}/{fname}" />
  <meta property="og:image" content="{OG_IMG}" />
  <meta name="twitter:card" content="summary_large_image" />
  <meta name="twitter:title" content="{title} | 1bit.MONSTER" />
  <meta name="twitter:description" content="{desc}" />
  <meta name="twitter:image" content="{OG_IMG}" />
  <link rel="canonical" href="{SITE_URL}/{fname}" />
  <script type="application/ld+json">
  {jsonld}
  </script>
{style}
  <script src="theme.js"></script>
</head>
<body>
  <header class="topnav" data-od-id="topnav">
    <div class="container topnav-inner" style="max-width: 1000px;">
      <a href="index.html" class="logo">
        <svg viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <rect class="relay r1" x="2" y="2" width="9" height="9" rx="2"/><rect class="relay r2" x="13" y="2" width="9" height="9" rx="2"/><rect class="relay r3" x="2" y="13" width="9" height="9" rx="2"/><rect class="relay r4" x="13" y="13" width="9" height="9" rx="2"/>
        </svg>
        <span>1bit.MONSTER</span>
      </a>
      {nav}
      <a class="btn btn-ghost" href="https://github.com/1bit-MONSTER/1bit-MONSTER">GitHub&nbsp;&#8599;</a>
    </div>
  </header>

  <main id="content">
    <article class="art container" data-od-id="{fname.replace('.html', '')}">
      <a class="back" href="1bit-blog.html">&larr; all posts</a>
      <div class="art-meta">
        <span>{date}</span>
        <span class="tag">{_slug(family or arch)}</span>
        <span class="tag">significant</span>
      </div>
      <h1>{title}</h1>
      <p class="lead">{desc}</p>
      <div class="prose">
{body}
      </div>
    </article>
  </main>

{foot}
  <script src="analytics.js"></script>
  <script src="https://context7.com/widget.js" data-library="/1bit-monster/1bit-monster"></script>
</body>
</html>
"""
    return fname, html


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", required=True, help="stripped architecture class, e.g. deepseek_v4_flash")
    ap.add_argument("--model", required=True, help="example HF model id, e.g. deepseek-ai/DeepSeek-V4-Flash")
    ap.add_argument("--family", default=None, help="family name for the title, e.g. DeepSeek V4")
    ap.add_argument("--date", default="2026-09-02")
    ap.add_argument("--desc", default=None)
    ap.add_argument("--mode", choices=["announcement", "covered"], default="covered",
                    help="announcement = arrival detected, support in progress; "
                         "covered = the engine now maps it")
    args = ap.parse_args()

    models = json.loads(STATE.read_text()) if STATE.exists() else {}
    key = f"{args.mode}:{args.model}"
    if key in models:
        print(f"[sigpost] already generated a {args.mode} post for {args.model} "
              f"({models[key]}) — skip")
        return 0

    family = args.family or args.arch
    if args.mode == "announcement":
        desc = args.desc or (f"A significant new model architecture just arrived — {family} "
                             f"(example: {args.model}). The engine is on it: this is the "
                             f"announcement, with support and decode validation in progress.")
        body = [
            f"<b>{family}</b> is a <b>significant architecture arrival</b> — a major-family or "
            f"vision/multimodal model we don't fully map yet. "
            f"Example checkpoint on HuggingFace: <span class=\"mono\">{args.model}</span> "
            f"(stripped class <span class=\"mono\">{args.arch}</span>).",
            "This is the announcement: the model exists, the engine is tracking it, and support + "
            "decode validation are underway. When it lands, a follow-up post says so.",
            "<b>Draft scaffold:</b> fill in what the architecture actually does and the "
            "support/validation status before publishing. This entry exists so the blog updates the "
            "way the Lemonade SDK loop does — a new post per significant arrival, not just a number change.",
        ]
    else:
        desc = args.desc or (f"A significant new model architecture arrived — {family} "
                             f"(example: {args.model}) — and the engine now maps it to a token. "
                             f"This post scaffolds the entry; details are a draft.")
        body = [
            f"<b>{family}</b> is a <b>significant architecture arrival</b> — "
            f"a major-family or vision/multimodal model the engine did not previously map. "
            f"Example checkpoint on HuggingFace: <span class=\"mono\">{args.model}</span> "
            f"(stripped class <span class=\"mono\">{args.arch}</span>).",
            "The engine maps it to an architecture token, so the whole class now resolves to one binary — "
            "the census claim stays at 100% coverage.",
            "<b>Draft scaffold:</b> fill in what the architecture actually does, kernel/backend support, "
            "and decode-validation status before publishing.",
            "The daily census keeps this live: the new-model watcher catches the arrival, the autopr drafts "
            "the alias, and this post is generated so the change is discoverable.",
        ]

    dry = os.getenv("CENSUS_DRY_RUN") == "1"
    tpl = TEMPLATE.read_text(encoding="utf-8")
    style, nav, foot = _carve(tpl)
    fname, html = render(args.model, args.arch, family, args.date, desc, body, style, nav, foot,
                         mode=args.mode)
    print(f"[sigpost] {'DRY-RUN ' if dry else ''}would write site/{fname} ({len(html)//1024} KB)")
    if dry:
        return 0
    (SITE / fname).write_text(html, encoding="utf-8")
    models[key] = {"arch": args.arch, "family": family, "date": args.date, "file": fname,
                   "mode": args.mode}
    STATE.write_text(json.dumps(models, indent=1, sort_keys=True))
    print(f"[sigpost] wrote site/{fname}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
