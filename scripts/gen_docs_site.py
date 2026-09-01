#!/usr/bin/env python3
"""gen_docs_site.py — build the docs.1bit.monster static documentation site.

Reads curated Markdown from the repo's ``docs/`` folder and emits a
self-contained static HTML site (no runtime dependencies, no JS framework) into
a ``docs-site/`` directory ready for Cloudflare Pages.

Design goals:
  * Curate only public-facing docs (guides, model families, wiki, reference).
    Internal notes (research/, journey.md, audit-trail.md, goals/, plans/,
    bugs/, bug-reports/, superpowers/, archive/, SEO drafts, ...) are excluded.
  * Rewrite relative ``.md`` links so they resolve to the generated HTML; any
    link that points outside the curated set becomes a GitHub blob/tree URL so
    nothing is a dead link.
  * Ship a small, dependency-free client-side full-text search.
  * Match the 1bit.MONSTER one-bit light identity (oklch tokens, modern-minimal).

Usage:
    python3 scripts/gen_docs_site.py --out docs-site [--site-name 1bit.MONSTER]

Requires ``markdown-it-py`` (pip install markdown-it-py).
"""
from __future__ import annotations

import argparse
import html as html_mod
import json
import posixpath
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

try:
    from markdown_it import MarkdownIt
except ImportError:  # pragma: no cover - exercised in CI
    sys.stderr.write(
        "gen_docs_site.py needs markdown-it-py. Install it first:\n"
        "  pip install markdown-it-py\n"
    )
    sys.exit(1)

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

REPO = "1bit-MONSTER/1bit-MONSTER"
DOCS_ROOT = "docs"
REPO_BASE = f"https://github.com/{REPO}"
BLOB_URL = f"{REPO_BASE}/blob/main/{DOCS_ROOT}"
TREE_URL = f"{REPO_BASE}/tree/main/{DOCS_ROOT}"

# (docs-rel-path, nav-group, optional display-title override)
CURATED: list[tuple[str, str, str | None]] = [
    # Overview
    ("README.md", "Overview", None),
    ("CODEBASE.md", "Overview", "Codebase reference"),
    ("jarvis.md", "Overview", "JARVIS — the voice pipeline"),
    # Guides
    ("guides/getting-started.md", "Guides", None),
    ("guides/building.md", "Guides", None),
    ("guides/architecture.md", "Guides", None),
    ("guides/launch.md", "Guides", "Launch & serving"),
    ("guides/windows.md", "Guides", "Windows notes"),
    ("guides/roadmap.md", "Guides", None),
    ("guides/Lemonade-Compat.md", "Guides", "Lemonade compatibility"),
    # Model families
    ("model-families/README.md", "Models", None),
    ("model-families/zyphra.md", "Models", None),
    ("model-families/qwen.md", "Models", None),
    ("model-families/llama.md", "Models", None),
    ("model-families/mistral.md", "Models", None),
    ("model-families/gemma.md", "Models", None),
    ("model-families/phi.md", "Models", None),
    ("model-families/falcon.md", "Models", None),
    ("model-families/olmo.md", "Models", None),
    ("model-families/granite.md", "Models", None),
    ("model-families/smollm.md", "Models", None),
    ("model-families/deepseek.md", "Models", None),
    ("model-families/gpt-oss.md", "Models", None),
    ("model-families/laguna.md", "Models", None),
    ("model-families/kimi.md", "Models", None),
    ("model-families/bitnet-bonsai.md", "Models", None),
    ("model-families/whisper.md", "Models", None),
    # Wiki
    ("wiki/models.md", "Wiki", "Supported models (SSOT)"),
    ("wiki/performance.md", "Wiki", "Benchmarks (SSOT)"),
    ("wiki/Installation.md", "Wiki", "Installation"),
    ("wiki/npu-architecture.md", "Wiki", "NPU architecture"),
    ("wiki/boot-configuration.md", "Wiki", "Boot configuration"),
    ("wiki/Network-Topology.md", "Wiki", "Network topology"),
    ("wiki/decisions.md", "Wiki", "Design decisions"),
    ("wiki/rdna4-gemm.md", "Wiki", "RDNA4 GEMM"),
    ("wiki/unsloth-dynamic-ggufs.md", "Wiki", "Dynamic GGUF Q4NX"),
    ("wiki/vek280-offline-box.md", "Wiki", "VEK280 offline box"),
    ("wiki/voice-trained-tutor.md", "Wiki", "Voice-trained tutor"),
    # Reference
    ("mesh-protocol.md", "Reference", "Mesh protocol"),
    ("sherry-format.md", "Reference", "Sherry model format"),
    ("llama.cpp-fork.md", "Reference", "llama.cpp fork"),
    ("engine_comparison_report.md", "Reference", "Engine comparison"),
    ("vendored-fastflowlm.md", "Reference", "Vendored FastFlowLM"),
    ("aiesim-debugging.md", "Reference", "AIE Sim debugging"),
    ("agentic-control-protocol.md", "Reference", "Agentic control protocol"),
    ("mobile/RUNBOOK.md", "Reference", "Mobile runbook"),
]

NAV_ORDER = ["Overview", "Guides", "Models", "Wiki", "Reference"]

# Files intentionally excluded from the public site, even if linked from
# curated pages. We never fail on these — links to them become GitHub URLs.
INTERNAL_FILES = {
    "journey.md",
    "audit-trail.md",
    "TRIAGE-ISSUES-2026-08.md",
    "AGENT-COORDINATION.md",
    "agent-role-prompts.md",
    "reddit-zaya-dflash-draft.md",
    "search-console-setup.md",
    "seo-gap-analysis.md",
    "validation-gaps.md",
    "e2e-token-verify-1699.md",
    "zyphra-handoff-2026-08-05.md",
}
INTERNAL_DIRS = (
    "research/",
    "archive/",
    "plans/",
    "goals/",
    "bugs/",
    "bug-reports/",
    "superpowers/",
)
INTERNAL_TOPCODES = ("goals", "plans", "bugs", "bug-reports", "research", "archive", "superpowers")

# --------------------------------------------------------------------------- #
# Link rewriting
# --------------------------------------------------------------------------- #

_BLOB = re.compile(r"^https?://")


def _ensure_docs_rel(target_repo: str) -> str:
    """Turn a repo-absolute path into a 'docs/'-relative path, or None."""
    if target_repo == DOCS_ROOT:
        return "README.md"
    if target_repo.startswith(DOCS_ROOT + "/"):
        return target_repo[len(DOCS_ROOT) + 1:]
    return None


def _looks_internal(docsrel: str) -> bool:
    if docsrel in INTERNAL_FILES:
        return True
    head = docsrel.split("/", 1)[0]
    return head in INTERNAL_TOPCODES


def rewrite_link(href: str, src_rel: str) -> str:
    """Rewrite a markdown/HTML link href for the generated site."""
    if not href:
        return href
    href = href.strip()
    if _BLOB.match(href) or href.startswith(("//", "mailto:", "tel:")):
        return href
    # Split into path / query / fragment.
    m = re.match(r"^(.*?)(\?[^#]*)?(#.*)?$", href)
    path = m.group(1) or ""
    query = m.group(2) or ""
    frag = m.group(3) or ""
    if not path or path.startswith("#"):
        return href

    src_repo = f"{DOCS_ROOT}/{src_rel}"
    # Resolve against the directory of the *source* file in repo space.
    target_repo = posixpath.normpath(posixpath.join(posixpath.dirname(src_repo), path))

    docsrel = _ensure_docs_rel(target_repo)
    if docsrel is not None and docsrel in RENDERED:
        out_target = RENDERED[docsrel]
        out_src_dir = posixpath.dirname(RENDERED[src_rel])
        rel = posixpath.relpath(out_target, start=out_src_dir)
        if rel == ".":
            rel = posixpath.basename(out_target)
        return rel + query + frag

    # directory link (e.g. 'model-families/') -> its README/index if rendered
    if docsrel is not None and not path.endswith(".md"):
        for cand in (docsrel.rstrip("/") + "/README.md", docsrel.rstrip("/") + "/index.md"):
            if cand in RENDERED:
                out_target = RENDERED[cand]
                out_src_dir = posixpath.dirname(RENDERED[src_rel])
                rel = posixpath.relpath(out_target, start=out_src_dir)
                return rel + query + frag

    # Internal/out-of-scope target -> GitHub blob (file) or tree (dir/base).
    if docsrel is not None and _looks_internal(docsrel):
        if path.endswith(".md"):
            return f"{BLOB_URL}/{target_repo[len(DOCS_ROOT)+1:]}" + query + frag
        return f"{TREE_URL}/{target_repo[len(DOCS_ROOT)+1:]}" + query + frag
    if docsrel is None:
        # Escaped the docs/ tree (e.g. ../../CONTRIBUTING.md, ../../benchmarks/...)
        if path.endswith(".md"):
            return f"{REPO_BASE}/blob/main/{target_repo}" + query + frag
        return f"{REPO_BASE}/tree/main/{target_repo}" + query + frag
    # Remaining docs-relative file that is not rendered -> GitHub blob.
    return f"{BLOB_URL}/{docsrel}" + query + frag


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def build_markdown() -> MarkdownIt:
    md = MarkdownIt("commonmark", {"html": True}).enable("table").enable("strikethrough")
    renderer = md.renderer  # capture for the closure below

    def _link_open(tokens, idx, options, env):
        token = tokens[idx]
        if isinstance(token.attrs, dict) and "href" in token.attrs:
            token.attrs["href"] = rewrite_link(token.attrs["href"], env.get("src_rel", ""))
        else:
            a_index = token.attrIndex("href")
            if a_index >= 0:
                token.attrs[a_index][1] = rewrite_link(token.attrs[a_index][1], env.get("src_rel", ""))
        return renderer.renderToken(tokens, idx, options, env)

    md.renderer.rules["link_open"] = _link_open
    return md


def slugify(text: str) -> str:
    """Approximate GitHub's heading-anchor slugger."""
    s = text.strip().lower()
    s = re.sub(r"[^a-z0-9 _-]", "", s)
    s = re.sub(r"[\s_]+", "-", s)
    return s.strip("-")


def anchor_headers(html: str) -> str:
    """Add id attributes to h1-h4 for in-page anchor links."""
    seen: dict[str, int] = {}
    counter = [0]

    def repl(m: re.Match) -> str:
        level = m.group(1)
        inner = m.group(2)
        text = re.sub(r"<[^>]+>", "", inner)
        base = slugify(html_mod.unescape(text)) or f"section-{counter[0]}"
        if base in seen:
            seen[base] += 1
            base = f"{base}-{seen[base]}"
        else:
            seen[base] = 0
        counter[0] += 1
        return f"<h{level} id=\"{base}\">{inner}</h{level}>"

    return re.sub(r"<h([1-4])[^>]*>(.*?)</h\1>", repl, html, flags=re.S)


def first_paragraph_text(html: str) -> str:
    m = re.search(r"<p[^>]*>(.*?)</p>", html, re.S)
    if not m:
        return ""
    text = html_mod.unescape(re.sub(r"<[^>]+>", "", m.group(1)))
    return re.sub(r"\s+", " ", text).strip()


def page_title(text: str, override: str | None, path: str) -> str:
    if override:
        return override
    m = re.search(r"^#\s+(.+)$", text, re.M)
    if m:
        return m.group(1).strip()
    return Path(path).stem.replace("_", " ").replace("-", " ").title()


# --------------------------------------------------------------------------- #
# HTML shell / nav / search
# --------------------------------------------------------------------------- #

CSS = """
:root{
  --bg:oklch(99% .002 240); --surface:oklch(100% 0 0); --fg:oklch(18% .012 250);
  --muted:oklch(54% .012 250); --border:oklch(92% .005 250); --accent:oklch(58% .18 255);
  --accent-soft:color-mix(in oklch,var(--accent) 12%, transparent);
  --code-bg:oklch(96% .004 250); --mono:ui-monospace,'SF Mono',SFMono-Regular,Menlo,Consolas,monospace;
  --font:-apple-system,BlinkMacSystemFont,'SF Pro Text',system-ui,sans-serif;
  --fs-body:16px; --fs-meta:13px; --radius:10px; --container:1200px; --gutter:28px;
}
*{box-sizing:border-box} html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--bg);color:var(--fg);font-family:var(--font);font-size:var(--fs-body);line-height:1.62}
a{color:var(--accent);text-decoration:none} a:hover{text-decoration:underline}
.header{position:sticky;top:0;z-index:20;background:color-mix(in oklch,var(--surface) 90%, transparent);backdrop-filter:blur(10px);border-bottom:1px solid var(--border)}
.header .wrap{max-width:var(--container);margin:0 auto;padding:14px var(--gutter);display:flex;align-items:center;gap:16px}
.brand{display:flex;align-items:center;gap:10px;font-weight:700;letter-spacing:-.02em;color:var(--fg);text-decoration:none}
.brand:hover{text-decoration:none}
.brand .dot{width:12px;height:12px;border-radius:3px;background:var(--accent);display:inline-block}
.brand .sub{color:var(--muted);font-weight:500}
.search{margin-left:auto;min-width:220px;position:relative}
.search input{width:100%;padding:8px 12px;border:1px solid var(--border);border-radius:9px;background:var(--surface);font-size:var(--fs-meta);color:var(--fg);outline:none}
.search input:focus{border-color:var(--accent)}
.results{position:absolute;top:calc(100% + 6px);right:0;left:0;background:var(--surface);border:1px solid var(--border);border-radius:10px;box-shadow:0 12px 40px color-mix(in oklch,var(--fg) 12%, transparent);overflow:hidden;display:none;max-height:60vh;overflow-y:auto}
.results a{display:block;padding:10px 14px;color:var(--fg);border-bottom:1px solid var(--border)}
.results a:last-child{border-bottom:none}
.results a:hover{background:var(--accent-soft);text-decoration:none}
.results .t{font-weight:600;font-size:var(--fs-body)}
.results .m{color:var(--muted);font-size:var(--fs-meta);margin-top:2px}
.results .empty{padding:14px;color:var(--muted);font-size:var(--fs-meta)}
.layout{max-width:var(--container);margin:0 auto;display:grid;grid-template-columns:270px minmax(0,1fr);gap:48px;padding:40px var(--gutter) 80px}
.nav{position:sticky;top:70px;align-self:start;max-height:calc(100vh - 90px);overflow-y:auto;padding-right:8px}
.nav h4{font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:var(--muted);margin:26px 0 8px}
.nav a{display:block;padding:6px 10px;border-radius:8px;color:var(--fg);font-size:14px}
.nav a:hover{background:var(--accent-soft);text-decoration:none}
.nav a.active{background:var(--accent-soft);color:var(--accent);font-weight:600}
.content{min-width:0}
.content h1{font-size:clamp(30px,4vw,44px);line-height:1.1;margin:0 0 8px;letter-spacing:-.02em}
.content .lead{color:var(--muted);font-size:18px;margin:0 0 8px}
.breadcrumb{font-size:var(--fs-meta);margin:0 0 18px;color:var(--muted)}
.breadcrumb a{color:var(--muted)}
.breadcrumb a:hover{color:var(--accent)}
.content h2{font-size:26px;margin:2.2em 0 .6em;letter-spacing:-.01em}
.content h3{font-size:19px;margin:1.8em 0 .5em}
.content h4{font-size:16px;margin:1.6em 0 .4em}
.content p{margin:0 0 1em}
.content blockquote{margin:0 0 1em;padding:2px 18px;border-left:3px solid var(--accent);color:var(--muted)}
.content code{font-family:var(--mono);font-size:14px;background:var(--code-bg);padding:2px 6px;border-radius:6px}
.content pre{background:var(--code-bg);border:1px solid var(--border);border-radius:var(--radius);padding:16px 18px;overflow-x:auto;margin:0 0 1em}
.content pre code{background:none;padding:0}
.content table{width:100%;border-collapse:collapse;margin:0 0 1em;font-size:15px}
.content th,.content td{border:1px solid var(--border);padding:9px 12px;text-align:left;vertical-align:top}
.content th{background:var(--code-bg);font-weight:600}
.content tr:nth-child(even) td{background:color-mix(in oklch,var(--fg) 2%, transparent)}
.content ul,.content ol{margin:0 0 1em;padding-left:24px}
.content li{margin:3px 0}
.content hr{border:none;border-top:1px solid var(--border);margin:2em 0}
.content img{max-width:100%;border-radius:10px}
.footer{border-top:1px solid var(--border);padding:40px var(--gutter) 60px;color:var(--muted);font-size:14px}
.footer .wrap{max-width:var(--container);margin:0 auto;display:flex;flex-wrap:wrap;gap:16px;justify-content:space-between;align-items:center}
.footer a{color:var(--muted)}
.footer a:hover{color:var(--accent)}
@media(max-width:900px){.layout{grid-template-columns:1fr}.nav{position:static;max-height:none}}
"""

SEARCH_JS = r"""
(function(){
  var input=document.getElementById('docs-search');
  var box=document.getElementById('search-results');
  if(!input||!box)return;
  var idx=[],xhr=new XMLHttpRequest();
  xhr.open('GET','search.json');
  xhr.onload=function(){
    try{idx=JSON.parse(xhr.responseText)||[];}catch(e){idx=[];}
  };
  xhr.send();
  function esc(s){return s.replace(/[&<>"]/g,function(c){return{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c];});}
  function show(list){
    if(!list.length){box.innerHTML='<div class="empty">No matches.</div>';box.style.display='block';return;}
    var h='';
    list.slice(0,12).forEach(function(r){
      h+='<a href="'+r.url+'"><div class="t">'+esc(r.title)+'</div><div class="m">'+esc(r.lead)+'</div></a>';
    });
    box.innerHTML=h;box.style.display='block';
  }
  input.addEventListener('input',function(){
    var q=input.value.trim().toLowerCase();
    if(q.length<2){box.style.display='none';return;}
    var terms=q.split(/\s+/);
    var res=idx.filter(function(r){
      return terms.every(function(t){return (r.title+' '+r.text).toLowerCase().indexOf(t)>=0;});
    });
    show(res);
  });
  document.addEventListener('click',function(e){
    if(!input.contains(e.target)&&!box.contains(e.target))box.style.display='none';
  });
})();
"""


def page_shell(out: Path, *, title: str, lead: str, breadcrumb: str, nav_groups: dict,
               active_url: str, body: str, site_name: str, cur_dir: str) -> None:
    # Every nav/site link must be computed relative to THIS page's directory,
    # otherwise links on a page in a subdirectory resolve incorrectly.
    nav_html = []
    for group in NAV_ORDER:
        items = nav_groups.get(group)
        if not items:
            continue
        nav_html.append(f"<h4>{html_mod.escape(group)}</h4>")
        for item in items:
            cls = " class=\"active\"" if item["url"] == active_url else ""
            href = posixpath.relpath(item["url"], start=cur_dir or ".")
            nav_html.append(f"<a{cls} href=\"{href}\">{html_mod.escape(item['title'])}</a>")
    nav = "\n".join(nav_html)

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>{html_mod.escape(title)} · {html_mod.escape(site_name)} Docs</title>
<meta name="description" content="{html_mod.escape(lead[:180])}"/>
<link rel="canonical" href="{html_mod.escape('https://docs.1bit.monster/'+active_url.lstrip('/'))}"/>
<style>{CSS}</style>
</head>
<body>
<header class="header"><div class="wrap">
  <a class="brand" href="https://1bit.monster"><span class="dot"></span>1bit.MONSTER<span class="sub">Docs</span></a>
  <a class="brand" href="https://github.com/{REPO}" style="font-weight:500;font-size:14px">GitHub ↗</a>
  <div class="search"><input id="docs-search" type="search" placeholder="Search docs…"/><div id="search-results" class="results"></div></div>
</div></header>
<div class="layout">
  <nav class="nav">{nav}</nav>
  <main class="content">
    <div class="breadcrumb">{breadcrumb}</div>
    {body}
  </main>
</div>
<footer class="footer"><div class="wrap">
  <span>1bit.MONSTER — one engine, any model, zero Python. MIT licensed.</span>
  <span>Docs generated from <a href="{REPO_BASE}">the repo</a> · <a href="https://1bit.monster">Main site</a> · <a href="https://discord.gg/Qy38d4Xu2h" target="_blank" rel="noopener">Discord</a></span>
</div></footer>
<script>{SEARCH_JS}</script>
<script src="https://context7.com/widget.js" data-library="/1bit-monster/1bit-monster"></script>
</body>
</html>"""
    out.write_text(html_doc, encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs-site", help="output directory")
    ap.add_argument("--site-name", default="1bit.MONSTER", help="brand name")
    ap.add_argument("--docs-root", default=DOCS_ROOT, help="docs/ source dir")
    args = ap.parse_args()

    docs = Path(args.docs_root)
    if not docs.is_dir():
        sys.stderr.write(f"docs root not found: {docs}\n")
        return 1
    site = Path(args.out)
    if site.exists():
        shutil.rmtree(site)
    site.mkdir(parents=True)

    global RENDERED
    RENDERED = {}  # docs-rel path -> output-rel path
    META = {}  # docs-rel path -> {title, url, group, order}
    index_of: dict[str, str] = {}  # docs-rel -> [index.html | README.html path]

    # Output path mapping (mirror the docs/ structure; README -> index.html)
    for rel, _group, _ov in CURATED:
        src = docs / rel
        if not src.is_file():
            sys.stderr.write(f"warning: missing curated doc: {rel}\n")
            continue
        if rel.endswith("README.md"):
            base = rel[: -len("README.md")]  # ''  or 'model-families/' etc.
            out_rel = base + "index.html"
        else:
            out_rel = rel[: -3] + ".html"
        RENDERED[rel] = out_rel

    md = build_markdown()
    nav_groups: dict[str, list[dict]] = {}

    # ── Pass 1: collect nav/titles for every page (so nav is complete for all) ──
    for rel, group, override in CURATED:
        if rel not in RENDERED:
            continue
        src = docs / rel
        text = src.read_text(encoding="utf-8")
        title = page_title(text, override, rel)
        nav_groups.setdefault(group, []).append({
            "url": RENDERED[rel], "title": title, "rel": rel,
        })

    order_by_rel = {rel: i for i, (rel, *_r) in enumerate(CURATED)}
    for group in nav_groups:
        nav_groups[group].sort(key=lambda it: order_by_rel.get(it["rel"], 10**9))

    # ── Pass 2: render every page (nav now fully populated) ──
    search_entries: list[dict] = []
    for rel, group, override in CURATED:
        if rel not in RENDERED:
            continue
        src = docs / rel
        text = src.read_text(encoding="utf-8")
        title = page_title(text, override, rel)
        body = md.render(text, env={"src_rel": rel})
        body = anchor_headers(body)
        lead = first_paragraph_text(body) or title

        out_rel = RENDERED[rel]
        out_file = site / out_rel
        out_file.parent.mkdir(parents=True, exist_ok=True)

        if out_rel == "index.html":
            breadcrumb = ""
            active_url = ""
        else:
            active_url = out_rel
            parent = posixpath.dirname(out_rel)
            up = posixpath.relpath("index.html", start=parent or ".")
            breadcrumb = f'<a href="{up}">Docs</a>'
            if group and group != "Overview":
                breadcrumb += f' <span>›</span> <span>{html_mod.escape(group)}</span>'
            breadcrumb += f' <span>›</span> <span>{html_mod.escape(title)}</span>'

        page_shell(out_file, title=title, lead=lead, breadcrumb=breadcrumb,
                   nav_groups=nav_groups, active_url=active_url, site_name=args.site_name,
                   body=body, cur_dir=posixpath.dirname(out_rel))

        plain = html_mod.unescape(re.sub(r"<[^>]+>", " ", body))
        plain = re.sub(r"\s+", " ", plain).strip()
        search_entries.append({"title": title, "url": "/" + out_rel,
                               "lead": lead, "text": plain[:20000]})

    # Write search index
    (site / "search.json").write_text(
        json.dumps(search_entries, ensure_ascii=False), encoding="utf-8")

    # Write _redirects.
    # Clean URLs (/path from /path.html, /section/ from index.html) are handled
    # by Cloudflare Pages automatically; do NOT add competing rewrite rules or
    # they loop (308) against Pages' own clean-URL behavior. Only convenience
    # redirects live here.
    redirects = [
        "# Cloudflare Pages redirect rules for docs.1bit.monster",
        "# Generated by scripts/gen_docs_site.py",
        "# Clean URLs are served by Cloudflare Pages automatically (X.html -> X).",
        "",
        "# Convenience redirects",
        "/github  https://github.com/1bit-MONSTER/1bit-MONSTER  302",
        "/gh      https://github.com/1bit-MONSTER/1bit-MONSTER  302",
    ]
    (site / "_redirects").write_text("\n".join(redirects), encoding="utf-8")

    # robots.txt + sitemap
    (site / "robots.txt").write_text(
        "User-agent: *\nAllow: /\nSitemap: https://docs.1bit.monster/sitemap.xml\n",
        encoding="utf-8")
    sitemap = ['<?xml version="1.0" encoding="UTF-8"?>',
               '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">']
    for rel, _g, _o in CURATED:
        if rel in RENDERED:
            loc = "https://docs.1bit.monster/" + RENDERED[rel]
            sitemap.append(f"  <url><loc>{loc}</loc></url>")
    sitemap.append("</urlset>")
    (site / "sitemap.xml").write_text("\n".join(sitemap), encoding="utf-8")

    print(f"built {len(RENDERED)} pages → {site}/")
    print(f"search entries: {len(search_entries)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
