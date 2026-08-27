#!/usr/bin/env python3
"""postgen.py — generate blog posts from content specs using the locked template.

Specs: dict of filename -> {title, description, tag, date, body: [("h2", ...), ("p", ...), ...]}
Usage: python3 scripts/postgen.py specs.json site/
"""
import json
import sys
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parent.parent / "site" / "1bit-post-one-engine.html"

STYLE_START = "<style>"
STYLE_END = "</style>"
NAV_START = '<nav>'
NAV_END = '</nav>'
FOOT = '  <footer class="pagefoot"'


def stamp(post: str, style: str, body: str) -> str:
    return post.replace(style, style).replace(NAV_START + "\n", NAV_START + "\n").replace(
        "      <a href=\"1bit-blog.html\" aria-current=\"page\">Blog</a>",
        "      <a href=\"1bit-blog.html\" aria-current=\"page\">Blog</a>")


def main() -> int:
    spec_path, site_dir = sys.argv[1], Path(sys.argv[2])
    specs = json.load(open(spec_path))
    tpl = (site_dir / "1bit-post-one-engine.html").read_text(encoding="utf-8")
    # carve out reusable pieces
    style = tpl[tpl.index(STYLE_START):tpl.index(STYLE_END) + len(STYLE_END)]
    nav = tpl[tpl.index(NAV_START):tpl.index(NAV_END) + len(NAV_END)]
    foot = tpl[tpl.index(FOOT):tpl.index("</body>")]

    for fname, spec in specs.items():
        lines = []
        for kind, text in spec["body"]:
            if kind == "h2":
                lines.append(f"        <h2>{text}</h2>")
            elif kind == "p":
                lines.append(f"        <p>{text}</p>")
            elif kind == "code":
                lines.append(f"        <pre>{text}</pre>")
            elif kind == "callout":
                lines.append(f'        <div class="callout"><p>{text}</p></div>')
            elif kind == "table":
                head, rows = text[0], text[1:]
                th = "".join(f"<th>{c}</th>" for c in head)
                tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
                lines.append(f"        <table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table>")
        body = "\n".join(lines)

        html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{spec["title"]} | 1bit.MONSTER</title>
  <meta name="description" content="{spec["description"]}" />
{style}
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
      <a class="btn btn-ghost" href="https://github.com/1bit-MONSTER/1bit-MONSTER">GitHub&nbsp;↗</a>
    </div>
  </header>

  <main id="content">
    <article class="art container" data-od-id="{fname.replace('.html','')}">
      <a class="back" href="1bit-blog.html">&larr; all posts</a>
      <div class="art-meta">
        <span class="tag">{spec.get("tag", "engine")}</span>
        <span>{spec.get("date", "2026-08-25")}</span>
      </div>
      <h1>{spec["title"]}</h1>
      <p class="lead">{spec.get("lead", spec["description"])}</p>
      <div class="prose">
{body}
      </div>
    </article>
  </main>

{foot}
  <script src="analytics.js"></script>
</body>
</html>
"""
        (site_dir / fname).write_text(html, encoding="utf-8")
        print(f"wrote site/{fname} ({len(html)//1024} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
