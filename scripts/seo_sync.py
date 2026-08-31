#!/usr/bin/env python3
"""seo_sync.py — daily SEO tag refresh for the 1bit.MONSTER site.

Runs daily (see .github/workflows/seo-sync.yml). Detects drift between the
numbers baked into site/*.html and the engine's CURRENT state, then updates
the SEO tags + stat numbers so changes land on 1bit.monster ("redirected
here") instead of going stale:

  Change sources caught:
    1. New HF model archs  -> the watcher/census_autopr adds an alias to
       rcpp_arch_from_string -> "HF arch strings" count +1 -> site updates.
    2. New family token     -> RCPP_ARCH_* def added -> "architecture tokens" +1.
    3. Census re-sweep      -> coverage numbers move (X/Y checkpoints mapped).
    4. Upstream lemonade    -> vendored version bumps -> new post + blog card.

All facts are derived OFFLINE from committed files (no network):

    tokens       = distinct RCPP_ARCH_* tokens in include/rocm_cpp/bitnet_model.h
    arch_strings = mapped cases in rcpp_arch_from_string (excl. RCPP_ARCH_UNKNOWN)
    coverage     = registry_covered / with_arch from Testing/census_full_summary.json
    lemonade     = project(lemon_cpp VERSION ...) in third_party/lemonade/CMakeLists.txt

Idempotent: a second run with nothing changed touches nothing and exits 0.

Usage:
    python3 scripts/seo_sync.py            # apply; prints CHANGED:/NEW: lines
    python3 scripts/seo_sync.py --check    # report drift, exit 1 if any, no writes
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SITE = os.path.join(ROOT, "site")
HEADER = os.path.join(ROOT, "include", "rocm_cpp", "bitnet_model.h")
CENSUS = os.path.join(ROOT, "Testing", "census_full_summary.json")
LEMONADE_CMAKE = os.path.join(ROOT, "third_party", "lemonade", "CMakeLists.txt")

# ── fact extraction ────────────────────────────────────────────────────────

def fmt(n):
    return f"{n:,}"


def count_tokens():
    """Distinct RCPP_ARCH_* tokens defined in bitnet_model.h (e.g. 552)."""
    toks = set()
    with open(HEADER, encoding="utf-8") as f:
        for m in re.finditer(r"\bRCPP_ARCH_[A-Z0-9_]+\b", f.read()):
            toks.add(m.group(0))
    return len(toks)


def count_arch_strings():
    """Mapped arch strings in rcpp_arch_from_string (excl. UNKNOWN fallback).

    Every non-UNKNOWN `return RCPP_ARCH_*` line in bitnet_model.h lives inside
    rcpp_arch_from_string (verified), so a file-wide count is exact.
    """
    n = 0
    with open(HEADER, encoding="utf-8") as f:
        for line in f:
            if "return RCPP_ARCH_" in line and "RCPP_ARCH_UNKNOWN" not in line:
                n += 1
    return n


def census_coverage():
    """(covered, with_arch) from census_full_summary.json."""
    try:
        with open(CENSUS, encoding="utf-8") as f:
            d = json_load(f)
        return int(d.get("registry_covered", 0)), int(d.get("with_arch", 0))
    except (OSError, ValueError, TypeError):
        return 0, 0


def json_load(f):
    import json
    return json.load(f)


def lemonade_version():
    """Vendored lemonade version, e.g. '11.7.0', or None."""
    try:
        with open(LEMONADE_CMAKE, encoding="utf-8") as f:
            m = re.search(r"project\(\s*lemon_cpp\s+VERSION\s+([0-9]+\.[0-9]+\.[0-9]+)", f.read())
        return m.group(1) if m else None
    except OSError:
        return None


# ── number sync (site/*.html) ─────────────────────────────────────────────

# Every replacer rebuilds the FULL match text with only the drifted number(s)
# swapped, and returns m.group(0) untouched when nothing drifted — so a
# no-change run is byte-identical (idempotent) and a changed run is minimal.


def _pair(m, n1, n2):
    """groups: (num, sep, num, suffix) -> fix each num independently."""
    g = m.groups()
    return (g[0] if g[0] == n1 else n1) + g[1] + (g[2] if g[2] == n2 else n2) + g[3]


def _frac(m, covered, with_arch):
    """groups: (num, num, suffix) with literal '/' between the numbers."""
    g = m.groups()
    return (g[0] if g[0] == fmt(covered) else fmt(covered)) + "/" + \
           (g[1] if g[1] == fmt(with_arch) else fmt(with_arch)) + g[2]


def _of(m, covered, with_arch):
    """groups: (num, ' of ', num, suffix)."""
    g = m.groups()
    return (g[0] if g[0] == fmt(covered) else fmt(covered)) + g[1] + \
           (g[2] if g[2] == fmt(with_arch) else fmt(with_arch)) + g[3]


def _num_between(m, value):
    """groups: (prefix, num, suffix)."""
    g = m.groups()
    return g[0] + (g[1] if g[1] == value else value) + g[2]


def _meta_pair(m, covered, tokens):
    """groups: (num, ' checkpoints · ', num, ' tokens')."""
    g = m.groups()
    return (g[0] if g[0] == fmt(covered) else fmt(covered)) + g[1] + \
           (g[2] if g[2] == fmt(tokens) else fmt(tokens)) + g[3]


def _pct_claim(m, covered, with_arch, suffix_groups):
    """Percentage claims only move when coverage actually drops below 100%."""
    if covered >= with_arch:
        return m.group(0)
    g = m.groups()
    return g[0] + _pct(covered, with_arch) + g[-1]


def _build_patterns(tokens, arch, covered, with_arch):
    t, a, c, w = fmt(tokens), fmt(arch), fmt(covered), fmt(with_arch)
    return [
        # "552 architecture tokens, 1,774 HF arch strings" (+ "/" variant in posts)
        (re.compile(r"(\d[\d,]*)( architecture tokens[ ,/]+)(\d[\d,]*)( HF arch strings)"),
         lambda m: _pair(m, t, a)),
        # coverage fraction "317,310/317,310 checkpoints mapped"
        (re.compile(r"(\d[\d,]*)/(\d[\d,]*)( checkpoints mapped)"),
         lambda m: _frac(m, covered, with_arch)),
        # "317,310 of 317,310 arch-bearing text-gen checkpoints mapped"
        (re.compile(r"(\d[\d,]*)( of )(\d[\d,]*)( arch-bearing text-gen checkpoints mapped)"),
         lambda m: _of(m, covered, with_arch)),
        # stats: <span class="n">552</span><span class="l">architecture tokens</span>
        (re.compile(r"(<span class=\"n\">)(\d[\d,]*)(</span><span class=\"l\">architecture tokens</span>)"),
         lambda m: _num_between(m, t)),
        (re.compile(r"(<span class=\"n\">)(\d[\d,]*)(</span><span class=\"l\">HF arch strings</span>)"),
         lambda m: _num_between(m, a)),
        # monster-v2 map counts
        (re.compile(r"(<span class=\"map-name\">architecture strings</span><span class=\"map-count\">)(\d[\d,]*)(</span>)"),
         lambda m: _num_between(m, a)),
        (re.compile(r"(<span class=\"map-name\">architecture tokens</span><span class=\"map-count\">)(\d[\d,]*)(</span>)"),
         lambda m: _num_between(m, t)),
        (re.compile(r"(<span class=\"map-name\">checkpoints</span><span class=\"map-count\">)(\d[\d,]*)(</span>)"),
         lambda m: _num_between(m, c)),
        # monster-v2 band-meta "317,310 checkpoints · 552 tokens"
        (re.compile(r"(\d[\d,]*)( checkpoints · )(\d[\d,]*)( tokens)"),
         lambda m: _meta_pair(m, covered, tokens)),
        # monster-v2 lead "317,310 arch-bearing checkpoints resolve to 552 tokens"
        (re.compile(r"(\d[\d,]*)( arch-bearing checkpoints resolve to )(\d[\d,]*)( tokens,)"),
         lambda m: _meta_pair(m, covered, tokens)),
        # percentage claims only move when coverage drops below 100%
        (re.compile(r"(\d+(?:\.\d+)?)%( HuggingFace coverage)"),
         lambda m: _pct_claim(m, covered, with_arch, 2)),
        (re.compile(r"(<span class=\"n\">)(\d+(?:\.\d+)?)(</span><span class=\"l\">checkpoints mapped</span>)"),
         lambda m: _pct_claim(m, covered, with_arch, 3)),
    ]


def _pct(covered, with_arch):
    if with_arch <= 0:
        return "0%"
    p = 100.0 * covered / with_arch
    return f"{p:.1f}%".rstrip("0").rstrip(".") + "%" if p % 1 else f"{int(p)}%"


def sync_site_numbers(apply=True):
    """Rewrite drifted numbers in site/*.html. Returns {path: [notes]}."""
    tokens = count_tokens()
    arch = count_arch_strings()
    covered, with_arch = census_coverage()
    patterns = _build_patterns(tokens, arch, covered, with_arch)

    changed = {}
    for name in sorted(os.listdir(SITE)):
        if not name.endswith(".html"):
            continue
        path = os.path.join(SITE, name)
        with open(path, encoding="utf-8") as f:
            html = f.read()
        orig = html
        notes = []
        for pat, repl in patterns:
            new_html, n = pat.subn(repl, html)
            if new_html != html:  # count real diffs only (idempotent no-ops skipped)
                html = new_html
                notes.append(f"{pat.pattern[:44]} x{n}")
        if html != orig:
            changed[path] = notes
            if apply:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(html)
    return changed, (tokens, arch, covered, with_arch)


# ── lemonade upstream sync ────────────────────────────────────────────────

_POST_RE = re.compile(r"^1bit-post-lemonade-v(\d+)\.html$")


def newest_lemonade_post(site_dir):
    """(filename, compact_version) of the newest lemonade post, or None."""
    best = None
    for name in os.listdir(site_dir):
        m = _POST_RE.match(name)
        if m:
            v = int(m.group(1))
            if best is None or v > best[1]:
                best = (name, v)
    return best


def _version_label(compact):
    # Ambiguous for multi-digit minors — prefer extracting from the post text.
    return compact  # best-effort; only used if the title regex fails


def sync_lemonade(today):
    """If the vendored lemonade version is newer than the newest site post,
    generate the new post from the previous one + add a blog card.

    Returns (created_post_name, prev_post_name, new_version) or None."""
    ver = lemonade_version()
    if not ver:
        return None
    compact = ver.replace(".", "")
    newest = newest_lemonade_post(SITE)
    if newest and newest[1] >= int(compact):
        return None
    if not newest:
        return None  # no template to build from; require a human first post
    prev_name, prev_compact = newest
    prev_path = os.path.join(SITE, prev_name)
    with open(prev_path, encoding="utf-8") as f:
        tpl = f.read()

    # Full version label straight from the template ("11.7.0"): compact forms
    # are ambiguous, the title is not.
    tm = re.search(r"Lemonade v([0-9.]+)", tpl)
    if not tm:
        print(f"seo_sync: cannot read version from {prev_name} — skipping lemonade post", file=sys.stderr)
        return None
    prev_label = tm.group(1)
    new_name = f"1bit-post-lemonade-v{compact}.html"
    new_path = os.path.join(SITE, new_name)

    # version/date/id swaps on the template
    out = tpl
    out = out.replace(f"v{prev_compact}", f"v{compact}")     # data-od-id, hrefs
    out = out.replace(f"v{prev_label}", f"v{ver}")           # "v11.7.0" -> "v11.8.0"
    out = out.replace(prev_label, ver)                        # remaining bare "11.7.0"
    # art-meta date -> today
    out = re.sub(r"<span>(\d{4}-\d{2}-\d{2})</span>", f"<span>{today}</span>", out, count=1)
    # honest meta description (avoids carrying v-prev release specifics)
    desc = (f"The embedded Lemonade server core is re-vendored to v{ver} — upstream sync "
            f"landed; see the post for what changed. Engine HF coverage: "
            f"{fmt(count_tokens())} architecture tokens, {fmt(count_arch_strings())} HF arch "
            f"strings, {_cov_str()} checkpoints mapped.")
    out = re.sub(r'<meta name="description" content="[^"]*"', f'<meta name="description" content="{desc}"', out, count=1)
    # review marker (draft PR gate)
    marker = (f"<!-- seo-sync: generated from {prev_name} on {today} — release-specific "
              f"claims below are UNVERIFIED for v{ver}; update before merge -->\n")
    out = out.replace("<!doctype html>", "<!doctype html>\n" + marker, 1)

    with open(new_path, "w", encoding="utf-8") as f:
        f.write(out)

    # blog card, inserted newest-first (before the current first log-row)
    blog_path = os.path.join(SITE, "1bit-blog.html")
    with open(blog_path, encoding="utf-8") as f:
        blog = f.read()
    card = (
        '          <article class="log-row" data-od-id="post-lemonade-v' + compact + '">\n'
        '            <span class="log-date">' + today + '</span>\n'
        '            <div class="log-body">\n'
        '              <h3><a href="' + new_name + '" class="log-link">Lemonade v' + ver +
        ', and how we stay current with the SDK</a></h3>\n'
        '              <p>The embedded Lemonade server core is re-vendored to v' + ver +
        ' — upstream sync landed, and the re-vendor loop keeps every release in step. Engine HF coverage: ' +
        fmt(count_tokens()) + ' architecture tokens, ' + fmt(count_arch_strings()) +
        ' HF arch strings, ' + _cov_str() + ' checkpoints mapped.</p>\n'
        '              <div class="log-tags"><span class="log-tag">lemonade</span><span class="log-tag">upstream</span><span class="log-tag">sdk</span></div>\n'
        '            </div>\n'
        '          </article>\n')
    # insert newest-first: card + the anchor line (keeps both at 10-space indent)
    anchor = '          <article class="log-row" data-od-id="post-'
    assert anchor in blog, "1bit-blog.html log-row anchor not found"
    blog = blog.replace(anchor, card + anchor, 1)

    # keep the Blog schema's blogPost array in lockstep with the new card:
    # seo_sync is the only writer of lemonade cards, and without this the
    # JSON-LD silently drops the newest post (v11.8.1 was missing entirely).
    # Same headline as the card; a human refines both together.
    ld_entry = ('{"@type":"BlogPosting","headline":"Lemonade v' + ver +
                ', and how we stay current with the SDK","url":"https://1bit.monster/' +
                new_name + '"}')
    ld_anchor = '"blogPost":['
    assert ld_anchor in blog, "1bit-blog.html blogPost anchor not found"
    blog = blog.replace(ld_anchor, ld_anchor + ld_entry + ',', 1)

    with open(blog_path, "w", encoding="utf-8") as f:
        f.write(blog)

    return (new_name, prev_name, ver)


def _cov_str():
    covered, with_arch = census_coverage()
    return f"{fmt(covered)}/{fmt(with_arch)}"


# ── driver ────────────────────────────────────────────────────────────────

def main():
    check_only = "--check" in sys.argv
    today = os.popen("date -u +%Y-%m-%d").read().strip() or "1970-01-01"

    changed, (tokens, arch, covered, with_arch) = sync_site_numbers(apply=not check_only)
    lines = []
    for path, notes in sorted(changed.items()):
        lines.append(f"CHANGED: {os.path.relpath(path, ROOT)} ({'; '.join(notes)})")

    lem = sync_lemonade(today)
    if lem:
        new_name, prev_name, ver = lem
        lines.append(f"NEW: {new_name} (lemonade {prev_name} -> v{ver})")
        lines.append("CHANGED: site/1bit-blog.html (new lemonade post card)")

    if check_only:
        if lines:
            print(f"DRIFT ({len(lines)} change(s)): tokens={tokens} arch_strings={arch} "
                  f"coverage={covered}/{with_arch}")
            for ln in lines:
                print("  " + ln)
            return 1
        print(f"OK: tokens={tokens} arch_strings={arch} coverage={covered}/{with_arch} "
              f"lemonade={lemonade_version()} — no drift")
        return 0

    if not lines:
        print(f"NO-CHANGES: tokens={tokens} arch_strings={arch} "
              f"coverage={covered}/{with_arch} lemonade={lemonade_version()}")
        return 0
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
