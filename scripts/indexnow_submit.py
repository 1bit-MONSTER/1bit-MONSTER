#!/usr/bin/env python3
"""indexnow_submit.py — tell Bing/Yandex which pages changed, via IndexNow.

The site has hosted the protocol key file since #1993/#1994
(`site/<key>.txt`, served at `https://1bit.monster/<key>.txt`) but nothing ever
submitted with it, so search engines only learned about changes by re-crawling
the sitemap on their own schedule. This is the missing half.

IndexNow is one POST:

    POST https://api.indexnow.org/indexnow
    {"host": "1bit.monster", "key": "<key>",
     "keyLocation": "https://1bit.monster/<key>.txt",
     "urlList": ["https://1bit.monster/<page>", ...]}

Protocol responses: 200 OK, 202 Accepted (key validation pending), 400 bad
request, 403 key not valid / key file not found, 422 URLs do not belong to the
host, 429 too many requests.

Usage:
    indexnow_submit.py --changed-since <sha> [--head <sha>]   # pages in that range
    indexnow_submit.py --all                                  # every sitemap URL
    indexnow_submit.py --urls https://1bit.monster/ ...

    --dry-run      print the payload and send nothing
    --endpoint U   override the endpoint (testing)
    --site-dir D   default: site

Exit codes: 0 = submitted, nothing to submit, or dry run; 1 = configuration
error, or a response the protocol defines as a refusal (400/403/422);
2 = transient (429/5xx), which the caller may choose to ignore.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

HOST = "1bit.monster"
BASE = f"https://{HOST}"
ENDPOINT = "https://api.indexnow.org/indexnow"
MAX_URLS = 10_000  # protocol limit per request
KEY_RE = re.compile(r"^[A-Za-z0-9-]{8,128}$")

# HTTP status -> (exit code, explanation). IndexNow's own semantics.
STATUS = {
    200: (0, "OK — URLs accepted"),
    202: (0, "Accepted — key validation pending on their side"),
    400: (1, "Bad request — malformed payload"),
    403: (1, "Forbidden — key not valid, or the key file is not reachable at keyLocation"),
    422: (1, "Unprocessable — URLs do not belong to the host, or the key does not match"),
    429: (2, "Too many requests — rate limited, retry later"),
}


def find_key(site_dir: Path) -> tuple[str, str]:
    """Locate the hosted IndexNow key: a site/*.txt whose name is the key and
    whose content is the same key (that is the protocol's own requirement)."""
    for p in sorted(site_dir.glob("*.txt")):
        stem = p.stem
        if not KEY_RE.match(stem):
            continue
        try:
            body = p.read_text(encoding="utf-8", errors="ignore").strip()
        except OSError:
            continue
        if body == stem:
            return stem, f"{BASE}/{p.name}"
    raise SystemExit("indexnow: no key file found (expected site/<key>.txt containing <key>)")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout


def url_for(name: str) -> str | None:
    """site/<page>.html -> canonical URL; index.html is the slash form."""
    if not name.endswith(".html"):
        return None
    page = name[len("site/"):] if name.startswith("site/") else name
    if page == "index.html":
        return f"{BASE}/"
    return f"{BASE}/{page}"


def urls_changed(site_dir: Path, since: str, head: str) -> list[str]:
    out = []
    try:
        names = git("diff", "--name-only", "--diff-filter=ACMR", since, head,
                    "--", f"{site_dir}/").splitlines()
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"indexnow: git diff failed: {e.stderr.strip()}") from e
    for n in names:
        u = url_for(n)
        if u:
            out.append(u)
    return sorted(set(out))


def urls_sitemap(site_dir: Path) -> list[str]:
    sm = site_dir / "sitemap.xml"
    if not sm.exists():
        raise SystemExit(f"indexnow: {sm} not found (needed for --all)")
    locs = re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", sm.read_text(encoding="utf-8"))
    return sorted({u for u in locs if u.startswith(BASE)})


def submit(endpoint: str, payload: dict, dry_run: bool) -> int:
    body = json.dumps(payload).encode()
    print(f"indexnow: {len(payload['urlList'])} url(s) -> {endpoint}")
    for u in payload["urlList"][:10]:
        print(f"  {u}")
    if len(payload["urlList"]) > 10:
        print(f"  ... and {len(payload['urlList']) - 10} more")
    if dry_run:
        print("indexnow: dry run — nothing sent")
        return 0
    req = urllib.request.Request(endpoint, data=body,
                                 headers={"Content-Type": "application/json; charset=utf-8"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            code = resp.status
    except urllib.error.HTTPError as e:
        code = e.code
    except Exception as e:  # network failure: transient by nature
        print(f"indexnow: transport error: {e}")
        return 2
    rc, why = STATUS.get(code, (2 if code >= 500 else 1, "unexpected status"))
    print(f"indexnow: HTTP {code} — {why}")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description="Submit changed URLs to IndexNow (Bing/Yandex).")
    ap.add_argument("--site-dir", default="site")
    ap.add_argument("--changed-since", metavar="SHA")
    ap.add_argument("--head", default="HEAD", metavar="SHA")
    ap.add_argument("--all", action="store_true", help="submit every sitemap URL")
    ap.add_argument("--urls", nargs="*", default=[], help="explicit URLs")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--endpoint", default=ENDPOINT)
    a = ap.parse_args()

    site_dir = Path(a.site_dir)
    if a.urls:
        urls = sorted({u for u in a.urls if u.startswith(BASE)})
    elif a.all:
        urls = urls_sitemap(site_dir)
    elif a.changed_since:
        urls = urls_changed(site_dir, a.changed_since, a.head)
    else:
        ap.error("one of --changed-since / --all / --urls is required")

    if not urls:
        print("indexnow: no shareable page changed — nothing to submit")
        return 0
    if len(urls) > MAX_URLS:
        print(f"indexnow: truncating {len(urls)} urls to the protocol limit {MAX_URLS}")
        urls = urls[:MAX_URLS]

    key, key_location = find_key(site_dir)
    payload = {"host": HOST, "key": key, "keyLocation": key_location, "urlList": urls}
    return submit(a.endpoint, payload, a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
