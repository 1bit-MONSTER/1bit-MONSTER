#!/usr/bin/env python3
"""census_sweep.py — full HuggingFace text-generation enumeration ("the archive train").

Walks EVERY page of /api/models?pipeline_tag=text-generation&config=true and
rebuilds the raw census inputs that Testing/census_coverage.py consumes:

    Testing/census_arch_counts.json         {total, no_arch, counts: {arch: n, "<none>": no_arch}}
    Testing/census_modeltype_aggregate.json {model_type: {count, top_archs: [{arch, n}]}}
    Testing/census_model_index.json         {model_id_lower: [model_type]}

These are the same three files the 2026-08-15 snapshot was built from.
Re-running the sweep makes the census exact again — the daily full sweep that
supersedes the incremental watcher delta (Testing/hf_new_models.py), which only
samples the newest models.

config=true returns each model's config inline (model_type + architectures),
so this needs NO per-model config fetch. model_type falls back to the _text
tag, then "<none>" (mirroring the aggregate's existing <none> bucket). The
arch strings are recorded RAW (case preserved) — the coverage/tail stages do
their own strip_arch().

Pagination is the Link-header cursor (limit=1000), the same auth-free walk the
tail sweep uses. Everything is built in memory and written atomically, so a
partial failure leaves the previous census files untouched.

--reset-delta zeroes delta_with_arch / delta_covered in
Testing/hf_new_models_state.json after a successful sweep: the snapshot is now
fresh, so the watcher's incremental baseline restarts from 0 (otherwise
seo_sync.py would double-count).

Usage:
    python3 Testing/census_sweep.py                    # full sweep (all pages)
    python3 Testing/census_sweep.py --max-pages 2      # test: first 2 pages
    python3 Testing/census_sweep.py --out-dir /tmp/x   # write to another dir
    python3 Testing/census_sweep.py --reset-delta      # + zero the watcher delta
"""
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API = "https://huggingface.co/api/models"
QUERY = "pipeline_tag=text-generation&config=true&limit=1000"


def hf_get(url, tries=4):
    """GET a URL, returning (parsed_json, Link_header). Retries w/ backoff."""
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "1bit-census-sweep"})
            tok = os.environ.get("HF_TOKEN", "")
            if tok:
                req.add_header("Authorization", "Bearer %s" % tok)
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.load(r), (r.headers.get("Link") or "")
        except Exception as e:  # noqa: BLE001
            if t == tries - 1:
                print("  fetch failed (%s): %s" % (url[:120], e), file=sys.stderr)
                return None, ""
            time.sleep(2 ** t)
    return None, ""


def next_cursor(link):
    """Extract the next-page cursor from a Link header, or None at the end."""
    m = re.search(r'<([^>]+)>;\s*rel="next"', link or "")
    if not m:
        return None
    q = urllib.parse.urlparse(m.group(1)).query
    return urllib.parse.parse_qs(q).get("cursor", [None])[0]


def model_type_of(model):
    """model_type from the inline config, falling back to the _text tag."""
    cfg = model.get("config") or {}
    mt = cfg.get("model_type")
    if mt:
        return str(mt).lower()
    for t in model.get("tags") or []:
        if t.endswith("_text"):
            return t[:-5]
    return None


def reset_watcher_delta():
    """Zero the watcher delta — the snapshot is now fresh, so the incremental
    baseline restarts from 0 (otherwise seo_sync.py double-counts)."""
    path = os.path.join(ROOT, "Testing", "hf_new_models_state.json")
    try:
        with open(path, encoding="utf-8") as f:
            state = json.load(f)
    except (OSError, ValueError):
        return
    state["delta_covered"] = 0
    state["delta_with_arch"] = 0
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=1, sort_keys=True)
    print("  reset watcher delta (snapshot baseline is now fresh)", flush=True)


def main():
    max_pages = None
    out_dir = os.path.join(ROOT, "Testing")
    reset_delta = False
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        a = args[i]
        if a.startswith("--max-pages="):
            max_pages = int(a.split("=", 1)[1])
        elif a == "--max-pages":
            i += 1
            max_pages = int(args[i])
        elif a == "--out-dir":
            i += 1
            out_dir = args[i]
        elif a == "--reset-delta":
            reset_delta = True
        i += 1

    counts = {}     # raw arch string -> checkpoint count
    mt_archs = {}   # model_type -> {arch: n}
    mt_count = {}   # model_type -> model count
    index = {}      # model_id_lower -> [model_type] ([] when unknown)
    total = 0
    no_arch = 0

    cursor = None
    page = 0
    while True:
        if max_pages is not None and page >= max_pages:
            break
        url = "%s?%s" % (API, QUERY)
        if cursor:
            url += "&cursor=" + urllib.parse.quote(cursor)
        batch, link = hf_get(url)
        if batch is None:
            print("census_sweep: page %d fetch failed — aborting (no writes)" % page,
                  file=sys.stderr)
            return 2
        if not batch:
            break
        for m in batch:
            total += 1
            mid = str(m.get("id") or "").lower()
            mt = model_type_of(m)
            index[mid] = [mt] if mt else []
            mt_key = mt or "<none>"
            mt_count[mt_key] = mt_count.get(mt_key, 0) + 1
            archs = (m.get("config") or {}).get("architectures") or []
            if not archs:
                no_arch += 1
            for a in archs:
                a = str(a)
                counts[a] = counts.get(a, 0) + 1
                bucket = mt_archs.setdefault(mt_key, {})
                bucket[a] = bucket.get(a, 0) + 1
        nxt = next_cursor(link)
        page += 1
        if page % 25 == 0:
            print("  page %d: %d models, %d archs" % (page, total, len(counts)),
                  flush=True)
        if not nxt:
            break
        cursor = nxt
        time.sleep(0.15)

    # aggregate: model_type -> {count, top_archs desc by n}
    aggregate = {}
    for mt, archs in mt_archs.items():
        top = [{"arch": a, "n": n} for a, n in
               sorted(archs.items(), key=lambda kv: -kv[1])]
        aggregate[mt] = {"count": mt_count.get(mt, 0), "top_archs": top}
    for mt, n in mt_count.items():  # model_types with no architectures
        aggregate.setdefault(mt, {"count": n, "top_archs": []})

    # Match the 2026-08-15 format: the no-arch bucket is also a counts key
    # (census_coverage.py skips it and prefers the top-level no_arch field).
    counts["<none>"] = no_arch

    out = {
        "census_arch_counts.json": {"total": total, "no_arch": no_arch, "counts": counts},
        "census_modeltype_aggregate.json": aggregate,
        "census_model_index.json": index,
    }
    for name, obj in out.items():
        path = os.path.join(out_dir, name)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=1, sort_keys=True)
        os.replace(tmp, path)

    if reset_delta:
        reset_watcher_delta()

    print("census_sweep: total=%d no_arch=%d archs=%d model_types=%d -> %s"
          % (total, no_arch, len(counts), len(aggregate), out_dir))
    return 0


if __name__ == "__main__":
    sys.exit(main())
