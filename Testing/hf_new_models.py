#!/usr/bin/env python3
"""hf_new_models.py — watch HF for new causal-LM models the registry doesn't cover.

The census (census_coverage.py) is a snapshot: 317,310/317,310 on 2026-08-15.
New models drop on HF daily; this watcher polls the newest text-generation
models, fetches each config.json, strips the architecture class, and probes
the REAL engine registry (rcpp_arch_from_string via the compiled probe). Any
new class the registry doesn't map is what silently breaks the 100% claim —
that is the alert.

Run daily (see scripts/jarvis-daily-routine.sh step 4):
    python3 Testing/hf_new_models.py [--limit N]   # N newest to check, default 120

Exit 0: no uncovered classes among the new batch. Exit 1: found some (alert).
State: Testing/hf_new_models_state.json (last run + seen model ids, capped).
"""
import json, os, sys, time, urllib.request, urllib.parse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "Testing"))
from census_coverage import strip_arch, NON_TEXT_GEN, build_mapper, probe

STATE = os.path.join(ROOT, "Testing", "hf_new_models_state.json")
API = "https://huggingface.co/api/models"
CFG = "https://huggingface.co/{mid}/resolve/main/config.json"
MAX_SEEN = 5000  # cap state growth; oldest dropped

# ponytail: tag-filtered scope only — new causal LMs carry the text-generation
# tag; encoder-decoder/TTS (NON_TEXT_GEN) are excluded by design. If HF ever
# stops tagging, broaden with a second filter= pass; daily volume is ~hundreds,
# so fetching configs for the newest N is cheap either way.


def hf_get(url, tries=3):
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "1bit-census-watch"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception:
            if t == tries - 1:
                return None
            time.sleep(1)


def main():
    limit = 120
    if len(sys.argv) > 1 and sys.argv[1] == "--limit":
        limit = int(sys.argv[2])

    state = {"last_run": None, "seen": {}}
    if os.path.exists(STATE):
        try:
            state = json.load(open(STATE))
        except Exception:
            pass
    seen = state.get("seen", {})
    state["last_run"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # newest models, text-generation tag, paginate until we cover `limit` unseen
    fresh = []  # models newer than the last run
    page, guard = 0, 20
    while len(fresh) < limit and page < guard:
        url = API + "?filter=text-generation&sort=createdAt&direction=-1&limit=100&full=true&p=%d" % page
        batch = hf_get(url)
        if not batch:
            break
        new = [m for m in batch if m["id"] not in seen]
        fresh.extend(new)
        if len(new) < len(batch):  # hit already-seen territory
            break
        page += 1
        time.sleep(0.3)
    fresh = fresh[:limit]

    mapper = build_mapper()
    new_classes = {}   # stripped class -> [model ids]
    uncovered = {}     # stripped class -> [model ids]
    n_in_scope = n_covered = 0

    for m in fresh:
        mid = m["id"]
        cfg = hf_get(CFG.format(mid=urllib.parse.quote(mid, safe="/")))
        archs = (cfg or {}).get("architectures") or []
        if not archs:
            seen[mid] = True  # no arch info -> nothing to track, don't re-fetch
            continue
        stripped = [strip_arch(str(a)) for a in archs]
        stripped = [s for s in stripped if s not in NON_TEXT_GEN]
        if not stripped:
            seen[mid] = True
            continue
        n_in_scope += 1
        toks = dict(zip(stripped, probe(mapper, stripped)))
        ok = any(t != 255 for t in toks.values())
        if ok:
            n_covered += 1
        for s, t in toks.items():
            (new_classes if t != 255 else uncovered).setdefault(s, []).append(mid)
        seen[mid] = True
        time.sleep(0.2)

    # cap state, persist
    if len(seen) > MAX_SEEN:
        seen = dict(list(seen.items())[-MAX_SEEN:])
    json.dump(state, open(STATE, "w"), indent=1, sort_keys=True)

    print(f"hf_new_models: {len(fresh)} new models checked, "
          f"{n_in_scope} in-scope, {n_covered} covered, "
          f"{len(uncovered)} uncovered class(es)")
    for s, ids in sorted(new_classes.items()):
        print(f"  covered family {s}: {len(ids)} model(s), e.g. {ids[0]}")
    for s, ids in sorted(uncovered.items()):
        print(f"  !! UNCOVERED {s}: {len(ids)} model(s), e.g. {ids[0]}")
        print(f"     -> add to include/rocm_cpp/bitnet_model.h + selfcheck, "
              f"then re-run census_coverage.py")

    return 1 if uncovered else 0


if __name__ == "__main__":
    sys.exit(main())
