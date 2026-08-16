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
from census_coverage import strip_arch, NON_TEXT_GEN, build_mapper, probe, UNKNOWN

STATE = os.path.join(ROOT, "Testing", "hf_new_models_state.json")
API = "https://huggingface.co/api/models"
CFG = "https://huggingface.co/{mid}/resolve/main/config.json"
MAX_SEEN = 5000  # cap state growth; oldest dropped

# Causal-decoder + VLM tags: new conditional-generation models (e.g. Muse
# Glimmer) carry image-text-to-text, not text-generation. Gated orgs 401 on
# config fetch, so tags are probed as fallback evidence (a model's tag IS its
# config model_type — same rule as the census model-index fallback).
FILTERS = ("text-generation", "image-text-to-text")

# ponytail: tag-filtered scope only — new causal LMs carry one of these two
# tags; encoder-decoder/TTS (NON_TEXT_GEN) are excluded by design. If HF ever
# stops tagging, broaden with a third filter= pass; daily volume is ~hundreds,
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


# pipeline/library tags that are NOT model_type tokens (probe noise)
PIPELINE_TAGS = frozenset((
    "transformers", "safetensors", "pytorch", "gguf", "executorch",
    "conversational", "feature-extraction", "text-generation",
    "image-text-to-text", "text-to-text", "text-to-image", "image-to-text",
    "automatic-speech-recognition", "text-to-speech", "fill-mask",
    "token-classification", "question-answering", "zero-shot-classification",
    "sentence-similarity", "summarization", "translation",
    "text-classification", "audio-classification", "image-classification",
    "object-detection", "image-segmentation", "depth-estimation",
    "visual-question-answering", "document-question-answering",
    "image-feature-extraction", "image-to-image", "image-to-video",
    "text-to-video", "video-classification", "voice-activity-detection",
    "tabular-classification", "tabular-regression",
    "reinforcement-learning", "robotics", "other",
))


def probe_tags(tags, mapper):
    """Probe plausible model_type tags (lowercase tokens, no license:/arxiv:/
    region:/base_model: noise). Returns (mapped_token, tag) or (None, None)."""
    cands = []
    for t in tags or []:
        if t.startswith(("license:", "arxiv:", "region:", "base_model:",
                         "datasets:", "pipeline:")):
            continue
        if t in PIPELINE_TAGS:
            continue
        if t and t.islower() and all(ch.isalnum() or ch in "_-" for ch in t):
            cands.append(t)
    # probe as-is and normalized (mirror the census model_type fallback)
    norm = [(t, t.replace("_", "").replace("-", "")) for t in cands]
    all_keys = [k for pair in norm for k in pair]
    toks = dict(zip(all_keys, probe(mapper, all_keys)))
    for t, n in norm:
        if toks.get(t, UNKNOWN) != UNKNOWN:
            return toks[t], t
        if toks.get(n, UNKNOWN) != UNKNOWN:
            return toks[n], t
    return None, None


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

    # newest models across causal-decoder tags, dedupe, until `limit` unseen
    fresh, seen_ids, guard = [], set(), 20
    for tag in FILTERS:
        page = 0
        while len(fresh) < limit and page < guard:
            url = (API + "?filter=%s&sort=createdAt&direction=-1&limit=100"
                   "&full=true&p=%d" % (tag, page))
            batch = hf_get(url)
            if not batch:
                break
            for m in batch:
                if m["id"] not in seen and m["id"] not in seen_ids:
                    seen_ids.add(m["id"])
                    fresh.append(m)
            # page entirely from previous runs -> walked past the horizon
            if all(m["id"] in seen for m in batch):
                break
            page += 1
            time.sleep(0.3)
    fresh = fresh[:limit]

    mapper = build_mapper()
    new_classes = {}   # stripped class -> [model ids]
    uncovered = {}     # stripped class -> [model ids]
    unverifiable = {}  # model id -> reason (gated/fetch-fail, no config)
    n_in_scope = n_covered = 0

    for m in fresh:
        mid = m["id"]
        cfg = hf_get(CFG.format(mid=urllib.parse.quote(mid, safe="/")))
        archs = (cfg or {}).get("architectures") or []
        if not archs:
            tags = m.get("tags") or []
            if "gguf" in tags or "lora" in tags or "peft" in tags or \
                    m.get("library_name") == "peft":
                # derivative (quant GGUF / LoRA adapter): no config.json by
                # design; the raw release carries the config and gets checked
                seen[mid] = True
                time.sleep(0.2)
                continue
            # gated/404/no-config: fall back to tag evidence; if tags don't
            # map either, surface it instead of silently swallowing
            tok, tag = probe_tags(tags, mapper)
            if tok is not None:
                n_in_scope += 1
                n_covered += 1
                new_classes.setdefault(tag, []).append(mid)
                seen[mid] = True
            else:
                if mid not in seen or not seen[mid]:
                    unverifiable[mid] = "no config / no mapped tag"
                seen[mid] = False  # retry next run
            time.sleep(0.2)
            continue
        stripped = [strip_arch(str(a)) for a in archs]
        stripped = [s for s in stripped if s not in NON_TEXT_GEN]
        if not stripped:
            seen[mid] = True
            continue
        n_in_scope += 1
        toks = dict(zip(stripped, probe(mapper, stripped)))
        ok = any(t != UNKNOWN for t in toks.values())
        if ok:
            n_covered += 1
        for s, t in toks.items():
            (new_classes if t != UNKNOWN else uncovered).setdefault(s, []).append(mid)
        seen[mid] = True
        time.sleep(0.2)

    # cap state, persist
    if len(seen) > MAX_SEEN:
        seen = dict(list(seen.items())[-MAX_SEEN:])
    json.dump(state, open(STATE, "w"), indent=1, sort_keys=True)

    print(f"hf_new_models: {len(fresh)} new models checked, "
          f"{n_in_scope} in-scope, {n_covered} covered, "
          f"{len(uncovered)} uncovered class(es), "
          f"{len(unverifiable)} unverifiable (gated/no-config)")
    for s, ids in sorted(new_classes.items()):
        print(f"  covered family {s}: {len(ids)} model(s), e.g. {ids[0]}")
    for s, ids in sorted(uncovered.items()):
        print(f"  !! UNCOVERED {s}: {len(ids)} model(s), e.g. {ids[0]}")
        print(f"     -> add to include/rocm_cpp/bitnet_model.h + selfcheck, "
              f"then re-run census_coverage.py")
    for mid, why in sorted(unverifiable.items()):
        print(f"  ? UNVERIFIABLE {mid} ({why}) — gated repos need a token; "
              f"retried next run")

    return 1 if (uncovered or unverifiable) else 0


if __name__ == "__main__":
    sys.exit(main())
