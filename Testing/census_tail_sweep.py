#!/usr/bin/env python3
"""census_tail_sweep.py — resolve the long-tail uncovered census classes (#1675).

Pass 1 (offline, no network): map uncovered classes through the saved census
model_type aggregate (Testing/census_modeltype_aggregate.json) — for each
uncovered class, find its dominant model_type and probe whether that
normalizes to a registry token.

Pass 2 (HTTP): crawl the HF models index (filter=text-generation, cursor
paginated) — each entry's tags carry the model_type (<mt>_text). For each
remaining tail class, find a candidate model whose id contains the class
name; classify by that model_type. Search-API fallback for unmatched classes.

Applies nothing by itself. Writes:
  Testing/census_tail_aliases.json  — {class: {token, model_type, source, checkpoints}}
  Testing/census_tail_skipped.json  — {class: reason}
With --apply, prints the C++ blocks for bitnet_model.h + selfcheck (they are
inserted into both repos by the caller when the output is reviewed).

Usage:
  python3 Testing/census_tail_sweep.py                 # pass 1 + pass 2 (full)
  python3 Testing/census_tail_sweep.py --limit 50      # cap pass-2 classes
  python3 Testing/census_tail_sweep.py --skip-http     # offline pass only
  python3 Testing/census_tail_sweep.py --apply         # print C++ blocks
"""
import json, os, subprocess, sys, time, urllib.request, urllib.parse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COUNTS = os.path.join(ROOT, "Testing", "census_arch_counts.json")
AGG = os.path.join(ROOT, "Testing", "census_modeltype_aggregate.json")
OUT_A = os.path.join(ROOT, "Testing", "census_tail_aliases.json")
OUT_S = os.path.join(ROOT, "Testing", "census_tail_skipped.json")
PROBE = os.path.join(ROOT, "Testing", "arch_mapper_probe.cpp")
INDEX = os.path.join(ROOT, "Testing", "census_model_index.json")

STRIP = ("forcausallm", "lmheadmodel", "model",
         "forconditionalgeneration", "forvisiontext2text")

VARIANTS = {
    "deepseekv3": "deepseek3", "deepseekv2": "deepseek2", "deepseekv4": "deepseekv4",
    "qwen2moe": "qwen2moe", "qwen3moe": "qwen3moe", "qwen35moe": "qwen35moe",
    "kimik3": "kimik3", "kimi3": "kimik3", "qwen35": "qwen35",
    "mamba2": "mamba", "bitnet": "bitnet", "granitemoe": "granitemoe",
    "step1moe": "step1", "glm4": "glm4", "falcon3": "falcon3",
    "olmo2": "olmo", "olmo3": "olmo",
}

KNOWN_HARD = {
    "granitemoehybrid", "glmmoedsa", "glm4moe", "glm4moelite", "nemotronh",
    "qwen3next", "minimaxm2", "cohere2", "falconh1", "llama4", "rwkv", "jais",
    "hyv3", "dynamicalibi", "opensci", "transformer", "dynamicforgetting",
    "dynamicslidingwindow", "nanochat", "afmoe", "lisa", "kormo", "gpt",
    "nmmaskmoellavaqwen3", "lfm2moe", "dflashdraft", "picodecoderhf",
    "seedoss", "t5with", "llavaqwen3next", "deepseekv32", "qwen3tdmoe",
}
OUT_OF_SCOPE = {
    "parlertts", "t5", "mt5", "bart", "mbart", "marian", "bert", "roberta",
    "t5with", "umt5", "longformerbart", "whisper",
}


def strip_arch(a):
    low = a.lower()
    for suf in STRIP:
        if len(low) > len(suf) and low.endswith(suf):
            return low[:-len(suf)]
    return low


def norm_mt(mt):
    key = mt.replace("_", "").replace("-", "")
    return VARIANTS.get(key, key)


def build_mapper():
    exe = os.path.join("/tmp", "arch_mapper_probe_sweep")
    subprocess.run(["g++", "-std=c++17", "-I", os.path.join(ROOT, "include"),
                    PROBE, "-o", exe], check=True)
    return exe


def probe(mapper, keys):
    out = subprocess.run([mapper], input="\n".join(keys), text=True,
                         capture_output=True, check=True)
    return [int(t) for t in out.stdout.split()]


def load_token_names():
    """Parse RCPP_ARCH_<NAME> = <N> from the live header (single source)."""
    import re
    hdr = open(os.path.join(ROOT, "include", "rocm_cpp", "bitnet_model.h")).read()
    return {n: name for name, n in re.findall(
        r"RCPP_ARCH_(\w+)\s*=\s*(\d+)", hdr)}


TOKEN_NAMES = load_token_names()
# Sentinel from the LIVE header (moved 255->988 by 6ad2947f; hardcoded 255
# would silently treat UNKNOWN as MAPPED)
UNKNOWN = int(next(n for n, name in TOKEN_NAMES.items() if name == "UNKNOWN"))


FAMILY_KEYS = ("deepseek", "qwen3", "qwen2", "llama", "mistral", "gemma",
              "phi", "gpt2", "falcon", "olmo", "mamba", "zamba", "opt",
              "gptj", "bloom", "step1", "gptoss", "kimi", "qwen35", "internlm",
              "minicpm", "hunyuan", "granite", "exaone", "glm", "xglm", "biogpt")


def double_evidence(s, mt, src):
    """Alias only when the class name AND the model_type both contain the
    same registry family key (e.g. mobilintqwen2 + mobilint-qwen2 -> qwen2).
    Never alias on model_type alone."""
    nm = norm_mt(mt)
    hit = None
    for k in FAMILY_KEYS:
        if k in s and k in nm:
            if hit is None or len(k) > len(hit):
                hit = k
    if not hit:
        return False
    t = probe(mapper, [norm_mt(hit)])[0]
    if t == UNKNOWN:
        return False
    fam = TOKEN_NAMES.get(str(t), "TOKEN%d" % t)
    aliases[s] = {"token": fam, "model_type": mt,
                  "source": src + " (double-evidence)",
                  "checkpoints": stripped[s]}
    return True


def hf_get(url, tries=3):
    for t in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=30) as r:
                return json.load(r)
        except Exception:
            if t == tries - 1:
                return None
            time.sleep(2 * (t + 1))


def crawl_text_gen(max_pages):
    """Paginate the models index via the Link-header cursor (auth-free);
    return {model_id: [model_type tags]} — the _text tags are the config's
    model_type, so this index classifies every crawled model."""
    import re as _re
    from urllib.parse import parse_qs
    out = {}
    cursor = None
    for page in range(max_pages):
        url = ("https://huggingface.co/api/models?filter=text-generation&limit=1000")
        if cursor:
            url += "&cursor=%s" % urllib.parse.quote(cursor)
        req = urllib.request.Request(url)
        try:
            r = urllib.request.urlopen(req, timeout=30)
            d = json.load(r)
            link = r.headers.get("Link", "")
        except Exception:
            break
        for m in d:
            mts = [t[:-5] for t in m.get("tags", []) if t.endswith("_text")]
            out[m["id"].lower()] = mts
        nxt = None
        m = _re.search(r'<([^>]+)>; rel="next"', link)
        if m:
            nxt = parse_qs(urllib.parse.urlparse(m.group(1)).query).get("cursor", [None])[0]
        if not nxt:
            break
        cursor = nxt
        if page % 25 == 0:
            print("  crawl page %d, %d models..." % (page, len(out)), flush=True)
        time.sleep(0.2)
    return out


def main():
    limit = None
    skip_http = False
    apply = False
    max_pages = 300
    for a in sys.argv[1:]:
        if a.startswith("--limit="):
            limit = int(a.split("=")[1])
        elif a == "--skip-http":
            skip_http = True
        elif a == "--apply":
            apply = True
        elif a.startswith("--max-pages="):
            max_pages = int(a.split("=")[1])

    counts = json.load(open(COUNTS))["counts"]
    agg = json.load(open(AGG))

    rev = {}
    for mt, info in agg.items():
        for a in info.get("top_archs", []):
            if a["arch"] not in rev or a["n"] > rev[a["arch"]][1]:
                rev[a["arch"]] = (mt, a["n"])

    stripped = {}
    for arch, cnt in counts.items():
        if arch == "<none>":
            continue
        s = strip_arch(arch)
        stripped[s] = stripped.get(s, 0) + cnt

    mapper = build_mapper()
    toks = probe(mapper, list(stripped))
    uncovered = [s for s, t in zip(stripped, toks) if t == UNKNOWN]

    aliases = json.load(open(OUT_A)) if os.path.exists(OUT_A) else {}
    skipped = json.load(open(OUT_S)) if os.path.exists(OUT_S) else {}

    def classify_mt(s, mt, src):
        t = probe(mapper, [norm_mt(mt)])[0]
        if t == UNKNOWN:
            return False
        fam = TOKEN_NAMES.get(str(t), "TOKEN%d" % t)
        aliases[s] = {"token": fam, "model_type": mt, "source": src,
                      "checkpoints": stripped[s]}
        return True

    # ── Pass 1: offline via the aggregate ──
    for s in uncovered:
        if s in aliases or s in skipped:
            continue
        if "moe" in s.lower():
            skipped[s] = "name suggests MoE — recon required"
            continue
        if s in KNOWN_HARD:
            skipped[s] = "filed issue — pass 2 evidence check"
            continue
        if s in OUT_OF_SCOPE:
            skipped[s] = "not a causal text decoder (denominator #1676)"
            continue
        best = None
        for arch, (mt, n) in rev.items():
            if strip_arch(arch) == s and (best is None or n > best[1]):
                best = (mt, n)
        if best and best[0] != "<none>":
            if not classify_mt(s, best[0], "aggregate"):
                skipped[s] = "model_type %r unknown to registry" % best[0]
        else:
            skipped[s] = "no model_type in aggregate (needs HTTP)"

    # ── Pass 2: HTTP — reconsider the classes pass 1 couldn't classify ──
    if not skip_http:
        todo = [s for s in skipped
                if skipped[s] == "no model_type in aggregate (needs HTTP)"
                or skipped[s].startswith("model_type ")
                or skipped[s] == "filed issue — pass 2 evidence check"
                or skipped[s].startswith("no single model_type tag")]
        for s in todo:
            del skipped[s]
        print("pass 2: %d classes to resolve; crawling index..." % len(todo),
              flush=True)
        index = {}
        if os.path.exists(INDEX) and "--fresh-crawl" not in sys.argv:
            index = json.load(open(INDEX))
            print("  reused saved index: %d models" % len(index), flush=True)
        else:
            index = crawl_text_gen(max_pages)
            print("  index: %d models" % len(index), flush=True)
            json.dump(index, open(INDEX, "w"))
        cands = {}
        for mid, mts in index.items():
            for s in todo:
                if s in cands or s in aliases or s in skipped:
                    continue
                if s in mid:
                    cands[s] = (mid, mts)
        print("  candidates found for %d classes" % len(cands), flush=True)
        done = 0
        for s in todo:
            if limit is not None and done >= limit:
                break
            done += 1
            if s in aliases or s in skipped:
                continue
            cand = cands.get(s)
            if cand:
                mts = [m for m in cand[1] if m != "none"]
                if len(mts) == 1:
                    if not classify_mt(s, mts[0], "crawl:" + cand[0]):
                        if not double_evidence(s, mts[0], "crawl:" + cand[0]):
                            skipped[s] = "model_type %r unknown to registry" % mts[0]
                elif len(mts) > 1:
                    # multiple tags: exact classify on a text-decoder family,
                    # else double-evidence per tag (class name must contain
                    # the family key too — never alias on tags alone)
                    done2 = False
                    for m in sorted(set(mts)):
                        if norm_mt(m) in ("llama", "qwen2", "qwen3") and classify_mt(s, m, "crawl:" + cand[0]):
                            done2 = True
                            break
                    if not done2:
                        for m in sorted(set(mts)):
                            if double_evidence(s, m, "crawl:" + cand[0]):
                                done2 = True
                                break
                    if not done2:
                        skipped[s] = "no single model_type tag (%s)" % mts
                else:
                    skipped[s] = "no single model_type tag (%s)" % (mts or cand[1])
            else:
                d = hf_get("https://huggingface.co/api/models?search=%s&limit=3"
                           % urllib.parse.quote(s))
                if d and d:
                    mts = [t[:-5] for t in d[0].get("tags", []) if t.endswith("_text")]
                    if len(mts) == 1:
                        if not classify_mt(s, mts[0], "search:" + d[0]["id"]):
                            if not double_evidence(s, mts[0], "search:" + d[0]["id"]):
                                skipped[s] = "model_type %r unknown to registry" % mts[0]
                    else:
                        skipped[s] = "no single model_type tag (search)"
                else:
                    skipped[s] = "no HF candidate"
                time.sleep(0.4)
            if done % 50 == 0:
                json.dump(aliases, open(OUT_A, "w"), indent=1, sort_keys=True)
                json.dump(skipped, open(OUT_S, "w"), indent=1, sort_keys=True)
                print("  %d/%d..." % (done, len(todo)), flush=True)

    json.dump(aliases, open(OUT_A, "w"), indent=1, sort_keys=True)
    json.dump(skipped, open(OUT_S, "w"), indent=1, sort_keys=True)

    tot_a = sum(v["checkpoints"] for v in aliases.values())
    tot_s = sum(stripped[s] for s in skipped)
    print("aliases: %d classes / %d ckpts" % (len(aliases), tot_a))
    print("skipped: %d classes / %d ckpts" % (len(skipped), tot_s))
    for s, v in sorted(aliases.items(), key=lambda kv: -kv[1]["checkpoints"])[:25]:
        print("  %6d  %-38s -> %-9s (%s)" % (
            v["checkpoints"], s, v["token"], v["model_type"]))

    if apply:
        lines = ["    // ── 2026-08-15 census tail sweep (auto-generated, model_type-verified) ──"]
        for s, v in sorted(aliases.items()):
            lines.append('    if (strcmp(s, "%s") == 0) return RCPP_ARCH_%s;  // %s' %
                         (s, v["token"], v["model_type"]))
        print("\n--- bitnet_model.h block ---")
        print("\n".join(lines))
        checks = ["    // ── census tail sweep checks ──"]
        for s, v in sorted(aliases.items()):
            checks.append('    check("%s", RCPP_ARCH_%s, "%s");' %
                          (s, v["token"], s))
        print("\n--- arch_mapping_selfcheck.cpp block ---")
        print("\n".join(checks))


if __name__ == "__main__":
    main()
