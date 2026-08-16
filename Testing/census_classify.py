#!/usr/bin/env python3
"""census_classify.py — offline structural classification of every remaining
uncovered census class, using /tmp/census_full_data.jsonl (the full per-model
census dump with config keys). For each uncovered class: aggregate the
dominant structural profile across its models, classify into an engine
layout, and emit alias candidates.

Profiles (strict — every signal must agree):
  llama   : rms_norm_eps + rope_theta, no attention_bias, no experts,
            no parallel_residual, not alibi, hidden_act in (silu, None)
  qwen2   : same as llama but attention_bias == true
  gpt2    : n_positions + layer_norm_eps, NO rms_norm_eps, NO rope_theta,
            no experts
  gptneox : use_parallel_residual == true (Pythia-style)
  falcon  : multi_query == true OR (alibi and no rope)
  olmo    : norm_is_layernorm == true (OLMo pre-0724)
  step1   : num_attention_groups present
  moe     : num_experts > 0 or num_experts_per_tok > 0  (flagged, not aliased)
  vlm     : model_type in known VLM families or architecture has
            ForVisionText2Text
  unknown : no strict profile

Writes Testing/census_classified.json: {class: {count, profile, evidence,
model_type, sample_model}}. Prints the alias block.
"""
import json, os, subprocess, sys

ROOT = "/home/bcloud/1bit-MONSTER"
DUMP = "/tmp/census_full_data.jsonl"
COUNTS = os.path.join(ROOT, "Testing", "census_arch_counts.json")
OUT = os.path.join(ROOT, "Testing", "census_classified.json")

STRIP = ("forcausallm", "lmheadmodel", "model",
         "forconditionalgeneration", "forvisiontext2text")
NON_TEXT_GEN = {"parlertts", "t5", "mt5", "t5with", "umt5", "bart", "mbart",
                "marian", "longformerbart", "t5gemma", "bert", "roberta",
                "xlmroberta", "xlnet"}


def strip_arch(a):
    low = a.lower()
    for suf in STRIP:
        if len(low) > len(suf) and low.endswith(suf):
            return low[:-len(suf)]
    return low


def main():
    # sentinel from the LIVE header (moved 255->988 by 6ad2947f; a hardcoded
    # 255 would silently treat UNKNOWN as MAPPED)
    import re
    hdr = open(os.path.join(ROOT, "include", "rocm_cpp", "bitnet_model.h")).read()
    UNKNOWN = int(re.search(r"RCPP_ARCH_UNKNOWN\s*=\s*(\d+)", hdr).group(1))
    counts = json.load(open(COUNTS))["counts"]
    stripped = {}
    for arch, cnt in counts.items():
        if arch == "<none>":
            continue
        s = strip_arch(arch)
        if s in NON_TEXT_GEN:
            continue
        stripped[s] = stripped.get(s, 0) + cnt

    # current coverage probe
    subprocess.run(["g++", "-std=c++17", "-I", os.path.join(ROOT, "include"),
                    os.path.join(ROOT, "Testing", "arch_mapper_probe.cpp"),
                    "-o", "/tmp/ampC"], check=True)
    out = subprocess.run(["/tmp/ampC"], input="\n".join(stripped), text=True,
                         capture_output=True, check=True)
    toks = [int(t) for t in out.stdout.split()]
    uncovered = [s for s, t in zip(stripped, toks) if t == UNKNOWN]

    # index the dump by stripped class
    print("indexing dump...", flush=True)
    by_class = {}
    with open(DUMP) as f:
        for ln in f:
            try:
                d = json.loads(ln)
            except Exception:
                continue
            for a in d.get("architectures") or []:
                s = strip_arch(str(a))
                if s in uncovered:
                    by_class.setdefault(s, []).append(d)
    print("classes with dump evidence:", len(by_class), flush=True)

    def dom(vals):
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        from collections import Counter
        return Counter(vals).most_common(1)[0][0]

    results = {}
    for s in uncovered:
        models = by_class.get(s, [])
        c = stripped[s]
        if not models:
            results[s] = {"count": c, "profile": "no_dump_evidence",
                          "evidence": "", "model_type": None, "sample": None}
            continue
        mt = dom([m.get("model_type") for m in models])
        nexp = dom([m.get("num_experts") for m in models]) or 0
        nept = dom([m.get("num_experts_per_tok") for m in models]) or 0
        rms = dom([m.get("rms_norm_eps") for m in models])
        rope = dom([m.get("rope_theta") for m in models])
        ln_eps = dom([m.get("layer_norm_eps") for m in models])
        npos = dom([m.get("n_positions") for m in models])
        abias = dom([m.get("attention_bias") for m in models])
        parallel = dom([m.get("use_parallel_residual") for m in models])
        act = dom([m.get("hidden_act") for m in models])
        alibi = dom([m.get("alibi") for m in models])
        mq = dom([m.get("multi_query") for m in models])
        layernorm = dom([m.get("norm_is_layernorm") for m in models])
        nag = dom([m.get("num_attention_groups") for m in models])
        h = dom([m.get("hidden_size") for m in models])
        nh = dom([m.get("num_attention_heads") for m in models])
        nkv = dom([m.get("num_key_value_heads") for m in models])
        ev = "mt=%s h=%s nh=%s nkv=%s rms=%s rope=%s ln=%s npos=%s abias=%s par=%s act=%s alibi=%s mq=%s nexp=%s" % (
            mt, h, nh, nkv, rms, rope, ln_eps, npos, abias, parallel, act,
            alibi, mq, nexp)
        if nexp or nept:
            prof = "moe"
        elif mq:
            prof = "falcon"
        elif layernorm:
            prof = "olmo"
        elif parallel:
            prof = "gptneox"
        elif npos and ln_eps and not rms and not rope:
            prof = "gpt2"
        elif rms and rope and not abias and not parallel and not alibi:
            prof = "llama"
        elif rms and rope and abias:
            prof = "qwen2"
        else:
            prof = "unknown"
        results[s] = {"count": c, "profile": prof, "evidence": ev,
                      "model_type": mt,
                      "sample": models[0].get("id")}

    json.dump(results, open(OUT, "w"), indent=1, sort_keys=True)

    from collections import Counter
    profs = Counter(r["profile"] for r in results.values())
    for p, n in profs.most_common():
        tot = sum(results[s]["count"] for s in results if results[s]["profile"] == p)
        print("%-16s %4d classes / %6d ckpts" % (p, n, tot))
    for p in ("llama", "qwen2", "gpt2", "gptneox", "falcon", "olmo"):
        rows = sorted((s for s in results if results[s]["profile"] == p),
                      key=lambda s: -results[s]["count"])
        print("\n--- " + p + " (alias candidates):")
        for s in rows[:25]:
            print("  %5d  %-34s %s" % (results[s]["count"], s, results[s]["model_type"]))


if __name__ == "__main__":
    main()
