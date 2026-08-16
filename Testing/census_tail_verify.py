#!/usr/bin/env python3
"""Verify remaining uncovered classes against real config.json files.
For each uncovered class with count >= MIN_C, find a candidate model (crawl
index substring, then search fallback), fetch config.json, extract structural
keys, and classify into a layout profile. Writes census_tail_verify.json:
  {class: {count, model_type, candidate, profile, evidence, decision}}
Decisions: ALIAS_LLAMA / ALIAS_QWEN2 / ALIAS_GPT2 / MOE / ENCODER_DECODER /
VLM / UNKNOWN.
"""
import json, os, re, subprocess, sys, time, urllib.request, urllib.parse

ROOT = "/home/bcloud/1bit-MONSTER"
COUNTS = os.path.join(ROOT, "Testing", "census_arch_counts.json")
AGG = os.path.join(ROOT, "Testing", "census_modeltype_aggregate.json")
INDEX = os.path.join(ROOT, "Testing", "census_model_index.json")
OUT = os.path.join(ROOT, "Testing", "census_tail_verify.json")
MIN_C = int(sys.argv[1]) if len(sys.argv) > 1 else 5
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 0

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
    counts = json.load(open(COUNTS))["counts"]
    agg = json.load(open(AGG))
    stripped = {}
    for arch, cnt in counts.items():
        if arch == "<none>":
            continue
        s = strip_arch(arch)
        if s in NON_TEXT_GEN:
            continue
        stripped[s] = stripped.get(s, 0) + cnt

    mt_of = {}
    for mt, info in agg.items():
        for a in info.get("top_archs", []):
            sk = strip_arch(a["arch"])
            if sk not in mt_of or a["n"] > mt_of[sk][1]:
                mt_of[sk] = (mt, a["n"])

    subprocess.run(["g++", "-std=c++17", "-I", os.path.join(ROOT, "include"),
                    os.path.join(ROOT, "Testing", "arch_mapper_probe.cpp"),
                    "-o", "/tmp/ampV"], check=True)

    def prb(keys):
        if not keys:
            return []
        out = subprocess.run(["/tmp/ampV"], input="\n".join(keys), text=True,
                             capture_output=True, check=True)
        return [int(t) for t in out.stdout.split()]

    toks = prb(list(stripped))
    uncovered = [s for s, t in zip(stripped, toks) if t == 255]
    todo = [s for s in uncovered if stripped[s] >= MIN_C]
    print("classes to verify:", len(todo), flush=True)

    index = json.load(open(INDEX)) if os.path.exists(INDEX) else {}
    # class -> candidate via substring (prefer models with a _text tag)
    cands = {}
    for mid, mts in index.items():
        for s in todo:
            if s in cands:
                continue
            if s in mid and (not cands.get(s) or mts):
                cands[s] = (mid, mts)
    print("candidates from index:", len(cands), flush=True)

    def hf_get(url, tries=3):
        for t in range(tries):
            try:
                with urllib.request.urlopen(url, timeout=30) as r:
                    return json.load(r)
            except Exception:
                if t == tries - 1:
                    return None
                time.sleep(2 * (t + 1))

    results = {}
    done = 0
    for s in todo:
        if LIMIT and done >= LIMIT:
            break
        done += 1
        mid, mts = cands.get(s, (None, None))
        if not mid:
            d = hf_get("https://huggingface.co/api/models?search=%s&limit=3"
                       % urllib.parse.quote(s))
            if d:
                mid = d[0]["id"]
            time.sleep(0.4)
        cfg = None
        if mid:
            cfg = hf_get("https://huggingface.co/%s/resolve/main/config.json" % mid)
            time.sleep(0.3)
        mt = mt_of.get(s, (None, 0))[0]
        if not cfg:
            results[s] = {"count": stripped[s], "model_type": mt,
                          "candidate": mid, "decision": "NO_CONFIG"}
        else:
            decision, evidence = classify(cfg, s, mt)
            results[s] = {"count": stripped[s], "model_type": mt,
                          "candidate": mid, "decision": decision,
                          "evidence": evidence}
        if done % 25 == 0:
            json.dump(results, open(OUT, "w"), indent=1, sort_keys=True)
            print("  %d/%d..." % (done, len(todo)), flush=True)

    json.dump(results, open(OUT, "w"), indent=1, sort_keys=True)
    from collections import Counter
    dec = Counter(r["decision"] for r in results.values())
    print("decisions:", dict(dec))
    for d in ("ALIAS_LLAMA", "ALIAS_QWEN2", "ALIAS_GPT2", "MOE", "ENCODER_DECODER", "VLM"):
        tot = sum(results[s]["count"] for s in results if results[s]["decision"] == d)
        print("  %-16s %5d ckpts" % (d, tot))


def classify(cfg, s, mt):
    mt = mt or ""
    if cfg.get("is_encoder_decoder"):
        return "ENCODER_DECODER", "is_encoder_decoder"
    nexp = cfg.get("num_local_experts") or 0
    if nexp > 0 or "moe" in s.lower() or "moe" in mt.lower():
        return "MOE", "num_local_experts=%d" % nexp
    if cfg.get("vision_config") or cfg.get("image_token_index") is not None:
        return "VLM", "vision tower present"
    h = cfg.get("hidden_size") or cfg.get("n_embd") or 0
    L = cfg.get("num_hidden_layers") or cfg.get("n_layer") or 0
    nh = cfg.get("num_attention_heads") or cfg.get("n_head") or 0
    nkv = cfg.get("num_key_value_heads") or 0
    act = (cfg.get("activation_function") or cfg.get("hidden_act") or "").lower()
    eps = cfg.get("rms_norm_eps")
    layernorm = cfg.get("layer_norm_epsilon")
    rope = cfg.get("rope_theta") or cfg.get("rope_scaling")
    if not h or not L or not nh:
        return "UNKNOWN", "no standard dims (h=%s L=%s nh=%s)" % (h, L, nh)
    # GPT-2 profile: n_embd/n_layer/n_head keys, gelu act, no rms_norm_eps
    if cfg.get("n_embd") and cfg.get("n_layer") and cfg.get("n_head") and \
            act in ("gelu", "gelu_new", "gelu_pytorch_tanh") and not eps:
        return "ALIAS_GPT2", "gpt2 profile (n_embd keys, %s act)" % act
    # Llama profile: RMSNorm + rope + silu/swiglu + no layernorm_epsilon
    if eps and rope and act in ("silu", "swiglu", "gelu") and h % nh == 0 and \
            (nkv == 0 or nh % nkv == 0) and not layernorm:
        return "ALIAS_LLAMA", "llama profile (rms %s rope %s act %s)" % (eps, rope, act)
    # qwen2 profile: rms + rope + silu, but with qkv biases — indistinguishable
    # from llama by config alone; model_type hint decides
    if eps and rope and act in ("silu", "swiglu") and "qwen" in mt:
        return "ALIAS_QWEN2", "qwen-family model_type %s" % mt
    return "UNKNOWN", "dims ok but no strict profile (act=%s eps=%s rope=%s ln=%s)" % (
        act, eps, rope, layernorm)


if __name__ == "__main__":
    main()
