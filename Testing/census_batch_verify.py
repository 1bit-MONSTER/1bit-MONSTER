#!/usr/bin/env python3
"""census_batch_verify.py — fetch real config.json for the top uncovered
classes and classify conservatively. Unlike census_tail_verify.py (which
trusts a single candidate), this one fetches MULTIPLE candidates per class
and only trusts a classification if all fetched configs agree. Writes
Testing/census_batch_verify.json.

Usage: python3 Testing/census_batch_verify.py [--classes N] [--min-count C]
"""
import ast
import json, os, re, subprocess, sys, time, urllib.request, urllib.parse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COUNTS = os.path.join(ROOT, "Testing", "census_arch_counts.json")
INDEX = os.path.join(ROOT, "Testing", "census_model_index.json")
OUT = os.path.join(ROOT, "Testing", "census_batch_verify.json")
UNCOV = "/tmp/uncovered_after_fallback.json"

STRIP = ("forcausallm", "lmheadmodel", "model",
         "forconditionalgeneration", "forvisiontext2text")


def hf_get(url, tries=3):
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "census-verify"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)
        except Exception:
            if t == tries - 1:
                return None
            time.sleep(2 * (t + 1))


def classify(cfg):
    """Return (family, evidence) or (None, reason). Conservative: any signal
    pointing to non-causal-decoder or mismatch wins."""
    mt = (cfg.get("model_type") or "").lower()
    if cfg.get("is_encoder_decoder"):
        return None, "is_encoder_decoder"
    if cfg.get("vision_config") or cfg.get("image_token_index") is not None:
        return None, "vision tower present (VLM)"
    if "moe" in mt and mt not in ("moe",):
        # MoE families that have native tokens
        pass
    nexp = cfg.get("num_local_experts") or 0
    nept = cfg.get("num_experts_per_tok") or 0
    # deepseek/instella-style: model_type deepseek_v3 -> DEEPSEEK
    if mt in ("deepseek_v3", "deepseek_v2", "deepseek"):
        return "DEEPSEEK", "model_type %s" % mt
    if mt in ("gpt_oss", "gptoss"):
        return "GPTOSS", "model_type %s" % mt
    if mt in ("qwen3", "qwen2", "qwen2_5"):
        return "QWEN" + mt[-1].upper() if mt in ("qwen3",) else "QWEN2", "model_type %s" % mt
    if mt in ("llama",):
        return "LLAMA", "model_type llama"
    if mt in ("mistral", "mixtral"):
        return "MISTRAL", "model_type %s" % mt
    if mt in ("gemma", "gemma2", "gemma3"):
        return "GEMMA", "model_type %s" % mt
    if mt in ("phi", "phi3", "phi4"):
        return "PHI", "model_type %s" % mt
    if mt in ("gpt2",):
        return "GPT2", "model_type gpt2"
    if mt in ("rwkv", "rwkv7", "rwkv6", "rwkv5"):
        return "RWKV", "model_type %s" % mt
    if mt.startswith("glm") or mt == "chatglm":
        return "GLM", "model_type %s" % mt
    if mt.startswith("kimi") or mt.startswith("moonshot"):
        return "KIMI_K3", "model_type %s" % mt
    if mt in ("step1", "step", "step3p5"):
        return "STEP1", "model_type %s" % mt
    if mt in ("minimax", "minimax_m2"):
        return "MINIMAXM2", "model_type %s" % mt
    if mt in ("zamba", "zamba2"):
        return "ZAMBA2" if mt == "zamba2" else "ZAMBA", "model_type %s" % mt
    if mt in ("mamba", "mamba2"):
        return "MAMBA", "model_type %s" % mt
    if mt in ("olmo", "olmo2", "olmo3"):
        return "OLMO", "model_type %s" % mt
    if mt in ("falcon", "falcon3"):
        return "FALCON", "model_type %s" % mt
    if mt in ("gpt_neox", "gptneox"):
        return "GPTNEOX", "model_type %s" % mt
    if mt in ("gpt_neo", "gptneo"):
        return "GPTNEO", "model_type %s" % mt
    if mt in ("opt",):
        return "OPT", "model_type %s" % mt
    if mt in ("gptj",):
        return "GPTJ", "model_type %s" % mt
    if mt in ("bloom",):
        return "BLOOM", "model_type %s" % mt
    if mt in ("codegen",):
        return "CODEGEN", "model_type %s" % mt
    if mt in ("qwen2_vl", "qwen3_vl"):
        return "QWEN2VL" if mt == "qwen2_vl" else "QWEN3VL", "model_type %s" % mt
    if mt in ("granite",):
        return "GEMMA", "model_type granite (gemma layout)"
    if mt in ("exaone",):
        return "LLAMA", "model_type exaone (llama layout)"
    if mt in ("sarvam_moe", "sarvam_mla"):
        return None, "sarvam family (no native token)"
    if mt in ("hgrn", "hgrn_bit", "retnet", "kormo", "daisy", "talkie", "quasar"):
        return None, "new family %s (no native token)" % mt
    # fall back to structural profile
    h = cfg.get("hidden_size") or cfg.get("n_embd") or 0
    L = cfg.get("num_hidden_layers") or cfg.get("n_layer") or 0
    nh = cfg.get("num_attention_heads") or cfg.get("n_head") or 0
    if not h or not L or not nh:
        return None, "no standard dims (h=%s L=%s nh=%s)" % (h, L, nh)
    act = (cfg.get("activation_function") or cfg.get("hidden_act") or "").lower()
    eps = cfg.get("rms_norm_eps")
    layernorm = cfg.get("layer_norm_epsilon")
    rope = cfg.get("rope_theta") or cfg.get("rope_scaling")
    npos = cfg.get("n_positions")
    if npos and layernorm and not eps and act in ("gelu", "gelu_new"):
        return "GPT2", "gpt2 profile (n_positions + ln, %s)" % act
    if eps and rope and act in ("silu", "swiglu") and not layernorm:
        return "LLAMA", "llama profile (rms %s rope %s act %s)" % (eps, rope, act)
    return None, "dims ok no strict profile (act=%s eps=%s rope=%s ln=%s)" % (
        act, eps, rope, layernorm)


def main():
    min_c = 10
    max_classes = 0
    args = sys.argv[1:]
    for a in args:
        if a.startswith("--min-count="):
            min_c = int(a.split("=")[1])
        elif a.startswith("--classes="):
            max_classes = int(a.split("=")[1])

    uncovered = json.load(open(UNCOV))
    counts = json.load(open(COUNTS))["counts"]
    stripped = {}
    for arch, cnt in counts.items():
        if arch == "<none>":
            continue
        low = arch.lower()
        for suf in STRIP:
            if len(low) > len(suf) and low.endswith(suf):
                low = low[:-len(suf)]
                break
        stripped[low] = stripped.get(low, 0) + cnt

    src = open(os.path.join(ROOT, "Testing", "census_coverage.py")).read()
    m = re.search(r"NON_TEXT_GEN = (\{.*?\n\})", src, re.S)
    excl = set(ast.literal_eval(m.group(1)))

    index = json.load(open(INDEX)) if os.path.exists(INDEX) else {}
    prev = json.load(open(OUT)) if os.path.exists(OUT) else {}

    todo = [s for s, info in sorted(uncovered.items(), key=lambda x: -x[1]["ckpts"])
            if info["ckpts"] >= min_c and s not in excl and s not in prev]
    if max_classes:
        todo = todo[:max_classes]
    print("classes to verify: %d (min-count=%d)" % (len(todo), min_c), flush=True)

    results = dict(prev)
    done = 0
    for s in todo:
        done += 1
        # gather up to 3 candidates: index substring first, then search
        cands = []
        for mid, mts in index.items():
            if s in mid:
                cands.append(mid)
                if len(cands) >= 3:
                    break
        if len(cands) < 3:
            try:
                d = hf_get("https://huggingface.co/api/models?search=%s&limit=3"
                           % urllib.parse.quote(s))
                if d:
                    for x in d:
                        if x["id"] not in cands:
                            cands.append(x["id"])
            except Exception:
                pass
            time.sleep(0.3)
        results[s] = {"ckpts": uncovered[s]["ckpts"], "candidates": cands[:3]}
        if not cands:
            results[s]["decision"] = "NO_CANDIDATE"
            continue
        fams = []
        evs = []
        for mid in cands[:3]:
            cfg = hf_get("https://huggingface.co/%s/resolve/main/config.json" % mid)
            time.sleep(0.3)
            if not cfg:
                continue
            fam, ev = classify(cfg)
            fams.append(fam)
            evs.append("%s:%s" % (mid.split("/")[-1], ev))
        if not fams:
            results[s]["decision"] = "FETCH_FAIL"
            continue
        if len(set(fams)) == 1 and fams[0] is not None:
            results[s]["decision"] = fams[0]
            results[s]["evidence"] = "; ".join(evs)
        elif all(f is None for f in fams):
            results[s]["decision"] = "UNKNOWN"
            results[s]["evidence"] = "; ".join(evs)
        else:
            results[s]["decision"] = "DISAGREE"
            results[s]["evidence"] = "; ".join(evs)
        if done % 10 == 0:
            json.dump(results, open(OUT, "w"), indent=1, sort_keys=True)
            print("  %d/%d..." % (done, len(todo)), flush=True)

    json.dump(results, open(OUT, "w"), indent=1, sort_keys=True)
    from collections import Counter
    dec = Counter(r.get("decision", "?") for r in results.values())
    print("decisions:", dict(dec))
    for d, n in dec.most_common():
        rows = [(s, r) for s, r in results.items() if r.get("decision") == d]
        tot = sum(r["ckpts"] for s, r in rows)
        print("  %-14s %3d classes / %5d ckpts" % (d, n, tot))
        for s, r in sorted(rows, key=lambda x: -x[1]["ckpts"])[:6]:
            print("      %5d  %-32s %s" % (r["ckpts"], s, r.get("evidence", "")[:60]))


if __name__ == "__main__":
    main()
