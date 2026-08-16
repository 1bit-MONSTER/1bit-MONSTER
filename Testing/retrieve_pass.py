import json, subprocess, os, sys, time, urllib.request, urllib.parse

ROOT = "/home/bcloud/1bit-MONSTER"
v = json.load(open(os.path.join(ROOT, "Testing/census_tail_verify.json")))
rem = json.load(open(os.path.join(ROOT, "Testing/census_true_remaining.json")))
rem_by = {s: c for s, c, _ in rem}

# ── pass 1: loose-profile aliases from the UNKNOWN bucket ──
loose = []
for s, r in v.items():
    if s not in rem_by:
        continue
    if r["decision"] != "UNKNOWN":
        continue
    e = r.get("evidence", "")
    # llama-ish: rms norm present + silu + no rope key (engine default 10000)
    if "no strict profile" in e and "act=silu" in e and "rms=" in e and "rope=None" in e:
        loose.append((s, "LLAMA", "loose llama profile (rms+silu, rope default)"))
print("pass1 loose aliases:", len(loose), "/", sum(rem_by[s] for s, _, _ in loose), "ckpts")

# ── pass 2: dump-ID config retry for NO_CONFIG + no-standard-dims ──
targets = set()
for s, r in v.items():
    if s not in rem_by:
        continue
    if r["decision"] == "NO_CONFIG" or (r["decision"] == "UNKNOWN" and "no standard dims" in r.get("evidence", "")):
        targets.add(s)

# dump model ids per class
ids_of = {}
with open("/tmp/census_full_data.jsonl") as f:
    for ln in f:
        try:
            d = json.loads(ln)
        except Exception:
            continue
        for a in d.get("architectures") or []:
            sk = a.lower()
            for suf in ("forcausallm", "lmheadmodel", "model", "forconditionalgeneration", "forvisiontext2text"):
                if len(sk) > len(suf) and sk.endswith(suf):
                    sk = sk[:-len(suf)]
                    break
            if sk in targets and d.get("id") and sk not in ids_of:
                ids_of[sk] = d["id"]

print("pass2 targets:", len(targets), "with dump ids:", len(ids_of))

def hf_get(url, tries=2):
    for t in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=25) as r:
                return json.load(r)
        except Exception:
            if t == tries - 1:
                return None
            time.sleep(1)

def classify_cfg(c, s, mt):
    if c.get("is_encoder_decoder"):
        return "ENCODER_DECODER"
    if (c.get("num_local_experts") or 0) > 0 or "moe" in s.lower() or "moe" in (mt or "").lower():
        return "MOE"
    if c.get("vision_config"):
        return "VLM"
    h = c.get("hidden_size") or c.get("n_embd") or 0
    L = c.get("num_hidden_layers") or c.get("n_layer") or 0
    nh = c.get("num_attention_heads") or c.get("n_head") or 0
    if not h or not L or not nh:
        return "UNKNOWN"
    act = (c.get("activation_function") or c.get("hidden_act") or "").lower()
    eps = c.get("rms_norm_eps") or c.get("norm_eps")
    ln_eps = c.get("layer_norm_epsilon")
    npos = c.get("n_positions")
    if npos and ln_eps and not eps:
        return "GPT2"
    if eps and act in ("silu", "swiglu"):
        return "LLAMA"
    return "UNKNOWN"

results = {}
done = 0
for s in sorted(targets):
    mid = ids_of.get(s)
    if not mid:
        results[s] = "NO_DUMP_ID"
        continue
    cfg = hf_get("https://huggingface.co/%s/resolve/main/config.json" % mid)
    if not cfg:
        results[s] = "FETCH_FAIL"
    else:
        mt = cfg.get("model_type")
        results[s] = classify_cfg(cfg, s, mt) + "|" + str(mid) + "|" + str(mt)
    done += 1
    if done % 50 == 0:
        print("  %d/%d..." % (done, len(targets)), flush=True)
    time.sleep(0.35)

json.dump(results, open(os.path.join(ROOT, "Testing/census_dump_retry.json"), "w"), indent=1, sort_keys=True)
from collections import Counter
dec = Counter(r.split("|")[0] for r in results.values())
print("pass2 decisions:", dict(dec))
for d in ("LLAMA", "GPT2", "VLM"):
    rows = [(s, r) for s, r in results.items() if r.split("|")[0] == d]
    tot = sum(rem_by[s] for s, _ in rows)
    print("  %s: %d classes / %d ckpts" % (d, len(rows), tot))
    for s, r in sorted(rows, key=lambda x: -rem_by[x[0]])[:10]:
        print("    %3d  %-26s %s" % (rem_by[s], s, r))
