#!/usr/bin/env python3
"""zaya_logit_check_ref.py — BF16 transformers reference for zaya_logit_check.cpp.

Replays the exact token-id sequence produced by the C++ side through the
Zyphra transformers (zaya1) model and compares per-step last-token logits.

Usage: zaya_logit_check_ref.py <model_dir> <ids.txt> <logits.bin> <tol_topk>
  ids.txt      : prompt ids + greedy continuation (one per line)
  logits.bin   : float32 vocab per step, steps concatenated (from C++ side)
  tol_topk     : required top-10 agreement fraction (default 1.0)
Exit 0 on pass, 1 on fail.
"""
import sys, struct
import numpy as np
import torch

def main():
    model_dir, ids_path, logits_path = sys.argv[1:4]
    tol_topk = float(sys.argv[4]) if len(sys.argv) > 4 else 1.0

    ids = [int(l) for l in open(ids_path)]
    ids = torch.tensor(ids, dtype=torch.long)

    from transformers import AutoTokenizer, AutoConfig
    tok = AutoTokenizer.from_pretrained(model_dir)
    cfg = AutoConfig.from_pretrained(model_dir)
    n_vocab = getattr(cfg, 'vocab_size', len(tok))

    with open(logits_path, 'rb') as f:
        raw = f.read()
    n_total = len(raw) // 4
    steps = n_total // n_vocab
    n_prompt = len(ids) - steps
    cpp_logits = np.frombuffer(raw, dtype=np.float32).reshape(steps, n_vocab)
    print(f"  ids={len(ids)} prompt={n_prompt} steps={steps} vocab={n_vocab}")

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16).eval()
    # replay: step t uses ids[:n_prompt + t]
    hits = 0
    worst_cos = 1.0
    with torch.no_grad():
        for t in range(steps):
            inp = ids[: n_prompt + t].unsqueeze(0)
            out = model(inp)
            ref = out.logits[0, -1, :n_vocab].float().numpy()
            # compare: top-10 agreement
            cpp_top = set(np.argsort(cpp_logits[t])[-10:])
            ref_top = set(np.argsort(ref)[-10:])
            agree = len(cpp_top & ref_top)
            hits += agree == 10
            # cosine
            c = float(np.dot(cpp_logits[t], ref) / (np.linalg.norm(cpp_logits[t]) * np.linalg.norm(ref) + 1e-12))
            worst_cos = min(worst_cos, c)
            if t < 3 or agree < 10:
                print(f"  step {t}: top10 agree {agree}/10 cos {c:.5f}")

    frac = hits / steps
    print(f"  top-10 agreement: {hits}/{steps} ({frac:.2%})  worst cos: {worst_cos:.5f}")
    if frac >= tol_topk and worst_cos > 0.9:
        print("PASS")
        return 0
    print("FAIL")
    return 1

if __name__ == '__main__':
    sys.exit(main())
