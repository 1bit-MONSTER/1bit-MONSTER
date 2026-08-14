#!/usr/bin/env python3
"""Step1 torch oracle — modeling_step1.py fallback path (CPU, no Optimus op →
build_alibi_cache sqrt-ALiBi + F.scaled_dot_product_attention). Mirrors the
engine's autoregressive loop: argmax at each prompt position, then N generated
tokens. Dumps final logits to E2E_FULL_LOGITS.

Usage: e2e_torch_oracle_step1.py <model_dir> <ids.txt> [N gen tokens]
"""
import os, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = sys.argv[1] if len(sys.argv) > 1 else "/tmp/onebit-e2e/step1"
ids = [int(x) for x in open(sys.argv[2] if len(sys.argv) > 2 else "/tmp/step1_ids.txt").read().split()]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 20

tok = AutoTokenizer.from_pretrained(MODEL)
m = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True, torch_dtype=torch.float32).eval()

def top8(lg):
    idx = torch.topk(lg, 8).indices.tolist()
    return " ".join(f"{i}:{lg[i]:.3f}" for i in idx)

t0 = time.time()
chain = []
with torch.no_grad():
    for i in range(len(ids)):
        inp = torch.tensor([ids[: i + 1]])
        lg = m(inp, use_cache=False).logits[0, -1]
        chain.append(int(lg.argmax()))
        print(f"ref-top8[{i}]: {top8(lg)}", flush=True)
    gen = []
    for g in range(N):
        inp = torch.tensor([ids + gen])
        lg = m(inp, use_cache=False).logits[0, -1]
        gen.append(int(lg.argmax()))
print(f"ref-chain: {' '.join(map(str, chain))}")
print(f"ref-gen: {' '.join(map(str, gen))}")
print(f"ref-final-top8: {top8(lg)}")
out = os.environ.get("E2E_FULL_LOGITS", "step1_ref_logits.txt")
with open(out, "w") as f:
    for i, v in enumerate(lg.tolist()):
        f.write(f"{i} {v}\n")
print(f"ref-logits: {out}  ({time.time()-t0:.1f}s)", flush=True)
