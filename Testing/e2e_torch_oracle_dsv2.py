#!/usr/bin/env python3
"""DeepSeek-V2-Lite torch oracle (modeling_deepseek_v2, bf16 CPU).
Mirrors the engine's autoregressive loop: argmax at each prompt position,
then N generated tokens. Dumps final logits to E2E_FULL_LOGITS.
"""
import os, sys, time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "deepseek-ai/DeepSeek-V2-Lite"
ids = [int(x) for x in open(sys.argv[1] if len(sys.argv) > 1 else "/tmp/dsv2_ids.txt").read().split()]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 20

tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
m = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16).eval()

def top8(lg):
    idx = torch.topk(lg, 8).indices.tolist()
    return " ".join(f"{i}:{lg[i]:.3f}" for i in idx)

t0 = time.time()
chain = []
with torch.no_grad():
    for i in range(len(ids)):
        lg = m(torch.tensor([ids[: i + 1]]).to(m.device)).logits[0, -1]
        chain.append(int(lg.argmax()))
        print(f"ref-top8[{i}]: {top8(lg)}", flush=True)
    gen = []
    for g in range(N):
        lg = m(torch.tensor([ids + gen]).to(m.device)).logits[0, -1]
        gen.append(int(lg.argmax()))
print(f"ref-chain: {' '.join(map(str, chain))}")
print(f"ref-gen: {' '.join(map(str, gen))}")
print(f"ref-final-top8: {top8(lg)}")
out = os.environ.get("E2E_FULL_LOGITS", "dsv2_ref_logits.txt")
with open(out, "w") as f:
    for i, v in enumerate(lg.tolist()):
        f.write(f"{i} {v}\n")
print(f"ref-logits: {out}  ({time.time()-t0:.1f}s)", flush=True)
