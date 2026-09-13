#!/usr/bin/env python3
"""HF oracle for the LFM2 q4nx dump.

Loads the tensors dumped by engine/npu/tools/lfm2_dump_weights.cpp into a real
transformers Lfm2ForCausalLM and runs the same prompt FLM was asked for. This
separates "are the weights right?" from "is my forward right?":

  * HF reproduces " Paris" (5242) -> the dump/layout is good; my C++ forward is the bug.
  * HF also misses           -> the dequant/layout/mapping is the bug.

Usage: lfm2_oracle.py <dumpdir> <id,id,...> [--swap-mlp] [--trust-remote]
"""
import sys, os, numpy as np, torch
from transformers import Lfm2Config, Lfm2ForCausalLM

dumpdir, ids_s = sys.argv[1], sys.argv[2]
swap = "--swap-mlp" in sys.argv

idx = {}
for line in open(os.path.join(dumpdir, "index.tsv")):
    name, N, K, fn, n, mode = line.rstrip("\n").split("\t")
    idx[name] = (int(N), int(K), fn, int(n), mode)

def get(name):
    N, K, fn, n, mode = idx[name]
    a = np.fromfile(fn, dtype=np.float32)
    shape = (N, K) if mode == "NK" else (K, N)
    a = a.reshape(shape)
    if mode == "KN":
        a = a.T
    return torch.from_numpy(a.copy())

cfg = Lfm2Config(
    vocab_size=65536, hidden_size=2048, intermediate_size=8192,
    num_hidden_layers=16, num_attention_heads=32, num_key_value_heads=8,
    head_dim=64, conv_L_cache=3, norm_eps=1e-5, rope_theta=1000000.0,
    full_attn_idxs=[2, 5, 8, 10, 12, 14],
    block_auto_adjust_ff_dim=False,          # the bundle's MLP is 8192, not 5632
    tie_word_embeddings=False,
)
model = Lfm2ForCausalLM(cfg).eval().float()

sd = {}
sd["model.embed_tokens.weight"] = get("model.token_embd.weight")
sd["model.norm.weight"] = get("model.norm.weight").reshape(-1)
sd["lm_head.weight"] = get("lm_head.weight")
for l in range(16):
    p = f"model.layers.{l}."
    sd[p + "operator_norm.weight"] = get(f"model.layers.{l}.input_layernorm.weight").reshape(-1)
    sd[p + "ffn_norm.weight"] = get(f"model.layers.{l}.post_attention_layernorm.weight").reshape(-1)
    g = get(f"model.layers.{l}.mlp.gate_proj.weight")
    u = get(f"model.layers.{l}.mlp.up_proj.weight")
    sd[p + "feed_forward.w1.weight"] = u if swap else g
    sd[p + "feed_forward.w3.weight"] = g if swap else u
    sd[p + "feed_forward.w2.weight"] = get(f"model.layers.{l}.mlp.down_proj.weight")
    if f"model.layers.{l}.self_attn.q_proj.weight" in idx:
        sd[p + "self_attn.q_proj.weight"] = get(f"model.layers.{l}.self_attn.q_proj.weight")
        sd[p + "self_attn.k_proj.weight"] = get(f"model.layers.{l}.self_attn.k_proj.weight")
        sd[p + "self_attn.v_proj.weight"] = get(f"model.layers.{l}.self_attn.v_proj.weight")
        sd[p + "self_attn.out_proj.weight"] = get(f"model.layers.{l}.self_attn.o_proj.weight")
        sd[p + "self_attn.q_layernorm.weight"] = get(f"model.layers.{l}.self_attn.q_norm.weight").reshape(-1)
        sd[p + "self_attn.k_layernorm.weight"] = get(f"model.layers.{l}.self_attn.k_norm.weight").reshape(-1)
    else:
        cw = get(f"model.layers.{l}.shortconv.conv.weight")      # [H,3]
        sd[p + "conv.conv.weight"] = cw.reshape(cw.shape[0], 1, cw.shape[1])
        sd[p + "conv.in_proj.weight"] = get(f"model.layers.{l}.shortconv.in_proj.weight")
        sd[p + "conv.out_proj.weight"] = get(f"model.layers.{l}.shortconv.out_proj.weight")

missing, unexpected = model.load_state_dict(sd, strict=False)
print("missing:", [m for m in missing][:6], "unexpected:", [u for u in unexpected][:6])

ids = [int(x) for x in ids_s.split(",")]
with torch.no_grad():
    out = model(torch.tensor([ids]))
    logits = out.logits[0, -1]
top = torch.topk(logits, 5)
print("swap_mlp:", swap, "| boot:", int(top.indices[0]),
      "| top5:", [(int(i), round(float(v), 3)) for i, v in zip(top.indices, top.values)])
