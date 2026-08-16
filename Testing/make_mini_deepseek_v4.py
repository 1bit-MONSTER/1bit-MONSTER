#!/usr/bin/env python3
"""make_mini_deepseek_v4.py — tiny DeepSeek-V4 fixture for the engine gate.

4 layers, all sliding_attention (no compressor/indexer machinery — those are
deferred; a real-checkpoint gate will exercise them). MLP schedule: first 2
layers hash_moe (tid2eid routing), last 2 moe (top-k sqrtsoftplus) — covers
BOTH routing paths. Saves state dict as safetensors (engine-readable) + bf16
.pt, config.json, tensor manifest, and the HF oracle logits.

Usage: python3 Testing/make_mini_deepseek_v4.py [outdir]
"""
import sys, os, json, torch
sys.path.insert(0, '/home/bcloud/models/venv-zaya/lib/python3.14/site-packages')
from transformers import DeepseekV4Config, DeepseekV4ForCausalLM
from safetensors.torch import save_file as st_save

OUT = sys.argv[1] if len(sys.argv) > 1 else '/tmp/onebit-dsv4'
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(0)
cfg = DeepseekV4Config(
    vocab_size=1000, hidden_size=64, moe_intermediate_size=32,
    num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=1,
    head_dim=16, q_lora_rank=8, o_lora_rank=8,
    n_routed_experts=8, n_shared_experts=1, num_experts_per_tok=2,
    max_position_embeddings=256, sliding_window=32,
    layer_types=["sliding_attention"] * 4,
    mlp_layer_types=["hash_moe", "hash_moe", "moe", "moe"],
)
m = DeepseekV4ForCausalLM(cfg)
m.eval()

ids = torch.tensor([[5, 7, 9, 11, 3]])
with torch.no_grad():
    out = m(ids)

torch.save(out.logits[0, -1].float().cpu(), os.path.join(OUT, 'logits_last.pt'))
import numpy as np
np.save(os.path.join(OUT, 'logits_last.npy'), out.logits[0, -1].float().cpu().numpy())
torch.save(ids, os.path.join(OUT, 'ids.pt'))

# state dict: bf16 like a real checkpoint, saved both as .pt and safetensors
sd = {k: v.detach().to(torch.bfloat16).contiguous() for k, v in m.state_dict().items()}
torch.save(sd, os.path.join(OUT, 'model.pt'))
st_save({k: v.float() for k, v in sd.items()}, os.path.join(OUT, 'model.safetensors'))

cfgd = cfg.to_dict()
cfgd['_mini_fixture'] = True
json.dump(cfgd, open(os.path.join(OUT, 'config.json'), 'w'), indent=2)

with open(os.path.join(OUT, 'tensors.txt'), 'w') as f:
    for k, v in sd.items():
        f.write(f"{k} {' '.join(str(x) for x in v.shape)}\n")

print(f"mini DeepSeek-V4 fixture -> {OUT}")
print(f"  layers={cfg.num_hidden_layers} H={cfg.hidden_size} heads={cfg.num_attention_heads} "
      f"q_lora={cfg.q_lora_rank} o_lora={cfg.o_lora_rank} o_groups={cfg.o_groups}")
print(f"  layer_types={cfg.layer_types}")
print(f"  mlp_layer_types={cfg.mlp_layer_types}")
print(f"  oracle top1 = {out.logits[0,-1].argmax().item()}")
print(f"  logits_last[0:5] = {[round(v,4) for v in out.logits[0,-1,:5].tolist()]}")
print(f"  {len(sd)} tensors")
