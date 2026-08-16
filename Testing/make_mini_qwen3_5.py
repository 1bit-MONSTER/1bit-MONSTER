#!/usr/bin/env python3
"""make_mini_qwen3_5.py — tiny Qwen3_5 text fixture for the engine gate.

4 layers: full_attention_interval=2 -> layer_types [linear, full, linear,
full] — exercises BOTH the GatedDeltaNet (linear) path and the gated GQA
(full attention) path. Saves safetensors (f32), bf16 .pt, config.json,
tensor manifest, HF oracle npy. Oracle runs on the bf16->f32 weights so it
matches what the engine loads.

Usage: python3 Testing/make_mini_qwen3_5.py [outdir]
"""
import sys, os, json, torch
sys.path.insert(0, '/home/bcloud/models/venv-zaya/lib/python3.14/site-packages')
from transformers import Qwen3_5TextConfig, Qwen3_5ForCausalLM
from safetensors.torch import save_file as st_save
import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else '/tmp/onebit-q35'
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(0)
cfg = Qwen3_5TextConfig(
    vocab_size=1000, hidden_size=64, intermediate_size=32,
    num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
    head_dim=16, max_position_embeddings=256,
    linear_conv_kernel_dim=4, linear_key_head_dim=8, linear_value_head_dim=8,
    linear_num_key_heads=4, linear_num_value_heads=4,
    full_attention_interval=2,
)
m = Qwen3_5ForCausalLM(cfg)
m.eval()

# bf16 state dict first; oracle on bf16->f32 weights (matches engine load)
with torch.no_grad():
    sd0 = {k: v.detach().to(torch.bfloat16) for k, v in m.state_dict().items()}
m.load_state_dict({k: v.to(torch.float32) for k, v in sd0.items()})
m.eval()

ids = torch.tensor([[5, 7, 9, 11, 3]])
with torch.no_grad():
    out = m(ids)

np.save(os.path.join(OUT, 'logits_last.npy'), out.logits[0, -1].float().cpu().numpy())
torch.save(ids, os.path.join(OUT, 'ids.pt'))

sd = {k: v.contiguous() for k, v in sd0.items()}
torch.save(sd, os.path.join(OUT, 'model.pt'))
st_save({k: v.float() for k, v in sd.items()}, os.path.join(OUT, 'model.safetensors'))

cfgd = cfg.to_dict()
cfgd['_mini_fixture'] = True
json.dump(cfgd, open(os.path.join(OUT, 'config.json'), 'w'), indent=2)

with open(os.path.join(OUT, 'tensors.txt'), 'w') as f:
    for k, v in sd.items():
        f.write(f"{k} {' '.join(str(x) for x in v.shape)}\n")

print(f"mini Qwen3_5 fixture -> {OUT}")
print(f"  layer_types={cfg.layer_types}")
print(f"  oracle top1 = {out.logits[0,-1].argmax().item()}")
print(f"  logits_last[0:5] = {[round(v,4) for v in out.logits[0,-1,:5].tolist()]}")
print(f"  {len(sd)} tensors")
