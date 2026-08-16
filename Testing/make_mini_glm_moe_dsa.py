#!/usr/bin/env python3
"""make_mini_glm_moe_dsa.py — tiny GLM-MoE-DSA fixture for the engine gate.

4 layers: layer 0 dense MLP + full indexer; layers 1-2 sparse MoE + SHARED
indexer (reuse layer 0's top-k); layer 3 sparse MoE + full indexer. Exercises
the DSA cross-layer sharing path AND both MLP types. Saves safetensors
(engine-readable f32), bf16 .pt, config.json, tensor manifest, HF oracle npy.

Usage: python3 Testing/make_mini_glm_moe_dsa.py [outdir]
"""
import sys, os, json, torch
sys.path.insert(0, '/home/bcloud/models/venv-zaya/lib/python3.14/site-packages')
from transformers import GlmMoeDsaConfig, GlmMoeDsaForCausalLM
from safetensors.torch import save_file as st_save
import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else '/tmp/onebit-glmdsa'
os.makedirs(OUT, exist_ok=True)

torch.manual_seed(0)
cfg = GlmMoeDsaConfig(
    vocab_size=1000, hidden_size=64, moe_intermediate_size=32,
    intermediate_size=32,  # dense MLP width (default 12288 would bloat the fixture)
    num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4,
    qk_nope_head_dim=8, qk_rope_head_dim=4, v_head_dim=8, kv_lora_rank=8, q_lora_rank=8,
    n_routed_experts=8, n_shared_experts=1, num_experts_per_tok=2,
    max_position_embeddings=256,
    index_topk=2,  # < seq_len(5): forces the DSA mask to actually drop tokens
    index_head_dim=8, index_n_heads=2,
    first_k_dense_replace=1,
    indexer_types=["full", "shared", "shared", "full"],
)
m = GlmMoeDsaForCausalLM(cfg)
m.eval()

ids = torch.tensor([[5, 7, 9, 11, 3]])
with torch.no_grad():
    out = m(ids)

np.save(os.path.join(OUT, 'logits_last.npy'), out.logits[0, -1].float().cpu().numpy())
torch.save(ids, os.path.join(OUT, 'ids.pt'))

sd = {k: v.detach().to(torch.bfloat16).contiguous() for k, v in m.state_dict().items()}
torch.save(sd, os.path.join(OUT, 'model.pt'))
st_save({k: v.float() for k, v in sd.items()}, os.path.join(OUT, 'model.safetensors'))

cfgd = cfg.to_dict()
cfgd['_mini_fixture'] = True
json.dump(cfgd, open(os.path.join(OUT, 'config.json'), 'w'), indent=2)

with open(os.path.join(OUT, 'tensors.txt'), 'w') as f:
    for k, v in sd.items():
        f.write(f"{k} {' '.join(str(x) for x in v.shape)}\n")

print(f"mini GLM-MoE-DSA fixture -> {OUT}")
print(f"  layers={cfg.num_hidden_layers} H={cfg.hidden_size} heads={cfg.num_attention_heads}")
print(f"  mlp_layer_types={cfg.mlp_layer_types}")
print(f"  indexer_types={cfg.indexer_types}")
print(f"  oracle top1 = {out.logits[0,-1].argmax().item()}")
print(f"  logits_last[0:5] = {[round(v,4) for v in out.logits[0,-1,:5].tolist()]}")
print(f"  {len(sd)} tensors")
