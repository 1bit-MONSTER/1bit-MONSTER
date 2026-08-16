#!/usr/bin/env python3
"""make_mini_mimo_v2.py — tiny MiMo-V2-Flash fixture for the engine gate.

4 layers: hybrid_layer_pattern [0,1,0,1] (0=full attention, 1=sliding-window),
moe_layer_freq [0,1,1,1] (layer 0 dense MLP, rest MoE). SWA layers carry the
attention sink bias; full layers don't. v_scale=0.707 on values.

Uses the vendored (API-patched) remote modeling code in Testing/vendor_mimo/
so the oracle matches the checkpoint's actual reference implementation.

Usage: python3 Testing/make_mini_mimo_v2.py [outdir]
"""
import sys, os, json, torch, importlib.util
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/vendor_mimo")
sys.path.insert(0, '/home/bcloud/models/venv-zaya/lib/python3.14/site-packages')
from safetensors.torch import save_file as st_save
import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else '/tmp/onebit-mimo'
os.makedirs(OUT, exist_ok=True)

VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor_mimo")

# load config as "mimo_cfg" (the vendored modeling imports it by that name)
spec_c = importlib.util.spec_from_file_location("mimo_cfg", os.path.join(VENDOR, "configuration_mimo_v2_flash.py"))
cfgmod = importlib.util.module_from_spec(spec_c)
sys.modules["mimo_cfg"] = cfgmod
spec_c.loader.exec_module(cfgmod)

spec_m = importlib.util.spec_from_file_location("mimo_mod", os.path.join(VENDOR, "modeling_mimo_v2_flash.py"))
mod = importlib.util.module_from_spec(spec_m)
spec_m.loader.exec_module(mod)

torch.manual_seed(0)
cfg = mod.MiMoV2FlashConfig(
    vocab_size=1000, hidden_size=64, intermediate_size=32,
    moe_intermediate_size=32, num_hidden_layers=4,
    num_attention_heads=4, num_key_value_heads=2,
    head_dim=16, v_head_dim=8, swa_head_dim=16, swa_v_head_dim=8,
    swa_num_attention_heads=4, swa_num_key_value_heads=2,
    partial_rotary_factor=0.5, rope_theta=10000.0, swa_rope_theta=10000.0,
    n_routed_experts=8, num_experts_per_tok=2, scoring_func='sigmoid',
    topk_method='noaux_tc', max_position_embeddings=256,
    sliding_window=32, sliding_window_size=32,
    hybrid_layer_pattern=[0, 1, 0, 1], moe_layer_freq=[0, 1, 1, 1],
    attention_value_scale=0.707, attention_bias=False,
    add_swa_attention_sink_bias=True, add_full_attention_sink_bias=False,
    routed_scaling_factor=1.0, n_group=1, topk_group=1, norm_topk_prob=True,
)
m = mod.MiMoV2FlashForCausalLM(cfg)
m.eval()

# The remote code never initializes e_score_correction_bias (torch.empty ->
# garbage huge values) nor gate.weight (torch.empty -> garbage). Real training
# inits bias to zeros and weight to normal(0, initializer_range). Do the same,
# then scale the router weights so random-init scores aren't all ~0.5 (which
# makes top-k an arbitrary tie between torch and the engine).
for layer in m.model.layers:
    if layer.mlp.__class__.__name__ == "MiMoV2MoE":
        with torch.no_grad():
            layer.mlp.gate.e_score_correction_bias.zero_()
            layer.mlp.gate.weight.normal_(0.0, 0.02)
            layer.mlp.gate.weight.mul_(20.0)

# build bf16 state dict FIRST, then run the oracle on the bf16->f32 weights
# (the engine loads the bf16 weights; the oracle must use the same numbers)
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

cfgd = {k: v for k, v in cfg.to_dict().items()}
cfgd['_mini_fixture'] = True
json.dump(cfgd, open(os.path.join(OUT, 'config.json'), 'w'), indent=2)

with open(os.path.join(OUT, 'tensors.txt'), 'w') as f:
    for k, v in sd.items():
        f.write(f"{k} {' '.join(str(x) for x in v.shape)}\n")

print(f"mini MiMo-V2 fixture -> {OUT}")
print(f"  layers={cfg.num_hidden_layers} H={cfg.hidden_size} pattern={cfg.hybrid_layer_pattern}")
print(f"  moe_freq={cfg.moe_layer_freq}")
print(f"  oracle top1 = {out.logits[0,-1].argmax().item()}")
print(f"  logits_last[0:5] = {[round(v,4) for v in out.logits[0,-1,:5].tolist()]}")
print(f"  {len(sd)} tensors")
