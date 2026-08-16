#!/usr/bin/env python3
"""export_metal_model.py — dump a real HF model into backend_metal's .bin
layout + emit the torch greedy-oracle tokens the Mac-side gate compares
against. Run on the host with torch; ship the .bin dir + oracle.json to the
Mac.

Usage:
    python3 Testing/export_metal_model.py <hf_model_id> <out_dir>
"""
import json
import os
import struct
import sys

import torch
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer


def write_f32(path, tensor):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    t = tensor.detach().to(torch.float32).cpu().numpy()
    with open(path, "wb") as f:
        f.write(t.tobytes())


def main():
    if len(sys.argv) < 3:
        print("usage: export_metal_model.py <hf_id> <out_dir>")
        return 1
    hf_id, wdir = sys.argv[1], sys.argv[2]
    cfg = AutoConfig.from_pretrained(hf_id)
    model = AutoModelForCausalLM.from_pretrained(hf_id, torch_dtype=torch.float32)
    model.eval()
    tok = AutoTokenizer.from_pretrained(hf_id)
    sd = model.state_dict()

    H = cfg.hidden_size
    L = cfg.num_hidden_layers
    V = cfg.vocab_size

    write_f32(f"{wdir}/model_embed_tokens_weight.bin", sd["model.embed_tokens.weight"])
    write_f32(f"{wdir}/model_norm_weight.bin", sd["model.norm.weight"])
    write_f32(f"{wdir}/model_input_hidden_states_scale.bin", torch.ones(H))
    write_f32(f"{wdir}/model_input_hidden_states_bias.bin", torch.zeros(H))
    for il in range(L):
        p = f"model.layers.{il}"
        write_f32(f"{wdir}/model_layers_{il}_self_attn_q_proj.weight.bin", sd[f"{p}.self_attn.q_proj.weight"])
        write_f32(f"{wdir}/model_layers_{il}_self_attn_k_proj.weight.bin", sd[f"{p}.self_attn.k_proj.weight"])
        write_f32(f"{wdir}/model_layers_{il}_self_attn_v_proj.weight.bin", sd[f"{p}.self_attn.v_proj.weight"])
        write_f32(f"{wdir}/model_layers_{il}_self_attn_o_proj.weight.bin", sd[f"{p}.self_attn.o_proj.weight"])
        write_f32(f"{wdir}/model_layers_{il}_mlp_gate_proj.weight.bin", sd[f"{p}.mlp.gate_proj.weight"])
        write_f32(f"{wdir}/model_layers_{il}_mlp_down_proj.weight.bin", sd[f"{p}.mlp.down_proj.weight"])
        write_f32(f"{wdir}/model_layers_{il}_mlp_up_proj.weight.bin", sd[f"{p}.mlp.up_proj.weight"])
        write_f32(f"{wdir}/model_layers_{il}_input_layernorm.weight.bin", sd[f"{p}.input_layernorm.weight"])
        write_f32(f"{wdir}/model_layers_{il}_post_attention_layernorm.weight.bin", sd[f"{p}.post_attention_layernorm.weight"])

    # ── Greedy oracle: seed → chain of N greedy tokens (torch, fp32) ──
    oracle = {"model": hf_id, "hidden": H, "layers": L, "heads": cfg.num_attention_heads,
              "kv_heads": cfg.num_key_value_heads, "head_dim": H // cfg.num_attention_heads,
              "intermediate": cfg.intermediate_size, "vocab": V,
              "seeds": {}}
    for seed in (5, 42, 99, 1000, 31337):
        with torch.no_grad():
            t = torch.tensor([[seed]])
            chain = [seed]
            for _ in range(6):
                out = model(t)
                nxt = int(out.logits[0, -1].argmax())
                chain.append(nxt)
                t = torch.tensor([[nxt]])
        oracle["seeds"][str(seed)] = chain
    with open(f"{wdir}/oracle.json", "w") as f:
        json.dump(oracle, f, indent=1)

    # sizes
    tot = 0
    for root, _, files in os.walk(wdir):
        for fn in files:
            tot += os.path.getsize(os.path.join(root, fn))
    print(f"exported {hf_id} → {wdir} ({tot/1e6:.0f} MB, oracle {len(oracle['seeds'])} seeds)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
