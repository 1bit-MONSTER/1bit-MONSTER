#!/usr/bin/env python3
"""HF safetensors (bf16) → INT8 QDQ ONNX — bypasses GGUFs entirely.
Some official qwen gguf releases carry corrupted norms AND linears; the
safetensors are the source of truth. Output structure matches the GGUF
converter (model.layers.%d.* names, per-row int8 + DQ, per-row int8
embed/lm_head over the 2GB limit).
"""
import sys, json, struct
import numpy as np
import onnx
from onnx import helper, numpy_helper

def read_st(path):
    f = open(path, 'rb')
    hdr_len = int.from_bytes(f.read(8), 'little')
    hdr = json.loads(f.read(hdr_len))
    return f, hdr  # f positioned at the data section start

def get_tensor(f, hdr, name, base):
    off, end = hdr[name]['data_offsets']
    f.seek(base + off)
    n = (end - off) // 2
    raw = np.frombuffer(f.read((end - off)), dtype='<u2').astype(np.uint32)
    return (raw << 16).astype(np.uint32).view(np.float32)  # bf16 → f32

def int8_quant(w):
    R, C = w.shape
    scale = np.abs(w).max(axis=1, keepdims=True) / 127.0
    scale = np.maximum(scale, 1e-10)
    wq = np.clip(np.round(w / scale), -128, 127).astype(np.int8)
    return wq, scale.astype(np.float32)

def main():
    st_path, outdir = sys.argv[1], sys.argv[2]
    import os
    os.makedirs(outdir, exist_ok=True)
    f, hdr = read_st(st_path)
    base = f.tell()
    names = set(hdr.keys())
    # config from the repo (fetched separately)
    import urllib.request
    cfg = json.loads(urllib.request.urlopen(
        "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct/resolve/main/config.json").read())
    L = cfg['num_hidden_layers']; hs = cfg['hidden_size']; is_ = cfg['intermediate_size']
    nh = cfg['num_attention_heads']; nkv = cfg['num_key_value_heads']
    print(f"[st] {L} layers hs={hs} is={is_} heads={nh} nkv={nkv}")

    def T(n):  # fetch + reshape [dims]
        w = get_tensor(f, hdr, n, base)
        return w.reshape(hdr[n]['shape'])

    inits, nodes = [], []
    def add(name, arr):
        inits.append(numpy_helper.from_array(arr, name))

    emb = T('model.embed_tokens.weight')
    # size estimate: int8 linears + embed + lm_head (2GB protobuf cap)
    wb = sum(np.prod(hdr[n]['shape']) for n in names if 'layers' in n and '.weight' in n and 'norm' not in n)
    eb = emb.size * 2
    big = (wb + eb + 8*1024*1024) > 1.9e9
    if big:
        q, s = int8_quant(emb)
        add('model.embed_tokens.weight', q)
        add('model.embed_tokens.weight_scale', s.reshape(-1))
        nodes.append(helper.make_node("DequantizeLinear", ["model.embed_tokens.weight", "model.embed_tokens.weight_scale"], ["model.embed_tokens.weight_dq"], name="dq_embed"))
    else:
        add('model.embed_tokens.weight', emb.astype(np.float16))
    add('model.norm.weight', T('model.norm.weight').astype(np.float32))

    linears = ['self_attn.q_proj.weight', 'self_attn.k_proj.weight', 'self_attn.v_proj.weight',
               'self_attn.o_proj.weight', 'mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight']
    for l in range(L):
        p = f'model.layers.{l}.'
        add(p + 'input_layernorm.weight', T(p + 'input_layernorm.weight').astype(np.float32))
        add(p + 'post_attention_layernorm.weight', T(p + 'post_attention_layernorm.weight').astype(np.float32))
        for b in ['self_attn.q_proj.bias', 'self_attn.k_proj.bias', 'self_attn.v_proj.bias']:
            if p + b in names:
                add(p + b, T(p + b).astype(np.float32))
        for ln in linears:
            w = T(p + ln)
            if w.ndim == 2 and w.shape[1] == hs and ln != 'mlp.down_proj.weight':
                pass  # already [N, K]
            q, s = int8_quant(w)
            full = p + ln
            add(full, q)
            add(full + '_scale', s.reshape(-1))
            nodes.append(helper.make_node("DequantizeLinear", [full, full + '_scale'], [full + '_dq'], name=f"dq_l{l}_{ln}"))

    graph = helper.make_graph(nodes, "g", [], [], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    onnx.save(model, os.path.join(outdir, "model_int8.onnx"))
    print(f"[st] wrote {outdir}/model_int8.onnx: {len(inits)} initializers, {len(nodes)} DQ nodes")
    cfg_out = {"model_type": "qwen2", "hidden_size": hs, "intermediate_size": is_,
               "num_attention_heads": nh, "num_key_value_heads": nkv, "head_dim": hs // nh,
               "max_position_embeddings": cfg['max_position_embeddings'],
               "rms_norm_eps": cfg['rms_norm_eps'], "rope_theta": cfg['rope_theta'],
               "vocab_size": cfg['vocab_size'], "tie_word_embeddings": 1}
    with open(os.path.join(outdir, "config.json"), "w") as f2:
        json.dump(cfg_out, f2, indent=2)
    print("[st] wrote config.json")

if __name__ == "__main__":
    main()
