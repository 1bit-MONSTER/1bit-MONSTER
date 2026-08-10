#!/usr/bin/env python3
"""HF safetensors (bf16) → INT8 QDQ ONNX — bypasses GGUFs entirely.
Some official qwen gguf releases carry corrupted norms AND linears; the
safetensors are the source of truth. Output structure matches the GGUF
converter (model.layers.%d.* names, per-row int8 + DQ, per-row int8
embed/lm_head over the 2GB limit).
"""
import sys, json, struct, os
import numpy as np
import onnx
from onnx import helper, numpy_helper

def read_shard(path):
    f = open(path, 'rb')
    hdr_len = int.from_bytes(f.read(8), 'little')
    hdr = json.loads(f.read(hdr_len))
    return f, hdr, f.tell()  # f positioned at the data section start

def load_source(src):
    """Returns (tensor_getter, name->shape map, extra_names). Supports a single
    .safetensors file or a directory / index.json with shards."""
    if os.path.isdir(src) or src.endswith('index.json'):
        idx_path = src if src.endswith('index.json') else os.path.join(src, 'model.safetensors.index.json')
        idx = json.load(open(idx_path))
        base_dir = os.path.dirname(idx_path)
        shards = {}
        shapes = {}
        for tname, shard in idx['weight_map'].items():
            if shard not in shards:
                f, h, b = read_shard(os.path.join(base_dir, shard))
                shards[shard] = (f, h, b)
                for k, v in h.items():
                    if 'shape' in v:
                        shapes.setdefault(k, v['shape'])
            else:
                f, h, b = shards[shard]
                for k, v in h.items():
                    if 'shape' in v:
                        shapes.setdefault(k, v['shape'])
        def get(name):
            shard = idx['weight_map'][name]
            f, h, base = shards[shard]
            off, end = h[name]['data_offsets']
            f.seek(base + off)
            raw = np.frombuffer(f.read(end - off), dtype='<u2').astype(np.uint32)
            return (raw << 16).astype(np.uint32).view(np.float32)
        return get, shapes
    f, hdr, base = read_shard(src)
    def get(name):
        off, end = hdr[name]['data_offsets']
        f.seek(base + off)
        raw = np.frombuffer(f.read(end - off), dtype='<u2').astype(np.uint32)
        return (raw << 16).astype(np.uint32).view(np.float32)
    return get, {k: v['shape'] for k, v in hdr.items()}

def int8_quant(w):
    R, C = w.shape
    scale = np.abs(w).max(axis=1, keepdims=True) / 127.0
    scale = np.maximum(scale, 1e-10)
    wq = np.clip(np.round(w / scale), -128, 127).astype(np.int8)
    return wq, scale.astype(np.float32)

def main():
    st_path, outdir = sys.argv[1], sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    T0, hdr = load_source(st_path)
    names = set(hdr.keys())
    # config: prefer a local config.json next to the source, else fetch from
    # the HF repo (models/<name> flow drops config.json alongside the shards).
    import urllib.request
    base_dir = st_path if os.path.isdir(st_path) else os.path.dirname(os.path.abspath(st_path))
    cfg_path = os.path.join(base_dir, 'config.json')
    if os.path.exists(cfg_path):
        cfg = json.load(open(cfg_path))
    else:
        repo = os.environ.get('HF_REPO', 'Qwen/Qwen2.5-1.5B-Instruct')
        cfg = json.loads(urllib.request.urlopen(
            f"https://huggingface.co/{repo}/resolve/main/config.json").read())
    # Layer count comes from the INDEX/tensors, never the config fallback —
    # a stale/wrong config silently truncates the model (the 3B's 8 missing
    # layers were the root cause of its garbage output).
    L = max(int(n.split('.')[2]) for n in names if '.layers.' in n) + 1
    hs = cfg['hidden_size']; is_ = cfg['intermediate_size']
    nh = cfg['num_attention_heads']; nkv = cfg['num_key_value_heads']
    print(f"[st] {L} layers hs={hs} is={is_} heads={nh} nkv={nkv}")

    def T(n):  # fetch + reshape [dims]
        return T0(n).reshape(hdr[n])

    inits, nodes = [], []
    def add(name, arr):
        inits.append(numpy_helper.from_array(arr, name))

    emb = T('model.embed_tokens.weight')
    # size estimate: int8 linears + embed + lm_head (2GB protobuf cap)
    wb = sum(np.prod(hdr[n]) for n in names if 'layers' in n and '.weight' in n and 'norm' not in n)
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

    # Untied LM head (e.g. qwen2.5-7b): the tied path silently uses the
    # embedding, producing garbage logits.
    if 'lm_head.weight' in names:
        lm = T('lm_head.weight')
        add('lm_head.weight', lm.astype(np.float16))
        print(f"[st] lm_head: {lm.shape} (fp16, untied)")
        tie = 0
    else:
        tie = 1

    graph = helper.make_graph(nodes, "g", [], [], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    out_path = os.path.join(outdir, "model_int8.onnx")
    if big:
        # >2GB protobuf cap: tensors go to a sibling .data file (loader supports it)
        onnx.save(model, out_path, save_as_external_data=True,
                  all_tensors_to_one_file=True, location="model_int8.onnx.data",
                  size_threshold=1024)
        print(f"[st] wrote {out_path} + model_int8.onnx.data ({len(inits)} initializers, external)")
    else:
        onnx.save(model, out_path)
        print(f"[st] wrote {out_path}: {len(inits)} initializers, {len(nodes)} DQ nodes")
    hd = cfg.get('head_dim', hs // nh)  # qwen2.5-3b: 36 heads × 128 ≠ hs/36
    cfg_out = {"model_type": "qwen2", "hidden_size": hs, "intermediate_size": is_,
               "num_attention_heads": nh, "num_key_value_heads": nkv, "head_dim": hd,
               "max_position_embeddings": cfg['max_position_embeddings'],
               "rms_norm_eps": cfg['rms_norm_eps'], "rope_theta": cfg['rope_theta'],
               "vocab_size": cfg['vocab_size'], "tie_word_embeddings": tie}
    with open(os.path.join(outdir, "config.json"), "w") as f2:
        json.dump(cfg_out, f2, indent=2)
    print("[st] wrote config.json")

if __name__ == "__main__":
    main()
