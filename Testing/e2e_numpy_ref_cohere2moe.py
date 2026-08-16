#!/usr/bin/env python3
"""Cohere2Moe numpy reference (port of transformers modeling_cohere2_moe.py 5.14).

Validates the engine's cohere2_moe path on the tiny random checkpoint.
Run: python3 Testing/e2e_numpy_ref_cohere2moe.py <model_dir> <ids.txt> <N>
Prints the N-token argmax chain (like the other numpy-ref oracles).
"""
import json, sys, os
import numpy as np

def load_safetensors(path):
    with open(path, 'rb') as f:
        n = int.from_bytes(f.read(8), 'little')
        hdr = json.loads(f.read(n))
        data = f.read()
    out = {}
    for k, v in hdr.items():
        if k == '__metadata__': continue
        dt = v['dtype']; sh = v['shape']
        raw = data[v['data_offsets'][0]:v['data_offsets'][1]]
        if dt == 'F32':
            a = np.frombuffer(raw, dtype=np.float32).reshape(sh).copy()
        elif dt == 'BF16':
            u = np.frombuffer(raw, dtype=np.uint16).reshape(sh).copy()
            a = (u.astype(np.uint32) << 16).view(np.float32).copy()
        else:
            raise ValueError(dt)
        out[k] = a
    return out

def rmsnorm(x, w, eps):
    return x / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps) * w

def layernorm_mean(x, w, eps):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.mean((x - mean)**2, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * w

def gelu(x):
    return 0.5 * x * (1.0 + np.vectorize(lambda t: __import__('math').erf(t / np.sqrt(2.0)))(x))

def rope(q, k, pos, theta):
    HD = q.shape[1]
    inv = 1.0 / (theta ** (np.arange(0, HD, 2) / HD))
    freq = pos * inv
    cos = np.cos(freq); sin = np.sin(freq)
    def apply(x):
        x1 = x[:, :HD//2]; x2 = x[:, HD//2:]
        return np.concatenate([x1*cos - x2*sin, x1*sin + x2*cos], axis=-1)
    return apply(q), apply(k)

def attention(hidden, l, w, cfg, pos, kv_k, kv_v, use_rope):
    pfx = f'model.layers.{l}.self_attn.'
    H = hidden.shape[0]
    NH = cfg['num_attention_heads']; NKV = cfg['num_key_value_heads']
    HD = cfg.get('head_dim', H // NH)
    q = (hidden @ w[pfx + 'q_proj.weight'].T).reshape(NH, HD)
    k = (hidden @ w[pfx + 'k_proj.weight'].T).reshape(NKV, HD)
    v = (hidden @ w[pfx + 'v_proj.weight'].T).reshape(NKV, HD)
    if use_rope:
        q, k = rope(q, k, pos, cfg['rope_parameters']['rope_theta'])
    kv_k.append(k); kv_v.append(v)
    T = len(kv_k)
    scale = HD ** -0.5
    out = np.zeros((NH, HD))
    for hh in range(NH):
        kh = hh // (NH // NKV)
        scores = np.array([np.dot(q[hh], kv_k[t][kh]) * scale for t in range(T)])
        scores = scores - scores.max()
        e = np.exp(scores); e /= e.sum()
        out[hh] = sum(e[t] * kv_v[t][kh] for t in range(T))
    return out.reshape(NH * HD) @ w[pfx + 'o_proj.weight'].T

def mlp(hidden, l, w, cfg):
    pfx = f'model.layers.{l}.mlp.'
    FF = cfg['intermediate_size']
    g = gelu(hidden @ w[pfx + 'gate_proj.weight'].T)
    u = hidden @ w[pfx + 'up_proj.weight'].T
    return (g * u) @ w[pfx + 'down_proj.weight'].T

def moe(hidden, l, w, cfg):
    pfx = f'model.layers.{l}.mlp.'
    E = cfg['num_experts']; TOPK = cfg['num_experts_per_tok']
    FF = cfg['intermediate_size']
    logits = hidden @ w[pfx + 'gate.weight'].T
    # expert_selection_fn softmax
    e = np.exp(logits - logits.max())
    probs = e / e.sum()
    idx = np.argsort(-probs)[:TOPK]
    rw = probs[idx]
    if cfg['norm_topk_prob']:
        rw = rw / rw.sum()
    out = np.zeros(hidden.shape[0])
    for ei, wt in zip(idx, rw):
        w1 = w[f'{pfx}experts.{ei}.gate_proj.weight']
        w2 = w[f'{pfx}experts.{ei}.down_proj.weight']
        w3 = w[f'{pfx}experts.{ei}.up_proj.weight']
        out += wt * (gelu(hidden @ w1.T) * (hidden @ w3.T) @ w2.T)
    return out

def block(hidden, l, w, cfg, pos, kv_k, kv_v):
    pfx = f'model.layers.{l}.'
    # LayerNorm (mean-centered weight-only) since rms_norm_eps is null
    x = layernorm_mean(hidden, w[pfx + 'input_layernorm.weight'], cfg['layer_norm_eps'])
    sliding = cfg['layer_types'][l] == 'sliding_attention'
    force = cfg['mlp_layer_types'][l] == 'dense' and cfg['prefix_dense_sliding_window_pattern'] == 1
    att = attention(x, l, w, cfg, pos, kv_k[l], kv_v[l], sliding or force)
    m = mlp(x, l, w, cfg) if cfg['mlp_layer_types'][l] == 'dense' else moe(x, l, w, cfg)
    return hidden + att + m  # parallel residual

def forward(cfg, w, ids, gen_len):
    H = cfg['hidden_size']; L = cfg['num_hidden_layers']
    kv_k = [[] for _ in range(L)]; kv_v = [[] for _ in range(L)]
    tok = ids.copy()
    gen = []
    for i, t in enumerate(tok):
        x = w['model.embed_tokens.weight'][t]
        for l in range(L):
            x = block(x, l, w, cfg, i, kv_k, kv_v)
        x = layernorm_mean(x, w['model.norm.weight'], cfg['layer_norm_eps'])
        logits = x @ w['lm_head.weight'].T
        nxt = int(np.argmax(logits))
    gen.append(nxt)
    tok.append(nxt)
    for _ in range(gen_len - 1):
        x = w['model.embed_tokens.weight'][tok[-1]]
        for l in range(L):
            x = block(x, l, w, cfg, len(tok) - 1, kv_k, kv_v)
        x = layernorm_mean(x, w['model.norm.weight'], cfg['layer_norm_eps'])
        logits = x @ w['lm_head.weight'].T
        nxt = int(np.argmax(logits))
        gen.append(nxt)
        tok.append(nxt)
    return gen

if __name__ == '__main__':
    model_dir = sys.argv[1]
    ids = [int(t) for t in open(sys.argv[2]).read().split()]
    gen_len = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    cfg = json.load(open(os.path.join(model_dir, 'config.json')))
    w = {}
    for f in [os.path.join(model_dir, 'model.safetensors')]:
        w.update(load_safetensors(f))
    gen = forward(cfg, w, ids, gen_len)
    print('ref-gen:', ' '.join(str(g) for g in gen))
