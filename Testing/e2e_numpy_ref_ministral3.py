#!/usr/bin/env python3
"""Ministral3 numpy reference (port of transformers modeling_ministral3.py 5.14).

Validates the engine's ministral3 path on the tiny random checkpoint.
Run: python3 Testing/e2e_numpy_ref_ministral3.py <model_dir> <ids.txt> <N>
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

def gelu(x):
    return 0.5 * x * (1.0 + np.vectorize(lambda t: __import__('math').erf(t / np.sqrt(2.0)))(x))

def yarn_inv_freq(cfg, HD):
    """YaRN inv_freq from transformers _compute_yarn_parameters."""
    rp = cfg['rope_parameters']
    base = rp['rope_theta']
    factor = rp['factor']
    beta_fast = rp.get('beta_fast', 32.0)
    beta_slow = rp.get('beta_slow', 1.0)
    orig_max = rp['original_max_position_embeddings']
    pos_freqs = base ** (np.arange(0, HD, 2) / HD)
    inv_extrap = 1.0 / pos_freqs
    inv_interp = 1.0 / (factor * pos_freqs)
    def find_dim(num_rot):
        return (HD * np.log(orig_max / (num_rot * 2 * np.pi))) / (2 * np.log(base))
    low = max(find_dim(beta_fast), 0)
    high = min(find_dim(beta_slow), HD - 1)
    if low == high:
        high += 0.001
    ramp = np.clip((np.arange(HD // 2) - low) / (high - low), 0, 1)
    inv_freq = inv_interp * (1 - ramp) + inv_extrap * ramp
    return inv_freq

def yarn_attn_factor(cfg):
    rp = cfg['rope_parameters']
    factor = rp['factor']
    mscale = rp.get('mscale', 1.0)
    mscale_all_dim = rp.get('mscale_all_dim', 1.0)
    def get_mscale(s, m=1.0):
        return 1.0 if s <= 1 else 0.1 * m * np.log(s) + 1.0
    if mscale and mscale_all_dim:
        return get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim)
    return get_mscale(factor)

def rope(q, k, pos, cfg):
    """YaRN rope with attention scaling, full head dim."""
    HD = q.shape[1]
    inv = yarn_inv_freq(cfg, HD)
    freq = pos * inv
    cos = np.cos(freq); sin = np.sin(freq)
    af = yarn_attn_factor(cfg)
    def apply(x):
        x1 = x[:, :HD//2]; x2 = x[:, HD//2:]
        return np.concatenate([x1*cos - x2*sin, x1*sin + x2*cos], axis=-1) * af
    return apply(q), apply(k)

def llama4_scale(pos, cfg):
    """1 + beta*log(1 + floor(pos/orig_max))"""
    rp = cfg['rope_parameters']
    beta = rp.get('llama_4_scaling_beta', 0.1)
    orig = rp.get('original_max_position_embeddings', 16384)
    return 1.0 + beta * np.log(1 + np.floor(pos / orig))

def attention(hidden, l, w, cfg, pos, kv_k, kv_v):
    pfx = f'model.layers.{l}.self_attn.'
    H = hidden.shape[0]
    NH = cfg['num_attention_heads']; NKV = cfg['num_key_value_heads']
    HD = cfg.get('head_dim', H // NH)
    q = (hidden @ w[pfx + 'q_proj.weight'].T).reshape(NH, HD)
    k = (hidden @ w[pfx + 'k_proj.weight'].T).reshape(NKV, HD)
    v = (hidden @ w[pfx + 'v_proj.weight'].T).reshape(NKV, HD)
    q, k = rope(q, k, pos, cfg)
    q = q * llama4_scale(pos, cfg)  # llama-4 attn scaling on q
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

def block(hidden, l, w, cfg, pos, kv_k, kv_v):
    pfx = f'model.layers.{l}.'
    eps = cfg['rms_norm_eps']
    x = rmsnorm(hidden, w[pfx + 'input_layernorm.weight'], eps)
    x = attention(x, l, w, cfg, pos, kv_k[l], kv_v[l])
    hidden = hidden + x
    x = rmsnorm(hidden, w[pfx + 'post_attention_layernorm.weight'], eps)
    x = mlp(x, l, w, cfg)
    return hidden + x

def forward(cfg, w, ids, gen_len):
    H = cfg['hidden_size']; L = cfg['num_hidden_layers']
    eps = cfg['rms_norm_eps']
    kv_k = [[] for _ in range(L)]; kv_v = [[] for _ in range(L)]
    tok = ids.copy()
    gen = []
    for i, t in enumerate(tok):
        x = w['model.embed_tokens.weight'][t]
        for l in range(L):
            x = block(x, l, w, cfg, i, kv_k, kv_v)
        x = rmsnorm(x, w['model.norm.weight'], eps)
        logits = x @ w['lm_head.weight'].T
        nxt = int(np.argmax(logits))
    gen.append(nxt)
    tok.append(nxt)
    for _ in range(gen_len - 1):
        x = w['model.embed_tokens.weight'][tok[-1]]
        for l in range(L):
            x = block(x, l, w, cfg, len(tok) - 1, kv_k, kv_v)
        x = rmsnorm(x, w['model.norm.weight'], eps)
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
