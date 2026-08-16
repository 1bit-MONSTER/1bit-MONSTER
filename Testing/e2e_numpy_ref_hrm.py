#!/usr/bin/env python3
"""HrmText numpy reference (port of transformers modeling_hrm_text.py 5.14).

Validates the engine's hrm_text path on the tiny random checkpoint.
Run: python3 Testing/e2e_numpy_ref_hrm.py <model_dir> <ids.txt> <N>
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

def rmsnorm(x, eps):
    return x / np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)  # weight-less

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

def attention(hidden, l, mod, w, cfg, pos, kv_k, kv_v):
    """HrmTextAttention: fused gqkv [q,k,v,gate each NH*HD], MHA, rope, sigmoid gate."""
    pfx = f'model.{mod}.layers.{l}.attn.'
    H = hidden.shape[0]
    NH = cfg['num_attention_heads']
    HD = cfg.get('head_dim', H // NH)
    dim = NH * HD
    g = hidden @ w[pfx + 'gqkv_proj.weight'].T  # [4*dim]
    q = g[:dim].reshape(NH, HD)
    k = g[dim:2*dim].reshape(NH, HD)
    v = g[2*dim:3*dim].reshape(NH, HD)
    gate = g[3*dim:]
    q, k = rope(q, k, pos, cfg['rope_parameters']['rope_theta'])
    kv_k.append(k); kv_v.append(v)
    T = len(kv_k)
    scale = HD ** -0.5
    out = np.zeros((NH, HD))
    for hh in range(NH):
        scores = np.array([np.dot(q[hh], kv_k[t][hh]) * scale for t in range(T)])
        scores = scores - scores.max()
        e = np.exp(scores); e /= e.sum()
        out[hh] = sum(e[t] * kv_v[t][hh] for t in range(T))
    out = out.reshape(dim)
    out = out * (1.0 / (1.0 + np.exp(-gate)))
    return out @ w[pfx + 'o_proj.weight'].T

def mlp(hidden, l, mod, w, cfg):
    pfx = f'model.{mod}.layers.{l}.mlp.'
    FF = cfg['intermediate_size']
    gu = hidden @ w[pfx + 'gate_up_proj.weight'].T
    g, u = gu[:FF], gu[FF:]
    return (gelu(g) * u) @ w[pfx + 'down_proj.weight'].T

def layer(hidden, l, mod, w, cfg, pos, kv_k, kv_v):
    eps = cfg['rms_norm_eps']
    x = rmsnorm(hidden, eps)
    a = attention(x, l, mod, w, cfg, pos, kv_k, kv_v)
    hidden = hidden + a
    x = rmsnorm(hidden, eps)
    m = mlp(x, l, mod, w, cfg)
    return hidden + m

def stack(hidden, mod, w, cfg, pos, kv_k, kv_v, nlayers):
    for l in range(nlayers):
        hidden = layer(hidden, l, mod, w, cfg, pos, kv_k[l], kv_v[l])
    return rmsnorm(hidden, cfg['rms_norm_eps'])  # stack final_norm (weight-less)

def forward(cfg, w, ids, gen_len):
    H = cfg['hidden_size']; eps = cfg['rms_norm_eps']
    L = cfg['num_layers_per_stack']
    HC = cfg['H_cycles']; LC = cfg['L_cycles']
    # torch cache slots: slot(h,l,layer) = (h*(L_cycles+1)+l)*num_layers_per_stack + layer
    # L stack at (h, l), H stack at (h, L_cycles) — SHARED slot space
    kv_k = [[[] for _ in range(L)] for _ in range(HC * (LC + 1))]
    kv_v = [[[] for _ in range(L)] for _ in range(HC * (LC + 1))]
    tok = ids.copy()
    gen = []
    for t in tok:
        zH = w['model.embed_tokens.weight'][t] * cfg['embedding_scale']
        zL = w['model.z_L_init'].copy()
        for hc in range(HC):
            for lc in range(LC):
                slot = hc * (LC + 1) + lc
                zL = stack(zL + zH, 'L_module', w, cfg, len(tok) - 1, kv_k[slot], kv_v[slot], L)
            slot = hc * (LC + 1) + LC
            zH = stack(zH + zL, 'H_module', w, cfg, len(tok) - 1, kv_k[slot], kv_v[slot], L)
        # H stack applies final_norm internally (weight-less RMSNorm)
        logits = zH @ w['lm_head.weight'].T
        nxt = int(np.argmax(logits))
        gen.append(nxt)
        tok.append(nxt)
        if len(gen) >= gen_len:
            break
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
