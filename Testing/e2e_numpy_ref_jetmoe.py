#!/usr/bin/env python3
"""JetMoE numpy reference (port of transformers modeling_jetmoe.py 5.14).

Validates the engine's jetmoe path on the tiny random checkpoint.
Run: python3 Testing/e2e_numpy_ref_jetmoe.py <model_dir> <ids.txt> <N>
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

def silu(x):
    return x / (1.0 + np.exp(-x))

def rope(q, k, pos, theta):
    HD = q.shape[1]
    inv = 1.0 / (theta ** (np.arange(0, HD, 2) / HD))
    freq = pos * inv
    cos = np.cos(freq); sin = np.sin(freq)
    def apply(x):
        x1 = x[:, :HD//2]; x2 = x[:, HD//2:]
        return np.concatenate([x1*cos - x2*sin, x1*sin + x2*cos], axis=-1)
    return apply(q), apply(k)

def gating(hidden, w, E, topk):
    """topk softmax gates + sorted topology (for single token, batch_index=0)."""
    logits = hidden @ w.T
    idx = np.argsort(-logits)[:topk]
    tl = logits[idx]
    e = np.exp(tl - tl.max())
    gates = e / e.sum()
    return idx, gates

def moa_attention(hidden, l, w, cfg, pos, kv_k, kv_v):
    """Mixture of Attention: routed query projections, shared KV, per-expert reduce."""
    pfx = f'model.layers.{l}.self_attention.'
    H = hidden.shape[0]
    NKV = cfg['num_key_value_heads']
    HD = cfg['kv_channels']
    NH = NKV  # effective q heads = kv_projection_size / kv_channels
    E = cfg['num_local_experts']; TOPK = cfg['num_experts_per_tok']
    # map: per-expert q projection [kv_channels * num_key_value_heads]
    idx, gates = gating(hidden, w[pfx + 'experts.router.layer.weight'], E, TOPK)
    bias = w[pfx + 'experts.bias']
    qs = np.zeros((TOPK, NH * HD))
    for k, e in enumerate(idx):
        qs[k] = hidden @ w[f'{pfx}experts.input_linear.weight'][e].T
    qs = qs.reshape(TOPK, NH, HD)
    kv = hidden @ w[pfx + 'kv_proj.weight'].T
    k, v = kv[:NKV*HD], kv[NKV*HD:]
    k = k.reshape(NKV, HD); v = v.reshape(NKV, HD)
    # rope on q (all topk) and k
    for kk in range(TOPK):
        qs[kk], _ = rope(qs[kk], k, pos, cfg['rope_parameters']['rope_theta'])
    k, _ = rope(k, k, pos, cfg['rope_parameters']['rope_theta'])
    kv_k.append(k); kv_v.append(v)
    T = len(kv_k)
    scale = HD ** -0.5
    # attention: query [TOPK*NH, HD], kv repeated TOPK times (repeat not interleave)
    outs = np.zeros((TOPK, NH, HD))
    for kk in range(TOPK):
        for hh in range(NH):
            kh = hh // (NH // NKV)
            scores = np.array([np.dot(qs[kk][hh], kv_k[t][kh]) * scale for t in range(T)])
            scores = scores - scores.max()
            e2 = np.exp(scores); e2 /= e2.sum()
            outs[kk][hh] = sum(e2[t] * kv_v[t][kh] for t in range(T))
    # reduce: per-expert output projection + gate weights
    flat = outs.reshape(TOPK, NH * HD)
    acc = np.zeros(H)
    for k, e in enumerate(idx):
        o = flat[k] @ w[f'{pfx}experts.output_linear.weight'][e].T
        acc += gates[k] * o
    return acc + bias

def moe(hidden, l, w, cfg):
    pfx = f'model.layers.{l}.mlp.'
    E = cfg['num_local_experts']; TOPK = cfg['num_experts_per_tok']
    FF = cfg['intermediate_size']
    bias = w[pfx + 'bias']
    idx, gates = gating(hidden, w[pfx + 'router.layer.weight'], E, TOPK)
    acc = np.zeros(hidden.shape[0])
    for k, e in enumerate(idx):
        gu = hidden @ w[f'{pfx}input_linear.weight'][e].T  # [2*FF]
        g, u = gu[:FF], gu[FF:]
        h = silu(g) * u
        acc += gates[k] * (h @ w[f'{pfx}output_linear.weight'][e].T)
    return acc + bias

def block(hidden, l, w, cfg, pos, kv_k, kv_v):
    pfx = f'model.layers.{l}.'
    eps = cfg['rms_norm_eps']
    x = rmsnorm(hidden, w[pfx + 'input_layernorm.weight'], eps)
    x = moa_attention(x, l, w, cfg, pos, kv_k[l], kv_v[l])
    hidden = hidden + x
    x = rmsnorm(hidden, w[pfx + 'post_attention_layernorm.weight'], eps)
    x = moe(x, l, w, cfg)
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
