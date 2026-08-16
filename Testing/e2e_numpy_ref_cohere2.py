#!/usr/bin/env python3
"""Cohere2 numpy reference (port of transformers modeling_cohere2.py 5.14).

Validates the engine's cohere2 path on trl-internal-testing/tiny-Cohere2ForCausalLM.
Run:
    python3 Testing/e2e_numpy_ref_cohere2.py <model_dir> <ids.txt> <N>
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

def layernorm(x, w, eps):
    """Cohere2LayerNorm: mean-centered, weight-only (no bias)."""
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.mean((x - mean)**2, axis=-1, keepdims=True)
    return (x - mean) * (1.0 / np.sqrt(var + eps)) * w

def silu(x):
    return x * (1.0 / (1.0 + np.exp(-x)))

def rotate_half_adjacent(x):
    """Cohere2 rotate_half: x[..., ::2] / x[..., 1::2] — ADJACENT pairs (2i,2i+1)."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return np.stack([-x2, x1], axis=-1).reshape(x.shape)

def apply_rope(q, k, pos, cfg):
    """Cohere2 rope: freqs interleaved (repeat_interleave 2), adjacent rotate."""
    HD = q.shape[-1]
    theta = cfg['rope_theta']
    dim = int(cfg.get('head_dim', HD))
    inv = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
    # freqs per position: [seq, dim/2], then repeat_interleave 2 -> [seq, dim]
    for t in range(q.shape[0]):
        freqs = t * inv
        emb = np.repeat(freqs, 2).astype(np.float32)  # interleave
        cos = np.cos(emb); sin = np.sin(emb)
        for h in range(q.shape[1]):
            q[t, h] = q[t, h] * cos + rotate_half_adjacent(q[t, h]) * sin
        for h in range(k.shape[1]):
            k[t, h] = k[t, h] * cos + rotate_half_adjacent(k[t, h]) * sin
    return q, k

def attention(x, l, w, cfg, kvcache):
    """GQA attention, adjacent rope, sliding window. x: [seq, H]."""
    H = cfg['hidden_size']; NH = cfg['num_attention_heads']; NKV = cfg['num_key_value_heads']
    HD = cfg['head_dim']
    pfx = f'model.layers.{l}.self_attn.'
    seq = x.shape[0]
    q = x @ w[pfx + 'q_proj.weight'].T  # [seq, NH*HD]
    k = x @ w[pfx + 'k_proj.weight'].T
    v = x @ w[pfx + 'v_proj.weight'].T
    q = q.reshape(seq, NH, HD)
    k = k.reshape(seq, NKV, HD)
    v = v.reshape(seq, NKV, HD)
    # rope only on sliding_attention layers
    if cfg['layer_types'][l] == 'sliding_attention':
        q, k = apply_rope(q, k, np.arange(seq, dtype=np.float64), cfg)
    qt = q.transpose(1, 0, 2)
    kt = k.transpose(1, 0, 2)
    vt = v.transpose(1, 0, 2)
    kt = np.repeat(kt, NH // NKV, axis=0); vt = np.repeat(vt, NH // NKV, axis=0)
    S = qt @ kt.transpose(0, 2, 1) * (HD ** -0.5)
    S = S + np.triu(np.ones((seq, seq)) * -1e9, k=1)[None]
    P = np.exp(S - S.max(-1, keepdims=True)); P /= P.sum(-1, keepdims=True)
    att = P @ vt
    att = att.transpose(1, 0, 2).reshape(seq, NH * HD)
    return att @ w[pfx + 'o_proj.weight'].T

def mlp(x, l, w, cfg):
    pfx = f'model.layers.{l}.mlp.'
    g = x @ w[pfx + 'gate_proj.weight'].T
    u = x @ w[pfx + 'up_proj.weight'].T
    d = w[pfx + 'down_proj.weight'].T
    return silu(g) * u @ d

def forward(cfg, w, ids, gen_len):
    H = cfg['hidden_size']; L = cfg['num_hidden_layers']
    EPS = cfg['layer_norm_eps']
    tok = ids.copy()
    gen = []
    for _ in range(gen_len):
        x = w['model.embed_tokens.weight'][tok].astype(np.float64)
        seq = len(tok)
        for l in range(L):
            pfx = f'model.layers.{l}.'
            xn = layernorm(x, w[pfx + 'input_layernorm.weight'], EPS)
            # parallel residual: attention + MLP from SAME normed input
            att = attention(xn, l, w, cfg, None)
            m = mlp(xn, l, w, cfg)
            x = x + att + m
        x = layernorm(x, w['model.norm.weight'], EPS)
        head = w['lm_head.weight'] if 'lm_head.weight' in w else w['model.embed_tokens.weight']  # tied
        logits = x[-1] @ head.T
        if cfg.get('logit_scale') and cfg['logit_scale'] != 1.0:
            logits = logits * cfg['logit_scale']
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
