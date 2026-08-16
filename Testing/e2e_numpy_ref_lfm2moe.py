#!/usr/bin/env python3
"""LFM2-MoE numpy reference (port of transformers modeling_lfm2_moe.py 5.14).

Validates the engine's lfm2_moe path on the tiny random checkpoint.
Run: python3 Testing/e2e_numpy_ref_lfm2moe.py <model_dir> <ids.txt> <N>
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
    """Full head-dim RoPE (rotate_half). q,k: [NH or NKV, HD]."""
    HD = q.shape[1]
    inv = 1.0 / (theta ** (np.arange(0, HD, 2) / HD))
    freq = pos * inv  # [HD/2]
    cos = np.cos(freq); sin = np.sin(freq)
    def apply(x):
        x1 = x[:, :HD//2]; x2 = x[:, HD//2:]
        return np.concatenate([x1*cos - x2*sin, x1*sin + x2*cos], axis=-1)
    return apply(q), apply(k)

def conv_layer(hidden, l, w, cfg, state):
    """ShortConv: in_proj -> B,C,x split -> Bx=B*x -> depthwise conv1d (L_cache=3) -> y=C*conv_out -> out_proj."""
    pfx = f'model.layers.{l}.conv.'
    H = hidden.shape[0]
    BCx = hidden @ w[pfx + 'in_proj.weight'].T
    B, C, x = BCx[:H], BCx[H:2*H], BCx[2*H:]
    Bx = B * x
    Lc = cfg['conv_L_cache']
    conv_w = w[pfx + 'conv.weight'][:, 0, :]  # [H, Lc] depthwise
    # causal conv with state (Lc-1 previous Bx values), w[0]=oldest
    out = np.zeros(H)
    for c in range(H):
        acc = conv_w[c, 0] * state[c][0] + conv_w[c, 1] * state[c][1]
        acc += conv_w[c, 2] * Bx[c]
        out[c] = acc
    if cfg['conv_bias']:
        out += w[pfx + 'conv.bias']
    for c in range(H):
        state[c][0] = state[c][1]
        state[c][1] = Bx[c]
    y = C * out
    return y @ w[pfx + 'out_proj.weight'].T

def attention(hidden, l, w, cfg, pos, kv_k, kv_v):
    pfx = f'model.layers.{l}.self_attn.'
    H = hidden.shape[0]
    NH = cfg['num_attention_heads']; NKV = cfg['num_key_value_heads']
    HD = H // NH
    q = (hidden @ w[pfx + 'q_proj.weight'].T).reshape(NH, HD)
    k = (hidden @ w[pfx + 'k_proj.weight'].T).reshape(NKV, HD)
    v = (hidden @ w[pfx + 'v_proj.weight'].T).reshape(NKV, HD)
    q = rmsnorm(q, w[pfx + 'q_layernorm.weight'], cfg['norm_eps'])
    k = rmsnorm(k, w[pfx + 'k_layernorm.weight'], cfg['norm_eps'])
    q, k = rope(q, k, pos, cfg['rope_parameters']['rope_theta'])
    kv_k.append(k); kv_v.append(v)
    T = len(kv_k)
    scale = HD ** -0.5
    out = np.zeros((NH, HD))
    for hh in range(NH):
        kh = hh % NKV
        scores = np.array([np.dot(q[hh], kv_k[t][kh]) * scale for t in range(T)])
        scores = scores - scores.max()
        e = np.exp(scores); e /= e.sum()
        out[hh] = sum(e[t] * kv_v[t][kh] for t in range(T))
    return out.reshape(H) @ w[pfx + 'out_proj.weight'].T

def mlp(hidden, l, w, cfg):
    pfx = f'model.layers.{l}.feed_forward.'
    return silu(hidden @ w[pfx + 'w1.weight'].T) * (hidden @ w[pfx + 'w3.weight'].T) @ w[pfx + 'w2.weight'].T

def moe(hidden, l, w, cfg):
    pfx = f'model.layers.{l}.feed_forward.'
    E = cfg['num_experts']; TOPK = cfg['num_experts_per_tok']
    FF = cfg['moe_intermediate_size']
    logits = hidden @ w[pfx + 'gate.weight'].T
    routing = 1.0 / (1.0 + np.exp(-logits))  # sigmoid
    bias = w.get(pfx + 'expert_bias')
    scores = routing + (bias if bias is not None else 0.0)
    idx = np.argsort(-scores)[:TOPK]
    rw = routing[idx]
    if cfg['norm_topk_prob']:
        rw = rw / (rw.sum() + 1e-6)
    rw = rw * cfg['routed_scaling_factor']
    out = np.zeros(hidden.shape[0])
    for ei, wt in zip(idx, rw):
        w1 = w[f'{pfx}experts.{ei}.w1.weight']
        w2 = w[f'{pfx}experts.{ei}.w2.weight']
        w3 = w[f'{pfx}experts.{ei}.w3.weight']
        out += wt * (silu(hidden @ w1.T) * (hidden @ w3.T) @ w2.T)
    return out

def block(hidden, l, w, cfg, states, pos, kv_k, kv_v):
    pfx = f'model.layers.{l}.'
    eps = cfg['norm_eps']
    x = rmsnorm(hidden, w[pfx + 'operator_norm.weight'], eps)
    if cfg['layer_types'][l] == 'conv':
        x = conv_layer(x, l, w, cfg, states[l])
    else:
        x = attention(x, l, w, cfg, pos, kv_k[l], kv_v[l])
    hidden = hidden + x
    x = rmsnorm(hidden, w[pfx + 'ffn_norm.weight'], eps)
    if l < cfg['num_dense_layers']:
        x = mlp(x, l, w, cfg)
    else:
        x = moe(x, l, w, cfg)
    return hidden + x

def forward(cfg, w, ids, gen_len):
    H = cfg['hidden_size']; L = cfg['num_hidden_layers']
    eps = cfg['norm_eps']
    Lc = cfg['conv_L_cache']
    states = [[np.zeros(Lc - 1) for _ in range(H)] for _ in range(L)]
    kv_k = [[] for _ in range(L)]; kv_v = [[] for _ in range(L)]
    tok = ids.copy()
    gen = []
    for t in tok:
        x = w['model.embed_tokens.weight'][t]
        for l in range(L):
            x = block(x, l, w, cfg, states, len(tok) - 1, kv_k, kv_v)
        x = rmsnorm(x, w['model.embedding_norm.weight'], eps)
        logits = x @ w['model.embed_tokens.weight'].T
        nxt = int(np.argmax(logits))
    gen.append(nxt)
    tok.append(nxt)
    for _ in range(gen_len - 1):
        x = w['model.embed_tokens.weight'][tok[-1]]
        for l in range(L):
            x = block(x, l, w, cfg, states, len(tok) - 1, kv_k, kv_v)
        x = rmsnorm(x, w['model.embedding_norm.weight'], eps)
        logits = x @ w['model.embed_tokens.weight'].T
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
