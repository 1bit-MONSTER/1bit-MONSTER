#!/usr/bin/env python3
"""GraniteMoeHybrid numpy reference (port of transformers modeling_granitemoehybrid.py 5.14).

Validates the engine's granitemoehybrid path on the tiny random checkpoint.
Run: python3 Testing/e2e_numpy_ref_granitemoehybrid.py <model_dir> <ids.txt> <N>
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

def mamba(hidden, l, w, cfg, state):
    """Single-token Mamba-2 mixer. state: dict with conv [conv_dim, d_conv-1], h [n_heads, d_head, d_state]."""
    pfx = f'model.layers.{l}.mamba.'
    H = hidden.shape[0]
    INT = int(cfg['mamba_expand'] * H)
    NH = cfg['mamba_n_heads']; DH = cfg['mamba_d_head']; DS = cfg['mamba_d_state']
    NG = cfg['mamba_n_groups']; DC = cfg['mamba_d_conv']
    conv_dim = INT + 2 * NG * DS
    proj = w[pfx + 'in_proj.weight'].T  # [H, INT+conv_dim+NH]
    p = hidden @ proj
    gate = p[:INT]
    hidden_BC = p[INT:INT + conv_dim]
    dt = p[INT + conv_dim:]
    # conv1d with state (w[0]=oldest)
    conv_w = w[pfx + 'conv1d.weight'][:, 0, :]  # [conv_dim, DC]
    conv_b = w[pfx + 'conv1d.bias'] if pfx + 'conv1d.bias' in w else None
    out = np.zeros(conv_dim)
    for c in range(conv_dim):
        acc = conv_w[c, 0] * state['conv'][c][-1] + conv_w[c, 1] * state['conv'][c][-2] + \
              conv_w[c, 2] * state['conv'][c][-3] + conv_w[c, 3] * hidden_BC[c]
        out[c] = acc
    if conv_b is not None: out += conv_b
    out = silu(out)
    # shift conv state
    for c in range(conv_dim):
        state['conv'][c][0] = state['conv'][c][1]
        state['conv'][c][1] = state['conv'][c][2]
        state['conv'][c][2] = hidden_BC[c]
    h = out[:INT]
    B = out[INT:INT + NG * DS]
    C = out[INT + NG * DS:INT + 2 * NG * DS]
    # SSM
    A = -np.exp(w[pfx + 'A_log'])  # [NH]
    dt_p = dt + w[pfx + 'dt_bias']
    dt_s = np.log1p(np.exp(dt_p))  # softplus
    # dt clamp: time_step_limit (0.0, inf) — no-op
    A_exp = A[:, None, None].repeat(DH, axis=1).repeat(DS, axis=2)  # [NH, DH, DS]
    dA = np.exp(dt_s[:, None, None] * A_exp)
    B2 = B.reshape(NG, DS)[None].repeat(NH // NG, axis=0).reshape(NH, DS)  # [NH, DS]
    dB = dt_s[:, None, None] * B2[:, None, :]  # [NH, 1, DS] * broadcast -> [NH, DH, DS]
    h2 = h.reshape(NH, DH)
    dBx = dB * h2[:, :, None]
    state['h'] = state['h'] * dA + dBx
    C2 = C.reshape(NG, DS)[None].repeat(NH // NG, axis=0).reshape(NH, DS)
    y = np.sum(state['h'] * C2[:, None, :], axis=-1)  # [NH, DH]
    D = w[pfx + 'D']
    y = y + h2 * D[:, None]
    y = y.reshape(INT)
    # gated RMSNorm: y * silu(gate) then RMS
    gated = y * silu(gate)
    y = rmsnorm(gated, w[pfx + 'norm.weight'], cfg['rms_norm_eps'])
    return y @ w[pfx + 'out_proj.weight'].T

def attention(hidden, l, w, cfg, kv_k, kv_v):
    """Single-token GQA, NoPE, scaling = attention_multiplier. Appends to kv cache."""
    pfx = f'model.layers.{l}.self_attn.'
    H = hidden.shape[0]
    NH = cfg['num_attention_heads']; NKV = cfg['num_key_value_heads']
    HD = H // NH
    q = (hidden @ w[pfx + 'q_proj.weight'].T).reshape(NH, HD)
    k = (hidden @ w[pfx + 'k_proj.weight'].T).reshape(NKV, HD)
    v = (hidden @ w[pfx + 'v_proj.weight'].T).reshape(NKV, HD)
    kv_k.append(k); kv_v.append(v)
    T = len(kv_k)
    scale = cfg['attention_multiplier']
    out = np.zeros((NH, HD))
    for hh in range(NH):
        kh = hh % NKV
        scores = np.zeros(T)
        for t in range(T):
            s = 0.0
            for d in range(HD):
                s += q[hh, d] * kv_k[t][kh, d]
            scores[t] = s * scale
        scores = scores - scores.max()
        e = np.exp(scores)
        e /= e.sum()
        acc = np.zeros(HD)
        for t in range(T):
            acc += e[t] * kv_v[t][kh]
        out[hh] = acc
    return out.reshape(H) @ w[pfx + 'o_proj.weight'].T

def moe(hidden, l, w, cfg):
    """MoE: router top-10 softmax + gate_up/down per expert."""
    pfx = f'model.layers.{l}.block_sparse_moe.'
    E = cfg['num_local_experts']; TOPK = cfg['num_experts_per_tok']
    FF = cfg['intermediate_size']
    H = hidden.shape[0]
    logits = hidden @ w[pfx + 'router.layer.weight'].T
    idx = np.argsort(-logits)[:TOPK]
    topk_logits = logits[idx]
    e = np.exp(topk_logits - topk_logits.max())
    wts = e / e.sum()
    out = np.zeros(H)
    for ei, wi in zip(idx, wts):
        gu = hidden @ w[pfx + 'input_linear.weight'][ei].T  # [2*FF]
        g, u = gu[:FF], gu[FF:]
        h = silu(g) * u
        out += wi * (h @ w[pfx + 'output_linear.weight'][ei].T)
    return out

def shared_mlp(hidden, l, w, cfg):
    pfx = f'model.layers.{l}.shared_mlp.'
    FF = cfg['shared_intermediate_size']
    gu = hidden @ w[pfx + 'input_linear.weight'].T
    g, u = gu[:FF], gu[FF:]
    return (silu(g) * u) @ w[pfx + 'output_linear.weight'].T

def block(hidden, l, w, cfg, states, kv_k, kv_v):
    pfx = f'model.layers.{l}.'
    eps = cfg['rms_norm_eps']
    x = rmsnorm(hidden, w[pfx + 'input_layernorm.weight'], eps)
    if cfg['layers_block_type'][l] == 'linear_attention':
        x = mamba(x, l, w, cfg, states[l])
    else:
        x = attention(x, l, w, cfg, kv_k[l], kv_v[l])
    hidden = hidden + x * cfg['residual_multiplier']
    x = rmsnorm(hidden, w[pfx + 'post_attention_layernorm.weight'], eps)
    if cfg['num_local_experts'] > 0:
        x = moe(x, l, w, cfg) + shared_mlp(x, l, w, cfg)
    else:
        x = shared_mlp(x, l, w, cfg)
    return hidden + x * cfg['residual_multiplier']

def forward(cfg, w, ids, gen_len):
    H = cfg['hidden_size']; L = cfg['num_hidden_layers']
    eps = cfg['rms_norm_eps']
    INT = int(cfg['mamba_expand'] * H); DS = cfg['mamba_d_state']
    NH = cfg['mamba_n_heads']; DH = cfg['mamba_d_head']; DC = cfg['mamba_d_conv']
    conv_dim = INT + 2 * cfg['mamba_n_groups'] * DS
    # mamba states per layer + kv caches per layer
    states = [{'conv': [np.zeros(DC - 1) for _ in range(conv_dim)],
               'h': np.zeros((NH, DH, DS))} for _ in range(L)]
    kv_k = [[] for _ in range(L)]; kv_v = [[] for _ in range(L)]
    tok = ids.copy()
    gen = []
    for t in tok:
        x = w['model.embed_tokens.weight'][t] * cfg['embedding_multiplier']
        for l in range(L):
            x = block(x, l, w, cfg, states, kv_k, kv_v)
        x = rmsnorm(x, w['model.norm.weight'], eps)
        logits = x @ w['model.embed_tokens.weight'].T * cfg['logits_scaling']
        nxt = int(np.argmax(logits))
    gen.append(nxt)
    tok.append(nxt)
    for _ in range(gen_len - 1):
        x = w['model.embed_tokens.weight'][tok[-1]] * cfg['embedding_multiplier']
        for l in range(L):
            x = block(x, l, w, cfg, states, kv_k, kv_v)
        x = rmsnorm(x, w['model.norm.weight'], eps)
        logits = x @ w['model.embed_tokens.weight'].T * cfg['logits_scaling']
        nxt = int(np.argmax(logits))
        gen.append(nxt)
        tok.append(nxt)
    return gen

if __name__ == '__main__':
    model_dir = sys.argv[1]
    ids = [int(t) for t in open(sys.argv[2]).read().split()]
    gen_len = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    cfg = json.load(open(os.path.join(model_dir, 'config.json')))
    # layers_block_type may come from 'layer_types'; normalize names
    if 'layers_block_type' not in cfg and 'layer_types' in cfg:
        cfg['layers_block_type'] = cfg['layer_types']
    cfg['layers_block_type'] = ['linear_attention' if t in ('mamba', 'linear_attention') else 'full_attention'
                                for t in cfg['layers_block_type']]
    w = {}
    for f in [os.path.join(model_dir, 'model.safetensors')]:
        w.update(load_safetensors(f))
    gen = forward(cfg, w, ids, gen_len)
    print('ref-gen:', ' '.join(str(g) for g in gen))
