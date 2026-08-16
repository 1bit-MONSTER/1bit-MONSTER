#!/usr/bin/env python3
"""Falcon-H1 numpy reference (port of transformers modeling_falcon_h1.py 5.14).

Validates the engine's falcon_h1 path on yujiepan/falcon-h1-tiny-random.
Run:
    python3 Testing/e2e_numpy_ref_falconh1.py <model_dir> <ids.txt> <N>
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
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * w

def silu(x):
    return x * (1.0 / (1.0 + np.exp(-x)))

def gated_rmsnorm(x, gate, w, eps, n_groups):
    """FalconH1RMSNormGated: norm_before_gate=False -> gate first, then
    group RMSNorm with weight view [groups, dim/groups]."""
    x = x * silu(gate)
    b, dim = x.shape
    x = x.reshape(b, n_groups, dim // n_groups)
    var = np.mean(x * x, axis=-1, keepdims=True)
    x = x * (1.0 / np.sqrt(var + eps))
    wv = w.reshape(n_groups, dim // n_groups)
    x = x * wv
    return x.reshape(b, dim)

def mamba(x, l, w, cfg, state):
    """Single token. x: [1,H]. state: dict(conv, rec, d_mlp_x/z)."""
    H = cfg['hidden_size']
    NH = cfg['mamba_n_heads']; NG = cfg['mamba_n_groups']
    DS = cfg['mamba_d_state']; DC = cfg['mamba_d_conv']
    DI = cfg['mamba_d_ssm'] if cfg.get('mamba_d_ssm') else int(cfg['mamba_expand'] * H)
    HD = cfg['mamba_d_head']
    conv_dim = DI + 2 * NG * DS
    n_heads = NH
    pfx = f'model.layers.{l}.mamba.'
    # ssm_in_multiplier then in_proj then mup_vector
    x = x * cfg['ssm_in_multiplier']
    proj = x @ w[pfx + 'in_proj.weight'].T   # [1, DI+conv_dim+NH]
    # mup vector: [z, x, B, C, dt] sections
    gts = NG * DS
    vec = np.ones(proj.shape[1])
    sm = cfg['ssm_multipliers']
    vec[:DI] *= sm[0]
    vec[DI:2*DI] *= sm[1]
    vec[2*DI:2*DI+gts] *= sm[2]
    vec[2*DI+gts:2*DI+2*gts] *= sm[3]
    vec[2*DI+2*gts:] *= sm[4]
    proj = proj * vec
    # split: d_mlp, z0, x0, gate, xBC, dt (d_mlp=0 for this config)
    d_mlp = (proj.shape[1] - (2*DI + 2*gts + n_heads)) // 2
    split = [d_mlp, d_mlp, DI, conv_dim, n_heads]
    parts = np.split(proj[0], np.cumsum(split)[:-1])
    z0, x0, gate, xbc, dt = parts
    # causal depthwise conv (w[0] = OLDEST tap)
    cw = w[pfx + 'conv1d.weight']  # [conv_dim, 1, DC]
    cb = w[pfx + 'conv1d.bias']
    conv = np.zeros(conv_dim)
    for c in range(conv_dim):
        acc = cb[c]
        acc += cw[c, 0, DC-1] * xbc[c]
        for j in range(1, DC):
            acc += cw[c, 0, DC-1-j] * state['conv'][c, -(j)]
        conv[c] = silu(acc)
    if DC > 1:
        state['conv'] = np.roll(state['conv'], -1, axis=1)
        state['conv'][:, -1] = xbc
    h, B, C = np.split(conv, [DI, DI + gts])
    B = B.reshape(NG, DS); C = C.reshape(NG, DS)
    A = -np.exp(w[pfx + 'A_log'].astype(np.float64))
    dtb = w[pfx + 'dt_bias'].astype(np.float64)
    D = w[pfx + 'D'].astype(np.float64)
    h2 = h.reshape(NH, HD)
    y_inner = np.zeros((NH, HD))
    hpg = NH // NG
    for g in range(NG):
        for hh in range(hpg):
            hid = g * hpg + hh
            dts = np.log1p(np.exp(dt[hid] + dtb[hid]))  # softplus
            dA = np.exp(dts * A[hid])
            dB = dts * B[g]
            xh = h2[hid]
            ssm = state['rec'][hid]  # [HD, DS]
            ssm_new = dA * ssm + dB[None, :] * xh[:, None]
            state['rec'][hid] = ssm_new
            y_inner[hid] = ssm_new @ C[g] + D[hid] * xh
    y = y_inner.reshape(-1)
    if cfg.get('mamba_rms_norm'):
        y = gated_rmsnorm(y[None], gate[None], w[pfx + 'norm.weight'], cfg['rms_norm_eps'], 1)
        y = y[0]
    if d_mlp > 0:
        y = np.concatenate([silu(z0) * x0, y])
    return y @ w[pfx + 'out_proj.weight'].T

def attention(x, l, w, cfg, kvcache):
    """GQA attention: k × key_multiplier, rope, causal. x: [seq, H]."""
    H = cfg['hidden_size']; NH = cfg['num_attention_heads']; NKV = cfg['num_key_value_heads']
    HD = cfg['head_dim']
    pfx = f'model.layers.{l}.self_attn.'
    seq = x.shape[0]
    q = x @ w[pfx + 'q_proj.weight'].T
    k = (x @ w[pfx + 'k_proj.weight'].T) * cfg['key_multiplier']
    v = x @ w[pfx + 'v_proj.weight'].T
    q = q.reshape(seq, NH, HD).transpose(1, 0, 2)
    k = k.reshape(seq, NKV, HD).transpose(1, 0, 2)
    v = v.reshape(seq, NKV, HD).transpose(1, 0, 2)
    k = np.repeat(k, NH // NKV, axis=0); v = np.repeat(v, NH // NKV, axis=0)
    # rope (full head dim)
    theta = cfg['rope_theta']
    inv = 1.0 / (theta ** (np.arange(0, HD, 2, dtype=np.float64) / HD))
    for t in range(seq):
        ang = t * inv
        cos = np.cos(ang).astype(np.float32); sin = np.sin(ang).astype(np.float32)
        half = HD // 2
        for h in range(NH):
            rot = q[h, t]
            x1, x2 = rot[:half], rot[half:]
            q[h, t] = np.concatenate([x1*cos - x2*sin, x1*sin + x2*cos])
        for h in range(NKV):
            rot = k[h, t]
            x1, x2 = rot[:half], rot[half:]
            k[h, t] = np.concatenate([x1*cos - x2*sin, x1*sin + x2*cos])
    S = q @ k.transpose(0, 2, 1) * (HD ** -0.5)
    S = S + np.triu(np.ones((seq, seq)) * -1e9, k=1)[None]
    P = np.exp(S - S.max(-1, keepdims=True)); P /= P.sum(-1, keepdims=True)
    att = P @ v
    att = att.transpose(1, 0, 2).reshape(seq, NH * HD)
    return att @ w[pfx + 'o_proj.weight'].T

def mlp(x, l, w, cfg):
    pfx = f'model.layers.{l}.feed_forward.'
    gm, dm = cfg['mlp_multipliers']
    g = x @ w[pfx + 'gate_proj.weight'].T * gm
    u = x @ w[pfx + 'up_proj.weight'].T
    d = w[pfx + 'down_proj.weight'].T
    return (silu(g) * u) @ d * dm

def forward(cfg, w, ids, gen_len):
    H = cfg['hidden_size']; L = cfg['num_hidden_layers']
    EPS = cfg['rms_norm_eps']
    NH = cfg['mamba_n_heads']; NG = cfg['mamba_n_groups']
    DS = cfg['mamba_d_state']; DC = cfg['mamba_d_conv']
    DI = cfg['mamba_d_ssm'] if cfg.get('mamba_d_ssm') else int(cfg['mamba_expand'] * H)
    HD = cfg['mamba_d_head']
    conv_dim = DI + 2 * NG * DS
    conv_state = [np.zeros((conv_dim, DC - 1)) for _ in range(L)]
    rec_state = [np.zeros((NH, HD, DS)) for _ in range(L)]
    tok = ids.copy()
    gen = []
    for _ in range(gen_len):
        x = w['model.embed_tokens.weight'][tok].astype(np.float64) * cfg['embedding_multiplier']
        seq = len(tok)
        for l in range(L):
            pfx = f'model.layers.{l}.'
            xn = rmsnorm(x, w[pfx + 'input_layernorm.weight'], EPS)
            # mamba + attention from SAME normed input
            st = {'conv': conv_state[l], 'rec': rec_state[l]}
            m = np.zeros_like(x)
            for t in range(seq):
                m[t] = mamba(xn[t:t+1], l, w, cfg, st)
            conv_state[l] = st['conv']; rec_state[l] = st['rec']
            m = m * cfg['ssm_out_multiplier']
            a = attention(xn * cfg['attention_in_multiplier'], l, w, cfg, None) * cfg['attention_out_multiplier']
            x = x + m + a
            # MLP
            xn = rmsnorm(x, w[pfx + 'pre_ff_layernorm.weight'], EPS)
            x = x + mlp(xn, l, w, cfg)
        x = rmsnorm(x, w['model.final_layernorm.weight'], EPS)
        head = w['model.embed_tokens.weight']  # tied
        logits = x[-1] @ head.T * cfg['lm_head_multiplier']
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
