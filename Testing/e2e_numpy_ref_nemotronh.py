#!/usr/bin/env python3
"""Nemotron-H numpy reference (port of transformers modeling_nemotron_h.py 5.14).

Validates the engine's nemotron_h path on tiny-NemotronHForCausalLM-nano.
Run:
    python3 Testing/e2e_numpy_ref_nemotronh.py <model_dir> <ids.txt> <N>
Prints the N-token argmax chain (like the other numpy-ref oracles).
"""
import json, sys, os, glob
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

def relu2(x):
    r = np.maximum(x, 0.0)
    return r * r

def mamba2_mixer(x, l, w, cfg, state):
    """x: [1, H]. NemotronH Mamba2 mixer, single token (selective_state_update path).
    state: dict with conv_state [d_conv-1, conv_dim] and ssm_state [heads, hd, d_state]."""
    H = cfg['hidden_size']; NH = cfg['mamba_num_heads']; HD = cfg['mamba_head_dim']
    NG = cfg['n_groups']; DS = cfg['ssm_state_size']; DC = cfg['conv_kernel']
    DI = NH * HD  # intermediate
    conv_dim = DI + 2 * NG * DS
    pfx = f'backbone.layers.{l}.mixer.'
    # in_proj [projection_size, H] = [DI + conv_dim + NH, H]
    proj = x @ w[pfx + 'in_proj.weight'].T  # [1, DI+conv_dim+NH]
    z = proj[0, :DI]
    xbc = proj[0, DI:DI + conv_dim]
    dt_raw = proj[0, DI + conv_dim:]
    # conv1d depthwise, kernel DC, padding DC-1 (causal).
    # state['conv']: [conv_dim, DC-1] (per-channel history, most recent last).
    cw = w[pfx + 'conv1d.weight']  # [conv_dim, 1, DC]
    cb = w[pfx + 'conv1d.bias']    # [conv_dim]
    # Causal depthwise conv with kernel K, left-pad K-1 (torch conv1d
    # groups=conv_dim, padding=K-1, slice [:seq]). out[t] = sum_j w[K-1-j]*x[t-j]:
    # w[0] is the OLDEST tap, w[K-1] the current (empirically verified vs torch).
    xbc_conv = np.zeros(conv_dim)
    for c in range(conv_dim):
        acc = cb[c]
        acc += cw[c, 0, DC - 1] * xbc[c]               # current input (w[K-1])
        for j in range(1, DC):                         # history
            acc += cw[c, 0, DC - 1 - j] * state['conv'][c, -(j)]
        xbc_conv[c] = acc
    # update conv state: shift left, append current input
    if DC > 1:
        state['conv'] = np.roll(state['conv'], -1, axis=1)
        state['conv'][:, -1] = xbc
    xbc_act = silu(xbc_conv)
    x_inner = xbc_act[:DI]
    B = xbc_act[DI:DI + NG * DS].reshape(NG, DS)
    C = xbc_act[DI + NG * DS:].reshape(NG, DS)
    A = -np.exp(w[pfx + 'A_log'].astype(np.float64))  # [NH]
    dt_bias = w[pfx + 'dt_bias'].astype(np.float64)
    D = w[pfx + 'D'].astype(np.float64)
    # selective scan per head: hd elements share A, dt, B, C (group)
    y_inner = np.zeros(DI)
    hpg = NH // NG
    for g in range(NG):
        for h in range(hpg):
            hid = g * hpg + h
            dt = dt_raw[hid] + dt_bias[hid]
            dts = np.log1p(np.exp(dt))  # softplus
            dts = max(dts, cfg['time_step_min'])         # clamp(dt, time_step_min)
            if np.isfinite(cfg.get('time_step_max')) and dts > cfg['time_step_max']:
                dts = cfg['time_step_max']
            xh = x_inner[hid * HD:(hid + 1) * HD].astype(np.float64)
            ssm = state['ssm'][hid]  # [HD, DS]
            Bg = B[g]; Cg = C[g]
            dA = np.exp(dts * A[hid])                    # [scalar]
            dB = dts * Bg                                # [DS] (Euler: dB = dt*B)
            ssm_new = dA * ssm + dB[None, :] * xh[:, None]  # [HD, DS]
            state['ssm'][hid] = ssm_new
            y_inner[hid * HD:(hid + 1) * HD] = ssm_new @ Cg + D[hid] * xh
    # silu(z) gate then group RMSNorm (Zamba2RMSNormGated, norm_before_gate=False)
    y_gated = y_inner * silu(z)
    # Zamba2RMSNormGated: y*silu(gate), then GROUP RMSNorm (consecutive chunks
    # of group_size), then elementwise weight (self.weight * x).
    nw = w[pfx + 'norm.weight']  # [DI]
    gs = DI // NG
    y_out = np.zeros(DI)
    for g in range(NG):
        seg = y_gated[g * gs:(g + 1) * gs]
        rms = np.sqrt(np.mean(seg * seg) + 1e-6)
        y_out[g * gs:(g + 1) * gs] = (seg / rms) * nw[g * gs:(g + 1) * gs]
    out = y_out @ w[pfx + 'out_proj.weight'].T
    return out

def attention(xn, l, w, cfg, kvcache):
    """NoPE GQA attention. xn: [seq,H]. kvcache: (k [NKV,seq,HD], v [NKV,seq,HD])."""
    H = cfg['hidden_size']; NH = cfg['num_attention_heads']; NKV = cfg['num_key_value_heads']
    HD = cfg['head_dim']
    pfx = f'backbone.layers.{l}.mixer.'
    q = xn @ w[pfx + 'q_proj.weight'].T  # [seq, NH*HD]
    k = xn @ w[pfx + 'k_proj.weight'].T
    v = xn @ w[pfx + 'v_proj.weight'].T
    seq = q.shape[0]
    q = q.reshape(seq, NH, HD).transpose(1, 0, 2)
    k = k.reshape(seq, NKV, HD).transpose(1, 0, 2)
    v = v.reshape(seq, NKV, HD).transpose(1, 0, 2)
    k = np.repeat(k, NH // NKV, axis=0); v = np.repeat(v, NH // NKV, axis=0)
    S = q @ k.transpose(0, 2, 1) * (HD ** -0.5)
    S = S + np.triu(np.ones((seq, seq)) * -1e9, k=1)[None]
    P = np.exp(S - S.max(-1, keepdims=True)); P /= P.sum(-1, keepdims=True)
    out = P @ v
    return out.transpose(1, 0, 2).reshape(seq, NH * HD) @ w[pfx + 'o_proj.weight'].T

def mlp(xn, l, w, cfg):
    pfx = f'backbone.layers.{l}.mixer.'
    return (relu2(xn @ w[pfx + 'up_proj.weight'].T)) @ w[pfx + 'down_proj.weight'].T

def moe(xn, l, w, cfg):
    """Nemotron-H MoE: sigmoid router, up/down-only experts (no gate), shared experts."""
    H = cfg['hidden_size']; NE = cfg['n_routed_experts']; TOPK = cfg['num_experts_per_tok']
    NSH = cfg['n_shared_experts']; MIE = cfg['moe_intermediate_size']
    NG = cfg['n_group']; TLG = cfg['topk_group']; NTP = cfg['norm_topk_prob']
    RSC = cfg['routed_scaling_factor']
    pfx = f'backbone.layers.{l}.mixer.'
    seq_t = xn.shape[0]
    acc = np.zeros((seq_t, H))
    for t in range(seq_t):
        x1 = xn[t]
        logits = x1 @ w[pfx + 'gate.weight'].T
        scores = 1.0 / (1.0 + np.exp(-logits))
        cb = w[pfx + 'gate.e_score_correction_bias']
        sc = scores + cb
        per = NE // NG
        grp = sc.reshape(NG, per)
        gs = np.sort(grp, axis=-1)[:, -2:].sum(-1)
        gidx = np.argsort(gs)[-TLG:]
        mask = np.zeros(NE, bool)
        for g in gidx: mask[g * per:(g + 1) * per] = True
        scc = sc.copy(); scc[~mask] = -1e30
        topk = np.argsort(scc)[-TOPK:]
        wt = scores[topk]
        if NTP: wt = wt / (wt.sum() + 1e-20)
        wt = wt * RSC
        for e, wei in zip(topk, wt):
            u = x1 @ w[pfx + f'experts.{e}.up_proj.weight'].T
            d = w[pfx + f'experts.{e}.down_proj.weight'].T
            acc[t] += wei * (relu2(u) @ d)
    SIM = MIE * NSH
    u = xn @ w[pfx + 'shared_experts.up_proj.weight'].T
    d = w[pfx + 'shared_experts.down_proj.weight'].T
    acc += (relu2(u) @ d)
    return acc

def forward(cfg, w, ids, gen_len):
    H = cfg['hidden_size']
    L = len(cfg['layers_block_type'])
    EPS = cfg['layer_norm_epsilon']
    bt = cfg['layers_block_type']
    # states: conv_state [L, d_conv-1, conv_dim], ssm_state [L, NH, HD, DS]
    DI = cfg['mamba_num_heads'] * cfg['mamba_head_dim']
    conv_dim = DI + 2 * cfg['n_groups'] * cfg['ssm_state_size']
    conv_state = [np.zeros((conv_dim, cfg['conv_kernel'] - 1)) for _ in range(L)]
    ssm_state = [np.zeros((cfg['mamba_num_heads'], cfg['mamba_head_dim'], cfg['ssm_state_size'])) for _ in range(L)]
    tok = ids.copy()
    gen = []
    for _ in range(gen_len):
        x = w['backbone.embedding.weight'][tok].astype(np.float64)  # [seq, H]
        for l in range(L):
            pfx = f'backbone.layers.{l}.'
            xn = rmsnorm(x, w[pfx + 'norm.weight'], EPS)
            if bt[l] == 'mamba':
                out = np.zeros_like(x)
                st = {'conv': conv_state[l], 'ssm': ssm_state[l]}
                for t in range(x.shape[0]):
                    out[t] = mamba2_mixer(xn[t:t+1], l, w, cfg, st)
                    conv_state[l] = st['conv']; ssm_state[l] = st['ssm']
                x = x + out
            elif bt[l] == 'attention':
                o = attention(xn, l, w, cfg, None)
                x = x + o
            elif bt[l] == 'mlp':
                x = x + mlp(xn, l, w, cfg)
            elif bt[l] == 'moe':
                x = x + moe(xn, l, w, cfg)
            else:
                raise ValueError(bt[l])
        x = rmsnorm(x, w['backbone.norm_f.weight'], EPS)
        logits = x[-1] @ w['lm_head.weight'].T
        nxt = int(np.argmax(logits))
        gen.append(nxt)
        tok.append(nxt)
    return gen

if __name__ == '__main__':
    model_dir = sys.argv[1]
    ids = [int(t) for t in open(sys.argv[2]).read().split()]
    gen_len = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    model = json.load(open(os.path.join(model_dir, 'config.json')))
    w = {}
    for f in glob.glob(os.path.join(model_dir, '*.safetensors')):
        w.update(load_safetensors(f))
    gen = forward(model, w, ids, gen_len)
    print('ref-gen:', ' '.join(str(g) for g in gen))
