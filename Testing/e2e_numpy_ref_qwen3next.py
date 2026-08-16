#!/usr/bin/env python3
"""Qwen3-Next numpy reference (port of transformers modeling_qwen3_next.py 5.14).

Validates the engine's qwen3_next path on tiny-random/qwen3-next-moe.
Run:
    python3 Testing/e2e_numpy_ref_qwen3next.py <model_dir> <ids.txt> <N>
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
    # Qwen3NextRMSNorm: norm(x) * (1 + weight) — delta convention
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * (1.0 + w)

def silu(x):
    return x * (1.0 / (1.0 + np.exp(-x)))

def l2norm(x, eps=1e-6):
    return x / (np.sqrt(np.sum(x * x, axis=-1, keepdims=True)) + eps)

def gated_delta_net(x, l, w, cfg, state):
    """Single token. x: [1,H]. state: dict(conv, rec)."""
    H = cfg['hidden_size']
    NK = cfg['linear_num_key_heads']; NV = cfg['linear_num_value_heads']
    KHD = cfg['linear_key_head_dim']; VHD = cfg['linear_value_head_dim']
    KD = KHD * NK; VD = VHD * NV
    conv_dim = KD * 2 + VD
    K = cfg['linear_conv_kernel_dim']
    pfx = f'model.layers.{l}.linear_attn.'
    qkvz = x @ w[pfx + 'in_proj_qkvz.weight'].T   # [1, KD*2+VD*2]
    ba = x @ w[pfx + 'in_proj_ba.weight'].T       # [1, NV*2]
    mq = qkvz.reshape(1, NK, 2 * KHD + 2 * (NV // NK) * VHD)
    mb = ba.reshape(1, NK, 2 * (NV // NK))
    query = mq[0, :, :KHD]
    key = mq[0, :, KHD:2*KHD]
    value = mq[0, :, 2*KHD:2*KHD + (NV//NK)*VHD]
    z = mq[0, :, 2*KHD+(NV//NK)*VHD:]
    b = mb[0, :, :(NV//NK)]
    a = mb[0, :, (NV//NK):]
    query = query.reshape(-1)   # [KD]
    key = key.reshape(-1)
    value = value.reshape(NV, VHD).reshape(-1)  # [VD] row-major
    z = z.reshape(NV, VHD)
    b = b.reshape(-1); a = a.reshape(-1)
    mixed = np.concatenate([query, key, value])
    # causal depthwise conv: out = w[K-1]*x[t] + ... + w[0]*x[t-(K-1)]
    # state['conv']: [conv_dim, K-1], index 0 = OLDEST
    cw = w[pfx + 'conv1d.weight']
    conv_out = np.zeros(conv_dim)
    for c in range(conv_dim):
        acc = cw[c, 0, K - 1] * mixed[c]
        for j in range(K - 1):
            acc += cw[c, 0, j] * state['conv'][c, j]
        conv_out[c] = silu(acc)
    if K > 1:
        state['conv'] = np.roll(state['conv'], -1, axis=1)
        state['conv'][:, -1] = mixed
    qq = conv_out[:KD].reshape(NK, KHD)
    kk = conv_out[KD:2*KD].reshape(NK, KHD)
    vv = conv_out[2*KD:].reshape(NV, VHD)
    rep = NV // NK
    qq = np.repeat(qq, rep, axis=0)
    kk = np.repeat(kk, rep, axis=0)
    qq = l2norm(qq) * (1.0 / np.sqrt(KHD))
    kk = l2norm(kk)
    beta = 1.0 / (1.0 + np.exp(-b))
    g = -np.exp(w[pfx + 'A_log'].astype(np.float64)) * np.log1p(np.exp(a.astype(np.float64) + w[pfx + 'dt_bias'].astype(np.float64)))
    # recurrent delta rule
    out = np.zeros((NV, VHD))
    for h in range(NV):
        st = state['rec'][h]  # [KHD, VHD]
        st = st * np.exp(g[h])
        kv_mem = np.sum(st * kk[h][:, None], axis=0)
        delta = (vv[h] - kv_mem) * beta[h]
        st = st + kk[h][:, None] * delta[None, :]
        state['rec'][h] = st
        out[h] = np.sum(st * qq[h][:, None], axis=0)
    # Qwen3NextRMSNormGated: DIRECT weight, norm before gate, then silu(z)
    nw = w[pfx + 'norm.weight']
    normed = np.zeros_like(out)
    for h in range(NV):
        seg = out[h]
        r = 1.0 / np.sqrt(np.mean(seg * seg) + cfg['rms_norm_eps'])
        normed[h] = seg * r * nw
    gated = normed * silu(z)
    flat = gated.reshape(-1)
    return flat @ w[pfx + 'out_proj.weight'].T

def full_attention(x, l, w, cfg, kvcache):
    """Qwen3-Next full attention: q_proj [2*hd] split q+gate, q/k RMSNorm
    (1+w), partial rope, attn * sigmoid(gate). x: [seq, H]."""
    H = cfg['hidden_size']; NH = cfg['num_attention_heads']; NKV = cfg['num_key_value_heads']
    HD = cfg['head_dim']
    pfx = f'model.layers.{l}.self_attn.'
    seq = x.shape[0]
    qg = x @ w[pfx + 'q_proj.weight'].T
    qg = qg.reshape(seq, NH, HD * 2)
    query = qg[:, :, :HD]
    gate = qg[:, :, HD:]
    k = (x @ w[pfx + 'k_proj.weight'].T).reshape(seq, NKV, HD)
    v = (x @ w[pfx + 'v_proj.weight'].T).reshape(seq, NKV, HD)
    qn = w[pfx + 'q_norm.weight']; kn = w[pfx + 'k_norm.weight']
    for t in range(seq):
        for h in range(NH):
            r = 1.0 / np.sqrt(np.mean(query[t, h]**2) + cfg['rms_norm_eps'])
            query[t, h] = query[t, h] * r * (1.0 + qn)
        for h in range(NKV):
            r = 1.0 / np.sqrt(np.mean(k[t, h]**2) + cfg['rms_norm_eps'])
            k[t, h] = k[t, h] * r * (1.0 + kn)
    qt = query.transpose(1, 0, 2)
    kt = k.transpose(1, 0, 2)
    vt = v.transpose(1, 0, 2)
    kt = np.repeat(kt, NH // NKV, axis=0); vt = np.repeat(vt, NH // NKV, axis=0)
    # partial rope
    dim = int(HD * cfg['partial_rotary_factor'])
    theta = cfg['rope_theta']
    inv = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.float64) / dim))
    for t in range(seq):
        ang = t * inv
        cos = np.cos(ang).astype(np.float32); sin = np.sin(ang).astype(np.float32)
        half = dim // 2
        for h in range(NH):
            rot = qt[h, t, :dim]
            x1, x2 = rot[:half], rot[half:dim]
            qt[h, t, :dim] = np.concatenate([x1*cos - x2*sin, x1*sin + x2*cos])
        for h in range(NKV):
            rot = kt[h, t, :dim]
            x1, x2 = rot[:half], rot[half:dim]
            kt[h, t, :dim] = np.concatenate([x1*cos - x2*sin, x1*sin + x2*cos])
    S = qt @ kt.transpose(0, 2, 1) * (HD ** -0.5)
    S = S + np.triu(np.ones((seq, seq)) * -1e9, k=1)[None]
    P = np.exp(S - S.max(-1, keepdims=True)); P /= P.sum(-1, keepdims=True)
    att = P @ vt
    att = att.transpose(1, 0, 2)
    gated = att * (1.0 / (1.0 + np.exp(-gate)))
    flat = gated.reshape(seq, NH * HD)
    return flat @ w[pfx + 'o_proj.weight'].T

def mlp(x, l, w, cfg):
    pfx = f'model.layers.{l}.mlp.'
    g = x @ w[pfx + 'gate_proj.weight'].T
    u = x @ w[pfx + 'up_proj.weight'].T
    d = w[pfx + 'down_proj.weight'].T
    return silu(g) * u @ d

def moe(x, l, w, cfg):
    """Qwen3-Next MoE: softmax router top-k + norm_topk, gated experts,
    shared expert + shared_expert_gate (sigmoid scalar). x: [seq, H]."""
    H = cfg['hidden_size']; NE = cfg['num_experts']; TOPK = cfg['num_experts_per_tok']
    MIE = cfg['moe_intermediate_size']; SIE = cfg['shared_expert_intermediate_size']
    pfx = f'model.layers.{l}.mlp.'
    seq_t = x.shape[0]
    acc = np.zeros((seq_t, H))
    for t in range(seq_t):
        x1 = x[t]
        logits = x1 @ w[pfx + 'gate.weight'].T
        probs = np.exp(logits - logits.max()); probs /= probs.sum()
        topk = np.argsort(probs)[-TOPK:]
        wts = probs[topk]
        if cfg.get('norm_topk_prob'):
            wts /= wts.sum()
        for e, wei in zip(topk, wts):
            g = x1 @ w[pfx + f'experts.{e}.gate_proj.weight'].T
            u = x1 @ w[pfx + f'experts.{e}.up_proj.weight'].T
            d = w[pfx + f'experts.{e}.down_proj.weight'].T
            acc[t] += wei * (silu(g) * u) @ d
        g = x1 @ w[pfx + 'shared_expert.gate_proj.weight'].T
        u = x1 @ w[pfx + 'shared_expert.up_proj.weight'].T
        d = w[pfx + 'shared_expert.down_proj.weight'].T
        sg = 1.0 / (1.0 + np.exp(-(x1 @ w[pfx + 'shared_expert_gate.weight'].T)[0]))
        acc[t] += sg * (silu(g) * u) @ d
    return acc

def forward(cfg, w, ids, gen_len):
    H = cfg['hidden_size']; L = cfg['num_hidden_layers']
    EPS = cfg['rms_norm_eps']
    lt = cfg['layer_types']
    NK = cfg['linear_num_key_heads']; NV = cfg['linear_num_value_heads']
    KHD = cfg['linear_key_head_dim']; VHD = cfg['linear_value_head_dim']
    KD = KHD * NK; VD = VHD * NV
    conv_dim = KD * 2 + VD
    K = cfg['linear_conv_kernel_dim']
    conv_state = [np.zeros((conv_dim, K - 1)) for _ in range(L)]
    rec_state = [np.zeros((NV, KHD, VHD)) for _ in range(L)]
    tok = ids.copy()
    gen = []
    for _ in range(gen_len):
        x = w['model.embed_tokens.weight'][tok].astype(np.float64)
        seq = len(tok)
        for l in range(L):
            pfx = f'model.layers.{l}.'
            xn = rmsnorm(x, w[pfx + 'input_layernorm.weight'], EPS)
            if lt[l] == 'linear_attention':
                out = np.zeros_like(x)
                st = {'conv': conv_state[l], 'rec': rec_state[l]}
                for t in range(seq):
                    out[t] = gated_delta_net(xn[t:t+1], l, w, cfg, st)
                conv_state[l] = st['conv']; rec_state[l] = st['rec']
                x = x + out
            elif lt[l] == 'full_attention':
                x = x + full_attention(xn, l, w, cfg, None)
            xn = rmsnorm(x, w[pfx + 'post_attention_layernorm.weight'], EPS)
            is_moe = (cfg['num_experts'] > 0 and (l + 1) % cfg['decoder_sparse_step'] == 0 and l not in cfg['mlp_only_layers'])
            f = moe(xn, l, w, cfg) if is_moe else mlp(xn, l, w, cfg)
            x = x + f
        x = rmsnorm(x, w['model.norm.weight'], EPS)
        logits = x[-1] @ w['lm_head.weight'].T
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
    for f in glob.glob(os.path.join(model_dir, '*.safetensors')):
        w.update(load_safetensors(f))
    gen = forward(cfg, w, ids, gen_len)
    print('ref-gen:', ' '.join(str(g) for g in gen))
