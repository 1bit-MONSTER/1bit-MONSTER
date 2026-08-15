#!/usr/bin/env python3
"""GLM-4-MoE numpy reference (port of transformers modeling_glm4_moe.py 5.14).

Validates the engine's glm4moe path on tiny-Glm4MoeForCausalLM. Run:
    python3 Testing/e2e_numpy_ref_glm4moe.py <model_dir> <ids.txt> <N>
Prints the N-token argmax chain (like the other numpy-ref oracles) and
dumps final logits to GLM4MOE_REF_LOGITS if set.
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
        elif dt == 'F16':
            a = np.frombuffer(raw, dtype=np.float16).reshape(sh).astype(np.float32).copy()
        else:
            raise ValueError(dt)
        out[k] = a
    return out

def rmsnorm(x, w, eps):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * w

def silu(x):
    return x * (1.0 / (1.0 + np.exp(-x)))

def rope_cos_sin(pos, inv_freq):
    """HF Glm4MoeRotaryEmbedding.forward: freqs = inv_freq @ pos;
    emb = cat([freqs, freqs]) -> rotary_dim = 2*len(inv_freq)."""
    ang = (pos.astype(np.float64) * inv_freq)  # [seq, half]
    freqs = np.concatenate([ang, ang], axis=-1).astype(np.float32)  # [seq, rotary_dim]
    return np.cos(freqs), np.sin(freqs)

def rotate_half(x):
    """HF rotate_half: splits at hd//2 (hd = width of the ROTATED slice)."""
    hd = x.shape[-1]
    x1 = x[..., :hd // 2]
    x2 = x[..., hd // 2:]
    return np.concatenate([-x2, x1], axis=-1)

def forward(model, w, ids, gen_len):
    H = model['hidden_size']; L = model['num_hidden_layers']
    NH = model['num_attention_heads']; NKV = model['num_key_value_heads']
    HD = model['head_dim']; IM = model['intermediate_size']
    EPS = model['rms_norm_eps']
    THETA = model['rope_theta']; ROPE_DIM = int(model['partial_rotary_factor'] * HD)
    NE = model['n_routed_experts']; TOPK = model['num_experts_per_tok']
    NSH = model['n_shared_experts']; FKD = model['first_k_dense_replace']
    MIE = model['moe_intermediate_size']
    NG = model['n_group']; TLG = model['topk_group']
    NTP = model['norm_topk_prob']; RSC = model['routed_scaling_factor']

    inv_freq = 1.0 / (THETA ** (np.arange(0, ROPE_DIM, 2, dtype=np.float64) / ROPE_DIM))

    def attn(xn, l, kcache, vcache, pos_all):
        """xn: [seq, H] (pre-normed). Full recompute; kcache = [NKV, seq, HD]."""
        pfx = f'model.layers.{l}.self_attn.'
        q = xn @ w[pfx + 'q_proj.weight'].T + w[pfx + 'q_proj.bias']
        k = xn @ w[pfx + 'k_proj.weight'].T + w[pfx + 'k_proj.bias']
        v = xn @ w[pfx + 'v_proj.weight'].T + w[pfx + 'v_proj.bias']
        q = q.reshape(-1, NH, HD)
        k = k.reshape(-1, NKV, HD)
        v = v.reshape(-1, NKV, HD)
        if model.get('use_qk_norm'):
            qn = w[pfx + 'q_norm.weight']; kn = w[pfx + 'k_norm.weight']
            q = rmsnorm(q, qn, EPS)
            k = rmsnorm(k, kn, EPS)
        # HF apply_rotary_pos_emb: rotary_dim = cos.shape[-1]; rotate first
        # rotary_dim of q/k, pass the rest through unchanged.
        cos, sin = rope_cos_sin(pos_all, inv_freq)  # [seq, rotary_dim]
        rdim = cos.shape[-1]
        qr = np.zeros_like(q); kr = np.zeros_like(k)
        for t in range(q.shape[0]):
            qr[t, :, :rdim] = q[t, :, :rdim] * cos[t] + rotate_half(q[t, :, :rdim]) * sin[t]
            qr[t, :, rdim:] = q[t, :, rdim:]
            kr[t, :, :rdim] = k[t, :, :rdim] * cos[t] + rotate_half(k[t, :, :rdim]) * sin[t]
            kr[t, :, rdim:] = k[t, :, rdim:]
        # causal scores with [head, seq, dim] layout (GQA: repeat KV heads)
        qt = qr.transpose(1, 0, 2)  # [NH, seq, HD]
        kt = kr.transpose(1, 0, 2)  # [NKV, seq, HD]
        kt = np.repeat(kt, NH // NKV, axis=0)  # -> [NH, seq, HD]
        S = qt @ kt.transpose(0, 2, 1)  # [NH, seq, seq]
        S *= HD ** -0.5
        seq = S.shape[1]
        mask = np.triu(np.ones((seq, seq), np.float32) * -1e9, k=1)
        S = S + mask[None]
        P = np.exp(S - S.max(-1, keepdims=True))
        P /= P.sum(-1, keepdims=True)
        vt = v.transpose(1, 0, 2)  # [NKV, seq, HD]
        vt = np.repeat(vt, NH // NKV, axis=0)  # -> [NH, seq, HD]
        out = P @ vt  # [NH, seq, HD]
        o = out.transpose(1, 0, 2).reshape(seq, NH * HD) @ w[pfx + 'o_proj.weight'].T
        return o

    def dense_mlp(xn, l):
        pfx = f'model.layers.{l}.mlp.'
        g = xn @ w[pfx + 'gate_proj.weight'].T
        u = xn @ w[pfx + 'up_proj.weight'].T
        d = w[pfx + 'down_proj.weight'].T
        return silu(g) * u @ d

    def moe_mlp(xn, l):
        """xn: [seq, H]; MoE FFN is per-token. Returns [seq, H]."""
        pfx = f'model.layers.{l}.mlp.'
        seq_t = xn.shape[0]
        acc = np.zeros((seq_t, H), np.float64)
        for t in range(seq_t):
            x1 = xn[t]
            logits = x1 @ w[pfx + 'gate.weight'].T  # [NE]
            scores = 1.0 / (1.0 + np.exp(-logits))   # sigmoid
            cb = w[pfx + 'gate.e_score_correction_bias']
            scores_choice = scores + cb
            # group-limited top-k
            per = NE // NG
            grp = scores_choice.reshape(NG, per)
            gs = np.sort(grp, axis=-1)[:, -2:].sum(-1)          # topk(2) per group, summed
            gidx = np.argsort(gs)[-TLG:]                        # topk_group groups
            mask = np.zeros(NE, np.bool_)
            for g in gidx:
                mask[g * per:(g + 1) * per] = True
            sc = scores_choice.copy()
            sc[~mask] = -1e30
            topk = np.argsort(sc)[-TOPK:]
            wt = scores[topk]
            if NTP:
                wt = wt / (wt.sum() + 1e-20)
            wt = wt * RSC
            for e, wei in zip(topk, wt):
                g = x1 @ w[pfx + f'experts.{e}.gate_proj.weight'].T
                u = x1 @ w[pfx + f'experts.{e}.up_proj.weight'].T
                d = w[pfx + f'experts.{e}.down_proj.weight'].T
                acc[t] += wei * (silu(g) * u) @ d
        # shared experts (fused, SIM = MIE * NSH) — same for all tokens
        SIM = MIE * NSH
        g = xn @ w[pfx + 'shared_experts.gate_proj.weight'].T  # [seq, SIM]
        u = xn @ w[pfx + 'shared_experts.up_proj.weight'].T
        d = w[pfx + 'shared_experts.down_proj.weight'].T
        acc += (silu(g) * u) @ d
        return acc

    tok = ids.copy()
    gen = []
    for _ in range(gen_len):
        x = w['model.embed_tokens.weight'][tok].astype(np.float64)  # [seq, H]
        pos_all = np.arange(len(tok), dtype=np.float64)
        kcache = [None] * L; vcache = [None] * L
        for l in range(L):
            pfx = f'model.layers.{l}.'
            xn = rmsnorm(x, w[pfx + 'input_layernorm.weight'], EPS)
            o = attn(xn, l, kcache, vcache, pos_all)
            x = x + o
            xn = rmsnorm(x, w[pfx + 'post_attention_layernorm.weight'], EPS)
            if l < FKD:
                f = dense_mlp(xn, l)
            else:
                f = moe_mlp(xn, l)
            x = x + f
        x = rmsnorm(x, w['model.norm.weight'], EPS)
        logits = x[-1] @ w['lm_head.weight'].T
        nxt = int(np.argmax(logits))
        gen.append(nxt)
        tok.append(nxt)
    return gen


# ── module-level helpers for the comparison harness (same math as forward) ──
def attn(xn, l, kcache, vcache, pos_all, w, model):
    H = model['hidden_size']; NH = model['num_attention_heads']; NKV = model['num_key_value_heads']
    HD = model['head_dim']; EPS = model['rms_norm_eps']
    THETA = model['rope_theta']; RD = int(model['partial_rotary_factor'] * HD)
    inv_freq = 1.0 / (THETA ** (np.arange(0, RD, 2, dtype=np.float64) / RD))
    pfx = f'model.layers.{l}.self_attn.'
    q = xn @ w[pfx + 'q_proj.weight'].T + w[pfx + 'q_proj.bias']
    k = xn @ w[pfx + 'k_proj.weight'].T + w[pfx + 'k_proj.bias']
    v = xn @ w[pfx + 'v_proj.weight'].T + w[pfx + 'v_proj.bias']
    q = q.reshape(-1, NH, HD); k = k.reshape(-1, NKV, HD); v = v.reshape(-1, NKV, HD)
    if model.get('use_qk_norm'):
        q = rmsnorm(q, w[pfx + 'q_norm.weight'], EPS)
        k = rmsnorm(k, w[pfx + 'k_norm.weight'], EPS)
    cos, sin = rope_cos_sin(pos_all, inv_freq)
    rdim = cos.shape[-1]
    qr = np.zeros_like(q); kr = np.zeros_like(k)
    for t in range(q.shape[0]):
        qr[t, :, :rdim] = q[t, :, :rdim] * cos[t] + rotate_half(q[t, :, :rdim]) * sin[t]
        qr[t, :, rdim:] = q[t, :, rdim:]
        kr[t, :, :rdim] = k[t, :, :rdim] * cos[t] + rotate_half(k[t, :, :rdim]) * sin[t]
        kr[t, :, rdim:] = k[t, :, rdim:]
    qt = qr.transpose(1, 0, 2); kt = kr.transpose(1, 0, 2)
    kt = np.repeat(kt, NH // NKV, axis=0)
    S = qt @ kt.transpose(0, 2, 1) * (HD ** -0.5)
    seq = S.shape[1]
    S = S + np.triu(np.ones((seq, seq), np.float32) * -1e9, k=1)[None]
    P = np.exp(S - S.max(-1, keepdims=True)); P /= P.sum(-1, keepdims=True)
    vt = np.repeat(v.transpose(1, 0, 2), NH // NKV, axis=0)
    out = P @ vt
    return out.transpose(1, 0, 2).reshape(seq, NH * HD) @ w[pfx + 'o_proj.weight'].T

def dense_mlp(xn, l, w, model):
    pfx = f'model.layers.{l}.mlp.'
    g = xn @ w[pfx + 'gate_proj.weight'].T
    u = xn @ w[pfx + 'up_proj.weight'].T
    d = w[pfx + 'down_proj.weight'].T
    return (silu(g) * u) @ d

def moe_mlp(xn, l, w, model):
    H = model['hidden_size']; NE = model['n_routed_experts']; TOPK = model['num_experts_per_tok']
    NSH = model['n_shared_experts']; MIE = model['moe_intermediate_size']
    NG = model['n_group']; TLG = model['topk_group']; NTP = model['norm_topk_prob']
    RSC = model['routed_scaling_factor']
    pfx = f'model.layers.{l}.mlp.'
    seq_t = xn.shape[0]
    acc = np.zeros((seq_t, H), np.float64)
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
        mask = np.zeros(NE, np.bool_)
        for g in gidx: mask[g * per:(g + 1) * per] = True
        scc = sc.copy(); scc[~mask] = -1e30
        topk = np.argsort(scc)[-TOPK:]
        wt = scores[topk]
        if NTP: wt = wt / (wt.sum() + 1e-20)
        wt = wt * RSC
        for e, wei in zip(topk, wt):
            g = x1 @ w[pfx + f'experts.{e}.gate_proj.weight'].T
            u = x1 @ w[pfx + f'experts.{e}.up_proj.weight'].T
            d = w[pfx + f'experts.{e}.down_proj.weight'].T
            acc[t] += wei * (silu(g) * u) @ d
    SIM = MIE * NSH
    g = xn @ w[pfx + 'shared_experts.gate_proj.weight'].T
    u = xn @ w[pfx + 'shared_experts.up_proj.weight'].T
    d = w[pfx + 'shared_experts.down_proj.weight'].T
    acc += (silu(g) * u) @ d
    return acc

if __name__ == '__main__':
    model_dir = sys.argv[1]
    ids = [int(t) for t in open(sys.argv[2]).read().split()]
    gen_len = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    model = json.load(open(os.path.join(model_dir, 'config.json')))
    keys = {'hidden_size','num_hidden_layers','num_attention_heads','num_key_value_heads',
            'head_dim','intermediate_size','rms_norm_eps','partial_rotary_factor',
            'n_routed_experts','num_experts_per_tok','n_shared_experts','first_k_dense_replace',
            'moe_intermediate_size','n_group','topk_group','norm_topk_prob','routed_scaling_factor'}
    m2 = {k: model[k] for k in keys}
    m2['rope_theta'] = model.get('rope_theta', 1000000.0)
    m2['use_qk_norm'] = model.get('use_qk_norm', False)
    import glob
    st = glob.glob(os.path.join(model_dir, '*.safetensors'))
    w = {}
    for f in st:
        w.update(load_safetensors(f))
    gen = forward(m2, w, ids, gen_len)
    print('ref-gen:', ' '.join(str(g) for g in gen))
