#!/usr/bin/env python3
"""MiniMax-M2 numpy reference (port of transformers modeling_minimax_m2.py 5.14).

Validates the engine's minimax_m2 path on tiny-random/minimax-m2.
Run:
    python3 Testing/e2e_numpy_ref_minimaxm2.py <model_dir> <ids.txt> <N>
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
    # MiniMaxM2RMSNorm: DIRECT weight (T5-style), no +1 delta
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * w

def silu(x):
    return x * (1.0 / (1.0 + np.exp(-x)))

def attention(x, l, w, cfg, kvcache):
    """GQA attention: q/k single RMSNorm over FLATTENED dims, partial rope.
    x: [seq, H]. kvcache: (k [NKV, seq, HD], v)."""
    H = cfg['hidden_size']; NH = cfg['num_attention_heads']; NKV = cfg['num_key_value_heads']
    HD = cfg['head_dim']
    pfx = f'model.layers.{l}.self_attn.'
    seq = x.shape[0]
    q = x @ w[pfx + 'q_proj.weight'].T      # [seq, NH*HD]
    k = x @ w[pfx + 'k_proj.weight'].T      # [seq, NKV*HD]
    v = x @ w[pfx + 'v_proj.weight'].T
    # single RMSNorm over flattened dims
    qn = w[pfx + 'q_norm.weight']           # [NH*HD]
    kn = w[pfx + 'k_norm.weight']           # [NKV*HD]
    q = rmsnorm(q, qn, cfg['rms_norm_eps'])
    k = rmsnorm(k, kn, cfg['rms_norm_eps'])
    q = q.reshape(seq, NH, HD).transpose(1, 0, 2)
    k = k.reshape(seq, NKV, HD).transpose(1, 0, 2)
    v = v.reshape(seq, NKV, HD).transpose(1, 0, 2)
    k = np.repeat(k, NH // NKV, axis=0); v = np.repeat(v, NH // NKV, axis=0)
    # partial rope (rotary_dim from config, theta)
    rdim = cfg.get('rotary_dim', int(HD * cfg.get('partial_rotary_factor', 1.0)))
    theta = cfg['rope_theta']
    inv = 1.0 / (theta ** (np.arange(0, rdim, 2, dtype=np.float64) / rdim))
    for t in range(seq):
        ang = t * inv
        cos = np.cos(ang).astype(np.float32); sin = np.sin(ang).astype(np.float32)
        half = rdim // 2
        for h in range(NH):
            rot = q[h, t, :rdim]
            x1, x2 = rot[:half], rot[half:rdim]
            q[h, t, :rdim] = np.concatenate([x1*cos - x2*sin, x1*sin + x2*cos])
        for h in range(NKV):
            rot = k[h, t, :rdim]
            x1, x2 = rot[:half], rot[half:rdim]
            k[h, t, :rdim] = np.concatenate([x1*cos - x2*sin, x1*sin + x2*cos])
    S = q @ k.transpose(0, 2, 1) * (HD ** -0.5)
    S = S + np.triu(np.ones((seq, seq)) * -1e9, k=1)[None]
    P = np.exp(S - S.max(-1, keepdims=True)); P /= P.sum(-1, keepdims=True)
    att = P @ v  # [NH, seq, HD]
    att = att.transpose(1, 0, 2).reshape(seq, NH * HD)
    return att @ w[pfx + 'o_proj.weight'].T

def moe(x, l, w, cfg):
    """Sigmoid router + correction bias, top-k on sigmoid+correction, weights
    from RAW sigmoid normalized (÷sum), gated w1/w2/w3 experts. x: [seq, H]."""
    H = cfg['hidden_size']; NE = cfg['num_local_experts']; TOPK = cfg['num_experts_per_tok']
    MIE = cfg['intermediate_size']
    pfx = f'model.layers.{l}.block_sparse_moe.'
    seq_t = x.shape[0]
    acc = np.zeros((seq_t, H))
    for t in range(seq_t):
        x1 = x[t]
        logits = x1 @ w[pfx + 'gate.weight'].T
        scores = 1.0 / (1.0 + np.exp(-logits))               # sigmoid
        cb = w[pfx + 'e_score_correction_bias']
        scores_choice = scores + cb
        topk = np.argsort(scores_choice)[-TOPK:]
        wts = scores[topk]
        wts /= wts.sum()                                     # normalize
        for e, wei in zip(topk, wts):
            g = x1 @ w[pfx + f'experts.{e}.w1.weight'].T
            u = x1 @ w[pfx + f'experts.{e}.w3.weight'].T
            d = w[pfx + f'experts.{e}.w2.weight'].T
            acc[t] += wei * (silu(g) * u) @ d
    return acc

def forward(cfg, w, ids, gen_len):
    H = cfg['hidden_size']; L = cfg['num_hidden_layers']
    EPS = cfg['rms_norm_eps']
    tok = ids.copy()
    gen = []
    for _ in range(gen_len):
        x = w['model.embed_tokens.weight'][tok].astype(np.float64)
        seq = len(tok)
        for l in range(L):
            pfx = f'model.layers.{l}.'
            xn = rmsnorm(x, w[pfx + 'input_layernorm.weight'], EPS)
            x = x + attention(xn, l, w, cfg, None)
            xn = rmsnorm(x, w[pfx + 'post_attention_layernorm.weight'], EPS)
            x = x + moe(xn, l, w, cfg)
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
