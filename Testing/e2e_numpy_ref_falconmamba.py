#!/usr/bin/env python3
"""FalconMamba numpy reference (port of transformers modeling_falcon_mamba.py 5.14).

Validates the engine's falcon_mamba path on the tiny random checkpoint.
Run: python3 Testing/e2e_numpy_ref_falconmamba.py <model_dir> <ids.txt> <N>
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

def softplus(x):
    return np.log1p(np.exp(np.clip(x, -30, 30)))

def mixer(hidden, l, w, cfg, state):
    """Single-token FalconMamba mixer. state: dict with conv [D, K-1], h [D, N]."""
    pfx = f'backbone.layers.{l}.mixer.'
    D = cfg['intermediate_size']; N = cfg['state_size']; K = cfg['conv_kernel']
    TR = cfg['time_step_rank']
    p = hidden @ w[pfx + 'in_proj.weight'].T
    hs, gate = p[:D], p[D:]
    # conv with state (w[0]=oldest)
    conv_w = w[pfx + 'conv1d.weight'][:, 0, :]  # [D, K]
    conv_b = w[pfx + 'conv1d.bias'] if pfx + 'conv1d.bias' in w else np.zeros(D)
    out = np.zeros(D)
    for d in range(D):
        acc = conv_w[d, 0] * state['conv'][d][0] + conv_w[d, 1] * state['conv'][d][1] + \
              conv_w[d, 2] * state['conv'][d][2] + conv_w[d, 3] * hs[d]
        out[d] = silu(acc + conv_b[d])
    for d in range(D):
        state['conv'][d][0] = state['conv'][d][1]
        state['conv'][d][1] = state['conv'][d][2]
        state['conv'][d][2] = hs[d]
    # SSM
    xp = out @ w[pfx + 'x_proj.weight'].T
    dt_p, B, C = xp[:TR], xp[TR:TR+N], xp[TR+N:]
    B = rmsnorm(B, np.ones(N), cfg['mixer_rms_eps'])
    C = rmsnorm(C, np.ones(N), cfg['mixer_rms_eps'])
    dt_p = rmsnorm(dt_p, np.ones(TR), cfg['mixer_rms_eps'])
    dt_raw = dt_p @ w[pfx + 'dt_proj.weight'].T + w[pfx + 'dt_proj.bias']
    dt = softplus(dt_raw)  # [D]
    A = -np.exp(w[pfx + 'A_log'])  # [D, N]
    dA = np.exp(A * dt[:, None])
    dB = dt[:, None] * B[None, :]
    dBx = dB * out[:, None]
    state['h'] = state['h'] * dA + dBx
    y = np.sum(state['h'] * C[None, :], axis=-1)
    y = y + out * w[pfx + 'D']
    y = y * silu(gate)
    return y @ w[pfx + 'out_proj.weight'].T

def block(hidden, l, w, cfg, state):
    pfx = f'backbone.layers.{l}.'
    x = rmsnorm(hidden, w[pfx + 'norm.weight'], cfg['layer_norm_epsilon'])
    x = mixer(x, l, w, cfg, state[l])
    return hidden + x  # residual (fp32)

def forward(cfg, w, ids, gen_len):
    H = cfg['hidden_size']; L = cfg['num_hidden_layers']
    D = cfg['intermediate_size']; K = cfg['conv_kernel']; N = cfg['state_size']
    states = [{'conv': [np.zeros(K - 1) for _ in range(D)], 'h': np.zeros((D, N))} for _ in range(L)]
    tok = ids.copy()
    gen = []
    for t in tok:
        x = w['backbone.embeddings.weight'][t]
        for l in range(L):
            x = block(x, l, w, cfg, states)
        x = rmsnorm(x, w['backbone.norm_f.weight'], cfg['layer_norm_epsilon'])
        logits = x @ w['lm_head.weight'].T
        nxt = int(np.argmax(logits))
    gen.append(nxt)
    tok.append(nxt)
    for _ in range(gen_len - 1):
        x = w['backbone.embeddings.weight'][tok[-1]]
        for l in range(L):
            x = block(x, l, w, cfg, states)
        x = rmsnorm(x, w['backbone.norm_f.weight'], cfg['layer_norm_epsilon'])
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
