#!/usr/bin/env python3
"""RWKV-4 numpy reference (port of transformers modeling_rwkv.py 5.14).

Validates the engine's rwkv path on the tiny random checkpoint. Run:
    python3 Testing/e2e_numpy_ref_rwkv.py <model_dir> <ids.txt> <N>
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

def layernorm(x, w, b, eps):
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.mean((x - mean)**2, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * w + b

def wkv(time_decay, time_first, key, value, num, den, mx):
    """Single token WKV update. All [D] vectors. Returns output, new state."""
    time_decay = -np.exp(time_decay)
    max_for_output = np.maximum(mx, key + time_first)
    e1 = np.exp(mx - max_for_output)
    e2 = np.exp(key + time_first - max_for_output)
    num_out = e1 * num + e2 * value
    den_out = e1 * den + e2
    output = num_out / den_out
    # state update uses the PRE-update num/den (torch keeps num_state separate)
    max_for_state = np.maximum(mx + time_decay, key)
    e1 = np.exp(mx + time_decay - max_for_state)
    e2 = np.exp(key - max_for_state)
    num = e1 * num + e2 * value
    den = e1 * den + e2
    mx = max_for_state
    return output, num, den, mx

def attention(hidden, l, w, state):
    """RwkvSelfAttention single token. state: dict with shift, num, den, mx."""
    H = hidden.shape[0]
    pfx = f'rwkv.blocks.{l}.attention.'
    # time mixing with shifted
    shifted = state['shift']  # [H]
    k = hidden * w[pfx + 'time_mix_key'][0,0] + shifted * (1 - w[pfx + 'time_mix_key'][0,0])
    v = hidden * w[pfx + 'time_mix_value'][0,0] + shifted * (1 - w[pfx + 'time_mix_value'][0,0])
    r = hidden * w[pfx + 'time_mix_receptance'][0,0] + shifted * (1 - w[pfx + 'time_mix_receptance'][0,0])
    state['shift'] = hidden
    # projections
    key = k @ w[pfx + 'key.weight'].T
    value = v @ w[pfx + 'value.weight'].T
    receptance = 1.0 / (1.0 + np.exp(-(r @ w[pfx + 'receptance.weight'].T)))
    # WKV
    out, num, den, mx = wkv(w[pfx + 'time_decay'], w[pfx + 'time_first'],
                             key, value, state['num'], state['den'], state['mx'])
    state['num'], state['den'], state['mx'] = num, den, mx
    return receptance * out @ w[pfx + 'output.weight'].T

def feed_forward(hidden, l, w, state):
    """RwkvFeedForward single token. state: dict with shift2."""
    H = hidden.shape[0]
    pfx = f'rwkv.blocks.{l}.feed_forward.'
    shifted = state['shift2']
    k = hidden * w[pfx + 'time_mix_key'][0,0] + shifted * (1 - w[pfx + 'time_mix_key'][0,0])
    r = hidden * w[pfx + 'time_mix_receptance'][0,0] + shifted * (1 - w[pfx + 'time_mix_receptance'][0,0])
    state['shift2'] = hidden
    key = np.square(np.maximum(k @ w[pfx + 'key.weight'].T, 0))
    value = key @ w[pfx + 'value.weight'].T
    receptance = 1.0 / (1.0 + np.exp(-(r @ w[pfx + 'receptance.weight'].T)))
    return receptance * value

def forward(cfg, w, ids, gen_len):
    H = cfg['hidden_size']; L = cfg['num_hidden_layers']
    EPS = cfg['layer_norm_epsilon']
    # inference weight rescale (transformers _rescale_layers): divide the
    # block output projections by 2^int(l/rescale_every) — applied at load
    # time, NOT as a hidden-state divide.
    rw = {}
    for l in range(L):
        k = 2 ** int(l // cfg.get('rescale_every', 6)) if cfg.get('rescale_every', 6) > 0 else 1
        for pfx, suffix in [('attention', 'output.weight'), ('feed_forward', 'value.weight')]:
            key = f'rwkv.blocks.{l}.{pfx}.{suffix}'
            rw[key] = w[key] / k
    w = {**w, **rw}
    # state per layer: shift, shift2, num, den, mx
    states = [{'shift': np.zeros(H), 'shift2': np.zeros(H),
               'num': np.zeros(cfg['attention_hidden_size']),
               'den': np.zeros(cfg['attention_hidden_size']),
               'mx': np.full(cfg['attention_hidden_size'], -1e30)} for _ in range(L)]
    tok = ids.copy()
    gen = []
    # feed the prompt first, then generate (like the engine: argmax is
    # computed DURING the last prompt token's step and fed back)
    for t in tok:
        x = w['rwkv.embeddings.weight'][t].astype(np.float64)
        for l in range(L):
            st = states[l]
            if l == 0:
                x = layernorm(x, w['rwkv.blocks.0.pre_ln.weight'], w['rwkv.blocks.0.pre_ln.bias'], EPS)
            att = attention(layernorm(x, w[f'rwkv.blocks.{l}.ln1.weight'], w[f'rwkv.blocks.{l}.ln1.bias'], EPS), l, w, st)
            x = x + att
            ff = feed_forward(layernorm(x, w[f'rwkv.blocks.{l}.ln2.weight'], w[f'rwkv.blocks.{l}.ln2.bias'], EPS), l, w, st)
            x = x + ff
        x = layernorm(x, w['rwkv.ln_out.weight'], w['rwkv.ln_out.bias'], EPS)
        logits = x @ w['head.weight'].T
        nxt = int(np.argmax(logits))  # prediction during this token's step
    gen.append(nxt)
    tok.append(nxt)
    for _ in range(gen_len - 1):
        x = w['rwkv.embeddings.weight'][tok[-1]].astype(np.float64)  # feed predicted token
        for l in range(L):
            st = states[l]
            if l == 0:
                x = layernorm(x, w['rwkv.blocks.0.pre_ln.weight'], w['rwkv.blocks.0.pre_ln.bias'], EPS)
            att = attention(layernorm(x, w[f'rwkv.blocks.{l}.ln1.weight'], w[f'rwkv.blocks.{l}.ln1.bias'], EPS), l, w, st)
            x = x + att
            ff = feed_forward(layernorm(x, w[f'rwkv.blocks.{l}.ln2.weight'], w[f'rwkv.blocks.{l}.ln2.bias'], EPS), l, w, st)
            x = x + ff
        x = layernorm(x, w['rwkv.ln_out.weight'], w['rwkv.ln_out.bias'], EPS)
        logits = x @ w['head.weight'].T
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
