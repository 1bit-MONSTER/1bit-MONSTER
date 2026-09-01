#!/usr/bin/env python3
"""Full Qwen3-0.6B forward in numpy for the 5-token prompt, with configurable
weight source: 'bf16' (original), 'dequant' (Q4NX-dequantized F32)."""
import json, sys, numpy as np

def bf16_to_f32(u):
    return (u.astype(np.uint32) << 16).view(np.float32)

def load_tensor(dump_dir, name):
    m = json.load(open(f'{dump_dir}/manifest.json'))
    x = {t['name']: t for t in m}[name]
    raw = np.fromfile(f"{dump_dir}/{x['file']}", dtype=np.uint8)
    if x['type'] == 30:
        return bf16_to_f32(np.frombuffer(raw, dtype=np.uint16))
    return raw.view(np.float32)

def rmsnorm(x, w, eps=1e-6):
    # x: [embd, nt]; w: [embd] -> normalize over embd (axis 0)
    return x / np.sqrt((x*x).mean(0, keepdims=True) + eps) * w[:, None]

def rope_apply(x, pos, n_rot, head_dim, freq_base=1000000.0):
    # x: [n_heads, head_dim, n_tokens] (already split)
    n_heads, hd, nt = x.shape
    inv_freq = 1.0 / (freq_base ** (np.arange(0, n_rot, 2, dtype=np.float64) / hd))
    freqs = np.outer(pos, inv_freq)  # [nt, n_rot/2]
    cos = np.cos(freqs).astype(np.float32)  # [nt, n_rot/2]
    sin = np.sin(freqs).astype(np.float32)
    x = x.copy()
    half = n_rot // 2
    # qwen3 neox: rotate first n_rot dims in pairs (i, i+half)
    x0 = x[:, :half, :].copy()   # [h, n_rot/2, nt]
    x1 = x[:, half:n_rot, :].copy()
    cos_t = cos.T[None, :, :]    # [1, n_rot/2, nt]
    sin_t = sin.T[None, :, :]
    x[:, :half, :] = x0 * cos_t - x1 * sin_t
    x[:, half:n_rot, :] = x1 * cos_t + x0 * sin_t
    return x

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'dequant'
    if mode == 'bf16':
        d = '/tmp/dump_src'
    elif mode == 'dequant':
        d = '/tmp/dump_v2'  # v2 = Q4NX tiles; dequant below
    else:
        d = mode
    def get(name, shape):
        t = load_tensor(d, name)
        return t.reshape(shape)
    n_layer, n_embd, n_head, n_head_kv, n_ff, head_dim, n_rot = 28, 1024, 16, 8, 3072, 128, 128
    # tokenize "The capital of France is" using the vocab from the BF16 GGUF
    # (hardcode the ids from the earlier tokenization: 5 tokens)
    tok_ids = np.array([2137, 29870, 263, 6159, 374], dtype=np.int32)  # placeholder; real ids from dump_logits run
    # Actually read real ids: from the logits header we know n=5; get ids by tokenize via llama? Hardcode from tokenizer: use the known output
    # "The capital of France is" -> ids [2137, 29870, 263, 6159, 374]? verify below by printing
    # We'll read them from a small C++ tokenize dump instead.
    import struct, subprocess
    ids_txt = subprocess.run(['./tok_ids', 'The capital of France is'], capture_output=True, text=True).stdout
    tok_ids = np.array([int(x) for x in ids_txt.split()], dtype=np.int32)
    nt = len(tok_ids)
    print("tokens:", tok_ids.tolist())
    # embeddings: token_embd [151936, 1024]
    emb = get('token_embd.weight', (151936, 1024))
    h = emb[tok_ids].T  # [1024, nt]
    pos = np.arange(nt, dtype=np.float32)
    for il in range(n_layer):
        pre = 'blk.%d.' % il
        h = rmsnorm(h, get(pre+'attn_norm.weight', (1024,)))
        # QKV
        Wq = get(pre+'attn_q.weight', (2048, 1024))
        Wk = get(pre+'attn_k.weight', (1024, 1024))
        Wv = get(pre+'attn_v.weight', (1024, 1024))
        Q = (Wq @ h).reshape(head_dim, n_head, nt).transpose(1,0,2)
        K = (Wk @ h).reshape(head_dim, n_head_kv, nt).transpose(1,0,2)
        V = (Wv @ h).reshape(head_dim, n_head_kv, nt).transpose(1,0,2)
        # q/k norms
        wq = get(pre+'attn_q_norm.weight', (head_dim,))
        Q = Q / np.sqrt((Q*Q).mean(1, keepdims=True) + 1e-6) * wq[None,:,None]
        wk = get(pre+'attn_k_norm.weight', (head_dim,))
        K = K / np.sqrt((K*K).mean(1, keepdims=True) + 1e-6) * wk[None,:,None]
        Q = rope_apply(Q, pos, n_rot, head_dim)
        K = rope_apply(K, pos, n_rot, head_dim)
        # attention with causal mask (GQA: n_head=16, n_head_kv=8)
        gqa = n_head // n_head_kv
        Qh = Q.transpose(2,0,1)                      # [nt, n_head, head_dim]
        Krep = K.repeat(gqa, axis=0)                 # [n_head, head_dim, nt]
        Vrep = V.repeat(gqa, axis=0)                 # [n_head, head_dim, nt]
        Kh = Krep.transpose(2,0,1)                   # [nt, n_head, head_dim]
        Vh = Vrep.transpose(2,0,1)                   # [nt, n_head, head_dim]
        S = np.einsum('tqd,kqd->tqk', Qh, Kh)        # [nt, n_head, nt]
        S = S / np.sqrt(head_dim)
        S = np.where(np.arange(nt)[:,None,None] >= np.arange(nt)[None,None,:], S, -1e9)
        P = np.exp(S - S.max(axis=-1, keepdims=True))
        P = P / P.sum(axis=-1, keepdims=True)
        O = np.einsum('tqk,kqd->tqd', P, Vh)   # [nt, n_head, head_dim]
        O = O.reshape(nt, head_dim, n_head).transpose(0,2,1).reshape(nt, n_head * head_dim).T
        Wo = get(pre+'attn_output.weight', (1024, 2048))
        h = Wo @ O + h  # residual (no bias in qwen3)
        # FFN
        h = rmsnorm(h, get(pre+'ffn_norm.weight', (1024,)))
        Wg = get(pre+'ffn_gate.weight', (3072, 1024))
        Wu = get(pre+'ffn_up.weight', (3072, 1024))
        Wd = get(pre+'ffn_down.weight', (1024, 3072))
        gate = Wg @ h
        silu = gate * (1.0 / (1.0 + np.exp(-gate)))
        h = Wd @ (silu * (Wu @ h)) + h
    h = rmsnorm(h, get('output_norm.weight', (1024,)))
    out = emb  # tied embeddings (Qwen3)
    logits = out @ h  # [151936, nt]
    np.save(f'/tmp/np_logits_{mode}.npy', logits[:, -1].astype(np.float32))
    top = np.argsort(-logits[:, -1])[:5]
    print("mode", mode, "top5:", top.tolist())

main()
