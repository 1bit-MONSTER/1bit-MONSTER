#!/usr/bin/env python3
"""OpenELM numpy reference — a direct port of apple/OpenELM modeling_openelm.py
(transformers 5.14 dropped OpenELM, so this is the oracle). Reads the HF
safetensors directly. Mirrors the autoregressive loop: per-prompt-position
argmax chain + N generated tokens. Dumps final logits to E2E_FULL_LOGITS.

Arch (from config.json): per-layer query/kv heads, per-layer FFN intermediate
(make_divisible(ffn_multiplier * model_dim, 256)), fused qkv with ROW split
[q|k|v], per-head RMSNorm on q/k (normalize_qk), HF half-split rope (chunk
pairing, freq_constant 10000, full head_dim), GQA repeat, SDPA scale
1/sqrt(head_dim), swish-GLU FFN, pre-norm RMSNorm, tied lm_head.
"""
import os, sys, json, time
import numpy as np
from safetensors.numpy import load_file

MODEL = sys.argv[1] if len(sys.argv) > 1 else "/tmp/onebit-e2e/openelm"
ids = [int(x) for x in open(sys.argv[2] if len(sys.argv) > 2 else "/tmp/openelm_ids.txt").read().split()]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 20

cfg = json.load(open(f"{MODEL}/config.json"))
D = cfg["model_dim"]; NL = cfg["num_transformer_layers"]; HD = cfg["head_dim"]
V = cfg["vocab_size"]
QH = cfg["num_query_heads"]; KH = cfg["num_kv_heads"]
FM = cfg["ffn_multipliers"]; DIV = cfg["ffn_dim_divisor"]

W = load_file(f"{MODEL}/model.safetensors")

def make_divisible(v, divisor):
    return int(v + divisor / 2) // divisor * divisor
INTER = [make_divisible(FM[i] * D, DIV) for i in range(NL)]

def rmsnorm(x, w, eps=1e-6):
    # x: [..., n]
    return x * w * (1.0 / np.sqrt((x * x).mean(-1, keepdims=True) + eps))

def rope_half_split(x, pos):
    # HF _apply_rotary_pos_emb on head_dim: (x*cos) + (rotate_half(x)*sin),
    # chunk pairing (i, i+dim/2). x: [nheads, hd]; pos int.
    dim = x.shape[-1]
    inv = 1.0 / (10000.0 ** (np.arange(0, dim, 2, dtype=np.float32) / dim))
    ang = pos * inv
    cos = np.concatenate([np.cos(ang), np.cos(ang)])  # [dim]
    sin = np.concatenate([np.sin(ang), np.sin(ang)])
    x1, x2 = x[..., : dim // 2], x[..., dim // 2 :]
    rot = np.concatenate([-x2, x1], axis=-1)
    return x * cos + rot * sin

def forward(x_ids, kcache, vcache):
    """Run the full sequence; return the logits at the last position."""
    S = len(x_ids)
    # embedding
    h = W["transformer.token_embeddings.weight"][x_ids].astype(np.float32)  # [S, D]
    for il in range(NL):
        p = f"transformer.layers.{il}."
        qh, kh = QH[il], KH[il]
        # attention (pre-norm)
        for s in range(S):
            # process positions one at a time, using the cache for past keys
            pass
        # vectorized prefill: compute q,k,v for all positions, then cache
        hq = rmsnorm(h, W[p + "attn_norm.weight"])  # [S, D]
        qkv = hq @ W[p + "attn.qkv_proj.weight"].T  # [S, (qh+kh+kh)*64]
        q, k, v = np.split(qkv, [qh * HD, (qh + kh) * HD], axis=-1)  # row split [q|k|v]
        q = q.reshape(S, qh, HD); k = k.reshape(S, kh, HD); v = v.reshape(S, kh, HD)
        q = rmsnorm(q, W[p + "attn.q_norm.weight"])  # per-head RMSNorm on hd
        k = rmsnorm(k, W[p + "attn.k_norm.weight"])
        # rope (per position)
        for s in range(S):
            q[s] = rope_half_split(q[s], s)
            k[s] = rope_half_split(k[s], s)
        # GQA repeat
        g = qh // kh
        kk = np.repeat(k, g, axis=1)   # [S, qh, HD]
        vv = np.repeat(v, g, axis=1)
        # SDPA scale 1/sqrt(hd), causal
        out = np.zeros((S, qh, HD), dtype=np.float32)
        for s in range(S):
            qs = q[s]  # [qh, HD]
            # scores[h, t] = qs[h] . kk[t, h] over HD, for t = 0..s
            scores = np.einsum('hd,thd->ht', qs, kk[: s + 1]) / np.sqrt(HD)  # [qh, s+1]
            scores = scores - scores.max(-1, keepdims=True)
            attn = np.exp(scores); attn /= attn.sum(-1, keepdims=True)
            out[s] = np.einsum('ht,thd->hd', attn, vv[: s + 1])  # [qh, HD]
        out = out.reshape(S, qh * HD)
        h = h + out @ W[p + "attn.out_proj.weight"].T
        # FFN (pre-norm)
        hf = rmsnorm(h, W[p + "ffn_norm.weight"])
        y12 = hf @ W[p + "ffn.proj_1.weight"].T  # [S, 2*inter]
        inter = INTER[il]
        y1, y2 = y12[:, :inter], y12[:, inter:]
        ffn = (y1 * (1.0 / (1.0 + np.exp(-y1))) * y2) @ W[p + "ffn.proj_2.weight"].T
        h = h + ffn
    h = rmsnorm(h, W["transformer.norm.weight"])
    logits = h[-1] @ W["transformer.token_embeddings.weight"].T  # tied lm_head
    return logits

def top8(lg):
    idx = np.argsort(-lg)[:8]
    return " ".join(f"{i}:{lg[i]:.3f}" for i in idx)

t0 = time.time()
chain = []
for i in range(len(ids)):
    lg = forward(ids[: i + 1], None, None)
    chain.append(int(lg.argmax()))
    print(f"ref-top8[{i}]: {top8(lg)}", flush=True)
gen = []
for g in range(N):
    lg = forward(ids + gen, None, None)
    gen.append(int(lg.argmax()))
print(f"ref-chain: {' '.join(map(str, chain))}")
print(f"ref-gen: {' '.join(map(str, gen))}")
print(f"ref-final-top8: {top8(lg)}")
out = os.environ.get("E2E_FULL_LOGITS", "openelm_ref_logits.txt")
with open(out, "w") as f:
    for i, v in enumerate(lg.tolist()):
        f.write(f"{i} {v}\n")
print(f"ref-logits: {out}  ({time.time()-t0:.1f}s)", flush=True)
