#!/usr/bin/env python3
"""zamba2_ref.py — independent numpy reference for Zamba2-1.2B, written from
transformers modeling_zamba2.py (NOT from the C++ engine). Reads the same GGUF
(Q4_0 dequant) and prints top-5 logits for token 0 (BOS). Used to verify the
C++ engine fixes (#1460 + hybrid rewrite) — logits must agree to ~1e-3.
"""
import struct, sys
from pathlib import Path
import numpy as np

# ── GGUF read + dequant ──
GGML_TYPE = {0: "F32", 1: "F16", 2: "Q4_0", 8: "Q8_0", 12: "Q4_K"}

def read_gguf(path):
    raw = Path(path).read_bytes()
    (version, n_t, n_kv) = struct.unpack_from("<IQQ", raw, 4)
    assert version == 3
    o = 24
    def rs(o):
        (n,) = struct.unpack_from("<Q", raw, o); return raw[o+8:o+8+n].decode(), o+8+n
    for _ in range(n_kv):
        _, o = rs(o)
        (vt,) = struct.unpack_from("<I", raw, o); o += 4
        if vt == 8: _, o = rs(o)
        elif vt == 9:
            (at, n) = struct.unpack_from("<IQ", raw, o); o += 12
            for _ in range(n):
                if at == 8: _, o = rs(o)
                else: o += {0:1,1:1,7:1,2:2,3:2,4:4,5:4,6:4}.get(at, 8)
        else: o += {0:1,1:1,7:1,2:2,3:2,4:4,5:4,6:4}.get(vt, 8)
    tensors = {}
    for _ in range(n_t):
        name, o = rs(o)
        (n_dims,) = struct.unpack_from("<I", raw, o); o += 4
        dims = list(struct.unpack_from(f"<{n_dims}Q", raw, o)); o += 8*n_dims
        (dtype,) = struct.unpack_from("<I", raw, o); o += 4
        (off,) = struct.unpack_from("<Q", raw, o); o += 8
        tensors[name] = {"type": dtype, "dims": dims, "off": off}
    base = (o + 31) & ~31
    for t in tensors.values(): t["off"] += base
    return raw, tensors

def dequant(raw, ti):
    t, dims = ti["type"], ti["dims"]
    n = int(np.prod(dims))
    off = ti["off"]
    if t == 0: return np.frombuffer(raw, dtype="<f4", count=n, offset=off).copy()
    if t == 1: return np.frombuffer(raw, dtype="<f2", count=n, offset=off).astype("<f4").copy()
    if t == 2:  # Q4_0 per ggml dequantize_row_q4_0: 18 B/block, 32 elems
        #   elements 0..15 = low nibbles, 16..31 = high nibbles; val = (nib-8)*d
        nblk = n // 32
        blk = np.frombuffer(raw, dtype=np.uint8, count=18*nblk, offset=off).reshape(nblk, 18)
        sc_u16 = blk[:, 0].astype(np.uint16) | (blk[:, 1].astype(np.uint16) << 8)
        scales = sc_u16.view("<f2").astype("<f4")
        q = blk[:, 2:]
        v = np.empty((nblk, 32), np.int16)
        v[:, :16] = (q & 0x0F).astype(np.int16) - 8
        v[:, 16:] = (q >> 4).astype(np.int16) - 8
        return (v * scales[:, None]).reshape(-1)
    if t == 14:  # Q6_K per ggml dequantize_row_q6_K: 210 B/block, 256 elems
        #   2 halves of 128; y = d*sc*(q-32); q = 4-bit nib | 2-bit<<4
        nblk = n // 256
        blk = np.frombuffer(raw, dtype=np.uint8, count=210*nblk, offset=off).reshape(nblk, 210)
        d_u16 = blk[:, 0].astype(np.uint16) | (blk[:, 1].astype(np.uint16) << 8)
        d = d_u16.view("<f2").astype("<f4")
        ql = blk[:, 2:130]; qh = blk[:, 130:194]
        sc = blk[:, 194:210].astype(np.int8).astype("<f4")
        j = np.arange(128); l = j % 32; quad = j // 32
        out = np.empty((nblk, 256), np.int16)
        for half in range(2):
            qlh = ql[:, half*64:(half+1)*64]; qhh = qh[:, half*32:(half+1)*32]; sch = sc[:, half*8:(half+1)*8]
            byte = qlh[:, l + 32 * (quad % 2)]
            nib = np.where(quad < 2, byte & 0x0F, byte >> 4)
            bits = (qhh[:, l] >> (2 * quad)) & 3
            q = (nib | (bits << 4)).astype(np.int16) - 32
            out[:, half*128:(half+1)*128] = q
            out[:, half*128:(half+1)*128] = (d[:, None] * sch[:, l // 16 + 2 * quad] * q)
        return out.reshape(-1)
    raise NotImplementedError(f"type {GGML_TYPE.get(t,t)}")

def load_model(path):
    raw, ts = read_gguf(path)
    W = {name: dequant(raw, ti) for name, ti in ts.items()}
    M = {"cfg": {}, "W": W}
    # config
    def kv(*keys, default=None):
        for k in keys:
            for cand in (k, f"zamba2.{k}"):
                pass
        return default
    # read KV again for config
    return W, ts

# ── Reference forward (modeling_zamba2.py) ──
def softplus(x): return np.log1p(np.exp(np.clip(x, -40, 20))) + np.where(x > 20, x - 20, 0)

def rms_norm(x, w, eps=1e-5):
    return x * (1.0 / np.sqrt(np.mean(x*x) + eps)) * w

def mamba2_block(x, L, cfg, conv_state, ssm_state):
    """Zamba2MambaMixer, per modeling_zamba2.py + mamba2.py step math."""
    d_model, d_inner, d_state, d_conv = cfg["d_model"], cfg["d_inner"], cfg["d_state"], cfg["d_conv"]
    n_head, n_group, head_dim = cfg["n_head"], cfg["n_group"], cfg["head_dim"]
    conv_dim = d_inner + 2*n_group*d_state
    d_in_proj = d_inner + conv_dim + n_head
    z = L["ssm_in"] @ x  # [d_in_proj] — GGUF [input, output] → output = W^T @ x
    z_seg, xbc, dt = z[:d_inner], z[d_inner:d_inner+conv_dim], z[d_inner+conv_dim:]
    # conv1d (causal, d_conv=4) with state
    xbc_conv = L["ssm_conv1d_b"].copy()
    xbc_conv += L["ssm_conv1d_w"][0] * xbc
    for k in range(1, d_conv):
        xbc_conv += L["ssm_conv1d_w"][k] * conv_state[k-1]
    conv_state[1:] = conv_state[:-1]; conv_state[0] = xbc
    xbc_act = xbc_conv / (1 + np.exp(-xbc_conv))
    x_inner = xbc_act[:d_inner]
    B = xbc_act[d_inner:d_inner+n_group*d_state].reshape(n_group, d_state)
    C = xbc_act[d_inner+n_group*d_state:].reshape(n_group, d_state)
    # scan
    dt_sp = softplus(dt + L["ssm_dt_bias"])
    A = L["ssm_a"]  # already negated
    y = np.empty(d_inner)
    for h in range(n_head):
        g = h // (n_head // n_group)
        dA = np.exp(dt_sp[h] * A[h])
        xh = x_inner[h*head_dim:(h+1)*head_dim]
        s = ssm_state[h]
        for hd in range(head_dim):
            s = dA * s + dt_sp[h] * B[g] * xh[hd]
            y[h*head_dim+hd] = C[g] @ s + L["ssm_d"][h] * xh[hd]
        ssm_state[h] = s
    # grouped gated RMSNorm (Zamba2RMSNormGated): norm layout [gs, n_group]
    gs = d_inner // n_group
    y = y.reshape(n_group, gs)
    y = y * (1.0 / np.sqrt(np.mean(y*y, axis=1, keepdims=True) + 1e-6))
    y = y * L["ssm_norm"].reshape(gs, n_group).T   # y[g,i] * norm[i,g]
    y = y.reshape(-1)
    y = y * (z_seg / (1 + np.exp(-z_seg)))
    return L["ssm_out"] @ y  # GGUF [d_model, d_inner] → W^T @ y? ssm_out GGUF dims [d_inner, d_model]

def main():
    path = sys.argv[1]
    raw, ts = read_gguf(path)
    W = {name: dequant(raw, ti).reshape(ti["dims"]) for name, ti in ts.items()}
    # config from known dims (1.2B)
    emb = W["token_embd.weight"]  # [d_model, vocab] in GGUF
    d_model = emb.shape[0]; vocab = emb.shape[1]
    ssm_in0 = W["blk.0.ssm_in.weight"]
    d_in_proj = ssm_in0.shape[0]; d_inner = 4096
    d_state, n_group, d_conv = 128, 1, 4
    n_head = 64; head_dim = 64
    conv_dim = d_inner + 2*n_group*d_state
    n_layers = len({n.split('.')[1] for n in W if n.startswith('blk.')})
    hybrid_ids = sorted({int(n.split('.')[1]) for n in W if 'attn_q' in n})
    n_attn, n_kv, hd = 32, 32, 128
    attn_in = n_attn * hd
    cfg = dict(d_model=d_model, d_inner=d_inner, d_state=d_state, d_conv=d_conv,
               n_head=n_head, n_group=n_group, head_dim=head_dim, n_layers=n_layers,
               hybrid_ids=hybrid_ids, n_attn=n_attn, n_kv=n_kv, hd=hd, attn_in=attn_in)
    print(f"layers={n_layers} hybrid={hybrid_ids}")

    # GGUF layout: linear tensors stored [input, output]; transpose for W^T@x
    def lin(name): return W[name].T  # [output, input]
    layers = {}
    for l in range(n_layers):
        p = f"blk.{l}."
        L = {
            "attn_norm": W[p+"attn_norm.weight"],
            "ssm_in": lin(p+"ssm_in.weight"),
            "ssm_conv1d_w": W[p+"ssm_conv1d.weight"],  # [d_conv, conv_dim]
            "ssm_conv1d_b": W[p+"ssm_conv1d.bias"],
            "ssm_dt_bias": W[p+"ssm_dt.bias"],
            "ssm_a": W[p+"ssm_a"].reshape(-1),
            "ssm_d": W[p+"ssm_d"].reshape(-1),
            "ssm_norm": W[p+"ssm_norm.weight"],
            "ssm_out": lin(p+"ssm_out.weight"),
        }
        if l in hybrid_ids:
            L.update({
                "post_attn_norm": W[p+"post_attention_norm.weight"],  # 2*d_model concat norm
                "ffn_norm": W[p+"ffn_norm.weight"],
                "attn_q": lin(p+"attn_q.weight"),
                "attn_k": lin(p+"attn_k.weight"),
                "attn_v": lin(p+"attn_v.weight"),
                "attn_o": lin(p+"attn_output.weight"),
                "ffn_gate": lin(p+"ffn_gate.weight"),
                "ffn_up": lin(p+"ffn_up.weight"),
                "ffn_down": lin(p+"ffn_down.weight"),
                "ssm_mix": lin(p+"ssm_mix.weight"),
            })
        layers[l] = L

    # forward token 0 (BOS)
    tok = 1
    hidden = emb[:, tok].copy()
    embedding = hidden.copy()
    conv_states = {l: np.zeros((d_conv-1, conv_dim)) for l in range(n_layers)}
    ssm_states = {l: np.zeros((n_head, d_state)) for l in range(n_layers)}
    kv_k = {h: np.zeros((4096, n_kv*hd)) for h in hybrid_ids}
    kv_v = {h: np.zeros((4096, n_kv*hd)) for h in hybrid_ids}
    pos = 0
    print("L-1 hidden0:", " ".join(f"{v:.6f}" for v in hidden[:4]))
    for l in range(n_layers):
        L = layers[l]
        if l in hybrid_ids:
            # Zamba2AttentionDecoderLayer: concat → norm → attn → pre_ff_norm → MLP
            x = np.concatenate([hidden, embedding])
            x = rms_norm(x, L["post_attn_norm"], 1e-5)
            q = L["attn_q"] @ x; k = L["attn_k"] @ x; v = L["attn_v"] @ x
            # RoPE (freqs: theta 10000, head_dim 128)
            q = q.reshape(n_attn, hd); k = k.reshape(n_kv, hd)
            for h in range(n_attn):
                for d in range(0, hd, 2):
                    fr = pos / 10000.0 ** (d / hd)
                    c, s = np.cos(fr), np.sin(fr)
                    q[h, d], q[h, d+1] = q[h,d]*c - q[h,d+1]*s, q[h,d]*s + q[h,d+1]*c
            for h in range(n_kv):
                for d in range(0, hd, 2):
                    fr = pos / 10000.0 ** (d / hd)
                    c, s = np.cos(fr), np.sin(fr)
                    k[h, d], k[h, d+1] = k[h,d]*c - k[h,d+1]*s, k[h,d]*s + k[h,d+1]*c
            kv_k[l][pos] = k.reshape(-1); kv_v[l][pos] = v.reshape(-1)
            # attention with scale sqrt(2/hd)
            scale = np.sqrt(2.0/hd)
            attn_out = np.zeros(n_attn*hd)
            for h in range(n_attn):
                kh = kv_k[l][:pos+1, h*hd:(h+1)*hd]  # [pos+1, hd], head h slice
                scores = (q[h] @ kh.T) * scale
                scores = scores - scores.max()
                pr = np.exp(scores); pr /= pr.sum()
                attn_out[h*hd:(h+1)*hd] = pr @ kv_v[l][:pos+1, h*hd:(h+1)*hd]
            attn_out = L["attn_o"] @ attn_out  # [d_model]
            ff = rms_norm(attn_out, L["ffn_norm"], 1e-5)
            g = L["ffn_gate"] @ ff; u = L["ffn_up"] @ ff
            th = L["ffn_down"] @ (g/(1+np.exp(-g)) * u)
            th = L["ssm_mix"] @ th
            hidden = hidden + th
            # mamba decoder
            normed = rms_norm(hidden, L["attn_norm"], 1e-5)
            mb = mamba2_block(normed, L, cfg, conv_states[l], ssm_states[l])
            hidden = hidden + mb
        else:
            normed = rms_norm(hidden, L["attn_norm"], 1e-5)
            mb = mamba2_block(normed, L, cfg, conv_states[l], ssm_states[l])
            hidden = hidden + mb
        print(f"L{l} hidden0:", " ".join(f"{v:.6f}" for v in hidden[:4]))
    hidden = rms_norm(hidden, W["output_norm.weight"], 1e-5)
    logits = emb.T @ hidden
    top = np.argsort(-logits)[:5]
    print("top5 (tied emb):", [(int(i), round(float(logits[i]), 4)) for i in top])
    print("logits min/max:", round(float(logits.min()),3), round(float(logits.max()),3))

if __name__ == "__main__":
    main()
