#!/usr/bin/env python3
"""GPT-OSS numpy reference — full 24-layer forward on the REAL checkpoint,
MXFP4 MoE kept packed (dequant per selected expert). Mirrors the engine's
autoregressive KV-cache loop and every quirk from modeling_gpt_oss.py:

  - YARN RoPE (theta 150000, factor 32, beta 32/1, orig_max 4096) with
    half-split (chunk) pairing; attention scale = (0.1*ln(f)+1)^2 / sqrt(hd)
  - GQA attention + per-head learned sinks (cat before softmax, dropped after)
  - router logits WITH bias, softmax over top-k only
  - MXFP4: value = FP4[nibble] * 2^(scale-127), low nibble->even, high->odd
  - gate = min(gate,7); up = clamp(up,±7); glu = gate*sigmoid(1.702*gate);
    gated = (up+1)*glu
  - RMSNorm (eps 1e-5), q/k/v/o biases, untied lm_head

Usage: e2e_numpy_ref_gptoss.py <model_dir> <ids.txt> [N gen tokens]
Prints the same chain:/engine-gen:/engine-top8: lines as e2e_seq_gen; dumps
final logits to E2E_FULL_LOGITS (or gptoss_ref_logits.txt).
"""
import glob, json, math, os, sys, time
import numpy as np
from safetensors import safe_open

MODEL = sys.argv[1] if len(sys.argv) > 1 else "/tmp/onebit-e2e/gptoss"
ids = [int(x) for x in open(sys.argv[2] if len(sys.argv) > 2 else "/tmp/gptoss_ids.txt").read().split()]
N = int(sys.argv[3]) if len(sys.argv) > 3 else 20

cfg = json.load(open(f"{MODEL}/config.json"))
H = cfg["hidden_size"]; NH = cfg["num_attention_heads"]; NKV = cfg["num_key_value_heads"]
HD = cfg["head_dim"]; L = cfg["num_hidden_layers"]; FF = cfg["intermediate_size"]
NE = cfg["num_local_experts"]; NEU = cfg["num_experts_per_tok"]
V = cfg["vocab_size"]; EPS = cfg["rms_norm_eps"]; SW = cfg["sliding_window"]
layer_types = cfg["layer_types"]
THETA = 150000.0; FACTOR = 32.0; BETA_FAST = 32.0; BETA_SLOW = 1.0; ORIG_MAX = 4096.0

FP4 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, -0, -.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)

# ── YARN inv_freq + attention scaling (transformers _compute_yarn_parameters) ──
# NOTE: the ramp runs over FREQ indices arange(dim//2) = [0..31] with
# dim-space bounds (lo,hi) — an easy-to-get-wrong 2x (fixed 2026-08-14:
# the engine was right, the first numpy port used even dim indices).
def yarn_dim(num_rot):
    return (HD * math.log(ORIG_MAX / (num_rot * 2 * math.pi))) / (2 * math.log(THETA))
lo = max(yarn_dim(BETA_FAST), 0.0); hi = min(yarn_dim(BETA_SLOW), HD - 1)
j = np.arange(0, HD // 2, dtype=np.float64)
ramp = np.clip((j - lo) / (hi - lo), 0, 1)
pf = THETA ** ((2 * j) / HD)
inv_freq = (1.0 / (FACTOR * pf)) * ramp + (1.0 / pf) * (1 - ramp)
ATTN_SCALE = (0.1 * math.log(FACTOR) + 1.0) ** 2 / math.sqrt(HD)

def rmsnorm(x, w):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + EPS) * w

def rope_half_split(q, pos):
    """q: [.., HD] — chunk(HD/2) half-split pairing (i with i+HD/2)."""
    d = HD // 2
    cos = np.cos(pos * inv_freq).astype(np.float32); sin = np.sin(pos * inv_freq).astype(np.float32)
    fh, sh = q[..., :d], q[..., d:]
    return np.concatenate([fh * cos - sh * sin, sh * cos + fh * sin], axis=-1)

def dequant(blocks, scales):
    """[R, nblk, 16] u8 + [R, nblk] u8 -> [R, 2880] f32 (32 vals per block)."""
    blk = blocks.reshape(blocks.shape[0], -1)
    s = np.ldexp(np.float32(1.0), scales.astype(np.int32) - 127)
    vals = np.empty((blocks.shape[0], blocks.shape[1] * 32), dtype=np.float32)
    vals[:, 0::2] = FP4[blk & 0x0F]
    vals[:, 1::2] = FP4[blk >> 4]
    return (vals.reshape(blocks.shape[0], blocks.shape[1], 32) * s[:, :, None]).reshape(blocks.shape[0], -1)

# ── load (attention/router/norms as f32; MoE blocks+scales stay U8) ──
import torch
print("loading weights...", flush=True)
W = {}
for shard in sorted(glob.glob(f"{MODEL}/model-*.safetensors")):
    with safe_open(shard, framework="torch") as f:
        for k in f.keys():
            t = f.get_tensor(k)
            W[k] = t.float().numpy() if t.dtype != torch.uint8 else t.numpy()

embed = W["model.embed_tokens.weight"]
final_norm = W["model.norm.weight"]
lm_head = W["lm_head.weight"]

def Lw(il, name):
    return W[f"model.layers.{il}.{name}"]

kcache = [[] for _ in range(L)]; vcache = [[] for _ in range(L)]

def step(x, pos):
    for il in range(L):
        xn = rmsnorm(x, Lw(il, "input_layernorm.weight"))
        q = xn @ Lw(il, "self_attn.q_proj.weight").T + Lw(il, "self_attn.q_proj.bias")
        k = xn @ Lw(il, "self_attn.k_proj.weight").T + Lw(il, "self_attn.k_proj.bias")
        v = xn @ Lw(il, "self_attn.v_proj.weight").T + Lw(il, "self_attn.v_proj.bias")
        q = rope_half_split(q.reshape(NH, HD), pos)
        k = rope_half_split(k.reshape(NKV, HD), pos)
        v = v.reshape(NKV, HD)
        kcache[il].append(k); vcache[il].append(v)

        att = np.zeros((NH, HD), dtype=np.float32)
        sinks = Lw(il, "self_attn.sinks")
        for h in range(NH):
            kh = h // (NH // NKV)
            t0 = max(0, pos - SW + 1) if layer_types[il] == "sliding_attention" else 0
            K = np.stack(kcache[il][t0:pos + 1]); Vv = np.stack(vcache[il][t0:pos + 1])
            scores = K[:, kh] @ q[h] * ATTN_SCALE
            combined = np.concatenate([scores, np.array([sinks[h]], dtype=np.float32)])
            combined = combined - combined.max()
            probs = np.exp(combined); probs /= probs.sum()
            att[h] = probs[:-1] @ Vv[:, kh]
        x = x + (att.reshape(-1) @ Lw(il, "self_attn.o_proj.weight").T + Lw(il, "self_attn.o_proj.bias"))

        x2 = rmsnorm(x, Lw(il, "post_attention_layernorm.weight"))
        router = x2 @ Lw(il, "mlp.router.weight").T + Lw(il, "mlp.router.bias")
        topk = np.argsort(router)[-NEU:][::-1]
        rs = router[topk] - router[topk].max()
        wts = np.exp(rs); wts /= wts.sum()
        acc = np.zeros(H, dtype=np.float32)
        for e, we in zip(topk, wts):
            gu = dequant(Lw(il, "mlp.experts.gate_up_proj_blocks")[e], Lw(il, "mlp.experts.gate_up_proj_scales")[e]) @ x2
            gu += Lw(il, "mlp.experts.gate_up_proj_bias")[e]
            gate = np.minimum(gu[0::2], 7.0); up = np.clip(gu[1::2], -7.0, 7.0)
            gated = (up + 1.0) * gate * (1.0 / (1.0 + np.exp(-1.702 * gate)))
            down = dequant(Lw(il, "mlp.experts.down_proj_blocks")[e], Lw(il, "mlp.experts.down_proj_scales")[e]) @ gated
            down += Lw(il, "mlp.experts.down_proj_bias")[e]
            acc += we * down
        x = x + acc
    return rmsnorm(x, final_norm) @ lm_head.T

def top8(lg):
    idx = np.argsort(lg)[-8:][::-1]
    return " ".join(f"{int(i)}:{lg[i]:.3f}" for i in idx)

t0 = time.time()
chain = []
for i, tid in enumerate(ids):
    x = embed[tid].copy()
    lg = step(x, i)
    chain.append(int(np.argmax(lg)))
    print(f"ref-top8[{i}]: {top8(lg)}", flush=True)
gen = []
for g in range(N):
    x = embed[gen[-1] if g else chain[-1]].copy()
    lg = step(x, len(ids) + g)
    gen.append(int(np.argmax(lg)))
print(f"ref-chain: {' '.join(map(str, chain))}")
print(f"ref-gen: {' '.join(map(str, gen))}")
print(f"ref-final-top8: {top8(lg)}")
out = os.environ.get("E2E_FULL_LOGITS", "gptoss_ref_logits.txt")
with open(out, "w") as f:
    for i, v in enumerate(lg):
        f.write(f"{i} {v}\n")
print(f"ref-logits: {out}  ({time.time()-t0:.1f}s)", flush=True)
