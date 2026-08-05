#!/usr/bin/env python3
"""Golden reference for one Qwen3.6 GatedDeltaNet (linear_attention) layer.

Loads the real Q4NX weights and runs the layer exactly as
transformers/models/qwen3_next/modeling_qwen3_next.py does, so the C++ engine
path has something to validate against.

Also resolves two ambiguities the file format does not document:
  * BF16 2-D layout is NOT uniform. `moe_router` [2048,256] uses a stride-8
    interleave (the layout npu_engine_universal.cpp already hardcodes), but
    `ssm_alpha_proj`/`ssm_beta_proj` [2048,32] are plain row-major.
    Evidence: scrambling a trained matrix mixes values across columns and
    drives per-column norms toward equality, so the correct read is the one
    with HIGHER column-norm CV. Control (router, known-good interleaved):
    0.173 interleaved vs 0.092 plain. alpha_proj: 0.037 interleaved vs 0.463
    plain — a 12x gap the other way. Re-check with --probe-layouts.
  * `ssm_a` is already-negated A (= -exp(A_log)), NOT raw A_log — the #1460
    trap. Evidence: 160/160 values across 5 layers are negative and 145/160
    fall in [-16,0), matching the reference init `A = uniform(0,16)`. Raw
    A_log under that init would be positive ~94% of the time. Using it as
    A_log silently destroys the state decay.

Usage:
  python3 tools/gdn_reference.py --probe-layouts
  python3 tools/gdn_reference.py --dump build/gdn_golden.npz
"""
import argparse
import json
import mmap
import struct

import numpy as np

MODEL = "/home/bcloud/.config/flm/models/Qwen3.6-35B-A3B-NPU2/model.q4nx"
CONFIG = "/home/bcloud/.config/flm/models/Qwen3.6-35B-A3B-NPU2/config.json"

H = 2048
NUM_V_HEADS = 32
NUM_K_HEADS = 16
HEAD_K = 128
HEAD_V = 128
CONV_K = 4
KEY_DIM = NUM_K_HEADS * HEAD_K      # 2048
VALUE_DIM = NUM_V_HEADS * HEAD_V    # 4096
CONV_DIM = KEY_DIM * 2 + VALUE_DIM  # 8192
EPS = 1e-6


class Q4nx:
    def __init__(self, path):
        self.f = open(path, "rb")
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        hsz = struct.unpack("<Q", self.mm[:8])[0]
        self.hdr = json.loads(self.mm[8:8 + hsz].decode("utf-8", "replace")[
            : self.mm[8:8 + hsz].decode("utf-8", "replace").rindex("}") + 1])
        self.df = 8 + hsz

    def raw(self, key):
        v = self.hdr[key]
        o = v["data_offsets"]
        return v, self.mm[self.df + o[0]: self.df + o[1]]

    def f32(self, key):
        v, b = self.raw(key)
        return np.frombuffer(b, dtype=np.float32).astype(np.float64), v["shape"]

    def bf16_1d(self, key):
        v, b = self.raw(key)
        u = np.frombuffer(b, dtype=np.uint16).astype(np.uint32) << 16
        return u.view(np.float32).astype(np.float64), v["shape"]

    def bf16_2d(self, key, interleave):
        """[in=H, out=N]. interleave=True uses the stride-8 cracked layout."""
        v, b = self.raw(key)
        u = np.frombuffer(b, dtype=np.uint16).astype(np.uint32) << 16
        flat = u.view(np.float32).astype(np.float64)
        n_in, n_out = v["shape"]
        if not interleave:
            return flat.reshape(n_in, n_out)
        # W[i][j] = flat[(i%8)*(n_out*n_in/8) + j*(n_in/8) + i/8]
        blk = n_out * (n_in // 8)
        i = np.arange(n_in)[:, None]
        j = np.arange(n_out)[None, :]
        idx = (i % 8) * blk + j * (n_in // 8) + i // 8
        return flat[idx]

    def q8_0(self, key):
        """I8 Q8_0 (8704 B/row) -> [out_features, in_features] row-major f64."""
        v, b = self.raw(key)
        shp = v["shape"]
        i8_rows = shp[0] * shp[1]
        in_features = shp[1] * 256
        TR, TC, RB = 32, 256, 8704
        ntc = in_features // TC
        ntr = i8_rows // ntc
        out = np.zeros((ntr * TR, ntc * TC), dtype=np.float64)
        buf = np.frombuffer(b, dtype=np.uint8)
        for ir in range(i8_rows):
            t = buf[ir * RB:(ir + 1) * RB]
            sc = (t[:512].view(np.uint16).astype(np.uint32) << 16).view(np.float32).astype(np.float64)
            vals = t[512:].view(np.int8).astype(np.float64).reshape(TR, TC)
            tr_, tc_ = ir // ntc, ir % ntc
            # scales[g*TR + r] for g in TC/32 groups
            s = sc.reshape(TC // 32, TR).T                       # [TR, groups]
            s = np.repeat(s, 32, axis=1)                          # [TR, TC]
            out[tr_ * TR:(tr_ + 1) * TR, tc_ * TC:(tc_ + 1) * TC] = vals * s
        return out


def full_attn_forward(m, layer, n_tokens=3, seed=7):
    """Golden for one full-attention layer (Qwen3.6: 16 heads x 256, 2 KV
    heads, q + output gate fused in q_proj per-head halves, partial rotary
    64-of-256 @ theta 1e7, attn_out *= sigmoid(gate)). Mirrors
    transformers Qwen3NextAttention exactly."""
    p = f"model.layer.{layer}.self_attn."
    q = m.q8_0(p + "q_proj.weight")        # [8192, 2048] = per-head [q 256 | gate 256]
    k = m.q8_0(p + "k_proj.weight")        # [512, 2048]
    v = m.q8_0(p + "v_proj.weight")        # [512, 2048]
    o = m.q8_0(p + "o_proj.weight")        # [2048, 4096]
    qn = m.bf16_1d(p + "q_norm.weight")[0]  # [256] (1+w RMSNorm weight)
    kn = m.bf16_1d(p + "k_norm.weight")[0]
    rng = np.random.default_rng(seed)
    xs = rng.standard_normal((n_tokens, H)) * 0.5
    outs = []
    THETA, ROT_DIM = 1e7, 64          # rope_parameters: theta, partial_rotary_factor 0.25
    for t in range(n_tokens):
        x = xs[t]
        qkv = q @ x                   # [8192]
        qq = np.empty((16, 256)); gate = np.empty((16, 256))
        for h in range(16):
            qq[h] = qkv[h * 512:h * 512 + 256]
            gate[h] = qkv[h * 512 + 256:h * 512 + 512]
        kk = k @ x                    # [512]
        vv = v @ x                    # [512]
        # q/k RMSNorm (1+w) per head
        for h in range(16):
            qq[h] = qq[h] / np.sqrt((qq[h] * qq[h]).mean() + 1e-6) * qn
        for h in range(2):
            kk[h * 256:(h + 1) * 256] = kk[h * 256:(h + 1) * 256] / np.sqrt(
                (kk[h * 256:(h + 1) * 256] ** 2).mean() + 1e-6) * kn
        # partial rotary: rotate first 64 dims (half-split within 64)
        def rot(xx):
            out = xx.copy()
            for d in range(32):
                f = 1.0 / THETA ** (d / 32.0)
                a = t * f
                c, s = np.cos(a), np.sin(a)
                x1, x2 = xx[d], xx[d + 32]
                out[d] = x1 * c - x2 * s
                out[d + 32] = x2 * c + x1 * s
            return out
        for h in range(16): qq[h] = rot(qq[h])
        for h in range(2): kk[h * 256:(h + 1) * 256] = rot(kk[h * 256:(h + 1) * 256])
        # GQA attention: 16 q heads, 2 kv heads, scale 1/sqrt(256)
        attn = np.zeros((16, 256))
        for h in range(16):
            kh = kk[(h // 8) * 256:(h // 8 + 1) * 256]
            vh = vv[(h // 8) * 256:(h // 8 + 1) * 256]
            # single position (decode step t): scores = q@k/scale, softmax over 1 -> 1.0
            attn[h] = vh * (np.dot(qq[h], kh) / np.sqrt(256.0))
        attn = attn * (1.0 / (1.0 + np.exp(-gate)))   # sigmoid gate, per-dim
        outs.append(o @ attn.reshape(-1))
    return np.array(outs), qq, gate, xs


def gdn_layers(m, n_layers=40):
    """Layer indices that are linear_attention (GDN); the rest are full attention."""
    return [l for l in range(n_layers)
            if f"model.layer.{l}.linear_attn.qkv_proj.weight" in m.hdr]


def l2norm(x, eps=1e-6):
    return x / np.sqrt((x * x).sum(-1, keepdims=True) + eps)


def softplus(x):
    return np.where(x > 20, x, np.log1p(np.exp(np.minimum(x, 20))))


def load_layer(m, layer, interleave=False):
    """interleave applies to alpha/beta only; see module docstring (default: plain)."""
    p = f"model.layer.{layer}.linear_attn."
    w = {}
    w["qkv"] = m.q8_0(p + "qkv_proj.weight")              # [8192, 2048]
    w["out"] = m.q8_0(p + "ssm_out_proj.weight")          # [2048, 4096]
    w["z"] = m.q8_0(f"model.layer.{layer}.self_attn.gate_proj.weight")  # [4096, 2048]
    w["alpha"] = m.bf16_2d(p + "ssm_alpha_proj.weight", interleave)     # [2048, 32]
    w["beta"] = m.bf16_2d(p + "ssm_beta_proj.weight", interleave)       # [2048, 32]
    w["conv"] = m.bf16_2d(p + "ssm_conv1d.weight", False)               # [4, 8192]
    w["norm"] = m.bf16_1d(p + "ssm_norm.weight")[0]                     # [128]
    w["ssm_a"] = m.f32(p + "ssm_a")[0]                                  # [32]
    w["dt_bias"] = m.f32(p + "ssm_dt.bias")[0]                          # [32]
    return w


def gdn_forward(w, xs, negate_a):
    """Run the GDN layer over a token sequence xs [T, H]. Returns out [T, H]."""
    T = xs.shape[0]
    conv_state = np.zeros((CONV_DIM, CONV_K))
    state = np.zeros((NUM_V_HEADS, HEAD_K, HEAD_V))
    A = -np.exp(w["ssm_a"]) if negate_a else w["ssm_a"]
    outs = []
    for t in range(T):
        x = xs[t]
        qkv = w["qkv"] @ x                    # [8192]
        z = w["z"] @ x                        # [4096]
        # causal depthwise conv1d (kernel 4) + silu
        conv_state = np.roll(conv_state, -1, axis=1)
        conv_state[:, -1] = qkv
        qkv = (conv_state * w["conv"].T).sum(-1)
        qkv = qkv / (1.0 + np.exp(-qkv))      # silu
        q = qkv[:KEY_DIM].reshape(NUM_K_HEADS, HEAD_K)
        k = qkv[KEY_DIM:2 * KEY_DIM].reshape(NUM_K_HEADS, HEAD_K)
        v = qkv[2 * KEY_DIM:].reshape(NUM_V_HEADS, HEAD_V)
        a = w["alpha"].T @ x                  # [32]
        b = w["beta"].T @ x                   # [32]
        beta = 1.0 / (1.0 + np.exp(-b))
        g = A * softplus(a + w["dt_bias"])    # [32]
        rep = NUM_V_HEADS // NUM_K_HEADS
        q = l2norm(np.repeat(q, rep, axis=0))
        k = l2norm(np.repeat(k, rep, axis=0))
        # recurrent gated delta rule
        eg = np.exp(g)[:, None, None]
        state = state * eg
        kv_mem = (state * k[:, :, None]).sum(axis=1)          # [32,128]
        delta = (v - kv_mem) * beta[:, None]
        state = state + k[:, :, None] * delta[:, None, :]
        core = (state * q[:, :, None]).sum(axis=1) / np.sqrt(HEAD_K)
        # gated RMSNorm (per v-head, head_v_dim) then out_proj
        zz = z.reshape(NUM_V_HEADS, HEAD_V)
        var = (core * core).mean(-1, keepdims=True)
        core = core * (1.0 / np.sqrt(var + EPS)) * w["norm"]
        core = core * (zz / (1.0 + np.exp(-zz)))
        outs.append(w["out"] @ core.reshape(-1))
    return np.array(outs), g, beta, state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--layer-type", default="gdn", choices=["gdn", "full"])
    ap.add_argument("--probe-layouts", action="store_true")
    ap.add_argument("--dump")
    ap.add_argument("--dump-bin", help="raw f32 golden for the C++ probe")
    args = ap.parse_args()

    m = Q4nx(MODEL)
    rng = np.random.default_rng(42)
    xs = rng.standard_normal((3, H)) * 0.5

    if args.probe_layouts:
        print("BF16 layout: higher column-norm CV = less scrambled = correct.")
        print("Control is moe_router, known-good interleaved.\n")
        rows = [("moe_router", "model.layer.0.moe_router.weight", 256),
                ("ssm_alpha_proj", "model.layer.0.linear_attn.ssm_alpha_proj.weight", 32),
                ("ssm_beta_proj", "model.layer.0.linear_attn.ssm_beta_proj.weight", 32)]
        print(f"{'tensor':16} {'interleaved CV':>15} {'plain CV':>10}   verdict")
        for name, key, _ in rows:
            cvs = {}
            for il in (True, False):
                W = m.bf16_2d(key, il)
                n = np.linalg.norm(W, axis=0)
                cvs[il] = n.std() / n.mean()
            verdict = "interleaved" if cvs[True] > cvs[False] else "plain"
            print(f"{name:16} {cvs[True]:15.4f} {cvs[False]:10.4f}   {verdict}")

        print("\nssm_a convention: A ~ uniform(0,16) at init, so raw A_log would be")
        print("mostly positive; already-negated A is all-negative in [-16,0).")
        a = np.concatenate([m.f32(f"model.layer.{l}.linear_attn.ssm_a")[0]
                            for l in gdn_layers(m)[:5]])
        print(f"  n={len(a)}  all_negative={bool((a < 0).all())}  "
              f"in[-16,0)={int(((a >= -16) & (a < 0)).sum())}/{len(a)}"
              f"  -> ssm_a is already -A (use directly)")
        return

    if args.layer_type == "full":
        out, qq, gate, xs = full_attn_forward(m, args.layer)
        print(f"layer {args.layer} (full-attn): out[0,:6] = {np.round(out[-1][:6], 5)}")
        if args.dump:
            np.savez(args.dump, xs=xs, out=out, **{f"w_{k}": v for k, v in w.items()})
        return

    w = load_layer(m, args.layer)
    out, g, beta, state = gdn_forward(w, xs, negate_a=False)
    print(f"layer {args.layer}: out[0,:6] = {np.round(out[-1][:6], 5)}")
    print(f"  exp(g): min={np.exp(g).min():.4f} max={np.exp(g).max():.4f}")
    if args.dump:
        np.savez(args.dump, xs=xs, out=out, g=g, beta=beta,
                 state=state, **{f"w_{k}": v for k, v in w.items()})
        print(f"wrote {args.dump}")
    if args.dump_bin:
        # header: int32 layer, T, H, NUM_V_HEADS  then xs[T,H], out[T,H], g[32], beta[32]
        with open(args.dump_bin, "wb") as f:
            f.write(struct.pack("<4i", args.layer, xs.shape[0], H, NUM_V_HEADS))
            for arr in (xs, out, g, beta):
                f.write(np.ascontiguousarray(arr, dtype=np.float32).tobytes())
        print(f"wrote {args.dump_bin}")


if __name__ == "__main__":
    main()
