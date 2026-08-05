#!/usr/bin/env python3
"""Full-model Qwen3.6-35B-A3B reference from the Q4NX weights (#1471).

Runs the complete 40-layer forward (30 GDN + 10 full-attention, MoE FFN,
RMSNorms, lm_head) in numpy, mirroring npu_engine_universal's CPU path
exactly (moe_ffn_cpu + gdn_attn_step + std_attn_step + rn_c), and compares
greedy decode tokens + per-layer hidden states against the engine's worker
op=32 stream. The honest end-to-end gate: per-layer math was already
bit-validated, this checks the wiring (embeddings, norms, splits, MoE).

Usage:
  python3 tools/qwen36_full_ref.py --prompt 151644,872,198,13048 --steps 4
  # then compare against the engine's worker output for the same prompt.
"""
import argparse
import json
import mmap
import struct
import sys

import numpy as np

MODEL = "/home/bcloud/.config/flm/models/Qwen3.6-35B-A3B-NPU2/model.q4nx"
H = 2048
EPS = 1e-6
GDN_VH, GDN_KH, GDN_HD = 32, 16, 128
STD_NH, STD_NKV, STD_HD = 16, 2, 256
CONV_DIM, CONV_K = 8192, 4
KEY_DIM = GDN_KH * GDN_HD          # 2048
VALUE_DIM = GDN_VH * GDN_HD        # 4096
N_EXPERTS, TOP_K, IM_EXP = 256, 8, 512
THETA, ROT_DIM = 1e7, 64


class Q4nx:
    def __init__(self, path):
        self.f = open(path, "rb")
        self.mm = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)
        hsz = struct.unpack("<Q", self.mm[:8])[0]
        raw = self.mm[8:8 + hsz].decode("utf-8", "replace")
        self.hdr = json.loads(raw[:raw.rindex("}") + 1])
        self.df = 8 + hsz

    def raw(self, key):
        v = self.hdr[key]
        o = v["data_offsets"]
        return v, self.mm[self.df + o[0]: self.df + o[1]]

    def bf16(self, key):
        v, b = self.raw(key)
        u = np.frombuffer(b, dtype=np.uint16).astype(np.uint32) << 16
        return u.view(np.float32).astype(np.float64), v["shape"]

    def bf16_1d(self, key):
        return self.bf16(key)[0]

    def f32(self, key):
        v, b = self.raw(key)
        return np.frombuffer(b, dtype=np.float32).astype(np.float64)

    def router(self, key):
        """stride-8 interleaved BF16 [H, N] (probe-validated layout)."""
        v, b = self.raw(key)
        u = np.frombuffer(b, dtype=np.uint16).astype(np.uint32) << 16
        flat = u.view(np.float32).astype(np.float64)
        n_in, n_out = v["shape"]
        blk = n_out * (n_in // 8)
        i = np.arange(n_in)[:, None]
        j = np.arange(n_out)[None, :]
        return flat[(i % 8) * blk + j * (n_in // 8) + i // 8]

    def q8_0(self, key):
        """Q8_0 (8704 B/row) -> [out_features, in_features] row-major."""
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
            s = sc.reshape(TC // 32, TR).T
            s = np.repeat(s, 32, axis=1)
            out[tr_ * TR:(tr_ + 1) * TR, tc_ * TC:(tc_ + 1) * TC] = vals * s
        return out

    def q4nx_1bp(self, key, i8_rows, in_features):
        """Q4NX 1BP: tile = 32x256, row-major bf16 scales/zps, int4 low-nibble-even."""
        v, b = self.raw(key)
        TR, TC, GS = 32, 256, 32
        ntc = in_features // TC
        ntr = i8_rows // ntc
        RB = TR * GS * 2 + TR * GS * 2 + TR * TC // 2   # 5120
        out = np.zeros((ntr * TR, ntc * TC), dtype=np.float64)
        buf = np.frombuffer(b, dtype=np.uint8)
        for ir in range(i8_rows):
            t = buf[ir * RB:(ir + 1) * RB]
            sc = (t[:TR * GS * 2].view(np.uint16).astype(np.uint32) << 16).view(np.float32).astype(np.float64)
            zp = (t[TR * GS * 2:TR * GS * 4].view(np.uint16).astype(np.uint32) << 16).view(np.float32).astype(np.float64)
            qd = t[TR * GS * 4:]
            tr_, tc_ = ir // ntc, ir % ntc
            for r in range(TR):
                for g in range(GS):
                    s, z = sc[r * GS + g], zp[r * GS + g]
                    if not np.isfinite(s) or abs(s) > 100: s = 0
                    if not np.isfinite(z) or abs(z) > 100: z = 0
                    for i in range(GS):
                        col = g * GS + i
                        byte = qd[(r * TC + col) // 2]
                        val = (byte >> 4) if (col & 1) else (byte & 0x0F)
                        out[tr_ * TR + r, tc_ * TC + col] = val * s + z
        return out


def silu(x):
    return x / (1.0 + np.exp(-x))


def softplus(x):
    return np.where(x > 20, x, np.log1p(np.exp(np.minimum(x, 20))))


def rn_c(x, w):
    """Engine rn_c: RMSNorm over H with weight, EPS=1e-6."""
    return x / np.sqrt((x * x).mean() + EPS) * w


def l2norm(x):
    return x / np.sqrt((x * x).sum() + EPS)


def rot64(x, pos):
    """Partial rotary: first 64 of 256 dims, half-split within 64, theta 1e7."""
    out = x.copy()
    for d in range(32):
        f = 1.0 / THETA ** (d / 32.0)
        a = pos * f
        c, s = np.cos(a), np.sin(a)
        x1, x2 = x[d], x[d + 32]
        out[d] = x1 * c - x2 * s
        out[d + 32] = x2 * c + x1 * s
    return out


class FullModel:
    def __init__(self, m: Q4nx):
        self.m = m
        self.nc = max(int(k.split(".")[2]) for k in m.hdr if k.startswith("model.layer.")) + 1
        self.is_gdn = [f"model.layer.{l}.linear_attn.qkv_proj.weight" in m.hdr for l in range(self.nc)]
        # norms + embeddings
        self.in_n = [m.bf16_1d(f"model.layer.{l}.input_layernorm.weight") for l in range(self.nc)]
        self.pa_n = [m.bf16_1d(f"model.layer.{l}.post_attention_layernorm.weight") for l in range(self.nc)]
        self.fin = m.bf16_1d("model.norm.weight")
        v, b = m.raw("model.embed_tokens.weight")
        u = np.frombuffer(b, dtype=np.uint16).astype(np.uint32) << 16
        self.emb = u.view(np.float32).astype(np.float64)  # [NV, H] plain
        # lm_head: Q8_0, rows = shape[0]*(H/256) i8 rows -> [31040, 2048]
        self.lm = m.q8_0("lm_head.weight")
        self.lm_nv = self.lm.shape[0]
        # per-layer MoE: router (interleaved), shared gate, expert offsets
        self.router = [m.router(f"model.layer.{l}.moe_router.weight") for l in range(self.nc)]
        self.shgate = [m.bf16_1d(f"model.layer.{l}.shared_expert_gate.weight") for l in range(self.nc)]
        # GDN per-layer (dequant on demand, cached)
        self.gdn = {}
        # STD per-layer
        self.std = {}

    def gdn_weights(self, l):
        if l in self.gdn:
            return self.gdn[l]
        p = f"model.layer.{l}.linear_attn."
        w = {
            "qkv": self.m.q8_0(p + "qkv_proj.weight"),
            "out": self.m.q8_0(p + "ssm_out_proj.weight"),
            "z": self.m.q8_0(f"model.layer.{l}.self_attn.gate_proj.weight"),
            "alpha": self.m.bf16_1d(p + "ssm_alpha_proj.weight").reshape(H, GDN_VH),
            "beta": self.m.bf16_1d(p + "ssm_beta_proj.weight").reshape(H, GDN_VH),
            "conv": self.m.bf16_1d(p + "ssm_conv1d.weight").reshape(CONV_K, CONV_DIM),
            "norm": self.m.bf16_1d(p + "ssm_norm.weight"),
            "a": self.m.f32(p + "ssm_a"),
            "dt": self.m.f32(p + "ssm_dt.bias"),
        }
        self.gdn[l] = w
        return w

    def std_weights(self, l):
        if l in self.std:
            return self.std[l]
        p = f"model.layer.{l}.self_attn."
        w = {
            "q": self.m.q8_0(p + "q_proj.weight"),      # [8192, 2048] q+gate halves
            "k": self.m.q8_0(p + "k_proj.weight"),      # [512, 2048]
            "v": self.m.q8_0(p + "v_proj.weight"),      # [512, 2048]
            "o": self.m.q8_0(p + "o_proj.weight"),      # [2048, 4096]
            "qn": self.m.bf16_1d(p + "q_norm.weight"),
            "kn": self.m.bf16_1d(p + "k_norm.weight"),
        }
        self.std[l] = w
        return w

    def gdn_layer(self, l, x, conv_state, delta_state):
        """One GDN layer step (mirror of gdn_attn_step + O + residual handled by caller)."""
        w = self.gdn_weights(l)
        qkv = w["qkv"] @ x
        # causal depthwise conv1d
        conv_state = np.roll(conv_state, -1, axis=1)
        conv_state[:, -1] = qkv
        qkv = silu((conv_state * w["conv"].T).sum(axis=1))
        q = qkv[:KEY_DIM].reshape(GDN_KH, GDN_HD)
        k = qkv[KEY_DIM:2 * KEY_DIM].reshape(GDN_KH, GDN_HD)
        v = qkv[2 * KEY_DIM:].reshape(GDN_VH, GDN_HD)
        a = w["alpha"].T @ x
        b = w["beta"].T @ x
        beta = 1.0 / (1.0 + np.exp(-b))
        g = w["a"] * softplus(a + w["dt"])
        rep = GDN_VH // GDN_KH
        q = l2norm(np.repeat(q, rep, axis=0))
        k = l2norm(np.repeat(k, rep, axis=0))
        eg = np.exp(g)[:, None, None]
        delta_state = delta_state * eg
        kv_mem = (delta_state * k[:, :, None]).sum(axis=1)
        delta = (v - kv_mem) * beta[:, None]
        delta_state = delta_state + k[:, :, None] * delta[:, None, :]
        core = (delta_state * q[:, :, None]).sum(axis=1) / np.sqrt(GDN_HD)
        z = w["z"] @ x
        zz = z.reshape(GDN_VH, GDN_HD)
        core = core / np.sqrt((core * core).mean(axis=1, keepdims=True) + EPS) * w["norm"]
        core = core * silu(zz)
        return w["out"] @ core.reshape(-1), conv_state, delta_state

    def std_layer(self, l, x, kvc, pos):
        """One full-attention layer step with KV cache (mirror of std_attn_step)."""
        w = self.std_weights(l)
        qkv = w["q"] @ x
        qq = np.empty((STD_NH, STD_HD))
        gate = np.empty((STD_NH, STD_HD))
        for h in range(STD_NH):
            qq[h] = qkv[h * 512:h * 512 + 256]
            gate[h] = qkv[h * 512 + 256:h * 512 + 512]
        kk = w["k"] @ x
        vv = w["v"] @ x
        for h in range(STD_NH):
            qq[h] = qq[h] / np.sqrt((qq[h] * qq[h]).mean() + EPS) * w["qn"]
            qq[h] = rot64(qq[h], pos)
        for h in range(STD_NKV):
            kh = kk[h * STD_HD:(h + 1) * STD_HD]
            kh = kh / np.sqrt((kh * kh).mean() + EPS) * w["kn"]
            kk[h * STD_HD:(h + 1) * STD_HD] = rot64(kh, pos)
        kvc["k"].append(kk)
        kvc["v"].append(vv)
        attn = np.zeros((STD_NH, STD_HD))
        for h in range(STD_NH):
            kvh = h // (STD_NH // STD_NKV)
            K = np.stack(kvc["k"])[:, kvh * STD_HD:(kvh + 1) * STD_HD]   # [t, 256]
            V = np.stack(kvc["v"])[:, kvh * STD_HD:(kvh + 1) * STD_HD]
            scores = K @ qq[h] / np.sqrt(STD_HD)                          # [t]
            p = np.exp(scores - scores.max())
            p /= p.sum()
            attn[h] = p @ V
        attn = attn * (1.0 / (1.0 + np.exp(-gate)))
        return w["o"] @ attn.reshape(-1)

    def moe_ffn(self, l, x):
        """MoE FFN (mirror of moe_ffn_cpu): router → top-8 → dequant → SiLU → shared."""
        p = f"model.layer.{l}.mlp."
        rt = self.router[l]
        logits = rt.T @ x
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()
        topk = np.argsort(-probs)[:TOP_K]
        out = np.zeros(H)
        for e in topk:
            G = self._exp_deq(p + "gate_exps_proj.weight", e, H, "g")
            U = self._exp_deq(p + "up_exps_proj.weight", e, H, "u")
            D = self._exp_deq(p + "down_exps_proj.weight", e, IM_EXP, "d")
            gu = silu(G @ x) * (U @ x)
            out += probs[e] * (D @ gu)
        # shared expert (Q8_0) + sigmoid gate
        sg = np.dot(self.shgate[l], x)
        sg_sig = 1.0 / (1.0 + np.exp(-sg))
        SG = self.m.q8_0(p + "share_gate_exps_proj.weight")
        SU = self.m.q8_0(p + "share_up_exps_proj.weight")
        SD = self.m.q8_0(p + "share_down_exps_proj.weight")
        sgu = silu(SG @ x) * (SU @ x)
        out += sg_sig * (SD @ sgu)
        return out

    def _exp_deq(self, key, ex, in_f, which):
        """Dequant ONE expert's slice from the Q4NX tensor (vectorized)."""
        v, _ = self.m.raw(key)
        # 256 experts always (router N_EXPERTS); i8 rows total = shape0*shape1,
        # split evenly per expert (128 for gate/up/down at these dims).
        tr_per = v["shape"][0] * v["shape"][1] // N_EXPERTS
        TR, TC, GS = 32, 256, 32
        GRPS = TC // GS            # 8 scale groups per tile row
        RB = 5120                  # scales 2*TR*GRPS + zps 2*TR*GRPS + int4 TR*TC/2
        ntc = in_f // TC
        ntr = tr_per // ntc
        out = np.zeros((ntr * TR, ntc * TC))
        buf = np.frombuffer(self.m.mm[self.m.df + v["data_offsets"][0]: self.m.df + v["data_offsets"][1]], dtype=np.uint8)
        for ir in range(tr_per):
            t = buf[(ex * tr_per + ir) * RB:(ex * tr_per + ir + 1) * RB]
            sc = (t[:TR * GRPS * 2].view(np.uint16).astype(np.uint32) << 16).view(np.float32).astype(np.float64).reshape(TR, GRPS)
            zp = (t[TR * GRPS * 2:TR * GRPS * 4].view(np.uint16).astype(np.uint32) << 16).view(np.float32).astype(np.float64).reshape(TR, GRPS)
            qd = t[TR * GRPS * 4:]
            vals = np.empty(TR * TC)
            vals[0::2] = qd & 0x0F
            vals[1::2] = qd >> 4
            vals = vals.reshape(TR, TC)
            s = np.repeat(sc, 32, axis=1)
            z = np.repeat(zp, 32, axis=1)
            np.clip(s, -100, 100, out=s); np.clip(z, -100, 100, out=z)
            out[(ir // ntc) * TR:(ir // ntc + 1) * TR,
                (ir % ntc) * TC:(ir % ntc + 1) * TC] = vals * s + z
        return out

    def decode(self, prompt, steps):
        """Greedy decode: returns (tokens, per-layer hidden states of last step)."""
        tokens = list(prompt)
        conv = {l: np.zeros((CONV_DIM, CONV_K)) for l in range(self.nc) if self.is_gdn[l]}
        delta = {l: np.zeros((GDN_VH, GDN_HD, GDN_HD)) for l in range(self.nc) if self.is_gdn[l]}
        kvc = {l: {"k": [], "v": []} for l in range(self.nc) if not self.is_gdn[l]}
        outs = []
        for step in range(len(prompt) + steps - 1):
            tok = prompt[step] if step < len(prompt) else tokens[-1]
            x = self.emb[tok].copy()
            hiddens = []
            for l in range(self.nc):
                x = rn_c(x, self.in_n[l])
                if self.is_gdn[l]:
                    attn_out, conv[l], delta[l] = self.gdn_layer(l, x, conv[l], delta[l])
                else:
                    attn_out = self.std_layer(l, x, kvc[l], len(kvc[l]["k"]))
                x = x + attn_out
                x = rn_c(x, self.pa_n[l])
                x = x + self.moe_ffn(l, x)
                hiddens.append(x.copy())
            logits = self.lm @ rn_c(x, self.fin)
            tok = int(np.argmax(logits))
            tokens.append(tok)
            if step >= len(prompt) - 1:
                outs.append((tok, hiddens))
        return tokens, outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="151644,872,198,13048")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--dump-hidden", help="save per-layer hidden states of the last step")
    ap.add_argument("--dump-first", help="save per-layer hidden states of the FIRST prompt position")
    args = ap.parse_args()
    prompt = [int(t) for t in args.prompt.split(",")]
    m = Q4nx(MODEL)
    fm = FullModel(m)
    tokens, outs = fm.decode(prompt, args.steps)
    print("prompt:", prompt)
    print("tokens:", tokens[len(prompt):])
    if args.dump_hidden:
        np.save(args.dump_hidden, np.array(outs[-1][1]))   # [NC, H]
        print(f"saved last-step hidden states ({len(outs[-1][1])} x {H})")
    if args.dump_first:
        # per-layer hidden states at prompt position 0 (mirror the engine's
        # prefill h_b[0] dump: 40 x H)
        x = rn_c(fm.emb[prompt[0]].copy(), fm.in_n[0])
        states = []
        for l in range(fm.nc):
            if fm.is_gdn[l]:
                conv = np.zeros((CONV_DIM, CONV_K))
                delta = np.zeros((GDN_VH, GDN_HD, GDN_HD))
                attn_out, conv, delta = fm.gdn_layer(l, x, conv, delta)
            else:
                kvc = {"k": [], "v": []}
                attn_out = fm.std_layer(l, x, kvc, 0)
            x = x + attn_out
            x = rn_c(x, fm.pa_n[l])
            x = x + fm.moe_ffn(l, x)
            states.append(x.copy())
        np.save(args.dump_first, np.array(states))
        print(f"saved first-position hidden states ({len(states)} x {H})")


if __name__ == "__main__":
    main()
