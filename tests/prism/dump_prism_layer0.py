#!/usr/bin/env python3
"""numpy counterpart of tests/prism/prism_layer0.cpp — first Bonsai-27B block.

Reads the same converted .1bp, uses Prism's own vendored transcode/unpack for the
quantised weights (so the decode path here is not our C++ one), and prints the same
diagnostic lines for compare_prism_layer0.py to check.

Single-token simplification, mirrored from the C++ side: the causal conv1d sees only
the current token (the recurrent conv history is zero), i.e. only the k=3 tap fires.

Usage: python3 dump_prism_layer0.py <model.1bp> [token_id]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import struct  # noqa: E402

from vendor_prism_codec import transcode, unpack  # noqa: E402
from verify_prism_1bp import parse_1bp  # noqa: E402

NK, NV, HK, HV = 16, 48, 128, 128
KD, VD, CD = NK * HK, NV * HV, 2 * NK * HK + NV * HV
QSIZE = {"Q1_0": 18, "PQ2_0": 34, "PTQ1_0": 28}
QNAME = {11: "Q1_0", 12: "PQ2_0", 13: "PTQ1_0"}


class Bp:
    def __init__(self, path: str) -> None:
        self.path = path
        self.hdr, self.entries, self.data_start = parse_1bp(path)
        self.by = {e["name"]: e for e in self.entries}
        self.f = open(path, "rb")
        et, es = self.by["__onebp_ext_prism_transform"], self.by["__onebp_ext_prism_signs"]
        self.f.seek(self.data_start + et["offset"]); blob = self.f.read(et["bytes"])
        self.f.seek(self.data_start + es["offset"]); self.sign_raw = self.f.read(es["bytes"])
        (magic, ver, kind, axis, self.block, self.grouped, nw, self.sign_count, nfold, ninv, _f, _r) = \
            struct.unpack_from("<12I", blob, 0)
        widths = struct.unpack_from("<%dI" % (2 * nw), blob, 48)
        self.widths = [(widths[2 * i], widths[2 * i + 1]) for i in range(nw)]
        self.signs = np.frombuffer(self.sign_raw, dtype=np.int8).astype(np.float32)

    def signs_for(self, width: int) -> np.ndarray:
        for w, off in self.widths:
            if w == width:
                return self.signs[off:off + w]
        raise KeyError(width)

    def raw(self, name: str, rows: int | None = None) -> bytes:
        e = self.by[name]
        nb = QSIZE[QNAME[e["quant"]]]
        cols = int(e["dims"][1])
        row_bytes = cols // 128 * nb
        self.f.seek(self.data_start + e["offset"])
        return self.f.read(row_bytes * (rows if rows else 1))

    def w(self, name: str) -> np.ndarray:
        """Full 2-D weight as float32 [rows, cols].

        Prism packings go through Prism's own transcode/unpack; aux tensors are F16
        tiles (the writer had no F32 tile branch — see the plan's P1.3 note).
        """
        e = self.by[name]
        if e["quant"] == 4:
            return self.f16_tile(name)
        if e["quant"] == 5:
            self.f.seek(self.data_start + e["offset"])
            rows, cols = int(e["dims"][0]), int(e["dims"][1])
            return np.frombuffer(self.f.read(rows * cols * 4), dtype=np.float32).copy().reshape(rows, cols)
        rows, cols = int(e["dims"][0]), int(e["dims"][1])
        words, scales, biases = transcode(self.raw(name, rows), (rows, cols), QNAME[e["quant"]])
        return unpack(words, scales, biases).astype(np.float32)

    def row(self, name: str, r: int) -> np.ndarray:
        e = self.by[name]
        cols = int(e["dims"][1])
        nb = QSIZE[QNAME[e["quant"]]]
        self.f.seek(self.data_start + e["offset"] + r * (cols // 128) * nb)
        raw = self.f.read(cols // 128 * nb)
        return unpack(*transcode(raw, (1, cols), QNAME[e["quant"]])).astype(np.float32)[0]

    def vec(self, name: str) -> np.ndarray:
        """1-D tensor: raw float32, `bytes` long.

        The quant field of an ndim==1 entry is advisory — the existing converter
        labels 1-D tensors with the *requested* quant (F16 by default) while writing
        raw f32, exactly as OnebpModel's ndim==1 branch assumes. The only ndim==1
        entries whose bytes are not f32 are the `__onebp_ext_*` metadata entries.
        """
        e = self.by[name]
        assert not name.startswith("__onebp_ext_"), name
        assert e["bytes"] == 4 * e["dims"][0], f"{name}: {e['bytes']} != 4*{e['dims'][0]}"
        self.f.seek(self.data_start + e["offset"])
        return np.frombuffer(self.f.read(e["bytes"]), dtype=np.float32).copy()

    def f16_tile(self, name: str) -> np.ndarray:
        """Whole 2-D F16-tiled aux tensor as [rows, cols]."""
        e = self.by[name]
        assert e["quant"] == 4, f"{name} is not F16 (quant {e['quant']})"
        rows, cols = int(e["dims"][0]), int(e["dims"][1])
        ntr, ntc = (rows + 31) // 32, (cols + 255) // 256
        self.f.seek(self.data_start + e["offset"])
        tiles = np.frombuffer(self.f.read(ntr * ntc * 32 * 256 * 2), dtype="<f2").astype(np.float32)
        tiles = tiles.reshape(ntr, ntc, 32, 256)
        out = np.zeros((ntr * 32, ntc * 256), dtype=np.float32)
        for tr in range(ntr):
            for tc in range(ntc):
                out[tr * 32:(tr + 1) * 32, tc * 256:(tc + 1) * 256] = tiles[tr, tc]
        return out[:rows, :cols]

    def hadamard_forward(self, x: np.ndarray) -> np.ndarray:
        sg = self.signs_for(len(x))
        out = np.empty_like(x)
        for base in range(0, len(x), self.block):
            v = (x[base:base + self.block] * sg[base:base + self.block]).astype(np.float64).copy()
            n, ln = self.block, 1
            while ln < n:
                for i in range(0, n, 2 * ln):
                    a = v[i:i + ln].copy(); b = v[i + ln:i + 2 * ln].copy()
                    v[i:i + ln] = a + b; v[i + ln:i + 2 * ln] = a - b
                ln *= 2
            out[base:base + self.block] = (v / np.sqrt(self.block)).astype(np.float32)
        return out

    def hadamard_inverse(self, x: np.ndarray) -> np.ndarray:
        sg = self.signs_for(len(x))
        out = np.empty_like(x)
        for base in range(0, len(x), self.block):
            v = x[base:base + self.block].astype(np.float64).copy()
            n, ln = self.block, 1
            while ln < n:
                for i in range(0, n, 2 * ln):
                    a = v[i:i + ln].copy(); b = v[i + ln:i + 2 * ln].copy()
                    v[i:i + ln] = a + b; v[i + ln:i + 2 * ln] = a - b
                ln *= 2
            out[base:base + self.block] = ((v / np.sqrt(self.block)) * sg[base:base + self.block]).astype(np.float32)
        return out


def rmsnorm_1pw(x: np.ndarray, w: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """GGUF/llama.cpp convention: PLAIN weights (LLM_NORM_RMS), i.e. y = x/rms * w.

    The HF convention (1 + w) — which our in-repo FP32 reference uses — is wrong for
    these packs and was the root cause of the P2.3 divergence (all 64 layers inflated).
    """
    r = 1.0 / np.sqrt(np.mean(x.astype(np.float64) ** 2) + eps)
    return (x * r * w).astype(np.float32)


def l2(v: np.ndarray) -> float:
    return float(np.sqrt(np.sum(v.astype(np.float64) ** 2)))


def silu(x):
    return x / (1.0 + np.exp(-x))


def main() -> int:
    path = sys.argv[1]
    token = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    bp = Bp(path)
    print(f"token={token} H={bp.hdr['hidden']} NK={NK} NV={NV} CD={CD} block={bp.block}")

    h = bp.hadamard_inverse(bp.row("token_embd.weight", token))
    print(f"h_embed_l2 {l2(h):.9e}")

    x = rmsnorm_1pw(h, bp.vec("blk.0.attn_norm.weight"))
    print(f"xnorm_l2 {l2(x):.9e}")
    xr = bp.hadamard_forward(x)

    qkv = bp.w("blk.0.attn_qkv.weight") @ xr
    print(f"qkv_l2 {l2(qkv):.9e}")
    z = bp.w("blk.0.attn_gate.weight") @ xr
    a = bp.w("blk.0.ssm_alpha.weight") @ xr
    b = bp.w("blk.0.ssm_beta.weight") @ xr

    cv = bp.f16_tile("blk.0.ssm_conv1d.weight")
    conv = silu(cv[:, 3] * qkv)            # single-token: only the newest tap
    print(f"conv_l2 {l2(conv):.9e}")

    q = conv[:KD].reshape(NK, HK).copy()
    k = conv[KD:2 * KD].reshape(NK, HK).copy()
    v = conv[2 * KD:].reshape(NV, HV)
    q /= np.sqrt(np.sum(q.astype(np.float64) ** 2, axis=1, keepdims=True) + 1e-6)
    k /= np.sqrt(np.sum(k.astype(np.float64) ** 2, axis=1, keepdims=True) + 1e-6)

    A = bp.vec("blk.0.ssm_a")
    dt = bp.vec("blk.0.ssm_dt.bias")
    g = A * np.log1p(np.exp(a + dt))
    beta = 1.0 / (1.0 + np.exp(-b))

    state = np.zeros((NV, HK, HV), dtype=np.float64)
    core = np.zeros((NV, HV), dtype=np.float32)
    nw = bp.vec("blk.0.ssm_norm.weight")
    zc = z.reshape(NV, HV)
    for vh in range(NV):
        kh = vh % NK
        state[vh] *= np.exp(g[vh])
        kv_mem = state[vh].T @ k[kh]                      # [HV]
        delta = (v[vh] - kv_mem) * beta[vh]
        state[vh] += np.outer(k[kh], delta)
        c = (state[vh].T @ q[kh]) / np.sqrt(HK)
        r = 1.0 / np.sqrt(np.mean(c ** 2) + 1e-6)
        core[vh] = (c * r * nw * silu(zc[vh])).astype(np.float32)   # RAW weight, not (1+w)
    print(f"state_l2 {l2(state.reshape(-1)):.9e}")
    print(f"core_l2 {l2(core.reshape(-1)):.9e}")

    crf = core.reshape(-1)
    rep = NV // NK
    pm = np.empty_like(crf)
    for rr in range(rep):
        for kk in range(NK):
            for hd in range(HV):
                pm[hd + HV * rr + HV * rep * kk] = crf[hd + HV * kk + HV * NK * rr]
    out = bp.w("blk.0.ssm_out.weight") @ bp.hadamard_forward(pm)
    print(f"gdn_out_l2 {l2(out):.9e}")
    h = h + out

    y = rmsnorm_1pw(h, bp.vec("blk.0.post_attention_norm.weight"))
    yr = bp.hadamard_forward(y)
    gate = silu(bp.w("blk.0.ffn_gate.weight") @ yr) * (bp.w("blk.0.ffn_up.weight") @ yr)
    down = bp.w("blk.0.ffn_down.weight") @ bp.hadamard_forward(gate)
    h = h + down

    print(f"layer_out_l2 {l2(h):.9e}")
    print("layer_out[0..7]" + "".join(f" {v:.9e}" for v in h[:8]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
