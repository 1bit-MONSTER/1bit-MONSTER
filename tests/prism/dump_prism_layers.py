#!/usr/bin/env python3
"""Full 64-layer numpy reference for the Prism packs — the P2 per-layer cosine gate.

Companion to tests/prism/prism_forward.cpp (our C++ low-bit forward). Both read the
same .1bp and, for one token, write the hidden state after each of the 64 layers as
raw float32 records (H floats each). compare_prism_layers.py then requires the
per-layer cosine to be >= 0.999, so a single wrong layer cannot hide behind a final
token match.

Why this is the reference: the in-repo C++ FP32 reference (src/qwen3_5.cpp) holds
every weight as dense f32 — 27B params = 108 GB, more than this box's unified budget
— and cannot read the *packed* Prism weights at all. This reference streams the same
1BP instead, and decodes it with Prism's own vendored transcode/unpack (so the decode
path is not our C++ one). Single-token by construction: the causal conv history and
the gated-delta / KV state are fresh, which is exactly the C++ forward's step 0.

Usage: python3 dump_prism_layers.py <model.1bp> <token_id> <out.bin>
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from vendor_prism_codec import transcode, unpack  # noqa: E402
from verify_prism_1bp import parse_1bp  # noqa: E402
from dump_prism_layer0 import rmsnorm_1pw, silu  # noqa: E402

H, NL, NH, NKV, HD = 5120, 64, 24, 4, 256
NK, NV, HK, HV = 16, 48, 128, 128
KD, VD, CD = NK * HK, NV * HV, 2 * NK * HK + NV * HV
ROPE = 64
QSIZE = {"Q1_0": 18, "PQ2_0": 34, "PTQ1_0": 28}
QNAME = {11: "Q1_0", 12: "PQ2_0", 13: "PTQ1_0"}


class Bp:
    def __init__(self, path: str) -> None:
        self.hdr, self.entries, self.data_start = parse_1bp(path)
        self.by = {e["name"]: e for e in self.entries}
        self.f = open(path, "rb")
        et = self.by.get("__onebp_ext_prism_transform")
        es = self.by.get("__onebp_ext_prism_signs")
        self.has_transform = et is not None and es is not None
        if self.has_transform:
            self.f.seek(self.data_start + et["offset"]); blob = self.f.read(et["bytes"])
            self.f.seek(self.data_start + es["offset"]); raw = self.f.read(es["bytes"])
            (magic, ver, kind, axis, self.block, self.grouped, nw, self.sign_count,
             _nfold, _ninv, _f, _r) = struct.unpack_from("<12I", blob, 0)
            widths = struct.unpack_from("<%dI" % (2 * nw), blob, 48)
            self.widths = [(widths[2 * i], widths[2 * i + 1]) for i in range(nw)]
            self.signs = np.frombuffer(raw, dtype=np.int8).astype(np.float32)
        else:
            self.block = 0
            self.signs = None

    def _resolve(self, e: dict) -> dict:
        # v4 dedup: an entry with bytes==0 is an alias whose offset is the INDEX of an
        # earlier tensor it shares data with. OnebpModel resolves these; so must we.
        hops = 0
        while e["bytes"] == 0:
            e = self.entries[e["offset"]]
            hops += 1
            if hops > 4096:
                raise RuntimeError("alias loop")
        return e

    def signs_for(self, width: int) -> np.ndarray:
        for w, off in self.widths:
            if w == width:
                return self.signs[off:off + w]
        raise KeyError(width)

    def _raw(self, name: str, rows: int) -> bytes:
        e = self._resolve(self.by[name])
        nb = QSIZE[QNAME[e["quant"]]]
        cols = int(e["dims"][1])
        rows = rows if rows else int(e["dims"][0])
        self.f.seek(self.data_start + e["offset"])
        return self.f.read(rows * (cols // 128) * nb)

    def _decode(self, raw: bytes, rows: int, cols: int, qname: str) -> np.ndarray:
        if qname == "Q1_0":
            # llama.cpp Q1_0: [fp16 d][16 B sign bits], bit l of byte il -> elem 8*il+l.
            # Prism's transcode() only knows the MLX-affine packs, so decode this one directly.
            nb, nblk = 18, cols // 128
            b = np.frombuffer(raw, dtype=np.uint8).reshape(rows, nblk, nb)
            d = (b[:, :, 0].astype(np.uint16) | (b[:, :, 1].astype(np.uint16) << 8)).view("<f2").astype(np.float32)
            bits = np.unpackbits(b[:, :, 2:18], axis=2, bitorder="little").astype(np.float32) * 2.0 - 1.0
            return (bits * d[:, :, None]).reshape(rows, cols)
        words, scales, biases = transcode(raw, (rows, cols), qname)
        return unpack(words, scales, biases).astype(np.float32)

    def w(self, name: str) -> np.ndarray:
        e = self._resolve(self.by[name])
        if e["quant"] == 4:
            return self._f16_tile(name)
        rows, cols = int(e["dims"][0]), int(e["dims"][1])
        return self._decode(self._raw(name, rows), rows, cols, QNAME[e["quant"]])

    def row(self, name: str, r: int) -> np.ndarray:
        e = self._resolve(self.by[name])
        cols = int(e["dims"][1])
        nb = QSIZE[QNAME[e["quant"]]]
        self.f.seek(self.data_start + e["offset"] + r * (cols // 128) * nb)
        raw = self.f.read((cols // 128) * nb)
        return self._decode(raw, 1, cols, QNAME[e["quant"]])[0]

    def vec(self, name: str) -> np.ndarray:
        e = self._resolve(self.by[name])
        assert not name.startswith("__onebp_ext_"), name
        self.f.seek(self.data_start + e["offset"])
        return np.frombuffer(self.f.read(e["bytes"]), dtype=np.float32).copy()

    def _f16_tile(self, name: str) -> np.ndarray:
        e = self._resolve(self.by[name])
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

    def _fwht_blocks(self, x: np.ndarray, sign_mode: str) -> np.ndarray:
        out = np.empty_like(x)
        for base in range(0, len(x), self.block):
            v = x[base:base + self.block].astype(np.float64).copy()
            n, ln = self.block, 1
            while ln < n:
                for i in range(0, n, 2 * ln):
                    a = v[i:i + ln].copy(); b = v[i + ln:i + 2 * ln].copy()
                    v[i:i + ln] = a + b; v[i + ln:i + 2 * ln] = a - b
                ln *= 2
            out[base:base + self.block] = (v / np.sqrt(self.block)).astype(np.float32)
        return out

    def hadamard_forward(self, x: np.ndarray) -> np.ndarray:
        sg = self.signs_for(len(x))
        return self._fwht_blocks(x * sg, "forward")

    def hadamard_inverse(self, x: np.ndarray) -> np.ndarray:
        sg = self.signs_for(len(x))
        return self._fwht_blocks(x, "inverse") * sg


def softplus(x: np.ndarray) -> np.ndarray:
    return np.where(x > 20.0, x, np.log1p(np.exp(np.minimum(x, 20.0))))


def main() -> int:
    if len(sys.argv) < 4:
        print(__doc__)
        return 2
    path, token, outpath = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    bp = Bp(path)

    h = bp.row("token_embd.weight", token)
    if bp.has_transform:
        h = bp.hadamard_inverse(h)

    gdn = [("blk.%d.attn_qkv.weight" % l) in bp.by for l in range(NL)]
    recs = []
    for l in range(NL):
        p = "blk.%d."
        xn = rmsnorm_1pw(h, bp.vec(p % l + "attn_norm.weight"))
        xr = bp.hadamard_forward(xn) if bp.has_transform else xn

        if gdn[l]:
            qkv = bp.w(p % l + "attn_qkv.weight") @ xr
            z = bp.w(p % l + "attn_gate.weight") @ xr
            # ssm_alpha/ssm_beta are NOT in the oracle folded set, so they take unrotated xn.
            a = bp.w(p % l + "ssm_alpha.weight") @ xn
            b = bp.w(p % l + "ssm_beta.weight") @ xn
            cv = bp._f16_tile(p % l + "ssm_conv1d.weight")
            conv = silu(cv[:, 3] * qkv)                  # single token: newest tap only
            q = conv[:KD].reshape(NK, HK).copy()
            k = conv[KD:2 * KD].reshape(NK, HK).copy()
            v = conv[2 * KD:].reshape(NV, HV)
            q /= np.sqrt(np.sum(q.astype(np.float64) ** 2, axis=1, keepdims=True) + 1e-6)
            k /= np.sqrt(np.sum(k.astype(np.float64) ** 2, axis=1, keepdims=True) + 1e-6)
            A = bp.vec(p % l + "ssm_a")
            dt = bp.vec(p % l + "ssm_dt.bias")
            g = A * softplus(a + dt)
            beta = 1.0 / (1.0 + np.exp(-b))
            nw = bp.vec(p % l + "ssm_norm.weight")
            zc = z.reshape(NV, HV)
            core = np.zeros((NV, HV), dtype=np.float32)
            for vh in range(NV):
                kh = vh % NK
                # fresh state: state = outer(k[kh], v*beta); c = state^T q / sqrt(HK)
                delta = (v[vh] * beta[vh]).astype(np.float32)
                c = ((k[kh] @ q[kh]) * delta / np.sqrt(HK)).astype(np.float32)
                r = 1.0 / np.sqrt(np.mean(c.astype(np.float64) ** 2) + 1e-6)
                core[vh] = (c * r * nw * silu(zc[vh])).astype(np.float32)
            crf = core.reshape(-1)
            if bp.has_transform:
                rep = NV // NK
                pm = np.empty_like(crf)
                for rr in range(rep):
                    for kk in range(NK):
                        for hd in range(HV):
                            pm[hd + HV * rr + HV * rep * kk] = crf[hd + HV * kk + HV * NK * rr]
                crf = pm
            crr = bp.hadamard_forward(crf) if bp.has_transform else crf
            h = h + (bp.w(p % l + "ssm_out.weight") @ crr)
        else:
            qbuf = bp.w(p % l + "attn_q.weight") @ xr
            kbuf = bp.w(p % l + "attn_k.weight") @ xr
            vbuf = bp.w(p % l + "attn_v.weight") @ xr
            qnw = bp.vec(p % l + "attn_q_norm.weight")
            knw = bp.vec(p % l + "attn_k_norm.weight")
            q3 = qbuf.reshape(NH, 2, HD)
            qs = q3[:, 0, :].copy()
            gt = q3[:, 1, :].copy()
            for hh in range(NH):
                qs[hh] = rmsnorm_1pw(qs[hh], qnw)
            kb = kbuf.reshape(NKV, HD).copy()
            for hh in range(NKV):
                kb[hh] = rmsnorm_1pw(kb[hh], knw)
            vb = vbuf.reshape(NKV, HD)
            groups = NH // NKV
            attn = np.empty((NH, HD), dtype=np.float32)
            for hh in range(NH):
                attn[hh] = vb[hh // groups]             # seq=1 -> softmax is 1, pos=0 RoPE identity
            attn = (attn * (1.0 / (1.0 + np.exp(-gt)))).astype(np.float32)
            ar = bp.hadamard_forward(attn.reshape(-1)) if bp.has_transform else attn.reshape(-1)
            h = h + (bp.w(p % l + "attn_output.weight") @ ar)

        yn = rmsnorm_1pw(h, bp.vec(p % l + "post_attention_norm.weight"))
        yr = bp.hadamard_forward(yn) if bp.has_transform else yn
        gate = silu(bp.w(p % l + "ffn_gate.weight") @ yr) * (bp.w(p % l + "ffn_up.weight") @ yr)
        gr = bp.hadamard_forward(gate) if bp.has_transform else gate
        h = h + (bp.w(p % l + "ffn_down.weight") @ gr)
        recs.append(h.astype(np.float32).copy())

    with open(outpath, "wb") as f:
        for r in recs:
            f.write(np.asarray(r, dtype="<f4").tobytes())
    print(f"{path.split('/')[-1]}: {len(recs)} layers x {H} floats -> {outpath}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
