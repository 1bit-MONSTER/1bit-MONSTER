#!/usr/bin/env python3
"""Which int4 convention does each *-NPU2 bundle need?

dequant_q4nx.cpp's two decoders differ on two axes (scale index layout and
nibble signedness), and the wheel that matters is that a WRONG pairing still
produces a plausible weight distribution — the model runs and confidently
answers with the wrong token. LFM2-1.2B turned out to need a third combination.

The oracle: a model with tie_word_embeddings=true must have lm_head ==
embed_tokens. Decoding lm_head under each convention and correlating against the
BF16 embedding rows identifies the convention (and, when nothing correlates,
tells you the bundle is untied or uses a convention we do not implement).

Usage: q4nx_decoder_map.py [models-dir]        (default ~/.config/flm/models)
"""
import json, os, struct, sys
import numpy as np

TILE_ROWS, TILE_COLS, ROW_BYTES = 32, 256, 5120
CONVENTIONS = [("group+unsigned", "group", False), ("group+signed", "group", True),
               ("row+unsigned", "row", False), ("row+signed", "row", True)]


def load(path):
    raw = open(path, "rb").read()
    hsz = struct.unpack("<Q", raw[:8])[0]
    return raw, json.loads(raw[8:8 + hsz].decode()), 8 + hsz


def bf16_rows(raw, base, off, nrows, K):
    n = nrows * K
    buf = np.frombuffer(raw, dtype=np.uint16, count=n, offset=base + off)
    return (buf.astype(np.uint32) << 16).view(np.float32).reshape(nrows, K)


def bf16_scalar(raw, o):
    return struct.unpack("<f", struct.pack("<I", struct.unpack("<H", raw[o:o + 2])[0] << 16))[0]


def decode_rows(raw, base, off, K, nrows, scale_major, signed):
    """Decode the first `nrows` matrix rows (nrows must be a multiple of 32)."""
    ntc = K // TILE_COLS
    ntiles = (nrows // TILE_ROWS) * ntc
    out = np.zeros((nrows, K), np.float32)
    for ir in range(ntiles):
        rd = base + off + ir * ROW_BYTES
        tr, tc = ir // ntc, ir % ntc
        scales = [bf16_scalar(raw, rd + 2 * i) for i in range(256)]
        zeros = [bf16_scalar(raw, rd + 512 + 2 * i) for i in range(256)]
        packed = rd + 1024
        for lr in range(TILE_ROWS):
            lane, lrow, nib = lr // 16, lr % 16, lr % 2
            bi = lrow // 2
            ld = packed + lane * (TILE_COLS * 8)
            row = out[tr * TILE_ROWS + lr]
            for col in range(TILE_COLS):
                g = col // 32
                si = (g * 32 + lr) if scale_major == "group" else (lr * 8 + g)
                s, z = scales[si], zeros[si]
                if not np.isfinite(s) or abs(s) > 100: s = 0.0
                if not np.isfinite(z) or abs(z) > 100: z = 0.0
                bv = raw[ld + col * 8 + bi]
                q = (bv & 0x0F) if nib == 0 else ((bv >> 4) & 0x0F)
                v = (q if q < 8 else q - 16) if signed else q
                row[tc * TILE_COLS + col] = v * s + z
    return out


def probe(model_dir):
    p = os.path.join(model_dir, "model.q4nx")
    if not os.path.exists(p):
        return None
    raw, m, base = load(p)
    emb_name = next((k for k in ("model.embed_tokens.weight", "model.token_embd.weight")
                     if k in m), None)
    if emb_name is None or "lm_head.weight" not in m or m[emb_name]["dtype"] != "BF16":
        return "no tied pair to test"
    NV, K = m[emb_name]["shape"][0], m[emb_name]["shape"][1]
    if K % TILE_COLS or m["lm_head.weight"]["dtype"] != "I8":
        return f"unsupported geometry (K={K})"
    rows = 64
    eo = m[emb_name]["data_offsets"][0]
    emb = bf16_rows(raw, base, eo, rows, K).ravel()
    lo = m["lm_head.weight"]["data_offsets"][0]
    scores = {}
    for name, sm, sg in CONVENTIONS:
        M = decode_rows(raw, base, lo, K, rows, sm, sg)
        scores[name] = float(np.corrcoef(emb, M.ravel())[0, 1])
    best = max(scores, key=lambda k: scores[k])
    return (best, scores[best], scores) if scores[best] > 0.9 else ("UNTIED-OR-UNKNOWN", scores[best], scores)


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/.config/flm/models")
    print(f"{'bundle':32s} {'winner':20s} {'corr':>7s}   all four")
    for name in sorted(os.listdir(d)):
        md = os.path.join(d, name)
        if not os.path.isdir(md):
            continue
        r = probe(md)
        if r is None:
            continue
        if isinstance(r, str):
            print(f"{name:32s} {r}")
            continue
        best, bc, sc = r
        detail = " ".join(f"{k.split('+')[1][:3]}={v:+.3f}" for k, v in sc.items())
        print(f"{name:32s} {best:20s} {bc:+7.4f}   {detail}")


if __name__ == "__main__":
    main()
