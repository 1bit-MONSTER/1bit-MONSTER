#!/usr/bin/env python3
"""verify_zaya_layout.py — settle a Q4NX container's decode convention against its
SOURCE checkpoint, element-wise, with no NPU and no engine build.

Why this exists: "the weights are corrupt" and "the decoder reads them wrong" are
different claims and a self-port of the decoder cannot distinguish them (ADR
engine-layering-and-lemonade-scope.md §9.9.3). The only witness that can is the
published checkpoint. This tool compares them directly, so the convention is
SETTLED BY MEASUREMENT rather than by argument.

It searches the 4-way convention space (scale index layout x nibble sign) crossed
with nibble/lane polarity and reports correlation + bias + error in units of the
per-group scale. A correct convention is recognisable without a threshold:
  corr -> 1, bias/scale -> 0, rms/scale -> 1/sqrt(12) = 0.2887
(the last is exactly int4 round-to-nearest error, so it proves the payload is the
quantised source and not merely correlated with it).

Usage:
  verify_zaya_layout.py --q4nx ~/ZAYA1-8B-Q4NX/zaya1-8b.q4nx \
                        --source-dir ~/zaya1-8b-hf \
                        [--tensor model.layers.0.self_attn.q_proj.weight] [--search]

Exit code 0 if the best convention is unambiguous (corr >= --min-corr, default 0.99).
"""
import argparse
import glob
import json
import os
import struct
import sys

import numpy as np

TILE_BYTES = 5120
TILE_ROWS = 32
TILE_COLS = 256
RNE_RMS = 1.0 / np.sqrt(12.0)  # expected rms/scale for correct int4 round-to-nearest


# ── container ────────────────────────────────────────────────────────────────
def read_q4nx_manifest(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def read_i8_rows(path, hdr, entry):
    """A Q4NX I8 tensor: shape [n_i8_rows, 5120] of raw tile bytes."""
    off = entry["data_offsets"]
    with open(path, "rb") as f:
        f.seek(hdr + off[0])
        buf = f.read(off[1] - off[0])
    return np.frombuffer(buf, dtype=np.uint8).reshape(entry["shape"][0], TILE_BYTES)


# ── source checkpoint (minimal safetensors reader) ───────────────────────────
def st_index(source_dir):
    idx = {}
    for s in sorted(glob.glob(os.path.join(source_dir, "*.safetensors"))):
        with open(s, "rb") as f:
            ln = struct.unpack("<Q", f.read(8))[0]
            h = json.loads(f.read(ln))
        base = 8 + ln
        for k, v in h.items():
            if k != "__metadata__":
                idx[k] = (s, base, v)
    return idx


def load_source(idx, name):
    s, base, v = idx[name]
    with open(s, "rb") as f:
        f.seek(base + v["data_offsets"][0])
        buf = f.read(v["data_offsets"][1] - v["data_offsets"][0])
    if v["dtype"] == "BF16":
        a = (np.frombuffer(buf, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
    else:
        raise SystemExit(f"source dtype {v['dtype']} not handled for {name}")
    return a.reshape(v["shape"]).astype(np.float32)


def resolve_source_name(idx, q4nx_key):
    """Q4NX names are HF-ish but not identical: the source folds q/k/v into
    `qkv_proj.` and drops/injects `.weight`. Match on the normalised tail."""
    def norm(k):
        k = k.replace(".weight", "").replace(".qkv_proj", "")
        return k
    if q4nx_key in idx:
        return q4nx_key
    want = norm(q4nx_key)
    for k in idx:
        if norm(k) == want:
            return k
    tail = ".".join(q4nx_key.split(".")[-3:])
    for k in idx:
        if tail in k:
            return k
    return None


# ── decode ───────────────────────────────────────────────────────────────────
def bf16(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32)


def decode_rows(tiles, n_tile_cols, scale_layout, nibble_mode, nib_pol=0, lane_pol=0):
    """Decode raw I8 rows to [out_rows, out_cols] float, plus the signed int4
    codes and per-group scales (so the caller can test exactness, not just corr).

    Layout facts this function encodes, all of them from the tile spec:
      scales  bytes [0:512]    256 bf16, indexed either group-major (g*32+lr) or
                               row-major (lr*8+g) — the ONE thing that differs
                               between the two in-tree decoders
      mins    bytes [512:1024] same indexing
      nibbles bytes [1024:5120] 2 lanes x 256 cols x 8 byte_idx, lane = lr//16,
                               byte_idx = (lr%16)//2, nibble = (lr%16)%2
    """
    ni = tiles.shape[0]
    n_tile_rows = ni // n_tile_cols
    out = np.zeros((n_tile_rows * TILE_ROWS, n_tile_cols * TILE_COLS), np.float32)
    Q = np.zeros_like(out)
    S = np.zeros_like(out)

    sc = bf16(np.frombuffer(tiles[:, 0:512].tobytes(), dtype="<u2").reshape(ni, 256))
    zp = bf16(np.frombuffer(tiles[:, 512:1024].tobytes(), dtype="<u2").reshape(ni, 256))
    packed = tiles[:, 1024:TILE_BYTES].reshape(ni, 2, TILE_COLS, 8)

    lr = np.arange(TILE_ROWS)
    col = np.arange(TILE_COLS)
    group = col // 32
    idx = group[None, :] * 32 + lr[:, None] if scale_layout == "group_major" \
        else lr[:, None] * 8 + group[None, :]
    lane = (lr // 16) if lane_pol == 0 else (1 - lr // 16)
    byte_idx = (lr % 16) // 2
    nib = ((lr % 16) % 2) if nib_pol == 0 else (1 - (lr % 16) % 2)

    for ir in range(ni):
        b = packed[ir][lane[:, None], col[None, :], byte_idx[:, None]]
        q = np.where(nib[:, None] == 0, b & 0x0F, (b >> 4) & 0x0F).astype(np.float32)
        if nibble_mode == "signed":
            q = np.where(q >= 8, q - 16, q)
        s = sc[ir][idx]
        w = q * s + zp[ir][idx]
        tr, tc = ir // n_tile_cols, ir % n_tile_cols
        sl = (slice(tr * TILE_ROWS, (tr + 1) * TILE_ROWS),
              slice(tc * TILE_COLS, (tc + 1) * TILE_COLS))
        out[sl], Q[sl], S[sl] = w, q, s
    return out, Q, S


def corr(a, b):
    a, b = a.ravel() - a.mean(), b.ravel() - b.mean()
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / d) if d else float("nan")


CONVENTIONS = [(sl, nm, np_, lp)
               for sl in ("group_major", "row_major")
               for nm in ("unsigned", "signed")
               for np_ in (0, 1)
               for lp in (0, 1)]


def best_convention(tiles, src, in_features):
    """Search the 4-way convention space x polarity. Also size the tensor against
    the source so a transposed orientation is reported, not silently zero-corr."""
    n_tile_cols = in_features // TILE_COLS
    W, _, _ = decode_rows(tiles, n_tile_cols, "row_major", "signed")
    orient = ""
    cmp_src = src
    if W.shape != src.shape:
        if W.shape == src.T.shape:
            cmp_src, orient = src.T, " (source compared TRANSPOSED)"
        else:
            return None, f"shape {W.shape} matches neither {src.shape} nor its transpose", None
    scored = []
    for sl, nm, np_, lp in CONVENTIONS:
        w, _, _ = decode_rows(tiles, n_tile_cols, sl, nm, np_, lp)
        scored.append((corr(w, cmp_src), sl, nm, np_, lp))
    scored.sort(reverse=True)
    return scored, orient, cmp_src


def exactness(tiles, src, in_features):
    """For the winning convention, report bias and error in units of the scale."""
    n_tile_cols = in_features // TILE_COLS
    W, Q, S = decode_rows(tiles, n_tile_cols, "row_major", "signed")
    if W.shape != src.shape:
        src = src.T
    d = W - src
    sm = float(S.mean())
    q_ideal = np.clip(np.rint(src / S), -8, 7)
    return {
        "corr": corr(W, src),
        "bias_over_scale": float(d.mean() / sm),
        "rms_over_scale": float(np.sqrt((d ** 2).mean()) / sm),
        "rne_ideal": RNE_RMS,
        "q_match": float((q_ideal == Q).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q4nx", default=os.path.expanduser("~/ZAYA1-8B-Q4NX/zaya1-8b.q4nx"))
    ap.add_argument("--source-dir", default=os.path.expanduser("~/zaya1-8b-hf"))
    ap.add_argument("--tensor", action="append", default=None,
                    help="Q4NX tensor key to test (repeatable); default a fixed probe set")
    ap.add_argument("--in-features", type=int, default=None,
                    help="override in_features when the container's geometry is ambiguous")
    ap.add_argument("--search", action="store_true", help="print the full convention search")
    ap.add_argument("--min-corr", type=float, default=0.99)
    args = ap.parse_args()

    man, hdr = read_q4nx_manifest(args.q4nx)
    idx = st_index(args.source_dir)
    keys = args.tensor or [
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        "model.layers.0.mlp.experts.gate_up_proj.weight",
    ]

    print(f"container : {args.q4nx}  ({len(man)} manifest entries)")
    print(f"source    : {args.source_dir}  ({len(idx)} tensors)")
    print(f"criterion : corr>={args.min_corr}, |bias/scale|<0.05, rms/scale~={RNE_RMS:.4f}")
    print()
    print(f"{'tensor':<44} {'corr':>7} {'bias/s':>8} {'rms/s':>7} {'qmatch':>7}  verdict")
    ok = True
    for k in keys:
        entry = man.get(k)
        if not isinstance(entry, dict):
            print(f"{k:<44} {'--':>7}  not in manifest")
            ok = False
            continue
        sk = resolve_source_name(idx, k)
        if sk is None:
            print(f"{k:<44} {'--':>7}  no source tensor matched")
            ok = False
            continue
        src = load_source(idx, sk)
        if src.ndim > 2:
            src = src.reshape(-1, src.shape[-1])
        tiles = read_i8_rows(args.q4nx, hdr, entry)

        # in_features comes from the tensor's OWN geometry when not forced: the
        # container stores tile rows, so this is in_features/256 tile columns.
        in_features = args.in_features
        if in_features is None:
            in_features = _infer_in_features(tiles.shape[0], src)

        scored, orient, cmp_src = best_convention(tiles, src, in_features)
        if scored is None:
            print(f"{k:<44} {'--':>7}  {orient}")
            ok = False
            continue
        top = scored[0]
        ex = exactness(tiles, src, in_features)
        verdict = "UNBIASED int4 rne" if abs(ex["bias_over_scale"]) < 0.05 \
            and abs(ex["rms_over_scale"] - RNE_RMS) < 0.05 else "off-convention"
        if top[0] < args.min_corr:
            ok = False
            verdict = "NO CONVENTION MATCHES"
        print(f"{k.split('.')[-2]+'.'+k.split('.')[-1]:<44} {ex['corr']:>7.4f} "
              f"{ex['bias_over_scale']:>8.4f} {ex['rms_over_scale']:>7.4f} "
              f"{ex['q_match']:>7.5f}  {verdict}{orient}")
        if args.search:
            print(f"    winner: scale={top[1]} nibble={top[2]} nibpol={top[3]} lanepol={top[4]}")
            for c, sl, nm, np_, lp in scored:
                print(f"    corr {c:7.4f}  scale={sl:<11} nibble={nm:<8} "
                      f"nibpol={np_} lanepol={lp}")
    print()
    print("SETTLED" if ok else "NOT SETTLED — inspect above")
    return 0 if ok else 1


def _infer_in_features(n_i8_rows, src):
    """n_i8_rows = (out/32) * (in/256). Try the shapes the source permits."""
    out_rows = src.shape[0]
    if src.shape[1] % TILE_COLS == 0:
        cand = src.shape[1]
        if n_i8_rows == (out_rows // TILE_ROWS) * (cand // TILE_COLS):
            return cand
        if n_i8_rows == (src.shape[1] // TILE_ROWS) * (out_rows // TILE_COLS):
            return out_rows
    raise SystemExit(f"cannot infer in_features: {n_i8_rows} tile rows vs source {src.shape}")


if __name__ == "__main__":
    sys.exit(main())
