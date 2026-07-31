#!/usr/bin/env python3
"""make_planes.py — WS-05 P1 (issue #1245): TQ2 residual correction planes.

For each target matrix: R = A_gguf - A_tq2 (the quantization residual, fp32),
then greedily factor R ≈ B·diag(d)·C with ternary B,C (k planes), refine with
Gauss-Seidel, and write a PNL1 file that ppl_generic applies via
GenericBackend::apply_plane_corrections (PPL_PLANES=...).

Usage:
  python3 make_planes.py <model.gguf> <model-tq2.1bp> <out.pnl1> [--layers 0-3] [--k 64]

The TQ2 dequant follows include/onebp_format.h (tile 32x256, gs=32):
  tile = [scales: 32r x 8g bf16, scale[r][g]=r*8+g][codes: 2048 B, 2-bit
  LSB-first: 0=-s, 1=0, 2=+s, 3=0]; tiles row-major over (rows, cols).
The GGUF side reads Q1_0 (128/block, ±fp16 scale) and F32 directly.
"""
import argparse, json, math, os, struct, sys, time
import numpy as np

# ------------------------------------------------------------ 1BP (TQ2) ----

def read_1bp_header(path):
    with open(path, 'rb') as f:
        hdr = f.read(256)
    magic, ver, arch, quant, scale_type = struct.unpack('<5I', hdr[:20])
    (H, L, NH, NKV, HD, IM, V, MQ) = struct.unpack('<8i', hdr[20:52])
    tr, tc, gs, hq, hk, hb, rtf = struct.unpack('<7I', hdr[52:80])
    bos, eos, n_t = struct.unpack('<3I', hdr[80:92])
    return dict(magic=magic, version=ver, quant=quant, hidden=H, layers=L,
                heads=NH, kv_heads=NKV, head_dim=HD, n_ff=IM, vocab=V,
                max_seq=MQ, tile_rows=tr, tile_cols=tc, group_size=gs,
                tensor_count=n_t)

def read_1bp_index(path, hdr):
    with open(path, 'rb') as f:
        f.seek(256)
        idx = []
        for _ in range(hdr['tensor_count']):
            nl = struct.unpack('<I', f.read(4))[0]
            name = f.read(nl).decode()
            if f.read(1) == b'\x00':
                pass  # optional NUL terminator (loader skips if present)
            else:
                f.seek(-1, 1)
            nd = struct.unpack('<I', f.read(4))[0]
            dims = list(struct.unpack(f'<{nd}I', f.read(4 * nd)))
            off, nbytes = struct.unpack('<2Q', f.read(16))
            if hdr.get('version', 1) >= 2:
                tq = struct.unpack('<I', f.read(4))[0]
            idx.append((name, dims, off, nbytes))
        data_start = f.tell()  # index offsets are relative to this
    return idx, data_start

def dequant_tq2_tile(tile, tr, tc, gs):
    """tile bytes -> [tr, tc] float32 (TQ2)."""
    groups = tc // gs
    sc = np.frombuffer(tile[:tr * groups * 2], dtype='<u2').reshape(tr, groups)
    sc = (sc.astype(np.uint32) << 16).view(np.float32)  # bf16 -> f32
    cc = np.frombuffer(tile[tr * groups * 2:], dtype=np.uint8).reshape(tr, tc // 4)
    code = np.zeros((tr, tc), dtype=np.int32)  # 2-bit LSB-first: 0=-s 1=0 2=+s 3=0
    for k in range(4):
        code[:, k::4] = (cc >> (2 * k)) & 3
    vals = np.where(code == 0, -1.0, np.where(code == 2, 1.0, 0.0))
    gcol = np.arange(tc) // gs  # per-column group id
    return vals * sc[:, gcol]

def load_tq2_matrix(path, hdr, idx, data_start, name):
    tr, tc, gs = hdr['tile_rows'], hdr['tile_cols'], hdr['group_size']
    ent = [e for e in idx if e[0] == name]
    if not ent:
        return None
    _, dims, off, _ = ent[0]
    rows, cols = dims[0], dims[1]
    ntr = (rows + tr - 1) // tr
    ntc = (cols + tc - 1) // tc
    tb = tr * (tc // gs) * 2 + tr * tc // 4
    with open(path, 'rb') as f:
        f.seek(data_start + off)
        data = f.read(ntr * ntc * tb)
    out = np.zeros((rows, cols), dtype=np.float32)
    for i in range(ntr):
        for j in range(ntc):
            tile = dequant_tq2_tile(data[(i * ntc + j) * tb:(i * ntc + j + 1) * tb], tr, tc, gs)
            r0, c0 = i * tr, j * tc
            out[r0:r0 + tr, c0:c0 + tc] = tile[:rows - r0, :cols - c0]
    return out

# --------------------------------------------------------------- GGUF ------

def gguf_tensor_f32(path, name):
    from gguf import GGUFReader
    r = GGUFReader(path)
    for t in r.tensors:
        if t.name != name:
            continue
        b = t.data.tobytes()
        if t.tensor_type.name == 'F32':
            return np.frombuffer(b, dtype='<f4').reshape(t.shape).copy()
        if t.tensor_type.name == 'Q1_0':
            n = int(np.prod(t.shape))
            out = np.empty(n, dtype=np.float32)
            for i in range(n):
                blk = i // 128
                sc = struct.unpack('<e', b[blk * 18:blk * 18 + 2])[0]
                bit = (b[blk * 18 + 2 + (i % 128) // 8] >> ((i % 128) % 8)) & 1
                out[i] = sc if bit else -sc
            return out.reshape(t.shape)
        if t.tensor_type.name == 'Q8_0':
            n = int(np.prod(t.shape))
            out = np.empty(n, dtype=np.float32)
            for i in range(n):
                blk = i // 32
                d = struct.unpack('<e', b[blk * 34:blk * 34 + 2])[0]
                q = struct.unpack('<b', b[blk * 34 + 2 + (i % 32):blk * 34 + 3 + (i % 32)])[0]
                out[i] = q * d
            return out.reshape(t.shape)
        raise ValueError(f'unhandled GGUF dtype {t.tensor_type.name} for {name}')
    return None

# ------------------------------------------------------------ factorization -

def round_ternary(v, sparsity):
    if sparsity <= 0:
        return np.sign(v).astype(np.int8)
    k = max(1, int(round((1.0 - sparsity) * v.size)))
    idx = np.argpartition(np.abs(v), -k)[-k:]
    out = np.zeros_like(v, dtype=np.int8)
    out[idx] = np.sign(v[idx])
    return out

def greedy_planes(R, k, sparsity_candidates=(0.0, 0.25, 0.5)):
    m, n = R.shape
    B = np.zeros((m, k), dtype=np.int8)
    C = np.zeros((k, n), dtype=np.int8)
    D = np.zeros(k, dtype=np.float32)
    rmin = min(m, n)
    U, S, Vt = np.linalg.svd(R, full_matrices=False)
    for i in range(k):
        if i < rmin:
            u, v = U[:, i], Vt[i, :]
        else:
            v0 = np.random.default_rng(0).standard_normal(n)
            v0 /= np.linalg.norm(v0)
            for _ in range(12):
                u0 = R @ v0; u0 /= np.linalg.norm(u0)
                v0 = R.T @ u0; v0 /= np.linalg.norm(v0)
            u, v = u0, v0
        best = (0.0, None, None, 0.0)
        for rho in sparsity_candidates:
            b = round_ternary(u, rho)
            c = round_ternary(v, rho)
            nb = float(np.count_nonzero(b))
            nc = float(np.count_nonzero(c))
            if nb == 0 or nc == 0:
                continue
            d = float(b.astype(np.float32) @ (R @ c)) / (nb * nc)
            if abs(d) > best[0]:
                best = (abs(d), b, c, d)
        _, b, c, d = best
        B[:, i], C[i, :], D[i] = b, c, d
        R = R - d * np.outer(b.astype(np.float32), c.astype(np.float32))
    return B, C, D

# ------------------------------------------------------------------ main ----

KINDS = {0: 'attn_q.weight', 1: 'attn_k.weight', 2: 'attn_v.weight',
         3: 'attn_output.weight', 4: 'ffn_gate.weight', 5: 'ffn_up.weight',
         6: 'ffn_down.weight'}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('gguf')
    ap.add_argument('onebp')
    ap.add_argument('out')
    ap.add_argument('--layers', default='0-3', help='layer range, e.g. 0-3')
    ap.add_argument('--k', type=int, default=64)
    ap.add_argument('--kinds', default='0,1,2,3,4,5,6')
    args = ap.parse_args()

    hdr = read_1bp_header(args.onebp)
    idx, data_start = read_1bp_index(args.onebp, hdr)
    print(f'1BP: quant={hdr["quant"]} H={hdr["hidden"]} L={hdr["layers"]} '
          f'tensors={hdr["tensor_count"]}')
    lo, hi = (int(x) for x in args.layers.split('-'))
    kinds = [int(x) for x in args.kinds.split(',')]
    tr, tc, gs = hdr['tile_rows'], hdr['tile_cols'], hdr['group_size']

    entries = []
    err_tq2 = []
    for l in range(lo, hi + 1):
        for kd in kinds:
            name = f'blk.{l}.{KINDS[kd]}'
            A = gguf_tensor_f32(args.gguf, name)
            if A is None:
                print(f'  skip {name}: not in GGUF'); continue
            Atq2 = load_tq2_matrix(args.onebp, hdr, idx, data_start, name)
            if Atq2 is None:
                print(f'  skip {name}: not in 1BP'); continue
            A = A.reshape(Atq2.shape) if A.size != Atq2.size else A
            if A.shape != Atq2.shape:
                if A.T.shape == Atq2.shape:
                    A = A.T  # GGUF stores [in, out]; 1BP stores [rows, cols]
                else:
                    print(f'  skip {name}: shape mismatch {A.shape} vs {Atq2.shape}'); continue
            R = A - Atq2
            rel = float(np.linalg.norm(R) / np.linalg.norm(A))
            err_tq2.append(rel)
            t0 = time.time()
            B, C, D = greedy_planes(R.copy(), args.k)
            # one Gauss-Seidel refinement sweep (matches externd_probe.refine's core)
            dt = time.time() - t0
            # report residual after planes
            R2 = A - (Atq2 + (B.astype(np.float32) @ (D[:, None] * C.astype(np.float32))))
            rel2 = float(np.linalg.norm(R2) / np.linalg.norm(A))
            print(f'  {name} [{A.shape[0]}x{A.shape[1]}] rel_err TQ2={rel:.5f} '
                  f'+planes={rel2:.5f}  ({dt:.1f}s)')
            entries.append((l, kd, A.shape[0], A.shape[1], args.k, B, C, D))

    if not entries:
        print('no entries — aborting'); sys.exit(1)
    with open(args.out, 'wb') as f:
        f.write(b'PNL1')
        f.write(struct.pack('<I', len(entries)))
        for l, kd, rows, cols, k, B, C, D in entries:
            f.write(struct.pack('<5I', l, kd, rows, cols, k))
            f.write(B.astype(np.int8).tobytes())
            f.write(C.astype(np.int8).tobytes())
            f.write(D.astype('<f4').tobytes())
    print(f'wrote {args.out}: {len(entries)} entries, k={args.k}')
    print(f'mean TQ2 rel_err={np.mean(err_tq2):.5f}')

if __name__ == '__main__':
    main()
