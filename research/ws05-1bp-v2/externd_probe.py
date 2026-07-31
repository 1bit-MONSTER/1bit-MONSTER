#!/usr/bin/env python3
"""ExTernD probe — WS-05 P0.

Tests the core claim of ExTernD (arXiv:2607.13511) on REAL LLM weight matrices:
  A ≈ B·diag(D)·C   with B,C ∈ {-1,0,+1} (ternary), D ∈ R^k real scales,
  expanded rank k = μ·min(m,n), μ > 1 → later planes correct earlier error,
  residual provably monotone-decreasing in k.

Measures the rate-distortion curve (relative Frobenius error vs effective
bits/weight) for μ ∈ {0.5, 1.0, 1.5, 2.0, 3.0} against:
  - BitNet-style per-tensor absmean ternary (the TQ2-family baseline)
  - per-row absmean ternary

Also exposes the TQ2 + residual-correction-planes hybrid (the actionable 1BP v2
format extension): `python3 externd_probe.py --hybrid` on a slice.

Usage:
  python3 externd_probe.py [--matrix q_proj|gate_proj|down_proj] [--layer N] [--mucap 2.0]
  python3 externd_probe.py --hybrid [--rows 512]
"""
import argparse, json, math, os, struct, sys, time
import numpy as np

SAFETENSORS = '/home/bcloud/checkpoints/Mage-VL/model-00001-of-00002.safetensors'
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'probe_results.json')

# ---------------------------------------------------------------- loading ---

def load_safetensor(path, name):
    with open(path, 'rb') as f:
        hdr_len = struct.unpack('<Q', f.read(8))[0]
        hdr = json.loads(f.read(hdr_len))
        e = hdr[name]
        f.seek(e['data_offsets'][0])
        raw = f.read(e['data_offsets'][1] - e['data_offsets'][0])
    dtype = e['dtype']
    if dtype == 'BF16':
        u16 = np.frombuffer(raw, dtype='<u2').reshape(e['shape'])
        return (u16.astype(np.uint32) << 16).view(np.float32)
    if dtype == 'F32':
        return np.frombuffer(raw, dtype='<f4').reshape(e['shape']).copy()
    if dtype == 'F16':
        return np.frombuffer(raw, dtype='<f2').reshape(e['shape']).astype(np.float32)
    raise ValueError(f'unhandled dtype {dtype}')

def load_matrix(kind, layer=0):
    P = f'model.language_model.layers.{layer}'
    table = {
        'q_proj':    f'{P}.self_attn.q_proj.weight',
        'gate_proj': f'{P}.mlp.gate_proj.weight',
        'down_proj': f'{P}.mlp.down_proj.weight',
    }
    return load_safetensor(SAFETENSORS, table[kind])

# ------------------------------------------------------------ factorization ---

def round_ternary(v, sparsity):
    """Round vector to {-1,0,+1} keeping exactly (1-sparsity) of elements nonzero.

    Uses exact top-k selection (argpartition), not a quantile threshold —
    quantiles degenerate when magnitudes cluster or repeat (e.g. unit vectors).
    """
    if sparsity <= 0:
        return np.sign(v).astype(np.int8)
    k = max(1, int(round((1.0 - sparsity) * v.size)))
    idx = np.argpartition(np.abs(v), -k)[-k:]
    out = np.zeros_like(v, dtype=np.int8)
    out[idx] = np.sign(v[idx])
    return out

def power_iteration(R, iters=12):
    m, n = R.shape
    v = np.random.default_rng(0).standard_normal(n)
    v /= np.linalg.norm(v)
    for _ in range(iters):
        u = R @ v
        nu = np.linalg.norm(u)
        if nu == 0:
            break
        u /= nu
        v = R.T @ u
        nv = np.linalg.norm(v)
        if nv == 0:
            break
        v /= nv
    return u, v

def greedy_ternary(A, k, sparsity_candidates=(0.0, 0.25, 0.5), progress=True):
    """Greedy deflation on the explicit residual.

    Planes i < min(m,n) are initialized from the SVD of A; planes i >= min(m,n)
    (expanded rank, the ExTernD trick) by power iteration on the *current
    residual* — those planes correct the error left by earlier ternary planes.
    """
    m, n = A.shape
    B = np.zeros((m, k), dtype=np.int8)
    C = np.zeros((k, n), dtype=np.int8)
    D = np.zeros(k, dtype=np.float32)
    R = A.astype(np.float32).copy()
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    rmin = min(m, n)
    for i in range(k):
        if i % 500 == 0 and progress:
            print(f'    plane {i}/{k}  |R|/|A| = {np.linalg.norm(R)/np.linalg.norm(A):.5f}')
        if i < rmin:
            u, v = U[:, i], Vt[i, :]
        else:
            u, v = power_iteration(R)
        best = (0.0, None, None)
        for rho in sparsity_candidates:
            b = round_ternary(u, rho)
            c = round_ternary(v, rho)
            nb = float(np.count_nonzero(b))   # exact: |b|² = nonzero count (int8 dots overflow)
            nc = float(np.count_nonzero(c))
            if nb == 0 or nc == 0:
                continue
            d = float(b.astype(np.float32) @ (R @ c)) / (nb * nc)
            if abs(d) > best[0]:
                best = (abs(d), b, c, d)
        _, b, c, d = best
        B[:, i], C[i, :], D[i] = b, c, d
        R -= d * np.outer(b.astype(np.float32), c.astype(np.float32))
    return B, C, D

def refine(A, B, C, D, sweeps=2):
    """Gauss-Seidel coordinate descent.

    Per plane i: re-solve d_i exactly (LS given fixed other planes), then
    re-round b_i against the residual that EXCLUDES plane i (R_{-i} c_i), then
    re-round c_i against R_{-i}^T b_i, re-solving d_i each time. Own-plane
    terms are cancelled explicitly — subtracting them twice (once via the Gram
    sum, once via the residual) halves d every sweep.
    """
    m, k = B.shape
    n = C.shape[1]
    Bf = B.astype(np.float32)
    Cf = C.astype(np.float32)
    for _ in range(sweeps):
        Gb = Bf.T @ Bf   # stale within the sweep (Gauss-Seidel approximation)
        Gc = Cf @ Cf.T
        for i in range(k):
            b_old = Bf[:, i].copy()
            c_old = Cf[i, :].copy()
            nb_old = Gb[i, i]
            nc_old = Gc[i, i]
            if nb_old == 0 or nc_old == 0:
                continue
            Aci = A @ c_old
            own_old = D[i] * Gb[i, i] * Gc[i, i]
            cross = float(D @ (Gb[i, :] * Gc[:, i])) - own_old
            D[i] = (float(b_old @ Aci) - cross) / (nb_old * nc_old)
            ri = Aci - (Bf @ (D * Gc[i, :])) + D[i] * b_old * nc_old
            best = (0.0, None, None)
            for rho in (0.25, 0.375, 0.5):
                b_new = round_ternary(ri, rho).astype(np.float32)
                nb = float(b_new @ b_new)
                if nb == 0:
                    continue
                d = float(b_new @ ri) / nb
                if abs(d) > best[0]:
                    best = (abs(d), b_new, d)
            b_new = best[1]
            if b_new is None:
                continue
            nb = float(b_new @ b_new)
            D[i] = best[2] / nc_old
            row = Bf.T @ b_new
            Atb = A.T @ b_new
            si = Atb - (Cf.T @ (D * row)) + D[i] * c_old * row[i]
            bestc = (0.0, None, None)
            for rho in (0.25, 0.375, 0.5):
                c_new = round_ternary(si, rho).astype(np.float32)
                nc = float(c_new @ c_new)
                if nc == 0:
                    continue
                d = float(c_new @ si) / nc
                if abs(d) > bestc[0]:
                    bestc = (abs(d), c_new, d)
            if bestc[1] is not None:
                nc = float(bestc[1] @ bestc[1])
                D[i] = bestc[2] / nb
                Bf[:, i], Cf[i, :] = b_new, bestc[1]
            else:
                Bf[:, i] = b_new
    return Bf.astype(np.int8), Cf.astype(np.int8), D

def error(A, B, C, D):
    """Exact relative Frobenius error, without materializing BDC."""
    m, k = B.shape
    n = C.shape[1]
    Gb = (B.astype(np.float32).T @ B.astype(np.float32))
    Gc = (C.astype(np.float32) @ C.astype(np.float32).T)
    g = np.array([B[:, i].astype(np.float32) @ (A @ C[i, :].astype(np.float32))
                  for i in range(k)], dtype=np.float32)
    l2_A = float((A * A).sum())
    l2_fit = float(np.einsum('i,ij,j->', D, Gb * Gc, D))
    cross = 2.0 * float(np.dot(D, g))
    rel = max(0.0, (l2_A - cross + l2_fit) / l2_A)
    return math.sqrt(rel)

# -------------------------------------------------------------- baselines ---

def per_tensor_ternary(A):
    alpha = float(np.abs(A).mean())
    T = np.round(A / alpha)
    return float(np.linalg.norm(A - alpha * T) / np.linalg.norm(A))

def per_row_ternary(A):
    alpha = np.abs(A).mean(axis=1, keepdims=True)
    T = np.round(A / alpha)
    return float(np.linalg.norm(A - alpha * T) / np.linalg.norm(A))

def bits_per_weight(m, n, k):
    return (2 * k * (m + n) + 32 * k) / (m * n)

# ------------------------------------------------------------------- main ---

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--matrix', default='q_proj', choices=['q_proj', 'gate_proj', 'down_proj'])
    ap.add_argument('--layer', type=int, default=0)
    ap.add_argument('--mucap', type=float, default=2.0)
    ap.add_argument('--rows', type=int, default=0, help='cap rows (0 = full)')
    ap.add_argument('--sweeps', type=int, default=2)
    ap.add_argument('--hybrid', action='store_true', help='run TQ2 + residual-correction-planes sweep')
    args = ap.parse_args()

    A = load_matrix(args.matrix, args.layer)
    if args.rows:
        A = A[:args.rows]
    m, n = A.shape
    print(f'matrix: {args.matrix} layer {args.layer}  shape {A.shape}  '
          f'mean|A|={np.abs(A).mean():.5f}  std={A.std():.5f}')
    print(f'baselines: per-tensor ternary  rel_err = {per_tensor_ternary(A):.5f}')
    print(f'           per-row ternary    rel_err = {per_row_ternary(A):.5f}')

    results = {'matrix': f'{args.matrix}_L{args.layer}', 'shape': [m, n],
               'baselines': {'per_tensor': per_tensor_ternary(A), 'per_row': per_row_ternary(A)}}
    if args.hybrid:
        alpha = float(np.abs(A).mean())
        E = A - alpha * np.round(A / alpha)
        base = float(np.linalg.norm(E) / np.linalg.norm(A))
        print(f'\nTQ2 baseline rel_err = {base:.4f}; sweeping correction planes:')
        rows = []
        for k in (16, 64, 256, 512):
            B, C, D = greedy_ternary(E, k, progress=False)
            B, C, D = refine(E, B, C, D, sweeps=2)
            rec = (B.astype(np.float32) @ (D[:, None] * C.astype(np.float32)))
            comb = float(np.linalg.norm(E - rec) / np.linalg.norm(A))
            extra = (2 * k * (m + n) + 32 * k) / (m * n)
            rows.append({'planes': k, 'extra_bpw': round(extra, 3), 'combined_rel_err': round(comb, 5)})
            print(f'  +{k:4d} planes (+{extra:.2f} b/w, total {2.0+extra:.2f}): combined={comb:.5f}')
        results['hybrid'] = rows
    else:
        rows = []
        for mu in (0.5, 1.0, 1.5, 2.0, 3.0):
            if mu > args.mucap:
                continue
            k = int(round(mu * min(m, n)))
            t0 = time.time()
            B, C, D = greedy_ternary(A, k)
            B, C, D = refine(A, B, C, D, sweeps=args.sweeps)
            rel = error(A, B, C, D)
            bps = bits_per_weight(m, n, k)
            rows.append({'mu': mu, 'k': k, 'rel_err': rel, 'bits_per_weight': bps,
                         'time_s': round(time.time() - t0, 1)})
            print(f'μ={mu:4.1f}  k={k:5d}  rel_err={rel:.5f}  {bps:.2f} bits/w  ({time.time()-t0:.1f}s)')
        results['externd'] = rows

    with open(OUT, 'w') as f:
        json.dump(results, f, indent=1)
    print(f'\nwrote {OUT}')

if __name__ == '__main__':
    main()
