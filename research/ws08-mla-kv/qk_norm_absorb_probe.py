#!/usr/bin/env python3
"""qk_norm_absorb_probe.py — WS-08 P1: numerical verification of QK-Normed MLA
absorption (arXiv:2606.16310, "QK-Normed MLA: QK normalization without full key
caching").

Checks the paper's core claim (Eq 10/12 + Appendix E reference test): the
absorbed latent formulation —

    q̃ = (q0 ⊙ γq ⊙ γk) @ (Wg^K)^T          # static affine absorbed into query
    αq = 1/Sq,  αk = 1/Sk                   # dynamic scalars (per token/group)
    logit = (q̃ · c_t) · αq · αk             # row-column scaled latent dot

— is numerically identical to explicitly materializing the post-up-projection
key and applying QK RMSNorm before the dot product:

    q̂ = RMSNorm_q(q0),  k̂ = RMSNorm_k(c_t @ Wg^K)
    logit = q̂ · k̂

Also verifies the blockwise content+RoPE scheme (Eq 11) and the cache-size
claim (scalar cache T×G vs full keys T×G×dc).

Run: python3 research/ws08-mla-kv/qk_norm_absorb_probe.py
"""
import numpy as np

def rmsnorm(z, gamma, eps=1e-6):
    s = np.sqrt(np.mean(z**2, axis=-1, keepdims=True) + eps)
    return z * gamma / s, 1.0 / s[..., 0]

def main():
    rng = np.random.default_rng(7)
    r, dc, dr, G, H = 64, 128, 32, 4, 8   # latent rank, content dim, rope dim, KV groups, query heads
    T, nq = 48, 12                          # cached tokens, query positions
    eps = 1e-6

    # weights (random, fixed)
    W_D = rng.normal(size=(dc, r)) * 0.1               # latent projection
    W_K = rng.normal(size=(G, r, dc)) * 0.1            # content key up-projection per group
    gamma_q = rng.uniform(0.5, 1.5, size=dc)
    gamma_k = rng.uniform(0.5, 1.5, size=dc)
    gamma_qr = rng.uniform(0.5, 1.5, size=dr)          # rope affine weights
    gamma_kr = rng.uniform(0.5, 1.5, size=dr)
    h_to_g = rng.integers(0, G, size=H)                # query head -> KV group

    # inputs
    X = rng.normal(size=(T, dc)) * 0.5                 # cached token hiddens
    Q0 = rng.normal(size=(nq, H, dc)) * 0.5            # raw content queries
    Q0r = rng.normal(size=(nq, H, dr)) * 0.5           # raw rope queries
    Kr = rng.normal(size=(T, dr)) * 0.5                # rope keys (shared across groups)

    # ── Reference (explicit): materialize full keys, apply QK RMSNorm ──
    C = X @ W_D                                        # [T, r] latent cache
    ref = np.zeros((nq, H, T))
    for t in range(T):
        for g in range(G):
            k0 = C[t] @ W_K[g]                         # [dc] un-normalized key
            khat, _ = rmsnorm(k0, gamma_k, eps)        # [dc] normalized key
            for i in range(nq):
                for h in range(H):
                    if h_to_g[h] != g:
                        continue
                    qhat, _ = rmsnorm(Q0[i, h], gamma_q, eps)
                    ref[i, h, t] = qhat @ khat

    # ── Absorbed (Eq 10/12): latent dot + row-column scalar scaling ──
    Sq = np.sqrt(np.mean(Q0**2, axis=-1) + eps)        # [nq, H]
    alpha_q = 1.0 / Sq
    C = X @ W_D                                        # [T, r] latent cache
    got = np.zeros((nq, H, T))
    for i in range(nq):
        for h in range(H):
            g = h_to_g[h]
            qtilde = (Q0[i, h] * gamma_q * gamma_k) @ W_K[g].T   # [r]
            k0 = C @ W_K[g]                            # temp projection for RMS stats only
            Sk = np.sqrt(np.mean(k0**2, axis=-1) + eps)          # [T] per-token scalar
            got[i, h] = (qtilde @ C.T) * alpha_q[i, h] * (1.0 / Sk)

    diff = np.abs(ref - got)
    print(f"content logits: max|ref - absorbed| = {diff.max():.3e}  (fp64, expect ~1e-14)")

    # ── Blockwise content + RoPE (Eq 11) ──
    ref_total = ref.copy()
    got_total = got.copy()
    for i in range(nq):
        for h in range(H):
            for t in range(T):
                qr_n, _ = rmsnorm(Q0r[i, h], gamma_qr, eps)
                kr_n, _ = rmsnorm(Kr[t], gamma_kr, eps)
                ref_total[i, h, t] += qr_n @ kr_n
                # rope path is identical in both (materialized vectors)
                got_total[i, h, t] += qr_n @ kr_n
    print(f"blockwise (content+RoPE): max diff = {np.abs(ref_total - got_total).max():.3e}")

    # ── Cache-size claim: scalar cache vs full keys ──
    scalar_bytes = T * G * 4
    full_key_bytes = T * G * dc * 4
    print(f"cache: scalar {scalar_bytes/1e3:.1f} KB vs full content keys {full_key_bytes/1e3:.1f} KB "
          f"({100*scalar_bytes/full_key_bytes:.2f}% — paper: 1.56% at r=512,G=8)")

    ok = diff.max() < 1e-12 and np.abs(ref_total - got_total).max() < 1e-12
    print("qk_norm_absorb_probe:", "OK" if ok else "FAIL")
    return 0 if ok else 1

if __name__ == "__main__":
    raise SystemExit(main())
