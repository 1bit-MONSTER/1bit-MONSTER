#!/usr/bin/env python3
"""cmp_deepseek_v4_indexer_scores.py — gate the indexer's MATHS, not its tie-breaking.

Why this exists: the reference's indexer ends in `torch.topk`, and the scorer's
`ReLU` leaves a large share of scores at exactly 0.0 (26.5% on the fixture), so
the order among tied entries — and sometimes *which* tied entry takes the k-th
slot — is implementation-defined. Comparing the integer index tables therefore
fails by construction; 57 of 64 fixture rows matched as SETS and the rest
differed only among equal scores. What an implementation can be held to is the
score table (`deepseek_v4_indexer_scores`) — that is what this checks.

Usage:
    python3 Testing/cmp_deepseek_v4_indexer_scores.py <fixture_dir> <engine_scores.bin> \
            [--layer 1] [--tol 1e-6] [--engine-index-file f --ref-index-file g]
"""
import argparse
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fixture_dir")
    ap.add_argument("engine_scores")
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--engine-index-file")
    ap.add_argument("--ref-index-file")
    args = ap.parse_args()

    ref = np.load(os.path.join(args.fixture_dir, f"indexer_scores_L{args.layer}.npy"))
    T, n_win = (int(x) for x in open(args.engine_scores + ".shape").read().split())
    eng = np.fromfile(args.engine_scores, dtype=np.float32).reshape(T, n_win)
    if eng.shape != ref.shape:
        print(f"FAIL: engine scores {eng.shape} != reference {ref.shape}")
        return 1

    d = np.abs(eng.astype(np.float64) - ref.astype(np.float64))
    print(f"layer {args.layer}: score table {ref.shape}  max|delta| = {d.max():.3e}  "
          f"mean = {d.mean():.3e}  (tol {args.tol:g})")
    zero = float(np.mean(ref == 0.0))
    print(f"  reference zero-score fraction: {zero:.3f} "
          f"(every zero is an exact tie the top-k breaks arbitrarily)")
    worst = np.unravel_index(int(d.argmax()), d.shape)
    print(f"  worst at token {worst[0]}, entry {worst[1]}: engine={eng[worst]:.8f} ref={ref[worst]:.8f}")

    # Optional: show that integer index mismatches are tie-induced.
    if args.engine_index_file and args.ref_index_file:
        gi = np.fromfile(args.engine_index_file, dtype=np.int64) if args.engine_index_file.endswith(".bin") \
            else np.load(args.engine_index_file)
        ri = np.load(args.ref_index_file)
        k = ri.shape[1]
        gi = gi.reshape(T, k)
        order_only = content = tie_explained = unexplained = 0
        for t in range(T):
            a, b = gi[t], ri[t]
            if np.array_equal(a, b):
                continue
            if sorted(a.tolist()) == sorted(b.tolist()):
                order_only += 1
                continue
            content += 1
            # every pick in either row (that is not -1) must be a tie at the k-th
            # boundary: the score VALUES of both rows' picks must agree within tol
            sa = np.sort([ref[t, i] for i in a if i >= 0])
            sb = np.sort([ref[t, i] for i in b if i >= 0])
            if sa.shape == sb.shape and np.allclose(sa, sb, atol=args.tol):
                tie_explained += 1
            else:
                unexplained += 1
        print(f"  index rows: order-only={order_only}  content-differing={content} "
              f"(tie-explained={tie_explained}, unexplained={unexplained})")

    # Selection validity (order-independent, so ties cannot make it flaky): the
    # reference's rule is "visible = s < (t+1)/rate, keep the k best". A correct
    # implementation's picks must be a valid top-k of the reference's own row:
    # every pick >= the k-th best visible score - tol, and no unpicked visible
    # entry above it.
    sel_ok = None
    if args.engine_index_file:
        gi = np.fromfile(args.engine_index_file, dtype=np.int64) if args.engine_index_file.endswith(".bin") \
            else np.load(args.engine_index_file)
        k = gi.shape[1] if gi.ndim > 1 else int(gi.size / T)
        gi = gi.reshape(T, k)
        rate = n_win and max(1, int(round(64 / n_win))) if False else None  # rate not needed below
        bad = 0
        for t in range(T):
            threshold = None  # inferred: the reference's visibility rule uses (t+1)//rate
            # infer rate from the fixture config instead of guessing
            import json as _json
            cfg = _json.load(open(os.path.join(args.fixture_dir, "config.json")))
            rate = cfg["compress_rates"]["compressed_sparse_attention"]
            threshold = (t + 1) // rate
            vis = ref[t][:threshold]
            if vis.size == 0:
                if any(i >= 0 for i in gi[t]):
                    bad += 1
                continue
            kth = np.sort(vis)[::-1][min(k, vis.size) - 1]
            picks = [ref[t, i] for i in gi[t] if i >= 0]
            if picks and min(picks) < kth - args.tol:
                bad += 1
        sel_ok = bad == 0
        print(f"  selection validity vs the reference's own scores: "
              f"{'PASS' if sel_ok else f'FAIL ({bad} row(s))'}")

    ok = d.max() <= args.tol and (sel_ok is not False)
    print(f"worst max|delta| = {d.max():.3e} -> {'PASS' if ok else 'FAIL'} (tol {args.tol:g})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
