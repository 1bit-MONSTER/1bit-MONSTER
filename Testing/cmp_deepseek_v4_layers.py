#!/usr/bin/env python3
"""cmp_deepseek_v4_layers.py — per-layer engine-vs-HF check for the V4 fixture.

The logits gate (`cmp_deepseek_v4`) only sees the final distribution, so it can
say "wrong" but not "wrong from layer 2 onward". This compares the engine's
per-layer residual streams (written by `cmp_deepseek_v4 <dir> <ids> <logits> 20
18 <states_out>`) against the HF oracle's `hidden_states.npz` — layer by layer,
which is what the ws13 P0.1 acceptance asks for (<=1e-6 max|delta| per layer).

Mapping (verified against transformers 5.16.1, not assumed): transformers
appends `hidden_states` BEFORE each decoder layer, so `hidden_states[i]` is the
input to layer i, and the final entry is the collapsed+normed output ([T, H],
no stream equivalent — the logits gate covers it).

Usage:
    python3 Testing/cmp_deepseek_v4_layers.py <fixture_dir> <engine_states_out> [--tol 1e-6]

Exit 0 = every compared state is within tolerance; 1 = some state is not.
"""
import argparse
import json
import os

import numpy as np


def stored_dtype(safetensors_path):
    """dtype of the fixture's stored weights, from the safetensors header."""
    try:
        with open(safetensors_path, "rb") as f:
            n = int.from_bytes(f.read(8), "little")
            header = json.loads(f.read(n))
        dtypes = {v["dtype"] for k, v in header.items() if k != "__metadata__"}
        return ",".join(sorted(dtypes))
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("fixture_dir")
    ap.add_argument("engine_states")
    ap.add_argument("--tol", type=float, default=1e-6)
    args = ap.parse_args()

    ref = np.load(os.path.join(args.fixture_dir, "hidden_states.npz"))
    keys = sorted(ref.files, key=lambda k: int(k.split("_")[1]))
    with open(args.engine_states + ".shape") as f:
        nstates, hc, H = (int(x) for x in f.read().split())
    eng = np.fromfile(args.engine_states, dtype=np.float32).reshape(nstates, hc, H)

    # stream-valued reference entries ([T, hc, H]); the last entry is [T, H].
    stream_keys = [k for k in keys if ref[k].ndim == 3]
    dtype = stored_dtype(os.path.join(args.fixture_dir, "model.safetensors"))

    print(f"fixture={args.fixture_dir}  weights={dtype}  tol={args.tol:g}")
    print(f"engine states: {nstates} x [hc={hc}, H={H}]  |  reference stream states: {len(stream_keys)}")
    if dtype not in ("F32",) and args.tol < 1e-4:
        print(f"  WARNING: weights stored as {dtype} — bf16 rounding alone perturbs a layer "
              f"state by ~1e-4, so a {args.tol:g} bound cannot be met by ANY implementation. "
              f"Regenerate with --weights-dtype float32.")

    worst, n_cmp = 0.0, min(nstates, len(stream_keys))
    for i in range(n_cmp):
        r = ref[stream_keys[i]][-1].astype(np.float64)   # last prompt token
        e = eng[i].astype(np.float64)
        d = float(np.abs(r - e).max())
        scale = float(np.abs(r).max()) or 1.0
        worst = max(worst, d)
        print(f"  state {i} (input to layer {i}): max|delta|={d:.3e}  rel={d/scale:.2e}  "
              f"|ref|max={scale:.3f}")

    print(f"worst max|delta| = {worst:.3e} over {n_cmp} state(s)   -> "
          f"{'PASS' if worst <= args.tol else 'FAIL'} (tol {args.tol:g})")
    if nstates > len(stream_keys):
        print(f"  note: {nstates - len(stream_keys)} engine state(s) beyond the reference's "
              f"stream entries (the final collapsed+normed state is covered by the logits gate)")
    return 0 if worst <= args.tol else 1


if __name__ == "__main__":
    raise SystemExit(main())
