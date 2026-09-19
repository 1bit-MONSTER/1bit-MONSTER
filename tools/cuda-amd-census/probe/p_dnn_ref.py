#!/usr/bin/env python3
"""p_dnn_ref.py — settle the conv2d disagreement with an INDEPENDENT reference.

Reads the state dumped by p_dnn_ref.cpp and recomputes the convolution with
numpy's own primitives (sliding_window_view + einsum). That shares no code, no
indexing loop and no language with either side under test, which is the point:
comparing MIOpen against a hand-rolled C++ reference cannot tell you which of the
two is wrong.

It reports three comparisons, not one:

    y_gpu  vs y_numpy   the real question — is MIOpen's conv2d correct?
    y_hand vs y_numpy   is the census's hand-rolled reference the buggy one?
    y_hand vs y_gpu     the original disagreement, reproduced for the record

and then tests the specific hypothesis that explains a mismatch which the
all-ones case cannot see (R==S==3 and H==W==8, so a spatial transpose is
invisible to a degenerate input): recompute with the filter transposed (r<->s)
and see whether *that* matches the outlier.
"""
import json
import sys

import numpy as np


def emit(result, detail):
    print("PROBE|dnn_ref|%s|%s" % (result, detail))
    sys.stdout.flush()


def conv_np(x, w, N, C, H, W, K, R, S, transpose_filter=False):
    xs = x.reshape(N, C, H, W)
    ws = w.reshape(K, C, R, S)
    if transpose_filter:
        # w[k,c,r,s] -> w[k,c,s,r]: the classic invisible-to-all-ones mistake.
        ws = np.transpose(ws, (0, 1, 3, 2))
    patches = np.lib.stride_tricks.sliding_window_view(xs, (R, S), axis=(2, 3))
    # patches: (N, C, OH, OW, R, S); ws: (K, C, R, S)
    return np.einsum("nchwrs,kcrs->nkhw", patches, ws, optimize=True)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/cudnn_ref.json"
    try:
        d = json.load(open(path))
    except Exception as e:
        emit("UNSUPPORTED", "cannot read %s: %s" % (path, e))
        return 3

    N, C, H, W = d["N"], d["C"], d["H"], d["W"]
    K, R, S = d["K"], d["R"], d["S"]
    OH, OW = d["OH"], d["OW"]

    x = np.array(d["x"], dtype=np.float64)
    w = np.array(d["w"], dtype=np.float64)
    gpu = np.array(d["y_gpu"], dtype=np.float64)
    hand = np.array(d["y_hand"], dtype=np.float64)

    ref = conv_np(x, w, N, C, H, W, K, R, S).reshape(-1)

    def cmp(a, b):
        m = float(np.max(np.abs(a - b)))
        scale = float(np.max(np.abs(b))) if np.max(np.abs(b)) > 0 else 1.0
        return m, m / scale

    gd, gr = cmp(gpu, ref)
    hd, hr = cmp(hand, ref)
    xd, xr = cmp(hand, gpu)

    TOL = 1e-5
    gpu_ok = gr < TOL
    hand_ok = hr < TOL

    detail = ("algo=%s | gpu_vs_numpy max_abs=%.3e rel=%.3e (%s) | "
              "hand_vs_numpy max_abs=%.3e rel=%.3e (%s) | hand_vs_gpu rel=%.3e"
              % (d.get("algo"), gd, gr, "MATCH" if gpu_ok else "DIFFER",
                 hd, hr, "MATCH" if hand_ok else "DIFFER", xr))

    if gpu_ok and hand_ok:
        emit("PASS", detail + " -> all three agree; the earlier discrepancy does not reproduce")
        return 0

    if gpu_ok and not hand_ok:
        # The interesting outcome: MIOpen is right, the census's reference is wrong.
        t = conv_np(x, w, N, C, H, W, K, R, S, transpose_filter=True).reshape(-1)
        td, tr = cmp(hand, t)
        cause = ("hand matches a TRANSPOSED filter (rel=%.3e)" % tr) if tr < TOL else \
                ("transposed filter does NOT explain it (best rel=%.3e)" % tr)
        emit("PASS", detail + " -> MIOpen agrees with numpy; the CENSUS REFERENCE is the wrong one. " + cause)
        return 0

    if hand_ok and not gpu_ok:
        emit("INCORRECT", detail + " -> MIOpen disagrees with numpy while the reference agrees: a real backend defect")
        return 4

    # Neither matches numpy: both are wrong, or the convention differs from both.
    t = conv_np(x, w, N, C, H, W, K, R, S, transpose_filter=True).reshape(-1)
    td, tr = cmp(gpu, t)
    emit("INCORRECT", detail + " -> neither matches numpy; gpu_vs_transposed_numpy rel=%.3e" % tr)
    return 4


if __name__ == "__main__":
    sys.exit(main())
