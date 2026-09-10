#!/usr/bin/env python3
"""Same-file correlation gate for device-vs-reference logit dumps (#2139 lanes).

Usage: corr_assert.py <gpu_dir> <cpu_dir> [chain_file] [floor]

Reference/cross-check tool: it asserts methodology BEFORE scoring, so a bad
comparison fails loudly instead of producing a plausible number. Dump dumps are
raw little-endian float32, one file per position (pNNNN.f32).

  - position counts must match on both sides
  - dump widths must match (derived from file size, not assumed: a 248,320-f32
    writer mixed with another width is a hard error)
  - if chain_file is given, dump count must equal the chain length; chains may be
    comma- OR whitespace-separated (both forms are used by neighbouring artifacts)
  - first-5 corr tripwire (strict, narrow): a misaligned chain collapses to ~0.56;
    below 0.99 the run aborts as METHODOLOGY SUSPECT, naming the offending index
  - per-position FLOOR on the minimum (loose, wide): catches a single collapsed
    position that a mean would hide. Default 0.90, an argument so cross-quant
    comparisons can lift or lower it.
  - relL2/maxabs are REPORTED but NOT GATED: pearson and argmax are
    affine-invariant, so a systematic gain/offset is invisible to them while it
    changes the sampling distribution above temperature 0 (dropped dequant scale,
    wrong group scale, bf16/fp16 scale rounding on a fused kernel are all affine
    classes that a GPU port introduces and argmax+corr cannot see). The good-run
    floor here is ~5% relative L2 (implementation difference), so gate only after
    observing a few runs' distribution.

Provenance: built for #2139 (q35 lanes) by agent-44437c, hardened with
@agent-ec855d (min-floor / affine blind spot / chain-separator and index
diagnostics) and @agent-ca60cf (alignment tripwire idea). Measured populations on
the Q8_0 102-position pair: good relL2 mean 0.048983, affine(gain 1.10/bias +0.5)
0.127397; good maxabs mean 0.958349, affine 2.606212 — with byte-identical
pearson/argmax lines, which is the point.
"""

import glob, os, struct, sys, re

import numpy as np


def width_of(path):
    return os.path.getsize(path) // 4


def main(gpu_dir, cpu_dir, chain_file=None, floor=0.90):
    floor = float(floor)
    g = sorted(glob.glob(os.path.join(gpu_dir, "p*.f32")))
    c = sorted(glob.glob(os.path.join(cpu_dir, "p*.f32")))
    assert g, "no dumps in %s" % gpu_dir
    assert c, "no dumps in %s" % cpu_dir
    assert len(g) == len(c), "position count mismatch: gpu=%d cpu=%d" % (len(g), len(c))

    wg, wc = width_of(g[0]), width_of(c[0])
    assert wg == wc, "WIDTH MISMATCH: gpu=%d f32 cpu=%d f32 (different writers)" % (wg, wc)

    if chain_file:
        raw = open(chain_file).read()
        # accept comma- OR whitespace-separated chains (neighbouring artifacts use
        # both: /tmp/ids130.txt is space-separated, ids.txt is comma-separated)
        ids = [t for t in re.split(r"[,\s]+", raw) if t.strip()]
        if len(ids) != len(g):
            head = ids[0] if ids else "<empty>"
            raise AssertionError(
                "CHAIN/Dump COUNT MISMATCH: parsed %d id(s) from %s vs %d dumps (first parsed "
                "token: %r). The chain must be comma- or whitespace-separated; if the count is "
                "1, the file probably uses a separator this parser does not split."
                % (len(ids), chain_file, len(g), head))
        print("chain: %d ids | first=%s last=%s" % (len(ids), ids[0], ids[-1]))
    print("positions=%d width=%d f32" % (len(g), wg))

    # first-5 tripwire (alignment sanity before scoring) — reports the offending index
    early = []
    for idx, (gi, ci) in enumerate(list(zip(g, c))[:5]):
        a = np.fromfile(gi, dtype=np.float32); b = np.fromfile(ci, dtype=np.float32)
        early.append((float(np.corrcoef(a, b)[0, 1]), idx))
    print("first-5 corr:", " ".join("%.6f" % e for e, _ in early))
    emin, eidx = min(early)
    if emin < 0.99:
        print("METHODOLOGY SUSPECT: first-%d min corr %.4f at position %d < 0.99 — chain "
              "misalignment or width mixup; refusing to score." % (len(early), emin, eidx))
        return 2

    corrs, agrees = [], 0
    worst = None
    rels, maxabs_v = [], []
    worst_rel = None
    for gi, ci in zip(g, c):
        a = np.fromfile(gi, dtype=np.float32); b = np.fromfile(ci, dtype=np.float32)
        r = float(np.corrcoef(a, b)[0, 1])
        am_a, am_b = int(a.argmax()), int(b.argmax())
        if am_a == am_b:
            agrees += 1
        corrs.append(r)
        if worst is None or r < worst[0]:
            worst = (r, os.path.basename(gi), am_a, am_b)
        # scale-sensitive statistics: pearson + argmax are affine-invariant, so a
        # systematic gain/offset is invisible to them (ec855d 2026-09-10). These are
        # REPORTED but deliberately NOT gated yet — the good-run floor measured here is
        # ~5% relative L2 (implementation difference), so gate only after a few runs.
        d = a - b
        denom = float(np.linalg.norm(b)) or 1.0
        rl = float(np.linalg.norm(d) / denom)
        ma = float(np.max(np.abs(d)))
        rels.append(rl); maxabs_v.append(ma)
        if worst_rel is None or rl > worst_rel[0]:
            worst_rel = (rl, os.path.basename(gi), ma)
    print("mean=%.6f min=%.6f argmax=%d/%d worst=%s corr=%.6f gpu=%d cpu=%d"
          % (float(np.mean(corrs)), float(np.min(corrs)), agrees, len(corrs),
             worst[1], worst[0], worst[2], worst[3]))
    print("[informational, not gated] relL2 mean=%.6f max=%.6f (worst %s %.6f) | "
          "maxabs mean=%.6f max=%.6f (worst %s)"
          % (float(np.mean(rels)), float(np.max(rels)), worst_rel[1], worst_rel[0],
             float(np.mean(maxabs_v)), float(np.max(maxabs_v)), worst_rel[1]))

    # per-position FLOOR: the mean can hide a single collapsed position (a scrambled
    # dump reads ~0.61 inside a 0.99 mean), so gate the minimum, not just the mean.
    if worst[0] < floor:
        print("SUSPECT: worst position %s corr=%.6f < floor %.2f (argmax gpu=%d cpu=%d) — "
              "refusing to pass. Override the floor as the 4th argument if this is an "
              "expected cross-quant case." % (worst[1], worst[0], floor, worst[2], worst[3]))
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
