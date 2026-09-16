#!/usr/bin/env python3
"""Independent audit of the captured attention kernel's mixing weights.

Companion to benchmarks/RESULTS-attn-kernel-audit-2026-09-13.md. Written from
scratch (numpy) so it does not merely re-run engine/npu/generators/check_attn.py;
it reproduces that tool's row-level numbers and then tests the assumptions the
"wrong softmax scale" claim rests on.

Inputs: the layer-0 attention I/O the engine dumps with NPU_DUMP_ATTNIO=1, from a
run in a *correct* configuration, e.g.

  NPU_RUNLIST=0 NPU_PREFILL_BF16=1 NPU_PREFILL_MAX=256 NPU_DUMP_ATTNIO=1 \
    engine/npu/build/npu_engine_qwen3_0_6b <model.q4nx> 1 ids256.txt
  -> expect boot=1614 at 256 tokens (the trusted paths' token)

  /tmp/eng_act.bin = Q   (256 x 2048 bf16, [token][head][dim], post-RoPE)
  /tmp/eng_kv.bin  = K/V (4 regions x 4194304 bf16; r0/r1 = K heads 0-3/4-7,
                          r2/r3 = V heads 0-3/4-7; 512 values per token)
  /tmp/eng_out.bin = the kernel's attention output (same shape as Q)

Copy the dumps somewhere persistent first: the engine hardcodes /tmp.

Usage: attn_audit.py <dumpdir> [--map]
"""
import numpy as np
import sys

NROWS, NH, NKV, HD = 256, 16, 8, 128
QOUT = NH * HD
GQA = NH // NKV
KV_REGION = 4194304
REF_SCALE = 1.0 / np.sqrt(HD)


def load_bf16(path, count):
    raw = np.fromfile(path, dtype="<u2", count=count)
    assert raw.size == count, (path, raw.size, count)
    return (raw.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def load(dumpdir):
    q = load_bf16(f"{dumpdir}/eng_act.bin", NROWS * QOUT).reshape(NROWS, NH, HD)
    kv = load_bf16(f"{dumpdir}/eng_kv.bin", 4 * KV_REGION)
    out = load_bf16(f"{dumpdir}/eng_out.bin", NROWS * QOUT).reshape(NROWS, NH, HD)
    return q, kv, out


def k_of(kv, tok, kvh):
    """K vector for kv-head kvh at token tok. K lives in regions 0/1."""
    reg, within = kvh // 4, kvh % 4
    off = reg * KV_REGION + tok * 512 + within * HD
    return kv[off:off + HD]


def v_of(kv, tok, kvh):
    """V vector for kv-head kvh at token tok. V lives in regions 2/3."""
    reg, within = kvh // 4, kvh % 4
    off = (reg + 2) * KV_REGION + tok * 512 + within * HD
    return kv[off:off + HD]


def reference(q, kv, scale=REF_SCALE, nrows=NROWS):
    """Causal GQA with the given softmax scale, from the dumped Q/K/V."""
    res = np.zeros((nrows, NH, HD))
    for i in range(nrows):
        for h in range(NH):
            kvh = h // GQA
            k = np.stack([k_of(kv, t, kvh) for t in range(i + 1)])
            v = np.stack([v_of(kv, t, kvh) for t in range(i + 1)])
            s = (k @ q[i, h]) * scale
            s -= s.max()
            w = np.exp(s)
            w /= w.sum()
            res[i, h] = w @ v
    return res


def span_residual(target, V):
    """Least-squares residual of target against the span of rows of V."""
    w, *_ = np.linalg.lstsq(V.T, target, rcond=None)
    return float(np.linalg.norm(V.T @ w - target) / np.linalg.norm(target))


def main():
    dumpdir = sys.argv[1] if len(sys.argv) > 1 else "/tmp"
    do_map = "--map" in sys.argv
    q, kv, out = load(dumpdir)

    ref = reference(q, kv)
    row_max = np.abs(ref - out).max(axis=(1, 2))
    matching = int((row_max < 0.05).sum())
    print(f"dumps: {dumpdir}   rows={NROWS} heads={NH}")
    print(f"reference scale 1/sqrt({HD}) = {REF_SCALE:.6f}")
    print(f"rows matching reference (<0.05): {matching}/{NROWS}")
    print("worst rows:", [(int(i), round(float(row_max[i]), 4))
                          for i in np.argsort(-row_max)[:8]])
    print("row 0 (DEGENERATE: one key, output is V0 by construction): "
          f"{float(row_max[0]):.4f}")
    print(f"row 255: {float(row_max[255]):.4f}")

    # per-head implied softmax scale on row 1 (two keys) under the checker's map
    i = 1
    print("\nrow 1 implied softmax scale per head (checker's kvh = head//2):")
    scales = []
    for h in range(NH):
        kvh = h // GQA
        tgt = out[i, h]
        v0, v1 = v_of(kv, 0, kvh), v_of(kv, 1, kvh)
        den = v1 - v0
        sel = np.abs(den) > 0.05 * np.abs(v1).max()
        if sel.sum() < 8:
            continue
        w1 = float(np.mean((tgt[sel] - v0[sel]) / den[sel]))
        l0 = float(k_of(kv, 0, kvh) @ q[i, h])
        l1 = float(k_of(kv, 1, kvh) @ q[i, h])
        if not (1e-6 < w1 < 1 - 1e-6) or abs(l1 - l0) < 1e-6:
            print(f"  h{h:02d}: implied w1={w1:.4f} — not invertible (peaked row)")
            continue
        s = np.log(w1 / (1 - w1)) / (l1 - l0)
        scales.append(s)
        print(f"  h{h:02d} kvh{kvh}: implied w1={w1:.4f}  implied scale={s:.5f}"
              f"  ({s / REF_SCALE:.2f}x reference)")
    if scales:
        a = np.array(scales)
        print(f"  spread: {a.min():.5f} .. {a.max():.5f} "
              f"({a.min() / REF_SCALE:.2f}x .. {a.max() / REF_SCALE:.2f}x). "
              "A single wrong scale would be uniform across heads.")

    if do_map:
        print("\nhead -> kv-head mapping check (is the checker's map the right one?)")
        wrong = 0
        for h in range(NH):
            fits = []
            for j in range(NKV):
                V = np.stack([v_of(kv, t, j) for t in range(i + 1)])
                fits.append((span_residual(out[i, h], V), j))
            fits.sort()
            best_res, best_j = fits[0]
            if best_j != h // GQA or best_res > 0.10:
                wrong += 1
                print(f"  h{h:02d}: best kvh={best_j} resid={best_res:.4f} "
                      f"(expected {h // GQA})")
        print(f"  heads whose output does NOT fit their expected kv-head V-span: "
              f"{wrong}/{NH}")


if __name__ == "__main__":
    main()
