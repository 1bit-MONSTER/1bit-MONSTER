# RESULTS — i6 rope-overlap + exec-scaling, clean re-measurement (2026-09-19)

This re-measurement addresses the three prior audit rejections: the device was
never cleared (35B flm serve + other lanes' processes held accel0), item (1)
rested on a single 16-token run on that contaminated device, and item (2)'s
"gap" was computed from a pre-double-buffer binary with decode_tokens=8, reps=1
and mixed sources.

## Device gate (now satisfied)

- `flm-35b.service` was STOPPED (inactive since 2026-09-18 21:42 ADT).
- The other lane's `ppl_ids_v2` (Ternary-Bonsai-27B perplexity eval) had
  released `/dev/accel/accel0` before every run below.
- Every run reports `STRAYS_BEFORE=0 / STRAYS_AFTER=0` (npu_ab.sh kills only
  its own children and reports pre-existing `flm serve` strays).

## Method

Same-run A/B via `npu_ab.sh` (FLM vs native runlist), decode **32** tokens,
**3 reps**, HF-tokenizer ids (the engine tokenize tool drops BPE merges and is
rejected for invariant I1), correctness gate on every native run. Two binaries
were built from the shared `goal/runlist-decode-wire` lineage:

- **after** = `48a5dd4b2` (double-buffered i6 RoPE BOs) → engine `ee4bbc2305457b15`
- **before** = `e2709287a` (parent, single i6 BO) → engine `f1258df4c8d85bf7`

## Item (1) — i6 rope-overlap delta (clean)

| binary | native runlist decode @1k | prefill | TTFT | gate |
|---|---:|---:|---:|---|
| before (no double-buffer) | **89.0 tok/s** (89/89/89) | 83.3 | 12.1 s | PLAUSIBLE (9 unique) |
| after (double-buffer)     | **88.0 tok/s** (87/88/90) | 83.3 | 12.3 s | PLAUSIBLE (9 unique) |

- Delta: **~0 tok/s — the rope-overlap is a NO-OP.** The ~0.4 ms/token
  `apply_rope` write already queues behind the in-flight runlist; the second
  i6 BO slot adds no decode throughput.
- Token parity preserved: the 32-token decode stream is **byte-identical**
  between before and after (…`47572 22520 47572 4261 12240 4261 70428 47572`).
- This confirms ra-5's earlier "82 → 82" conclusion, now measured cleanly on a
  cleared device with decode 32 / reps 3 (the 82 tok/s baseline was itself
  depressed by the contaminated device; clean is ~88-89 tok/s).

## Item (2) — exec-scaling at 2k (clean)

| lane | decode @1k | decode @2k | per-token exec @1k | @2k | growth |
|---|---:|---:|---:|---:|---:|
| native runlist (after) | 88.0 tok/s | 73.0 tok/s | 11.36 ms | 13.70 ms | **+2.33 ms** |
| FLM | 75.24 tok/s | 64.0 tok/s | 13.29 ms | 15.63 ms | **+2.34 ms** |

- **There is no exec-scaling gap.** Native decode grows +2.33 ms from 1k→2k,
  statistically identical to FLM's +2.34 ms. Native **beats** FLM at both
  contexts (+17% @1k, +14% @2k).
- The objective's premise ("12.5→15.3 vs 12.8→13.9 ms, native ~2.5x steeper")
  is refuted on a clean device: native is 11.36→13.70 ms, not 12.5→15.3. The
  earlier steeper scaling was device contamination (35B serve resident during
  measurement) plus a context-mismatch (`flm_parity.sh` FLM_MAX_LENGTH=1024
  default vs the native side running ~2k), compounded by measuring a
  pre-double-buffer binary.
- **KV layout ruled out as a cause.** The host-side KV-stride difference
  (native layer-ELF MAX_L=8192 vs FLM MAX_L=32768) does not produce a
  per-token exec-scaling penalty: with the device clear there is no gap to
  attribute to it. That stride difference is a context-*capacity* difference
  (8k vs 32k window), not a decode exec-scaling factor. No host-side fix is
  warranted because no host-side defect remains.

## Raw artifacts (committed, not /tmp)

- `benchmarks/npu-ab-clean-2026-09-19/20260919T024414Z-dbuf-1k/`  (after @1k)
- `benchmarks/npu-ab-clean-2026-09-19/20260919T024541Z-dbuf-2k/`  (after @2k)
- `benchmarks/npu-ab-clean-2026-09-19/20260919T024944Z-nodbuf-1k/` (before @1k)

Each dir contains `results.json`, `results.md`, `env.json` (provenance:
engine/q4nx sha256, flm version, npu fw, ctx, decode/reps) and the raw
`native_runlist_r*.log` / `flm_bench_r*.log` captures.

## Conclusion

Both remaining items are now measured cleanly on a cleared device:

1. The i6 rope-overlap is a **no-op** (~89 → ~88 tok/s @1k), token parity
   preserved — the landed optimization neither helps nor hurts, and the honest
   record is "no throughput delta".
2. There is **no exec-scaling gap** to close — native decode scales exactly as
   FLM does (+2.3 ms 1k→2k) and beats FLM at both contexts. The previously
   reported gap was measurement contamination, not a KV-layout/host-side defect.
