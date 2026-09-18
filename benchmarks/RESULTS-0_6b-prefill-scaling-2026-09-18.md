# The one (c) cell that does not pass: 0.6B TTFT at 8k, classified — 2026-09-18

Goal `mu35shsg-i3hlyi`. With the 8k rows in hand (`RESULTS-8k-six-model-2026-09-18.md`,
`RESULTS-8k-decode-top-of-window-2026-09-18.md`), 17 of the 18 model-metric cells at ~8k context
pass. The exception is **Qwen3-0.6B TTFT**, where native is 1.8–3.0% slower than FLM. This note
classifies it rather than leaving it as an unexplained miss.

## Four-point scaling, 0.6B, serial, same session

Native `Prefill: X ms` (two runs) against FLM `ttft` at the same context, `--decode-tokens 1`:

| ctx | native (2 runs) | native median | FLM ttft | native/FLM |
|---|---:|---:|---:|---:|
| 1k (1024) | 540, 554 ms | 547 ms | 0.751975 s | **0.73x (native 27% faster)** |
| 2k (2048) | 863, 887 ms | 875 ms | 0.997534 s | **0.88x (native 12% faster)** |
| 4k (4096) | 1714, 1758 ms | 1736 ms | 1.785345 s | **0.97x (native 2.7% faster)** |
| 8k (8192) | 4169, 4028 ms | 4098 ms | 3.979976 s | **1.03x (native 3% slower)** |

The campaign's own 8k pair (median of accepted runs) was 4139 ms vs 4066.5 s → 1.018x, so the
deficit at 8k is 1.8–3.0% depending on which FLM sample is taken. It is **not** a cold-start or
first-touch artifact: the deficit appears only at the top of the range and grows monotonically
with context, while 1k–4k are all native wins.

## Decomposition

Marginal cost between the measured points (ms per additional token):

| interval | native | FLM | native/FLM |
|---|---:|---:|---:|
| 1k → 2k | 0.320 | 0.240 | 1.33x |
| 2k → 4k | 0.420 | 0.384 | 1.09x |
| 4k → 8k | 0.577 | 0.536 | 1.08x |

Both curves are super-linear (attention work grows with context), and native's marginal cost is
8–33% higher across the whole range. What native has instead is a **much smaller fixed
overhead**: extrapolating the 1k–2k chord back to ctx=0 gives native ≈ 220 ms and FLM ≈ 500 ms,
i.e. FLM carries ~200 ms more context-independent startup per request. That is the entire reason
1k looks like a 27% native win: at 1k the fixed term dominates, and by ~5k tokens the marginal
term takes over:

```
native(t) ≈ 220 ms + m_n(t)·t         FLM(t) ≈ 500 ms + m_f(t)·t
crossover where 220 + 0.32t = 500 + 0.24t  ->  t ≈ 3.6k tokens   (1k-2k chords)
crossover where 220 + 0.58t = 500 + 0.54t  ->  t ≈ 8.1k tokens   (4k-8k chords)
```

so the crossing sits in the 4k–8k band, which is exactly what is measured: native ahead at 4k
(0.97x), behind at 8k (1.02–1.03x).

## Where the marginal cost is

Native's own 8k prefill breakdown for this model is `Prefill: 4139ms [GEMM 379ms, attn 2347ms,
conv+other 3968ms]` — the parts overlap across the host threads, but the **attention term is
2347 ms of the total and is the term that grows fastest with context**. The 1k→2k slope
difference (1.33x) is set by the long-context attention path, not by the GEMMs (379 ms at 8k for
8192 tokens is 0.046 ms/token, nowhere near the 0.58 ms/token marginal).

So the residual cell's optimisation target is the same subsystem as phase-2's attention work: the
generated attention kernel's cost at long context, where the fixed ~6 s per-launch device cost
and the `N<=512` softmax contract live. Nothing in this note is a reason to reopen the passing
cells — it names the one address at which native is behind and the reason the deficit is invisible
at 1k.

## Status of criterion (c) after this note

- **1k: met**, all three metrics for all six models (prefill 1.27–1.47x, TTFT faster,
  decode 1.01–1.11x).
- **8k (~8192): 17/18 cells met** — prefill ahead 1.04–1.10x for all six; decode at parity or
  ahead for all six; TTFT at parity-or-faster for five.
- **The one failing cell:** 0.6B TTFT at 8k, 1.8–3.0% behind, classified above as a
  marginal-long-context-attention cost offset by ~200 ms less fixed startup, with the crossover
  at ~4k–8k tokens and the target (the attn term at long context) named.

## Reproduce

```
python3 -c 'ids=[int(x) for x in open("/tmp/p_8k.txt").read().split()]
for k in (1,2,4): open("/tmp/p_%dk.txt"%k,"w").write(" ".join(map(str,ids[:k*1024])))'
# native: for k in 1 2 4 8; do for r in 1 2; do
#   env NPU_PREFILL_BF16=1 NPU_BF16=1 NPU_GREEDY=1 NPU_PREFILL_MAX=8192 NPU_PROMPT_MAX=8192 \
#     ./engine/npu/build/npu_engine_qwen3_0_6b ~/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx 1 /tmp/p_${k}k.txt
# done; done
# FLM: npu_ab.sh --model qwen3_0_6b --flm-tag qwen3:0.6b --prompt /tmp/p_${k}k.txt --ctx-k $k \
#        --decode-tokens 1 --reps 1 --skip-native
```
