# Why 0.6B TTFT at 8k is 1–3% behind: the prefill is host-bound, and the device path already hides — 2026-09-18

Goal `mu35shsg-i3hlyi`, criterion (c). `RESULTS-0_6b-prefill-scaling-2026-09-18.md` classified the
one non-passing cell as a marginal-long-context cost. This note localises it to a term and rules
out the two candidate fixes, using the engine's own prefill breakdown (the breakdown is computed
from the same invocation as the number, per I2).

## The breakdown, four contexts, 0.6B, serial, one process per point

| ctx | total | GEMM | attn | conv+other |
|---|---:|---:|---:|---:|
| 1k | 554 ms | 65 | 188 | **547** |
| 2k | 893 ms | 115 | 326 | **883** |
| 4k | 1738 ms | 212 | 769 | **1723** |
| 8k | 4030 / 4076 / 4094 ms | 390 / 396 / 402 | 2339 / 2341 / 2341 | **4007 / 4052 / 4071** |

In every row the total equals the `conv+other` term (547 vs 554, 883 vs 893, 1723 vs 1738, 4007
vs 4030). The attention term is not only not the bottleneck, it is **almost entirely hidden**:
`conv+other + attn` would be 6380 ms at 8k against a measured 4030 ms, i.e. ~96% of the device
attention is overlapped with host work. The exposed device cost at 8k is ~25 ms.

`conv+other` costs ~**0.42–0.53 ms per prompt token** at every context (flat, not growing), so it
is a per-token host cost, not a per-request setup: 8k costs less per token than 1k (0.489 vs
0.541), which also rules out a one-time 8k setup charge hiding in the prefill.

## What that means for the cell

At 8192 tokens, per-token prefill:

| arm | per-token |
|---|---|
| native (3 runs) | 0.4919, 0.4976, 0.4998 ms |
| FLM (3 samples) | 0.4858, 0.4931, 0.4964 ms |

So the cell is a **near-tie at ~0.49 ms/token**, native 0.9–2.9% higher, having been 27% *ahead*
at 1k (0.541 vs 0.734 ms/token) on the strength of ~200 ms less fixed cost. Native's advantage
is fixed-cost; FLM's is a marginally lower per-token cost that only shows once context is long
enough to amortise its startup.

## The two candidate fixes, both ruled out by this data

1. **Pipeline the prefill batches (double-buffer the activation BO).** The structural reason to
   expect a win was real — the async path in this engine is serialised by a single activation BO
   per context (`npu_engine_universal.cpp:1771` documents it), and the same double-buffering is
   what produced the decode-overlap fix. But the ceiling is the *exposed* device time, which is
   already down to ~25 ms at 8k. Best case it recovers 0.6% of a 1–3% gap. Not worth the risk.
2. **Run the prefill on the device instead of the host.** The int8 runlist layer kernels do
   exactly that and are fast, but this is the path whose oracle accuracy was **dense 1/20** — the
   reason the bf16 host prefill exists at all. Every speed number must ship with a passed
   correctness gate, so this trade is not available as a parity claim.

## The lever that is actually left

`conv+other` at ~0.45 ms/token for a 0.6B model on 8 threads is roughly **3 GFLOPS effective**
(1.7 GFLOP/token), i.e. two to three orders of magnitude under what an 8-core AVX-512 part does
on dense bf16 arithmetic. The host prefill path is therefore memory- or format-bound, and the
lever is in that path (its GEMM blocking, bf16↔int8 tile conversion, and per-layer
RMSNorm/RoPE/SiLU/quantise pipeline), not in the attention kernel. Host threads are already at
their optimum: `NPU_HOST_THREADS` default 8, with 12/16/24 all worse and **32 a ~5x regression
(20.4 s vs 4.1 s)** — a cliff, recorded here because `nproc`-many threads looks like a free win.

This is a real optimisation project on the host prefill path, not a bounded verification, and it
is outside the two phases the objective names (which target the attention kernel and the window).
It is therefore raised with the user rather than started: it would occupy the box for hours of
development plus measurement, past the objective's ~2 h stop-and-ask threshold.

## Status it leaves criterion (c) in

- **1k: met**, all six models, all three metrics.
- **8k: 17 of 18 cells met** (prefill ahead 1.04–1.10x for all six; decode parity-or-ahead for
  all six; TTFT parity-or-faster for five).
- **The one cell:** 0.6B TTFT at 8k, per-token 0.492–0.500 ms vs FLM 0.486–0.496 ms, i.e. a
  near-tie falling 1–3% short, localised to the host `conv+other` prefill term, with the two
  candidate fixes ruled out above and the remaining lever named.

## Cross-model evidence: the host term is per-element, not per-FLOP

The host term's cost is **proportional to `layers × hidden`**, not to `layers × hidden²`. Per-token
prefill at 8k across all six models (native medians from `RESULTS-8k-six-model-2026-09-18.md`),
with the geometry each model declares in `~/.config/flm/models/<M>/config.json`:

| model | layers | hidden | native ms/tok @8k | µs per 1000 layer·hidden elements |
|---|---:|---:|---:|---:|
| Qwen3-0.6B | 28 | 1024 | 0.500 | 17.4 |
| Qwen3-1.7B | 28 | 2048 | 0.691 | 12.1 |
| Qwen3-4B | 36 | 2560 | 1.605 | 17.4 |
| Qwen3-VL-4B | 36 | 2560 | 1.568 | 17.0 |
| Qwen3-8B | 36 | 4096 | 2.163 | 14.7 |
| Llama-3.1-8B | 32 | 4096 | 2.162 | 16.5 |

That ratio is constant to ±30% across a 4x span of hidden size and 28–36 layers. Arithmetic-bound
work cannot behave that way: at fixed layer count, doubling hidden (0.6B → 1.7B) doubles the
matmul FLOPs but only moves the per-token cost 0.500 → 0.691 ms (1.38x), and 4x the hidden
(0.6B → 8B) moves it 0.500 → 2.163 ms (4.3x) rather than ~16x. The host term is per-element
traffic, not per-FLOP maths.

**And it is large compared with the part that does scale with FLOPs.** At 8k on 0.6B, the
device bf16 GEMMs — the `H²` work, all 28 layers — are measured at 390–402 ms, while the
per-element host path is 4007–4071 ms. For this model the per-element bookkeeping costs **ten
times** the matrix multiplies.

The KV write alone is the right order of magnitude to explain it: every model here writes
`layers × 2 × 8 kv-heads × 128 head-dim` bf16 elements per prompt token, i.e. 4 KB per token per
layer, so 8192 tokens × 28 layers = **918 MB at 0.6B**, which at the box's measured triad
(205–217 GB/s) is ~4.2 s against a measured host term of 4.0 s. That is a consistency check, not
a proven attribution — the same traffic is also what the int8 tile conversion and the activation
round-trips between the bf16 prefill and the runlist decode consume — but it does exclude
"the conv kernels are slow at arithmetic" as the explanation.

Recorded as the state of the lever, not as a fix: any change here alters the prefill path that
carries the accuracy gate, so it needs the oracle scoreboard re-run per model and is a multi-hour
step (raised with the user).
