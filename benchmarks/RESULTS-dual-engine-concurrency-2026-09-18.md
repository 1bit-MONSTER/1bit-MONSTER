# Post-goal validation: several models loaded at once on the NPU, and what a local MoE makes of it — 2026-09-18

Not part of the audited goal (tip `f20cb3da1`); this is follow-on validation requested after it.
Question: with the engine now at parity with FastFlowLM single-instance, what happens when **two or
three engine processes hold the NPU at the same time** — does throughput aggregate, and do the
answers stay correct?

## Method

`~/dual2.sh` (1k context) and `~/dual8k.sh` (8160-token context). The engine takes an exclusive
`flock` on `/tmp/1bit-npu-device.lock` for its whole lifetime, so concurrent legs opt out with
`NPU_NO_DEVICE_LOCK=1` **deliberately** — the default behaviour is to wait for the lock and
serialise, which is the right default for measurements and the wrong one for this experiment.
Rates are each instance's own reported `ms/tok`, so process startup and weight dequantise are not
counted. Every concurrent instance's token stream was compared byte-for-byte with its serial run
of the same model, prompt and seed.

## Concurrency at 1k context (1024-token prompt, 128 decode tokens, greedy)

| config | per-instance decode | aggregate | each instance vs its solo rate |
|---|---|---:|---|
| serial: 0.6B / 1.7B / 4B | 77 / 39 / 18 tok/s | — | — |
| 2 × 0.6B | 34 + 34 | **68 tok/s** | 44% each (one device's worth of work, split) |
| 0.6B + 1.7B | 56 + 39 | **95 tok/s** | 0.6B 73%, **1.7B 100%** |
| 0.6B + 1.7B + 4B | 54 + 32 + 18 | **104 tok/s** | 70% / 82% / **100%** |

**Tokens: 6 of 6 instances byte-identical to their serial runs. No ERT/TDR/timeout line in any
log.** Correctness survives concurrency; this is the part that validates the single-instance work,
because the same binary and the same kernels produce the same tokens when a second engine is
competing for the device.

What the numbers say for serving: the device **shares, it does not serialise** (2×0.6B = 68 tok/s,
i.e. 88% of one instance's 77), and the productive case is *mixed sizes* — a small model riding
alongside a large one hardly taxes it, while the large one keeps its full solo rate. As system
throughput for a mixed workload (finish one 0.6B request and one 1.7B request, each 128 tokens):
**78 tok/s concurrent vs 52 tok/s running them one after the other = 1.5x**, and for the three-model
mix 54 vs 32 tok/s = **1.7x**.

## Concurrency at 8160 tokens of context — where the prefill is host-bound

| instance | serial prefill / decode | concurrent prefill / decode |
|---|---|---|
| Qwen3-0.6B | 3897 ms / 35 tok/s | **8676 ms (+123%)** / 27 tok/s (−23%) |
| Qwen3-1.7B | 5523 ms / 24 tok/s | **9911 ms (+79%)** / 24 tok/s (0%) |

Tokens still identical, still no device errors — but the wall clock for the pair is **18.3 s
concurrent against 11.7 s serial**, so at this context concurrency is a **0.64x loss**. The reason
is the one the parity work already localised: at 8192 tokens the prefill is host-bound per-element
work (`RESULTS-0_6b-prefill-hostbound-2026-09-18.md`), and two processes simply compete for the
same 16 cores. Decode still shares fine (27 + 24 tok/s).

**Operational reading: multi-model concurrency pays for decode-bound (short-context) work and
punishes long-context prefill.** Add a second model for chat-shaped traffic; do not for
long-document traffic.

## What the local MoE made of it

The box's own MoE (`qwen3.6-moe:35b-a3b` on FastFlowLM) was handed the table and asked to aggregate
and to say what the identity result does and does not validate. It got the arithmetic right — the
efficiencies (92% / 82% / 85% by its chosen definition), the 104 tok/s winner, and the 5.78x gain of
the three-model mix over the 4B alone — and it correctly flagged that a "sharing efficiency" against
independent linear scaling is not the right baseline for identical models on one device.

**It then inverted the direction of the parity result**, reading native's 1.34–1.51x *prefill rate*
ratio as a "1.34–1.51x slowdown" and FLM as having the better TTFT, when the shipped convention is
ratio = native/FLM with <1 meaning native is faster (native TTFT is 19–29% faster at 1k and 0.7–5.9%
faster at 8k). Given the convention spelled out explicitly, it stated the correction itself.

Recorded as-is because it is the useful lesson: **the MoE is a competent arithmetic aggregator and
an unreliable reader of ratio conventions.** Any use of a local model as a benchmark validator has
to state the direction of every ratio; left implicit, it will confidently report the opposite
conclusion.

## Reproduce

```bash
bash ~/dual2.sh      # 1k context, 2x same model, 0.6B+1.7B, 0.6B+1.7B+4B, correctness per instance
bash ~/dual8k.sh     # 8160-token context, 0.6B+1.7B, prefill and decode
# engine knobs: NPU_NO_DEVICE_LOCK=1 (concurrent legs only), NPU_PREFILL_BF16=1 NPU_BF16=1
# NPU_GREEDY=1, NPU_PREFILL_MAX/NPU_PROMPT_MAX=8192 for the long-context run
```
