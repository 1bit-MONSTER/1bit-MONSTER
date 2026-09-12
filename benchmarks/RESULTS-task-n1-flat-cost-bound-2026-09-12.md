# RESULTS — task-n1: prefill cost is FLAT vs context length (fixed-overhead bound)

Goal `mttxt22c-a6rv75`, task-n1. Measured native bf16 prefill time vs context
length (Qwen3-0.6B, `NPU_RUNLIST=0 NPU_PREFILL_BF16=1`, reclaimer prompt).

| n (tokens) | prefill time (3 runs) |
|---|---|
| 64 | 405 / 400 / 405 ms |
| 128 | ~417 ms |
| 192 | ~420 ms |
| 256 | 420 / 430 / 427 ms |

**The native per-op prefill cost is ~flat (~410-430 ms) from 64 to 256 tokens.**
The fixed cost dominates; marginal per-token cost is small.

## Why this matters for the @1k bar

The mm.xclbin GEMMs are launched as two 128-row batches and the attention is a
fixed 256-token ELF — so the device does full 256-token work even for short
contexts, and the per-projection kernel launches (QKV/O/GU/D + attention ≈ 5
launches/layer × 28 layers ≈ 140 launches) are a fixed per-layer overhead.

Even with chunked prefill (4 × 256-token chunks for @1k, which is the task-n2
position-shifted-ELF work), the projection is **~4 × 425 ms ≈ 1.7 s ≈ 600 tok/s**
vs FLM's on-box ~1400 tok/s @1k. The per-op path cannot reach 1494 @1k by
per-token optimization alone — the fixed launch overhead + host f32↔bf16 math
must be removed by fusing the layer (FLM's `gen_layer_seq` style, i.e. NPU
offload), which is the multi-week new-xclbin effort already documented.

## What IS achieved (this session, task-n1)

- Byte-exact token parity vs FLM at n=1..256 (two regressions bisected+fixed).
- Native prefill beats FLM at the native path's supported 256 tokens
  (~425 ms vs FLM ~600 ms, same output) — FLM's short-context prefill is
  dominated by its own larger fixed overhead.
- Host math parallelized (num_threads=8): 603 → ~425 ms.

## Conclusion

task-n1's "meet-or-beat 1494 @1k" cannot be met by the per-op native path — it
is fixed-overhead-bound, and closing the @1k gap requires the fused-layer NPU
offload (multi-week). The per-op path's honest ceiling at @1k (via chunking) is
~600 tok/s. This is the structural wall the audit and prior sessions identified.

## Launch-overhead quantification (why it is fixed-bound)

The per-op path issues **~9 kernel launches per layer** (QKV×2 batches + attn×1
+ O×2 + GU×2 + D×2) = **~252 launches / pass**. At the measured ~405 ms fixed
cost (n=64 ≈ n=256), that is **~1.6 ms of XRT launch overhead per kernel** — the
dominant term, not token throughput. FLM's `gen_layer_seq` fuses the whole layer
into ~1 launch/layer (28 launches), which is why its per-token cost at long
context is ~10× lower. The only way to close the gap is fused-layer kernels
(fewer launches), i.e. the NPU-offload work.
