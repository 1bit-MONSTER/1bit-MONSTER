# BitNet / Bonsai — Ternary-Native (TQ2)

Deepgrove's Bonsai models are ternary b1.58 — weights constrained to {−1, 0, +1}. These are the **only** family that uses the 1BP **TQ2** 2-bit format natively (dense models use Q4NX; see the [format policy](../wiki/models.md#1bp-format-policy-2026-07-31-verdict-ppl-measured)). TQ2 gives a 4× DDR-bandwidth saving over INT8.

## Models

| Model | Params | 1BP Size | Backend(s) | Perf |
|-------|:------:|:--------:|------------|:----:|
| **Bonsai-1.7B** | 1.7B | 841 MB | HIP / Vulkan ZINC | 21.9 tok/s (HIP), 21.7 (ZINC) |
| **Bonsai-4B** | 4B | 2.2 GB | HIP | 🚧 integration in progress |
| **Bonsai-8B** | 8B | 4.1 GB | HIP | 🚧 integration in progress |
| **Bonsai-27B** | 27B | 15 GB | HIP | 🔬 experimental |

## Notes

- **NPU:** ternary 1.58-bit / TQ1 via `mm_ternary_tq1.cc` (base-3 LUT decode + ping-pong); TQ2 models via `gemm_generate_sequence_tq2()` and `--native-tq2` in `npu_ternaryd.cpp`.
- **GPU:** Q1_0 1024-block kernel — 433 tok/s synthetic (kernel-level, HIP); 318 tok/s kernel-level on Vulkan ZINC.
- **CPU:** universal GGUF backend.

**See also:** [block-scaled ternary format](../research/block-scaled-ternary-format.md) · [benchmarks SSOT](../wiki/performance.md) · [all families](README.md)
