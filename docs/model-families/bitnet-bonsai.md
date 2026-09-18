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

- **NPU:** ternary 1.58-bit / TQ1 and TQ2 bridged to INT8 via `ternary_npu_bridge.h` (`pack_tq1_to_npu_int8()` / `pack_tq2_to_npu_int8()`) onto existing INT8 xclbin kernels; `mm_ternary_tq1.cc` / `mm_ternary_tq2.cc` are the on-tile LUT-decode microkernels for a true native 2-bit path, not yet the default — see the [NPU ternary roadmap](../research/npu-ternary-roadmap.md).
- **GPU:** Q1_0 1024-block kernel — 433 tok/s synthetic (kernel-level, HIP); 318 tok/s kernel-level on Vulkan ZINC.
- **CPU:** universal GGUF backend.

## Prism ML Bonsai 27B - hybrid GatedDeltaNet 1BP (2026-09-18) [in-repo]

A different family branch from the dense Deepgrove Bonsai above. Prism ML's 27B is a
**hybrid GatedDeltaNet** (Qwen3.6/3.8 base): 64 layers, 48 GDN + 16 gated-GQA, hidden
5120, 24 q / 4 kv heads, head_dim 256, GDN 16 k-heads / 48 v-heads (HK=HV=128), vocab
248320. Served through **1BP v5** with the three verbatim Prism packings plus the
folded-basis transform metadata (`__onebp_ext_prism_transform`). No MLX and no Prism
llama.cpp fork in the runtime loop; those are oracles/baselines only.

| Pack | Base | Weights | 1BP | Correctness | Greedy decode |
|---|---|---|---|---|---|
| `Bonsai-27B-Q1_0` | Qwen3.6 | 1-bit g128 (nb=18) | 3.80 GB | fork oracle 5/5; device forward 5/5 | **15.05 tok/s** |
| `Ternary-Bonsai-2-27B-PTQ1_0` | Qwen3.8 | base-3 g128 (nb=28), folded | 5.95 GB | fork oracle 5/5; device forward 5/5; per-layer cos >= 0.999 | **10.15 tok/s** |
| `Ternary-Bonsai-27B-PQ2_0` | Qwen3.6 | ternary g128 (nb=34) | 7.17 GB | fork oracle 5/5; device forward 5/5 | **12.20 tok/s** |

**Honesty tags.** Correctness results are on strixhalo gfx1151 and timing-immune: the
fork's own *generated* tokens are matched 5/5 on all three packs by both the CPU floor
and the full device forward, and the folded pack agrees per-layer with the in-repo numpy
reference at cosine >= 0.999 across all 64 layers. The **tok/s figures were measured with
other lanes active (load 3-20) and are relative only**; the lane's targets (>=42 / >=27 /
>=22 tok/s) require a quiet box and are not yet claimed. Kernel-level GEMV improved
23->80 (Q1_0), 25->93 (PTQ1_0), 35->93 (PQ2_0) GB/s with the tile kernel - ~40-46% of
the ~201 GB/s device triad ceiling. Plan of record:
[docs/plans/prism-bonsai-27b-custom-build.md](../plans/prism-bonsai-27b-custom-build.md).

**See also:** [block-scaled ternary format](../research/block-scaled-ternary-format.md) · [benchmarks SSOT](../wiki/performance.md) · [all families](README.md)
