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
| `Bonsai-27B-Q1_0` | Qwen3.6 | 1-bit g128 (nb=18) | 3.80 GB `[Bonsai-27B-Q1_0\|Q1_0\|1BP v5\|cpu-host\|-\|-\|2026-09-18]` | fork oracle 5/5; device forward 5/5 `[Bonsai-27B-Q1_0\|Q1_0\|CPU floor + HIP forward\|strixhalo-busy\|5\|capital-of-France\|2026-09-18]` | **15.05 tok/s** `[Bonsai-27B-Q1_0\|Q1_0\|HIP device forward\|strixhalo-busy\|16\|capital-of-France\|2026-09-18]` |
| `Ternary-Bonsai-2-27B-PTQ1_0` | Qwen3.8 | base-3 g128 (nb=28), folded | 5.95 GB `[Ternary-Bonsai-2-27B-PTQ1_0\|PTQ1_0\|1BP v5\|cpu-host\|-\|-\|2026-09-18]` | fork oracle 5/5; device forward 5/5; per-layer cos >= 0.999 `[Ternary-Bonsai-2-27B-PTQ1_0\|PTQ1_0\|CPU floor + HIP forward\|strixhalo-busy\|5\|capital-of-France\|2026-09-18]` | **10.15 tok/s** `[Ternary-Bonsai-2-27B-PTQ1_0\|PTQ1_0\|HIP device forward\|strixhalo-busy\|16\|capital-of-France\|2026-09-18]` |
| `Ternary-Bonsai-27B-PQ2_0` | Qwen3.6 | ternary g128 (nb=34) | 7.17 GB `[Ternary-Bonsai-27B-PQ2_0\|PQ2_0\|1BP v5\|cpu-host\|-\|-\|2026-09-18]` | fork oracle 5/5; device forward 5/5 `[Ternary-Bonsai-27B-PQ2_0\|PQ2_0\|CPU floor + HIP forward\|strixhalo-busy\|5\|capital-of-France\|2026-09-18]` | **12.20 tok/s** `[Ternary-Bonsai-27B-PQ2_0\|PQ2_0\|HIP device forward\|strixhalo-busy\|16\|capital-of-France\|2026-09-18]` |
| `Ternary-Bonsai-2-27B-PQ2_0` | Qwen3.8 | ternary g128 (nb=34), folded | 7.14 GB `[Ternary-Bonsai-2-27B-PQ2_0\|PQ2_0\|1BP v5\|cpu-host\|-\|-\|2026-09-18]` | container byte-exact vs source GGUF; not yet oracle-gated `[Ternary-Bonsai-2-27B-PQ2_0\|PQ2_0\|1BP v5 verifier\|cpu-host\|-\|-\|2026-09-18]` | not measured |

**Measured quality (PPL, timing-immune).** Perplexity over the WS-00 gate corpus at 128
tokens through our own device forward:
Q1_0 5.93, PQ2_0 6.08 `[Q1_0+PQ2_0\|Q1_0/PQ2_0\|HIP PrismEngine + .htok\|strixhalo-busy\|128\|WS-00 gate corpus\|2026-09-18]`
PTQ1_0 5.69 `[Ternary-Bonsai-2-27B-PTQ1_0\|PTQ1_0\|HIP PrismEngine + .htok\|strixhalo-busy\|128\|WS-00 gate corpus\|2026-09-18]`
(Qwen3.8 base, so not like-for-like with the Qwen3.6 rows). The `.htok` tokenizer files are
byte-identical across packs.

**Honesty tags.** Correctness results are on strixhalo gfx1151 and timing-immune.

- fork's own *generated* tokens matched 5/5 by both the CPU floor and the full device forward, all three packs
  `[3-packs\|verbatim\|CPU floor + HIP forward\|strixhalo-busy\|5\|capital-of-France\|2026-09-18]`
- per-layer cosine against the in-repo numpy streaming reference, all 64 layers
  `[3-packs\|verbatim\|numpy streaming ref\|cpu-host\|5\|capital-of-France\|2026-09-18]`:
  min 1.000000 PTQ1_0 / 0.999979 PQ2_0 / 1.000000 Q1_0 `[3-packs\|verbatim\|numpy streaming ref\|cpu-host\|5\|capital-of-France\|2026-09-18]`
- kernel-level GEMV, tile kernel (Q1_0 23->80, PTQ1_0 25->93, PQ2_0 35->93) GB/s
  `[3-packs\|verbatim\|HIP prism_gemv_tile.hip\|strixhalo-unknown\|-\|synthetic x\|2026-09-18]`
  = 40-46% of the device triad ceiling `[n/a\|probe\|HIP hip_bw_probe\|strixhalo-quiet\|-\|-\|2026-09-18]`

The **tok/s figures above were measured with other lanes active and are relative only** (`strixhalo-busy`);
they are **not** gate numbers. The lane's targets >=42 / >=27 / >=22 tok/s `[P3-gate-target\|3-packs\|spec\|n/a\|-\|-\|2026-09-18]`
require a quiet box: at 2026-09-18 12:40Z two peer NPU engines were live and the device triad read
`139.3-170.0 GB/s `[n/a\|probe\|HIP hip_bw_probe\|strixhalo-busy\|-\|-\|2026-09-18]` against `201-219 GB/s idle `[n/a\|probe\|HIP hip_bw_probe\|strixhalo-quiet\|-\|-\|2026-09-18]`,
so **the P3 gate is NOT YET MEASURED** and a ~6 min exclusive window is requested.
Results of record with the full tag set: [tests/prism/PRISM_RESULTS.md](../../tests/prism/PRISM_RESULTS.md),
enforced by `tests/prism/check_honesty_tags.py`. Plan of record:
[docs/plans/prism-bonsai-27b-custom-build.md](../plans/prism-bonsai-27b-custom-build.md).

**See also:** [block-scaled ternary format](../research/block-scaled-ternary-format.md) · [benchmarks SSOT](../wiki/performance.md) · [all families](README.md)
