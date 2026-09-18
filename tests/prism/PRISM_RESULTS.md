# Prism ML Bonsai 27B — results of record (lane `feat/prism-bonsai-27b`)

Every numeric claim in this file carries the P6 honesty tag
`[model | format | backend | box | tokens | prompt | date]`.

* `box` is one of `cpu-host`, `strixhalo-quiet`, `strixhalo-busy`, `strixhalo-unknown`.
  **`strixhalo-quiet` may only be claimed together with a triad evidence line below** — §4 carries the idle
  and loaded triad states as tagged rows; no timing may be quoted across those two states (risk R16).
* A row tagged `strixhalo-busy` or `strixhalo-unknown` is **relative-only**: it may be compared
  against another row measured in the same state, never against the P3 gate or the outside baseline.
* `tests/prism/check_honesty_tags.py` enforces all of the above and fails on an untagged number.

## 1. Container — 1BP v5 with verbatim Prism payloads

Source GGUFs → `~/models/prism/1bp/*.1bp`, regeneration **≈ 7 s/model** `[3-packs | GGUF -> 1BP converter | CPU-host gguf_to_onebp | cpu-host | - | - | 2026-09-18]`. Each verbatim tensor is
`memcmp`'d against its source GGUF tensor.

| measurement | value | tag |
|---|---|---|
| payload compared byte-identical (4 GGUFs) | 23.9 GB, 0 failures | `[all-4-packs\|1BP-v5\|CPU-host converter\|cpu-host\|- \| - \| 2026-09-18]` |
| folded PTQ1_0 pack | 402 tensors / 5.878 GB | `[Ternary-Bonsai-2-27B-PTQ1_0\|PTQ1_0 nb=28\|CPU-host\|cpu-host\|- \| - \| 2026-09-18]` |
| folded PQ2_0 pack | 402 tensors / 7.137 GB | `[Ternary-Bonsai-2-27B-PQ2_0\|PQ2_0 nb=34\|CPU-host\|cpu-host\|- \| - \| 2026-09-18]` |
| unfolded ternary pack | 498 tensors / 7.144 GB | `[Ternary-Bonsai-27B-PQ2_0\|PQ2_0 nb=34\|CPU-host\|cpu-host\|- \| - \| 2026-09-18]` |
| binary pack | 498 tensors / 3.782 GB | `[Bonsai-27B-Q1_0\|Q1_0 nb=18\|CPU-host\|cpu-host\|- \| - \| 2026-09-18]` |

## 2. Correctness — CPU reference forward (P2)

Prompt `760 6511 314 9338 369` ("The capital of France is") throughout this section.

| measurement | value | tag |
|---|---|---|
| exact top-1 per position, 3 packs | 15/15 | `[3-packs\|verbatim\|CPU own forward\|cpu-host\|5\|capital-of-France\|2026-09-18]` |
| per-layer cosine, 64 layers, PTQ1_0 | min 1.000000 (rel-L2 2.7e-05) | `[Ternary-Bonsai-2-27B-PTQ1_0\|PTQ1_0\|numpy streaming ref\|cpu-host\|5\|capital-of-France\|2026-09-18]` |
| per-layer cosine, 64 layers, PQ2_0 | min 0.999979 (rel-L2 6.4e-03) | `[Ternary-Bonsai-27B-PQ2_0\|PQ2_0\|numpy streaming ref\|cpu-host\|5\|capital-of-France\|2026-09-18]` |
| per-layer cosine, 64 layers, Q1_0 | min 1.000000 (rel-L2 6.1e-04) | `[Bonsai-27B-Q1_0\|Q1_0\|numpy streaming ref\|cpu-host\|5\|capital-of-France\|2026-09-18]` |
| fork top-5 agreement, ternary pack | 24/25 ids, positions 1-4 identical in order | `[Ternary-Bonsai-27B-PQ2_0\|PQ2_0\|our forward vs Prism fork\|cpu-host\|5\|capital-of-France\|2026-09-18]` |

## 3. Kernels — HIP on gfx1151 (P3)

| measurement | value | tag |
|---|---|---|
| GEMV decode parity, 3 packs | corr 1.000000000 vs f64 CPU dot | `[3-packs\|Q1_0/PQ2_0/PTQ1_0\|HIP prism_gemv.hip\|strixhalo-unknown\|- \| synthetic x \| 2026-09-18]` |
| 64-layer forward argmax | 5/5 vs fork oracle, 3 packs | `[3-packs\|verbatim\|HIP prism_forward_hip\|strixhalo-unknown\|5\|capital-of-France\|2026-09-18]` |
| full device forward, 16-token decode | 15.05 tok/s | `[Bonsai-27B-Q1_0\|Q1_0\|HIP\|strixhalo-busy\|16\|capital-of-France\|2026-09-18]` |
| full device forward, 16-token decode | 10.15 tok/s | `[Ternary-Bonsai-2-27B-PTQ1_0\|PTQ1_0\|HIP\|strixhalo-busy\|16\|capital-of-France\|2026-09-18]` |
| full device forward, 16-token decode | 12.20 tok/s | `[Ternary-Bonsai-27B-PQ2_0\|PQ2_0\|HIP\|strixhalo-busy\|16\|capital-of-France\|2026-09-18]` |
| GEMV bandwidth, `blk.0.ffn_gate` 17408x5120 | 23 -> 80 GB/s (Q1_0), 35 -> 93 (PQ2_0), 25 -> 93 (PTQ1_0) | `[3-packs\|verbatim\|HIP prism_gemv_tile.hip\|strixhalo-unknown\|- \| synthetic x \| 2026-09-18]` |

## 4. Bandwidth evidence (the quietness proof)

| measurement | value | tag |
|---|---|---|
| triad, box idle (baseline) | 201-219 GB/s | `[n/a\|probe\|HIP hip_bw_probe\|strixhalo-quiet\|- \| - \| 2026-09-18]` |
| triad, 2 peer NPU engines live | 139.3-170.0 GB/s | `[n/a\|probe\|HIP hip_bw_probe\|strixhalo-busy\|- \| - \| 2026-09-18]` |

## 5. P3 gate — MEASURED, **NOT MET** (2026-09-18)

The box was clean-rebooted to clear the peer lanes, and the backend decoded 32 tokens per pack in the
post-reboot window (no `npu_engine_*` / `pf` / `python3` lane). No triad reading was taken in that
window, so the box is tagged `strixhalo-unknown`, **not** `strixhalo-quiet`.

| pack | decoded (32 tokens) | gate | tag |
|---|---|---|---|
| Bonsai-27B-Q1_0 3.80 GB | 16 tok/s | >=42 tok/s | `[Bonsai-27B-Q1_0 | Q1_0 | HIP bench_hip_1bp | strixhalo-unknown | 32 | capital-of-France | 2026-09-18]` |
| Ternary-Bonsai-2-27B-PTQ1_0 5.95 GB | 11 tok/s | >=27 tok/s | `[Ternary-Bonsai-2-27B-PTQ1_0 | PTQ1_0 | HIP bench_hip_1bp | strixhalo-unknown | 32 | capital-of-France | 2026-09-18]` |
| Ternary-Bonsai-27B-PQ2_0 7.17 GB | 12 tok/s | >=22 tok/s | `[Ternary-Bonsai-27B-PQ2_0 | PQ2_0 | HIP bench_hip_1bp | strixhalo-unknown | 32 | capital-of-France | 2026-09-18]` |

**Result: MISSED, and it is a kernel limit, not a measurement artifact.** The tile GEMV's own best is the
section-3 row tagged `[3-packs | verbatim | HIP prism_gemv_tile.hip | strixhalo-unknown | - | synthetic x | 2026-09-18]`;
at that bandwidth the Q1_0 model projects to 21 tok/s `[Bonsai-27B-Q1_0 | Q1_0 | HIP projected from GEMV BW | strixhalo-unknown | - | capital-of-France | 2026-09-18]`,
already short of the gate before any non-GEMV overhead. Closing the gap needs a new kernel decomposition
(in the tile design each weight byte is paired with an x reload per four rows), not a measurement rerun.

Outside baseline to beat: the fork's own Vulkan numbers, 28.8 tok/s `[Bonsai-27B-Q1_0 | Q1_0 | Prism llama.cpp fork + Vulkan | strixhalo-unknown | 32 | - | 2026-09-18]`
and 4.4 tok/s `[Ternary-Bonsai-2-27B-PTQ1_0 | PTQ1_0 | Prism llama.cpp fork + Vulkan | strixhalo-unknown | 32 | - | 2026-09-18]`.
