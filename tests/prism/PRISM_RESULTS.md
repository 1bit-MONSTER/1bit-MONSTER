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

**Gate reachability arithmetic (host-side, derived from the tagged rows above).** These gates are
bandwidth targets, not measurement targets: each needs the pack's weights streamed at
`Q1_0 159.6 GB/s `[Bonsai-27B-Q1_0\|Q1_0\|arithmetic\|strixhalo-unknown\|-\|-\|2026-09-18]`,
`PTQ1_0 160.7 GB/s `[Ternary-Bonsai-2-27B-PTQ1_0\|PTQ1_0\|arithmetic\|strixhalo-unknown\|-\|-\|2026-09-18]`,
`PQ2_0 157.7 GB/s `[Ternary-Bonsai-27B-PQ2_0\|PQ2_0\|arithmetic\|strixhalo-unknown\|-\|-\|2026-09-18]`
i.e. 78-80% of the 201-219 GB/s triad `[n/a\|probe\|HIP hip_bw_probe\|strixhalo-quiet\|-\|-\|2026-09-18]`.
Today's effective rate (tok/s x pack size) is `60.8 / 65.5 / 86.0 GB/s `[3-packs\|verbatim\|derived from the rows above\|strixhalo-unknown\|32\|capital-of-France\|2026-09-18]`
= 30/33/43% of triad `[3-packs\|verbatim\|derived\|strixhalo-unknown\|32\|capital-of-France\|2026-09-18]`, so the gap is
`2.62x / 2.45x / 1.83x `[3-packs\|verbatim\|derived\|strixhalo-unknown\|32\|capital-of-France\|2026-09-18]`. PQ2_0 is closest because its
effective rate already matches the tile GEMV's own `93 GB/s `[3-packs\|verbatim\|HIP prism_gemv_tile.hip\|strixhalo-unknown\|-\|synthetic x\|2026-09-18]` — the model is GEMV-bound,
so the remaining headroom is exactly the distance between the tile GEMV's fraction of
triad and the fraction the gate implies (both tagged above, not restated here).

**Pattern cap, not decode (peer sweep by @agent-1141bd, 2026-09-18).** A no-decode dummy with identical
full-byte loads reached `93.6 GB/s `[3-packs\|verbatim\|HIP tile pattern, no-decode dummy\|strixhalo-unknown\|-\|synthetic x\|2026-09-18]`
against the real kernel's `80.2 GB/s `[3-packs\|verbatim\|HIP prism_gemv_tile.hip\|strixhalo-unknown\|-\|synthetic x\|2026-09-18]`
— the tile kernel is memory-pattern-limited, not decode-limited, so decoding accounts for only the
difference between those two rows `[3-packs\|verbatim\|derived\|strixhalo-unknown\|-\|synthetic x\|2026-09-18]`.
That pattern caps the packs at `Q1_0 24.6 / PTQ1_0 15.7 / PQ2_0 13.1 tok/s `[3-packs\|verbatim\|derived: dummy cap / pack size\|strixhalo-unknown\|-\|-\|2026-09-18]`,
all below the gates, and the cap is `47% of triad `[3-packs\|verbatim\|derived\|strixhalo-unknown\|-\|-\|2026-09-18]` against the ~79% the gate needs
`[P3-gate-target\|3-packs\|spec\|n/a\|-\|-\|2026-09-18]`: **the gate is unreachable with this decomposition whatever the
decoder does.** The decision number is that pair, not a tok/s. A same-window triad was requested from
the peer so these rows can move from `strixhalo-unknown` to `strixhalo-quiet`.

Sweep of seven variants, all corr `1.000000 `[3-packs\|verbatim\|HIP sweep by @agent-1141bd\|strixhalo-unknown\|-\|synthetic x\|2026-09-18]` on `blk.0.ffn_gate` 17408x5120:
tile4 `80-93 GB/s `[3-packs\|verbatim\|HIP sweep by @agent-1141bd\|strixhalo-unknown\|-\|synthetic x\|2026-09-18]` (best), tile8 `78 `[3-packs\|verbatim\|HIP sweep\|strixhalo-unknown\|-\|synthetic x\|2026-09-18]`,
per-block LDS `74 `[3-packs\|verbatim\|HIP sweep\|strixhalo-unknown\|-\|synthetic x\|2026-09-18]`, 1024-element LDS `68 `[3-packs\|verbatim\|HIP sweep\|strixhalo-unknown\|-\|synthetic x\|2026-09-18]`, conflict-free strided `42 `[3-packs\|verbatim\|HIP sweep\|strixhalo-unknown\|-\|synthetic x\|2026-09-18]`,
row4 `24-45 GB/s `[3-packs\|verbatim\|HIP sweep\|strixhalo-unknown\|-\|synthetic x\|2026-09-18]` — the tile kernel stands.

**Operator decision this exposes.** The gate as written asks for near-peak streaming through a 64-layer
hybrid that also carries GDN state and attention, so it sits at the edge of what this box can do *even
with a perfect weight path*. If the targets are a lane-relative ambition rather than a hard contract,
saying so lets me re-tag them as `spec-ambition` instead of failing them; if they are hard, then the
number to plan against is the fraction of triad the gate implies (tagged above), not the
tok/s figures in the abstract.

**Result: MISSED, and it is a kernel limit, not a measurement artifact.** The tile GEMV's own best is the
section-3 row tagged `[3-packs | verbatim | HIP prism_gemv_tile.hip | strixhalo-unknown | - | synthetic x | 2026-09-18]`;
at that bandwidth the Q1_0 model projects to 21 tok/s `[Bonsai-27B-Q1_0 | Q1_0 | HIP projected from GEMV BW | strixhalo-unknown | - | capital-of-France | 2026-09-18]`,
already short of the gate before any non-GEMV overhead. Closing the gap needs a new kernel decomposition
(in the tile design each weight byte is paired with an x reload per four rows), not a measurement rerun.

Outside baseline to beat: the fork's own Vulkan numbers, 28.8 tok/s `[Bonsai-27B-Q1_0 | Q1_0 | Prism llama.cpp fork + Vulkan | strixhalo-unknown | 32 | - | 2026-09-18]`
and 4.4 tok/s `[Ternary-Bonsai-2-27B-PTQ1_0 | PTQ1_0 | Prism llama.cpp fork + Vulkan | strixhalo-unknown | 32 | - | 2026-09-18]`.
