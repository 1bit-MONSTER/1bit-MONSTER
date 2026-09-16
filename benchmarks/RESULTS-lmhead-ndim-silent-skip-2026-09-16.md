# The MoE (and dense) runtime-layer engines silently drop a non-2-D lm_head — 2026-09-16

**Verified by running, not by reading.** Built `moe_smoke` from the branch that fixes the
layer-mismatch (PR #2434) and ran it on the 35B MoE model, one invocation:

```
layer: 1 (moe_layer_ctx1.elf), set via argv[5] or MOE_LAYER
MoERuntimeLayer: packing LAYER 1 weights (moe_layer_ctx1.elf)
engine init OK (packed layer 1)
forward(1): EXECUTED
logits: argmax=0 max=0.0000 NaN=0 (of 248320)
```

The layer/ELF mismatch fix works — the packed layer and the executed ELF now agree. But the logits
are **all zero**, and `init` never printed its `lm_head packed (N tiles)` line.

## Cause: `lm_head.weight` loads as 3-D, and everything that touches it requires 2-D

A probe of the loaded descriptor (no device needed):

```
lm_head_weight: ndim=3
  shape = 7760 8 8704
  data_size = 540344320
embed_tokens: ndim=2 shape=248320 2048   data_size = 1017118720
```

The model header *does* contain `lm_head.weight`, so this is not a missing tensor. It is the
**dimensionality guard**, which appears in four places and rejects `ndim != 2` in all of them:

| site | guard |
|---|---|
| `runtime_layer_moe.cpp:131` | `ndim == 2 ? shape[0] : 0` → `n_tiles = 0` → no weight BO |
| `runtime_layer_moe.cpp:228` | `if (kern_lmhead_ && bo_lmhead_w_)` → never true → **lm_head not added to the runlist** |
| `runtime_layer.cpp:164` (dense) | `mw_->lm_head_weight.ndim == 2` → same skip in the dense engine |
| `model.c:699` `npu_pack_lmhead_bo` | `ndim != 2 → return 0` |
| `model.c:211` `npu_desc_tiles` | `if (!d || d->ndim != 2) return 0` |

## Why this is the silent-failure class, not just a bug

`forward()` returns **true** and `moe_smoke` prints `DONE`, because nothing failed — the lm_head
submission was simply never added. `bo_logits_` is `memset` to zero at init and never written, so
`get_logits()` reads zeros. The harness reports

```
logits: argmax=0 max=0.0000 NaN=0
```

which reads like a clean, non-NaN result. It is not a result at all: **no lm_head ran.**

## What this does and does not say about the NaN

**It does not say the layer-mismatch fix removed the NaN**, and it would be easy to claim that from
the `NaN=0` above. That zero is explained entirely by `bo_logits_` never being written — the BO is
zeroed at init and nothing else touches it. **This run is evidence about the lm_head path, not about
the layer.** Whether a now-consistent layer produces NaN is still unmeasured, and cannot be measured
until the lm_head actually runs.

The prior NaN observations were made through the same harness, so they inherit the same doubt.

## The fix direction

`npu_desc_tiles` already derives the tile count from the **byte extent**
(`(rows * rb + NPU_TILE_BYTES - 1) / NPU_TILE_BYTES`) rather than from `shape[0]`, and the comment
above it says why:

> `THE SAME RULE MUST BE USED EVERYWHERE a tile count is derived from a tensor: the source read, the
> destination offsets, and the BO size. Fixing one of the three and not the others moves the fault
> rather than removing it.`

The same byte-extent rule generalises to `ndim > 2` (flatten `shape[0..ndim-1]` to a row count), so
the guards above should be widened rather than the tensors reshaped. **Not done here** — it changes
weight layout for every model that has a 2-D lm_head, and it needs the per-model byte-parity check
the repo uses for packer changes before it goes anywhere near `main`.

## Scope

- Reproduced on **Qwen3.6-35B-A3B-NPU2** only. Whether other models in the set load a >2-D
  `lm_head.weight` is **not** established; the dense guard means any that do are silently affected.
- Nothing here changes the MoE decode rate. It says the instrument that was measuring the MoE could
  not produce logits at all.

---

# Follow-up: which models are affected, and why the fix is NOT a guard widening

## Every model probed (18 models, host-side, no device)

`lm_head_weight.ndim` from `model_load()`, all models under `~/.config/flm/models/`:

| model | ndim | shape | tiles (5120 B) | |
|---|---|---|---|---|
| **Qwen3.6-35B-A3B-NPU2** | **3** | `[7760, 8, 8704]` | 105,536 | **silently skipped** |
| **Qwen3.5-4B-NPU2** | **3** | `[7760, 10, 8704]` | 131,920 | **silently skipped** |
| Gemma3-1B | 2 | `[147456, 1280]` | 36,864 | ok |
| Gemma3-4B | 2 | `[81940, 5120]` | 81,940 | ok |
| Gemma4-E2B-IT | 2 | `[49152, 5120]` | 49,152 | ok |
| Gemma4-E4B-IT | 2 | `[81920, 5120]` | 81,920 | ok |
| LFM2-1.2B | 2 | `[16384, 5120]` | 16,384 | ok |
| LFM2-2.6B | 2 | `[16384, 5120]` | 16,384 | ok |
| Llama-3.1-8B | 2 | `[64128, 5120]` | 64,128 | ok |
| Llama-3.2-1B | 2 | `[32064, 5120]` | 32,064 | ok |
| Llama-3.2-3B | 2 | `[48096, 5120]` | 48,096 | ok |
| Nanbeige4.1-3B | 2 | `[51920, 5120]` | 51,920 | ok |
| Phi4-mini-Instruct | 2 | `[75024, 5120]` | 75,024 | ok |
| Qwen3-0.6B | 2 | `[18992, 5120]` | 18,992 | ok |
| Qwen3-1.7B | 2 | `[37984, 5120]` | 37,984 | ok |
| Qwen3-4B | 2 | `[47480, 5120]` | 47,480 | ok |
| Qwen3-8B | 2 | `[75968, 5120]` | 75,968 | ok |
| Qwen3-VL-4B-Instruct | 2 | `[47480, 5120]` | 47,480 | ok |

**Two of eighteen — and they are exactly the two MoE-family models.** Qwen3.6-35B-A3B is parity gap
#1 (the dominant one) and Qwen3.5-4B is gap #4. Both are the tied-embedding bundles that FLM's own
`qwen3_5vl` is also documented as unable to load. **The two MoE-family parity rows are being measured
through an engine whose lm_head never runs.**

Scope note: this shows the *guard* skips them. Whether a given model's runs actually reach that path
depends on which engine it uses — verified by running for the MoE (`moe_smoke`, all-zero logits),
not verified per-model for the dense path.

## Correction to the fix direction I gave earlier in this same document

I wrote that the guards "should be widened rather than the tensors reshaped", because
`npu_desc_tiles` already derives tiles from the byte extent. **That was too optimistic and I am
retracting it.** The arithmetic says the 3-D layout is a different format, not a 2-D one viewed
differently:

| model | shape | size | size/5120 | tiles ÷ shape[0] |
|---|---|---|---|---|
| Qwen3-4B (works) | `[47480, 5120]` | 243,097,600 | 47,480 | **1.0** |
| Qwen3.6-35B-A3B | `[7760, 8, 8704]` | 540,344,320 | 105,536 | **13.6** |
| Qwen3.5-4B | `[7760, 10, 8704]` | 675,430,400 | 131,920 | **17.0** |

All three are exactly tile-aligned, so `size/5120` gives a tile count — but for the working 2-D models
`tiles == shape[0]` (one row per tile) while the 3-D models are 17.0 and **13.6** tiles per `shape[0]`.
A non-integer ratio means `shape[0]` is not a tile dimension there at all.

`npu_pack_lmhead_bo` feeds that count to `npu_reorder_tiles(..., G = hidden/128)`, a permutation
defined for the 2-D form. **Widening the guard would hand that permutation a count and a layout it
was not written for — producing silently wrong weights, which is the exact failure class this lane
keeps rediscovering.** The 3-D layout has to be understood first (FLM's runtime packing, or the
captured lm_head BO), and until it is, `n_tiles = 0` is *failing closed*, which is better than the
alternative.

So: the defect is real and the impact is now precisely scoped, but the fix is a layout
investigation, not a one-line guard change.
