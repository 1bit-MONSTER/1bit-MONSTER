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
