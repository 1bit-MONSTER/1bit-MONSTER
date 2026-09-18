# MoE NaN survey across every BO the engine syncs — 2026-09-16

Ran `moe_smoke` (this branch: layer-mismatch fix + host lm_head) with `NPU_DUMP_BOS=1` and inspected
**every** BO the engine syncs, to localise the NaN without assuming which one matters.

## Result

| BO | bytes | as bf16 | non-finite | finite non-zero | max abs finite |
|---|---|---|---|---|---|
| **act** (layer output) | 4,096 | 2,048 | **2,048 (100%)** | 0 | 0 |
| **router** | 12,288 | 6,144 | **0** | 6,144 | 1.344 |
| **norms** | 328,192 | 164,096 | 0 | 164,064 | 3.905e+36 |
| **kv** (state) | 1,048,576 | 524,288 | **255,030 (48.6%)** | 252,815 | 3.376e+38 |
| weightA | 1,048,576 | 524,288 | 1,171 | 522,740 | 3.39e+38 |
| weightB | 2,097,152 | 1,048,576 | 11,875 | 1,035,866 | 3.39e+38 |
| logits | 1,048,576 | 524,288 | 0 | 0 | 0 |

## What is solid

- **`act` is 100% NaN.** That is the layer's own output, and it needs no layout knowledge to read.
  Consistent with the earlier finding that a *clean* input (`moe_act_pre.bin`, 0 NaN) becomes all-NaN
  after `forward(1)`.
- **`router` is clean** — 6,144 finite values, max 1.344, all non-zero. So the router BO is not
  poisoned and the NaN is not coming from there.
- **`logits` is all-zero** — expected: the host lm_head path is in use, so `bo_logits_` is never
  written. Correct behaviour, not a finding.
- **The state BO contains large non-finite content** under a bf16 read.

## What I am NOT claiming, and why

**The `weightA` / `weightB` non-finite counts are NOT meaningful.** Those BOs hold *packed quantised
weights* — int4/int8 payloads with interleaved bf16 scales — so reading the raw blob as bf16 produces
arbitrary values, including ones that look non-finite. The ~0.2% and ~1.1% figures are an artefact of
my interpretation and should not be quoted. Their `max` of 3.39e38 is simply the top of the bf16 range
appearing somewhere in packed bytes.

**The per-head breakdown of the state is suspicious and I could not resolve it.** Slicing the state as
`[32 heads][128][128]` gives:

```
head  0    : all zero
head  1    : 7,551 inf, 1,719 NaN, 7,114 finite (max 3.38e38)
heads 2-31 : exactly 8,192 inf and 8,192 finite, every finite value 9.18e-41
```

**"Exactly half inf, the other half pinned to 9.18e-41 (the bf16 minimum denormal), identically for
30 of 32 heads" is not what numeric overflow produces.** Overflow is irregular. A pattern that exact
points at a *layout or element-width* mismatch in my reading, not at the arithmetic.

And there is a concrete reason to suspect the width: `decode_txn` reports the ELF's arg-4 lengths in
**4-byte words** (`arg4 @49152 len=524288` = 2 MB), while `dump_bos` dumps only the first **0x100000
= 1 MB** of that BO. So the region the ELF actually reads is twice the window I inspected, and if the
state is f32 — which `mamba_ssm_dtype=float32` says it should be — then 1 MB is **half** of it, and the
"half inf" could be the boundary rather than the data.

## Consequences

- **Solid:** the layer's output is 100% NaN while its input is clean, and the router BO is clean. The
  poisoning happens inside the layer.
- **Suggestive but unresolved:** the state BO contains inf/NaN. Whether that is the *cause* or a
  *consequence* cannot be told from this dump, because the slice may not cover the state the kernel
  uses.
- **Not usable:** the weight-BO figures.

## The next check, and it is cheap

Dump the **full** arg-4 region (2 MB at the ELF's stated length) and read it at **both** widths before
drawing anything from it. `dump_bos`'s kv size is a one-line change (`0x100000` → `0x200000` in
`runtime_layer_moe.cpp:355`), and this branch builds in minutes. Until that is done, the state figures
above should not be used to argue a mechanism — which is exactly the mistake this lane has made
repeatedly, and the reason this note separates what is solid from what is not.

---

# RESOLVED: the state is f32, and it is 98.2% NaN — the "half inf" was my slice

Ran the cheap check this note said was needed: dump the **full** 2 MB arg-4 region (one line,
`runtime_layer_moe.cpp:355`, `0x100000` → `0x200000`) and read it at both widths.

```
kv.bin = 2,097,152 bytes

  as bf16 : n=1,048,576  inf=515,455  nan=1,719   (49.3% non-finite)
  as f32  : n=  524,288  inf=     14  nan=514,588 (98.2% non-finite)
```

**The f32 element count is 524,288 — exactly 32 v-heads × 128 × 128, the state size this repo's own
`gdn_host_recurrence.h` documents.** The bf16 count (1,048,576) is exactly double, which is what you
get reading f32 data as half-width. Combined with `mamba_ssm_dtype=float32` in the model config, the
reading is unambiguous:

**The recurrent state is stored in float32 and 514,588 of its 524,288 entries — 98.2% — are NaN.**

## What this settles

1. **The "exactly 8192 inf per head, identically for 30 of 32 heads" pattern was an artefact of
   dumping half the tensor.** Half the window, half the state: it read as a suspiciously regular
   overflow signature because it *was* a boundary, not data. The lesson generalises — the pattern was
   too regular for arithmetic, and that was the clue that the reading was wrong.
2. **The state is already float32 and it still NaNs.** This is the strongest possible form of the
   earlier refutation: it is no longer "bf16 and f32 have the same range so float32 would not help",
   it is "**the state is f32 and is 98% NaN anyway**". A float32 GDN kernel cannot fix a float32 state
   that has already gone NaN.
3. Only **1,492** of 524,288 state entries are finite and non-zero. This is not partial corruption or
   a few heads overflowing — it is essentially the whole state.

## What it still does not establish

**The mechanism.** A state that is ~98% NaN says the recurrence diverged; it does not say why. The
candidates remain open, and the ones this session has already ruled out are the dtype (above) and a
wrong sign *in the model data* (`ssm_a` is correctly negative):

- `ssm_a` (or the other GDN gates) not reaching the kernel — `npu_pack_moe_linear5_bo` writes
  `ssm_conv1d`/`ssm_norm`/`ssm_a`/`ssm_dt_bias` into `norms[0, 66048)`, and the ELF's own S2MM write
  clobbers that region and never reads it. If the kernel's gate inputs come from anywhere near there,
  it is reading clobbered bytes;
- the region-B base differing between the v0.9.46 and v1.0.x layouts (the goal lane's addendum 160);
- something outside the recurrence.

**Distinguishing them needs the gate values the kernel actually consumes**, not the state it produced
— and that is the next measurement, not another inference.
