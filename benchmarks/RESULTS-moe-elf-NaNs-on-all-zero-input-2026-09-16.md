# The committed MoE whole-layer ELF produces NaN from an ALL-ZERO input — 2026-09-16

**Every input the harness controls has now been zeroed by direct intervention, and the NaN does not
go away.** That is a statement about the ELF, not about the packing.

## The elimination, by intervention rather than argument

Each row is a probe that memsets a region and re-runs `moe_smoke`. All report
`logits: ALL NaN (of 248320)`.

| region zeroed | probe | result |
|---|---|---|
| `act` — the data path | `MOE_ZERO_ACT=1` | **still all-NaN** |
| the whole weight BO (481 MB) | `MOE_ZERO_WEIGHTS=1` | **still all-NaN** |
| `norms[0,66048)` — the GDN gates | `MOE_ZERO_NORMS_HEAD=1` | **still all-NaN** |
| **the entire norms BO**, incl. `ssm_out` | `MOE_ZERO_NORMS=1` | **still all-NaN** |

Independently, a *read* of each region at its true dtype found nothing wrong:

- **router**: 0 non-finite, 6,144 finite, max 1.344;
- **`ssm_conv1d`** (bf16, 32,768): 0 non-finite, max 1.164;
- **`ssm_norm`** (bf16, 128): 0 non-finite, max 1.023;
- **`ssm_a`** (f32, 32): 0 non-finite, all negative, max 27.03;
- **`ssm_dt_bias`** (f32, 32): 0 non-finite, max 7.0;
- **`alpha_proj` / `beta_proj`** (65,536 / 32,768): 0 non-finite, max 0.139 / 0.158;
- **the state**: f32, 524,288 elements, 98.2% NaN.

**With every controlled input at zero, a correct layer must produce zeros.** The committed
`moe_layer_ctx1.elf` produces NaN. Nothing the harness packs, shapes, scales or dtypes can change that,
because it is not being computed from those values.

## What this rules out — cumulatively, not one at a time

- **Weight packing / repacking.** Zeroing all 481 MB of packed weights changes nothing. The whole
  reorder-and-repack workstream cannot be the fix.
- **The GDN gate tensors.** Zeroed, and clean when read.
- **`ssm_out`**, the one region the earlier dumps did not cover — now zeroed, still NaN.
- **The data path.** A zero activation leaves the state NaN, where `state += k⊗delta` with `k = v = 0`
  forces zero.
- **dtype.** The state is already f32 and is 98.2% NaN; a float32 kernel cannot fix a float32 state.
- **The host lm_head.** It faithfully propagates the act NaN; the poisoning is upstream of it.

## What it implies

The NaN is produced **inside the ELF** — from a constant baked into it, an operation on zeros
(`0 × inf` or `0/0`), or memory outside the BOs the harness fills. The read set is fully in-bounds
against the BO sizes, so "reads past the BO" is not the obvious explanation.

**Consequence for the goal lane: the fix is a new ELF, and a packing fix is not.** That is consistent
with the direction already chosen (a rebuilt whole-layer ELF) — but for a different and much stronger
reason than addendum 159 gave. It is *not* that the recurrence needs float32; it is that this ELF
cannot produce a finite result from zero input.

## What is NOT claimed

- **Not** which internal operation produces it. Zeroing establishes *that* the ELF does this, not
  *why*.
- **Not** that a rebuilt ELF will work — only that the current one cannot be fixed in the harness.
- **Not** that addendum 160's v0.9.46-vs-v1.0.x difference is irrelevant. A different library version
  generating a *correct* sequence is entirely consistent with "this particular ELF is broken", and is
  in fact supporting evidence for it.

## Reproduce

Probes are in this branch's `runtime_layer_moe.cpp` (`MOE_ZERO_ACT`, `MOE_ZERO_WEIGHTS`,
`MOE_ZERO_NORMS_HEAD`, `MOE_ZERO_NORMS`), all env-gated and off by default. Build `moe_smoke` and run
with each.
