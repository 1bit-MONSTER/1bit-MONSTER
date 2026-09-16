# The "GDN recurrence is bf16, needs float32" root cause does not hold — 2026-09-16

**An independent check of addendum 159, run because the stated path forward is an expensive rebuild.**

## What addendum 159 claims

> the bf16 NPU kernel … gets wrong: the recurrent gated-delta-state update. The NPU runs the whole
> layer in bf16 and the state update NaNs … while the model's config declares mamba_ssm_dtype=float32.

→ *"PATH FORWARD … build a single-launch whole-layer MoE ELF whose GDN recurrence runs in float32."*

**The premise about the config is correct** — verified directly:

```
Qwen3.6-35B-A3B-NPU2:  dtype = bfloat16   mamba_ssm_dtype = float32
Qwen3.5-4B-NPU2:       dtype = bfloat16   mamba_ssm_dtype = float32
```

**The inference from it does not hold.**

## Why not, structurally

**bf16 has the same exponent range as f32** (8-bit exponent, max ≈ 3.39e38 vs 3.40e38). bf16 loses
*mantissa* bits, not *range*. Losing precision cannot turn finite values into NaN in a recurrence whose
f32 version stays finite — it can only make them slightly different.

## Measured, using this repo's own recurrence

Reimplemented `gdn_host_recurrence.h`'s `recurrence_core`/`recurrence_step` verbatim at the model's real
geometry (32 v-heads, 128×128 state = 524,288 floats), then re-ran it with every state write and
intermediate rounded to bf16 after each step. 64 steps, both `ssm_a` signs, three input scales:

| input scale | f32 state | bf16 state |
|---|---|---|
| 0.5 | finite (max 0.0301) | finite (max 0.0300) |
| 2.0 | **NON-FINITE** | **NON-FINITE** |
| 10.0 | **NON-FINITE** | **NON-FINITE** |

**They reach non-finite together at every scale. There is no scale where bf16 NaNs and f32 does not.**

With `ssm_a` positive (wrong sign, i.e. growth instead of decay) both grow to ~1e27–1e32 at scale 0.5 —
still finite, still *both*, and still not separable by dtype.

## Consequence

**A float32 GDN kernel would not fix the NaN.** The bf16-vs-f32 distinction cannot be the mechanism,
so rebuilding the whole-layer ELF with a float32 recurrence — a substantial piece of work requiring the
MLIR-AIE toolchain — is aimed at a cause this test rejects.

## What the NaN could be instead (not established here)

The measurement above is a *negative* result and narrows the space rather than filling it:

- **A wrong-sign `ssm_a`** would give `exp(g) > 1` and unbounded growth. The engine's own header notes
  `ssm_a` is *already* `-A` (`#1460`); if the kernel's copy is not negated, the recurrence grows instead
  of decaying. My sweep shows growth does reach non-finite — and that it does so in **both** dtypes.
  That is a sign/packing question, not a precision one.
- **`exp(g)` overflow** for a large `g`, which again is dtype-independent.
- Something outside the recurrence entirely.

## Scope and caveats

- Synthetic inputs, a scale sweep, not the model's real tensors. What generalises is the **structural**
  argument (bf16 range == f32 range) plus the observation that no scale separates the two dtypes.
- The bf16 rounding here is round-to-nearest-even on the high 16 bits, matching `f32_to_bf16` in the
  header.
- This does **not** show the NaN's true cause. It shows the float32 rebuild is not justified by this
  evidence.

## Reproduce

`/tmp/gdntest/t.cpp` — 60 lines, `g++ -O2 -o t t.cpp && ./t`. Kept out of the repo because it is a
scratch check, not an artifact.
