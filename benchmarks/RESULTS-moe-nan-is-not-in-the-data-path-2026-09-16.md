# The MoE NaN is NOT in the data path, and not in the GDN gates — 2026-09-16

Two measurements that together pin the NaN away from where it has been assumed to be.

## 1. A ZERO input still produces a NaN state

Ran `moe_smoke` with `MOE_ZERO_ACT=1` (zeroes the act BO before the submit) and dumped the state:

```
MoERuntimeLayer: ACT ZEROED (input-independence probe)
DUMPBO kv -> /tmp/bo/kv.bin (2097152 @0)

as f32: n=524288  non-finite=512625   nonzero=513173
```

**With a zero activation the recurrent state is still 512,625 / 524,288 non-finite.**

This matters because the state update is `state *= exp(g)` then `state += k ⊗ delta` with
`delta = (v − kv_mem)·β`. With a zero input, `k = v = 0`, so `delta = 0` and **the state must stay
zero**. It does not. **The NaN is therefore not produced by the data path at all.** That is a
mechanism-level statement, not another correlation.

## 2. Every raw GDN tensor in the norms BO is clean

`npu_pack_moe_linear5_bo` memcpy's six tensors **raw** (not as packed blocks), so unlike the weight BO
these can be read at their true dtype. Checked each at its documented offset:

| tensor | dtype | n | non-finite | max abs |
|---|---|---|---|---|
| `ssm_conv1d` | bf16 | 32,768 | **0** | 1.164 |
| `ssm_norm` | bf16 | 128 | **0** | 1.023 |
| `ssm_a` | f32 | 32 | **0** | 27.03 (all negative) |
| `ssm_dt_bias` | f32 | 32 | **0** | 7.0 |
| `alpha_proj` @66048 | bf16/f32 | 65,536 / 32,768 | **0** | 0.139 |
| `beta_proj` @197120 | bf16/f32 | 65,536 / 32,768 | **0** | 0.158 |

**None of the GDN gate tensors contains a non-finite value**, and all have plausible magnitudes. So
the gates the recurrence depends on are not the poison either.

## What that leaves, and why `inf × 0` is the shape of it

`0 × inf = NaN`. A single **infinite value in a packed weight** would poison *every* output
*regardless of the input* — which is exactly the input-independence measured above, and exactly what
addendum 158 recorded ("structural-NaN verdict"). It also explains why the dtype argument failed: the
state's range is not what overflows, the *weights* are.

The remaining candidates, in order:

1. **An inf/NaN scale in the packed weights** (region B, the expert pool, or `ssm_out`). These are the
   only regions I have not been able to read at their true dtype, because they are packed blobs.
2. The kernel reading a region outside what the packers fill.
3. Something in `ssm_out` specifically — **not covered by this dump**: the ELF's `ssm_out` reads start
   at 328,192 and `dump_bos` dumps exactly `[0, 328192)`, so the boundary is precisely where my
   coverage ends.

## What is NOT claimed

- **Not** that the weights are inf — not measured. Those regions are packed and reading them as bf16
  produces arbitrary values (the earlier `weightA`/`weightB` non-finite counts were an artefact of
  that and remain unusable).
- **Not** the mechanism's location beyond "not the data path, not the gates".
- The `ssm_out` region is genuinely unexamined here, and it is the last raw-ish region before the
  packed weights.

## The next measurement

Extend the norms dump past 328,192 to cover `ssm_out`, and check the packed regions' **scale fields**
specifically rather than the blob — the scales are the only part of a packed block that is a float,
and an inf scale is the one thing that turns a zero input into NaN. pi's `/tmp/lin5dump0946` dumps of
the vendor's own `load_linear_weights` are the right reference for those fields.
