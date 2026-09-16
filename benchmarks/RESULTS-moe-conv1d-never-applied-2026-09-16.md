# The MoE layer path never applies the linear-attention conv1d — 2026-09-16

**Answers addendum 148 finding 3** ("where the conv1d/SSM weights ARE read from is still
unidentified") and **finding 2** (the conv1d weights are packed into a region the ELF clobbers).

**Answer: they are read from nowhere on the device, and the host never applies them either.**
The MoE linear-attention conv1d is a step that the harness does not perform.

## 1. The working reference does it on the host

`npu_engine_universal.cpp`'s `gdn_attn_step` — the dense GDN path used by Qwen3.5-4B, whose
fallback prefill boots correctly (`[0] boot=22069`) — applies the causal depthwise conv1d in plain
C++ on the host:

```cpp
// causal depthwise conv1d on the fused QKV (kernel 4)
memmove(conv_state, conv_state + gdn_conv_dim[l], (size_t)gdn_conv_dim[l] * (gdn_conv_k[l] - 1) * 4);
memcpy(conv_state + (size_t)gdn_conv_dim[l] * (gdn_conv_k[l] - 1), fqo, ...);
const float* cw = gdn_conv_w[l].data();
for (int cc = 0; cc < gdn_conv_dim[l]; cc++) {
    double s = 0;
    for (int kk = 0; kk < gdn_conv_k[l]; kk++)
        s += (double)conv_state[(size_t)kk * gdn_conv_dim[l] + cc] * cw[(size_t)kk * gdn_conv_dim[l] + cc];
    fqo[cc] = silu_f((float)s);
}
```

`gdn_conv_w[l]` is loaded once as host floats (`npu_engine_universal.cpp:2437-2439`) straight from
the model's `ssm_conv1d` tensor. **Conv is a host op in the reference implementation, not a device
op.** That is why the layer ELF has no read for it.

## 2. The MoE path does it nowhere

| file | mentions of `conv` |
|---|---|
| `npu-infer/src/runtime_layer_moe.cpp` | **1** — a comment at line 74, no code |
| `npu-infer/tools/moe_smoke.cpp` | 0 |
| `npu-infer/include/runtime_layer_moe.h` | 0 |

`MoERuntimeLayerEngine::forward` packs the BOs and submits the layer ELF. There is no host conv
step. So in the MoE path the conv1d is **neither applied on the host nor read by the device**.

## 3. It cannot be hiding inside the ELF

Addendum 148 already recorded that the ELF's arg-3 read set is
`@66048 alpha_proj`, `@197120 beta_proj`, `@328192+ ssm_out` with **no read of `[0, 66048)`** — the
region the packer fills with `ssm_conv1d`, `ssm_norm`, `ssm_a`, `ssm_dt_bias`.

Independently: `ssm_conv1d` alone is `[4, 8192]` bf16 = **65,536 B = 32,768 words**, while the
whole TXN is **24,636 words** (addendum 148's own word accounting). The weights are therefore
*arithmetically* unable to fit in the instruction stream. The ELF cannot be applying this conv.

So the packer's 65,536-byte prefix is not "dead weight in the wrong half of the BO" so much as
**weights for a step that never runs**, written into the exact region the ELF uses as S2MM scratch
— which is why the clobber is harmless: nothing reads it either way.

## 4. What this means for the MoE NaN and the parity gap

This is a **missing computation**, not a binding or packing error, which fits the observation that
the arg binding decoded clean and consistent (addendum 148 finding 1) while the output is wrong.
A linear-attention layer whose conv is skipped feeds the SSM a different signal than the model
expects; every downstream token is then wrong.

It is **not** established here that this is the *only* defect, nor that it is the NaN source
specifically — the harness also had a wrong-layer ELF/weights mismatch (fixed separately in this
branch) which confounds any earlier conclusion drawn from its output.

**Concrete next step, and it is cheap:** apply the same host conv1d the dense path already
implements, using the `ssm_conv1d` the packer already loads, and re-run `moe_smoke`. Then the
logits comparison is between two paths that at least compute the same function.

## 5. Method note

This was found by looking for the **working reference** — the dense GDN path that boots correctly —
instead of continuing to decode the vendor ELF for a read that is not there. The question "where is
X read from?" presupposed X is read on the device; asking "does anything compute X at all?"
answered it in one comparison.
