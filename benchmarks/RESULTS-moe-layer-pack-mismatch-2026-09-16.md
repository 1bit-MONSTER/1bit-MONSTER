# The MoE harness packed layer 0's weights and ran layer 1's ELF — 2026-09-16

**Finding: `MoERuntimeLayerEngine` packed the weights of ONE layer and executed the ELF of
ANOTHER.** `init()` called all four packers with `layer=0`, then warmed
`moe_layer_ctx1.elf`; `moe_smoke` then called `forward(1)`, running layer 1's program with
layer 0's weights. Nothing errored, because the two layers have identical layouts — so it
computed plausible, wrong output.

This is a **harness correctness** defect. It is *not* the cause of the MoE NaNs (addendum 148
already said so, and this note does not contradict that): it is the reason any
correct-output comparison against a per-layer reference could not have been valid.

## 1. The convention that hid it

The parameter was named `ctx_len` and documented as "the 1-based context length". For the
**dense** engine that is true — `RuntimeLayerEngine` loads `layer_ctx<N>.elf` where N really is a
context length, and its directory holds `layer_ctx444.elf`, `layer_ctx1111.elf`,
`layer_ctx2086.elf`, …

The **MoE** engine reuses that filename shape with a different meaning:

```
$ ls npu-infer/captures/txn-elfs-moe35b/moe_layer_ctx*.elf | wc -l
40
$ ... indices present:  0 1 2 3 4 5 6 7 ... 38 39
$ model config num_hidden_layers = 40
```

**Forty files, indices 0..39, and the model has forty layers.** `moe_layer_ctx<N>.elf` is
**layer N**, not context length N. Two conventions sharing a filename prefix, and the code
believed the wrong one.

## 2. The mismatch, as the code had it

```
runtime_layer_moe.cpp:62    if (!ensure_layer_kernel(1)) return false;      // warms ELF layer 1
runtime_layer_moe.cpp:82    npu_pack_moe_expert_pool(w, mw_, 0);            // packs layer 0
runtime_layer_moe.cpp:87    npu_pack_moe_region_b(w + REGION_B_BASE, mw_, 0);
runtime_layer_moe.cpp:98    npu_pack_moe_router_bo(r, mw_, 0);
runtime_layer_moe.cpp:105   npu_pack_moe_linear5_bo(n, mw_, 0);
moe_smoke.cpp:51            eng.forward(1);                                  // runs ELF layer 1
```

The header's own doc comment said "Run one layer forward (single layer, **linear layer 0**)" —
which matched the packed weights and contradicted the ELF actually loaded. The two halves of
the harness disagreed and the code had no way to notice.

## 3. Why it ran instead of failing

`moe_layer_ctx0.elf` and `moe_layer_ctx1.elf` are **both 103,104 bytes** — identical size,
because both layers are `linear_attention` (addendum 148) and therefore share a layout. A
wrong-layer run produces a *wrong answer*, never a *loud failure*. That is the same
silent-failure class as the two attention guards this lane already added
(`K > PV output tile n`; model q-heads > kernel columns), both of which were added after a
wrong kernel built, loaded and ran silently.

## 4. What changed

- The parameter is renamed `ctx_len` → `layer` throughout the MoE engine, and the header now
  documents the two conventions explicitly, because the old name is what allowed the confla-
  tion.
- `init(..., int layer = 1)` selects which layer is packed, and passes that **same** value to
  `ensure_layer_kernel`, so the ELF warmed at init is the layer whose weights were packed.
- The engine records `packed_layer_` and **refuses** to run any other layer's ELF, naming both
  layers and the fix. The weight/router/norms BOs hold exactly one layer, so this is never a
  legitimate request under the current single-BO design.
- `moe_smoke` takes the layer from `argv[5]` or `MOE_LAYER` (default **1**, the layer it has
  always loaded) and uses that one value for both packing and forward, printing it.

**Default 1 is deliberate**: it is the single-variable change. The harness already loaded layer
1's ELF; the only thing altered is that the weights now match it. Layer 0's alternative would
have changed which layer is exercised as well.

## 5. Verification

**Done, without the device:**

- `g++ -std=c++17 -fsyntax-only -Wall -Wextra` on both translation units: **rc=0**, no new warnings.
- Full compile **and link** with the runlist XRT 2.26.0 set: **rc=0**, binary produced.
- The premise itself, from the artifacts: 40 `moe_layer_ctx0..39` files against a model with
  exactly 40 layers.

**Not done — stated plainly:** the binary has **not been run**. `/dev/accel/accel0` is held by
the dense lane (rule 4), so the refusal path and the effect of a now-consistent layer on the
logits are **unexercised**. What is verified is that it compiles, links, and that the layer it
packs is the layer it loads.

## 6. Incidental — the harness's build recipe is wrong

`moe_smoke.cpp`'s header comment gives:

```
g++ -std=c++17 -O2 -I npu-infer/include moe_smoke.cpp npu-infer/src/model.c ...
```

`model.c` is **C**, and `g++` compiles `.c` as C++, which fails on `void*` conversions
(`mmap`, `calloc`) — reproduced here. `gcc -c` for `model.c` and `g++` for the rest links
cleanly. The recipe cost me one failed build; recording it so it costs nobody else one.

## 7. Consequence for the MoE correctness work

Any comparison of this harness's logits against a per-layer reference (e.g. `moe_ffn_cpu`, or a
CPU reference for layer 1) was being made against **the other layer's weights**. Such a
comparison could not have passed, and its failure was not evidence about the ELF, the kernels,
or the device. The all-NaN symptom is a separate and still-open question.
