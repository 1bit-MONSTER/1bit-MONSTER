# Undiscovered code and undiscovered math — a survey (2026-09-15)

Prompted by "undiscovered code & undiscovered math". This is what the tree holds
that nothing currently drives, with the evidence for each claim and a note on what
would have to be true for it to matter.

## Method, and the trap that comes first

A literal `grep -l <basename>` over `engine/npu/xclbins/` reports **~60 of 145
xclbins as unreferenced**, including `final_i8_D_K14336_N4096.xclbin`,
`final_i8_GU_…`, and the `_m1`/`_m8`/`_m32` variants. That is a false positive:
the engine *constructs* those names (`"final_i8_"+t+"_K"+K+"_N"+N+".xclbin"`), so
no literal appears anywhere. A name is not a reference — the same lesson the
repo's own notes record for instruments and fixtures.

What follows is from three searches that do not have that failure mode:

1. every `getenv("…")` in the engine sources (169 flags), diffed against every
   `NPU_*/RT_*/BF16MM_*` token that appears in any `.md`, `.sh`, `.py` or `.json`
   in the tree (275 tokens) — **~70 flags appear in no document or script**;
2. sources not compiled by any build script and not `#include`d anywhere;
3. the builders in `engine/npu/generators/` (31 of them) versus what the engine
   actually loads.

## 1. `engine/npu/src/kv_quant.h` — declared, never implemented, no caller

29 lines: an **INT4 K/V cache with per-32-element group scale + zero-point in
bf16**, and the constants for the shape (`KQ_HD 128`, `KQ_NKV 8`, `KQ_MAX_CTX
4096`). `KVQuant::append`, `dequant_K_all`, `dequant_V_all`, `quantize_group` and
`dequant_group` are **declared and defined nowhere** — a repo-wide grep for
`KVQuant` finds only this header, two docs, and the GitNexus parse cache.

The design has a doc that measured it once, `npu-infer/docs/quantized-kv-benchmarks.md`:
**6.4× KV memory** (896 MB → 140 MB at 28 layers / 4k), "16K tokens fits in
<600 MB", and a note that the quantisation error (~0.34/element) sits below the
model's own noise floor.

**Why it matters here:** the remaining capability gap is context **> 8192**, and
that limit is the KV window the artifacts are baked for (`MAX_L=8192`, 8 MB per
region × 4). Halving the KV does not by itself lift it — the attention kernels
bake the layout — but it is the only lever in the tree that changes the *storage*
term rather than the *kernel* term.

**Status: a design, not code.** Anything built on it starts by writing the
implementation.

## 2. `npu_attn_ctx.h` + `build_attn.sh` + `n1_core_attn.py` — a generated, LENGTH-PARAMETRIC attention that only the Zaya path uses

This is the most interesting find, because it is the **only attention path in the
tree that is not baked to a captured length**:

- `n1_core_attn.py` takes `-N` = `MAX_SEQ` (default 256) and builds the GQA
  flash-attention kernel — QK^T int8→int32, **softmax on-core with a causal mask
  and a LUT**, PV int8→int32 — for that length;
- `build_attn.sh` builds it and states the design is **VERIFIED on strixhalo**
  (QK^T, causal-mask softmax and PV all match the x86 contract, `test_attn.cpp`);
- `npu_attn_ctx.h` is the host driver, with `NPU_ATTN_MAX_SEQ` overriding a
  `MAX_SEQ` that defaults to **512** — "kernel-baked N (shipped attn.xclbin =
  N=512 build)", i.e. the override must match the build.

So the machine to build this at *any* length already exists. But:

- the **shipped `attn.xclbin` is an N=512 build**, far behind the captured ELFs
  the dense path uses (1024 / 2048 / 4096 / 8192);
- it is included by **`zaya_decode.cpp` only** — the dense Qwen3 path never calls
  it;
- it is **int8 KV** while the dense bf16 path uses bf16 KV, so adopting it is a
  dtype change, not a drop-in;
- and a caution from this repo's own history: a *different* generator's attention
  ELF was "both wrong and ~1200× slower (223050 ms)" (`SESSION-FINDINGS-2026-09-14`
  §2). Generated attention has a bad track record here; this one is verified
  correct on the hardware but its **speed for dense shapes is unmeasured**.

**Why it matters here:** it is the natural route past 8192 — the captured ELFs
cannot go further without new FLM captures at those lengths, while this can be
built at 16k tomorrow. The open question is cost, not feasibility.

## 3. `NPU_QWEN_I4` + `NPU_GUSILU_BF16PAIR` — a fused int4 GU→SiLU for the DENSE FFN, opt-in and unexercised

Code comment (2026-09-15), verbatim intent: *"env-gated int4 fused GU->SiLU
(GUSILU_i4) for the DENSE FFN (qwen3-0.6b). Kernel contract silicon-verified
(zaya 0.999336); this inits the fused P1 context so the dense GU->host-SiLU->D can
be swapped for the single GU+SiLU launch. **Opt-in (NPU_QWEN_I4=1) until its
per-weight fused corr gate passes.**"*

Note the two flags are a *pair*: `pack_gu_fused_i4` emits the `I4_BF16_PAIR`
layout, and `NPU_GUSILU_BF16PAIR=1` selects the matching `_bf16pair` xclbin — with
the other one the kernel "consumes the OLD layout (garbage h2 / no C1 emit)".

**Why it matters here:** it halves the dense FFN's weight traffic and launch count
— but only on the int8 split path, which nothing uses by default any more (the
default is bf16 prefill + runlist decode). Its value today is small; it would have
been the lever in round 1's landscape.

## 4. Unwired artifacts with a builder

- `final_cascade_fused_qwen3_0_6b.xclbin` + `build_iron_cascade_qwen3.sh` — a
  fused QKV/O/GU/D cascade for Qwen3-0.6B, built and never wired. Same argument as
  (3): it targets the 112-launch split path, which the runlist and bf16 paths
  replaced.
- the `_m1`/`_m8`/`_m32` small-M xclbins — present, unused, and **documented as a
  dead end**: the source comment says the `_m1` path gives *"garbage decode, no
  perf win — launch-bound"*.

## 5. The flags are mostly instruments, and I checked the two that looked like levers

Of the ~70 flags in no document, the plausible non-diagnostic ones are
`BF16MM_W_STRIDE`, `NPU_BS`, `NPU_PRF`, `NPU_GUSILU_BF16PAIR` (see 3) and the
`NPU_ATTN_*` set (see 2). The first two do not survive reading:

- **`BF16MM_W_STRIDE`** is only read inside the `BF16MM_W_FILE` branch — a
  *diagnostic* that substitutes one captured weight BO to test whether FLM's W
  reproduces the reference boot. Not a performance knob.
- **`NPU_BS`** overrides a batch size that already defaults to 8.

The rest are dumps, timings, dim overrides and self-tests (`NPU_DUMP_*`, `RT_DUMP_*`,
`NPU_*_TIMING`, `NPU_NC/NH/IM/HD`, `NPU_RUNTIME_SELFTEST`). Recording that is part
of the answer: the undiscovered-code population is real but it is mostly
instrumentation, and the interesting items are the four above.

## What I would do next, in order

1. **Measure the generated attention's cost at a length where it is already
   correct** — build `NPU_ATTN_N=1024` (or 2048) and time it against the captured
   `attn_mha_1024_nh16.elf` on the same shape and prompt. One build and one
   measurement decides whether (2) is the route past 8192 or another 1200× dead
   end. Everything else here is blocked behind that number.
2. Only if (1) is competitive: `NPU_ATTN_N=16384`, and the int8-KV dtype change
   that adopting it implies.
3. `kv_quant.h` is a larger, separate project (implement + new kernels); it is not
   a prerequisite for 2, and it does not by itself lift the 8192 window.
