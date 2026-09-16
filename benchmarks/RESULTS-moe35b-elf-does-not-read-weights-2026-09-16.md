# The "whole-layer" MoE ELF never consumes the weight BO — so it is not a whole layer

Decision-relevant conclusion of round 15. It supersedes the arg-order "fix" in 1ae82a6e1,
which removed a symptom without producing a valid layer.

## The measurement

Weight-BO size swept on the finite (swapped) path, **only** the size varying:

| weight BO bytes | logits |
| --- | --- |
| 481,935,360 (harness) | argmax=193722 max=0.0242 NaN=0 nonzero=248320 |
| 536,870,912 (FLM kernel A) | **byte-identical** |
| 542,113,792 (FLM kernel B) | **byte-identical** |
| 1,073,741,824 (1 GB) | **byte-identical** |

Together with the one-BO-at-a-time probes:

| zeroed BO | effect on output |
| --- | --- |
| act | **ALL ZERO — changed** |
| weight (481 MB) | none |
| router | none |
| norms (5 MB) | none |

**The weight BO's contents do not matter and its size does not matter.** A layer that never
reads its expert weights is not computing a MoE FFN.

This is why the NaN disappeared and why that is not a fix: the swap moved the real activation
into the slot this ELF reads as its activation (arg3), so the input stopped being packed
quantized bytes. But the layer still never touches the weights, so the finite output is not a
valid layer output.

## What this means for the 35B gap

The harness drives `moe_layer_ctx1.elf` (from `layer.xclbin`) as a whole layer. FLM ships, for
this model, **separately**: `layer.xclbin`, `mm.xclbin`, `dequant_mm.xclbin`, `attn.xclbin`,
`conv.xclbin`, `GateDeltaNet_prefill.xclbin`, `lm_head.xclbin`. The expert matmuls are the
obvious work for `mm`/`dequant_mm`. So the working hypothesis is that **the captured
`layer.xclbin` stage is a fragment of a layer, and the expert FFN is a different kernel we have
not captured or generated.** That would also explain the ELF's 13.75 MB of DDR traffic against
~12 MB of per-layer expert weights it needs and never reads.

Until that is settled, tuning the arg order or the BO sizes of this ELF cannot make the 35B MoE
correct, and any throughput number measured through it is measuring the wrong computation.

## FLM's real bindings, now captured (ground truth)

`run_qwen3_6_moe` under `cap_interposer.so` gives, for one 1-token forward:

```
idx=3: 1048576, 536870912, 9437184
idx=4: 1048576, 2097152, 542113792
idx=5: 1048576, 2097152, 536870912
idx=6: 1048576, 5242880
idx=7: 134217728, 3145728
```

**Different kernels, different arg orders** — two of them carry ~512 MB weight BOs, one at idx3
and one at idx4. Any claim that "the host binds what FLM binds" must name the kernel. Recipe:

```
LD_PRELOAD=/home/bcloud/1bit-MONSTER/npu-infer/tools/capture/cap_interposer.so \
CAP_DIR=/tmp/capmoe FLM_XCLBIN_PATH=/home/bcloud/.local/flm-v0946 \
  ./run_qwen3_6_moe /home/bcloud/.local/flm-v0946/model/Qwen3.6-35B-A3B-NPU2 1
```

`FLM_XCLBIN_PATH` must point at the directory *containing* `xclbins`; the process SIGABRTs after
the forward but the SETARG log is complete.

## Not established

- That the expert FFN is in `mm`/`dequant_mm` — that is the hypothesis, not a measurement.
- Numerical correctness in any configuration.
- Which captured binding belongs to which kernel: the interposer records run pointers and sizes
  but never the kernel name. Fixing that (it already hooks `run::run(const xrt::kernel&)`) is
  the cheapest way to close the arg-order question for good.

## Two retractions this round, for the record

1. An intermediate probe reported "255 pages with finite data → CHECK THE DUMP OFFSET". Wrong:
   the predicate counted `isfinite(0.0f)`, which is true, so the init memset read as data. The
   pages were 255 byte-identical all-zero blocks.
2. The BO-size hypothesis (harness weight BO 54.9 MB smaller than FLM's, earlier ELF reading to
   533,293,056 B, so out-of-bounds → NaN) was well motivated and is **refuted**: FLM's exact
   sizes with FLM's region-B base still give ALL NaN.

The patch table produced two wrong conclusions this session. `patch arg0` shows
`max(arg_offset+len)` = 481,878,528 B, which matches the weight BO and contradicts act@3 — so
either `arg_idx` is not the XRT argument index or `arg_offset` is not a byte offset into that
buffer. Do not reason from the patch table until that is settled.
