# FLM's REAL 35B MoE argument bindings, and what they do and do not explain

Captured by driving FLM's own runtime under the interposer. This is ground truth for the
35B MoE, which the Qwen3-4B log could never supply.

## The capture recipe (works; the process aborts afterwards but the log is written)

```
LD_PRELOAD=/home/bcloud/1bit-MONSTER/npu-infer/tools/capture/cap_interposer.so \
CAP_DIR=/tmp/capmoe \
FLM_XCLBIN_PATH=/home/bcloud/.local/flm-v0946 \
  ./run_qwen3_6_moe /home/bcloud/.local/flm-v0946/model/Qwen3.6-35B-A3B-NPU2 1
```

Two traps, both hit: `FLM_XCLBIN_PATH` must point to the directory that **contains** `xclbins`
(`/home/bcloud/.local/flm-v0946`), not to a model's xclbin dir — `utils.cpp:120-141` appends
`/xclbins` itself; and it exits 134 (SIGABRT) after "forward 1 tokens", which is harmless here.

## FLM binds DIFFERENT arg orders for DIFFERENT kernels

Distinct `idx=size` pairs seen in one 1-token forward:

```
idx=3: 1048576, 536870912, 9437184
idx=4: 1048576, 2097152, 542113792
idx=5: 1048576, 2097152, 536870912
idx=6: 1048576, 5242880
idx=7: 134217728, 3145728
```

Note the two large-weight bindings: **idx3=536,870,912** and **idx4=542,113,792**. FLM has both
orders, for different kernels. This is the fact that makes every earlier "the host binds what
FLM binds" comparison unsafe: the dense engine's `arg3=act, arg4=weights` is one kernel, and a
binding with the weights at idx3 is another. arg order is per-kernel caller order.

One binding, `{3:536870912, 4:1048576, 5:2097152, 6:5242880, 7:3145728}`, is the shape one would
expect of the MoE layer: a 512 MB weight buffer, a 1 MB activation, a 5 MB norms buffer.

## The harness's BO sizes vs FLM's

| arg | FLM size | harness size | verdict |
| --- | --- | --- | --- |
| 3 | 536,870,912 | 481,935,360 | **54,935,552 B short (54.9 MB)** |
| 4 | 1,048,576 | 1,048,576 | match |
| 5 | 2,097,152 | 1,060,864 | **1,036,288 B short (1.0 MB)** |
| 6 | 5,242,880 | 5,242,880 | match |
| 7 | 3,145,728 | 134,217,728 | oversized (harmless) |

## The size hypothesis is REFUTED

It was well motivated: the harness's weight BO is 54.9 MB smaller than FLM's, and an earlier
measurement had the regenerated v0.9.46 ELF reading to 533,293,056 B — which fits inside FLM's
536,870,912 and overflows the harness's 481,935,360 by 51 MB. Out-of-bounds reads would explain
a NaN. **It does not survive contact with the device:**

| configuration (order = LEGACY, i.e. arg3=weight, arg4=act) | logits |
| --- | --- |
| harness sizes (481,935,360 / base 0x1bc00000) | ALL NaN |
| **FLM's exact sizes (0x20000000 / base 0x1E000000)** | **ALL NaN** |
| big BO (0x20000000 / base 0x1bc00000) | ALL NaN |

Enlarging the weight BO to FLM's exact size, and moving region B to the offset the v0.9.46 ELF
reads, changes nothing. **Retracted: BO size and region-B base are not the NaN cause.**

## What IS still true, empirically

| order | logits |
| --- | --- |
| LEGACY: arg3=weight, arg4=act | ALL NaN |
| swapped: arg3=act, arg4=weight | finite (argmax=193722, NaN=0) |

So **this ELF wants the activation at arg3.** Under the swapped order the ELF also reads the
real activation (1 MB of bf16) instead of packed quantized weight bytes, which is why the NaN
goes away — ~0.4% of arbitrary 16-bit patterns are bf16 NaN, and one NaN in a matmul row NaNs
the whole row, which the norm then spreads.

**This contradicts the `{3:536870912, 4:1048576, ...}` binding only if that binding is the MoE
layer's.** It may belong to a different kernel — the interposer records run pointers and sizes
but never the kernel name, so this is not yet determined. That is the single open question.

## What is NOT established

- **Which captured binding is `moe_layer_ctx1.elf`'s.** Resolving it is the next step, and it is
  what decides whether the arg order is act@3 (swapped) or weight@3 (legacy).
- **Correctness either way.** Finite is not correct. The single-layer argmax is 193722 and the
  tool's stored reference is 76740; that reference was derived under a now-discredited order and
  must be re-derived.
- **The patch table's `arg_idx` semantics.** `patch arg0` shows `max(arg_offset+len)` =
  481,878,528 B, which matches the weight BO almost exactly and is inconsistent with act@3. Either
  `arg_idx` is not the XRT argument index, or `arg_offset` is not a byte offset into that buffer.
  Until this is nailed down, reasoning from the patch table is unsafe — it produced two wrong
  conclusions in this session.
- The `idx5` (router) BO is 1 MB short of FLM's; that is a real fidelity gap to fix even though it
  is not the NaN cause.

## Concrete next step

Make the interposer record the **kernel name** alongside each `SETARG`/`RUNLIST_ADD` (it has the
run pointer; `xrt::run` can be correlated with the kernel it was constructed from — the interposer
already hooks `run::run(const xrt::kernel&)`). Then the MoE layer's binding is identified by name
rather than inferred from sizes, and the arg order question closes.
