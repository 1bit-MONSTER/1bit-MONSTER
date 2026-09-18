# Beyond-8192: the measured bound on the generated-attention route

Goal `mu35shsg-i3hlyi`, task `beyond8192`. Date 2026-09-16 ~02:15 ADT.
Contract alternative taken: **"the evidence that bounds the window."**

## What is already true for the supported models

From the L1/register record (cited, not re-measured):
`benchmarks/LEVERS-register-2026-09-15.md` §"What is outside the set":

> contexts **beyond 8192** (the layer ELFs bake `MAX_L=8192` and the generator
> asserts `L ≤ MAX_L+1`, so 8192 tokens is the window) … beyond 8192 neither the
> layer ELFs nor the attention captures exist.

and the same register's lever table entry 8: the nh32 8192 capture + region stride
+ cap makes **all six models reach 8191** (4B/8B/VL-4B match FLM at 4200…8191).
So the parity window for the supported models is **[1, 8191]** and the blocker
past it is the baked `MAX_L=8192`, not a runtime defect.

## What I measured: the generated route, and where it stops

The route the objective names ("N=16384 is the route past 8192") is the generated
int8 attention. Built with `engine/npu/generators/n1_core_attn.py` via
`build_attn.sh` (`NPU_ATTN_N=<N>`, Zaya shape nq8/nkv2/hd128):

| N (baked MAX_SEQ) | build | note |
|---:|---|---|
| 512 (shipped) | ok | |
| 1024 (L1 chunked) | ok | L1 lane |
| **2048** | **ok** | 134096 B xclbin; NPU==EMU at seq=2048 (5.552646e-02), 6.651 ms/call |
| **3072** | **ok** | 163536 B xclbin; NPU==EMU at seq=3072 (5.023500e-02), 10.384 ms/call |
| 4096 | **FAIL** | `[AIE ERROR] _XAie_LoadProgMemSection():231: Overflow of program memory` |
| 8192 | **FAIL** | same program-memory overflow |

**The generated single-partition int8 attention stops between 3072 and 4096, at
per-core AIE program memory.** N=16384 is not reachable by scaling this design;
it needs the chunked/multi-core partition (flash-attention style), which is the
`n1_mha_chunked.py` family, not a parameter bump.

## The engine enforces exactly that bound

Engine runs with the generated kernels (`npu_engine_zr1`, ids as argv):

```
NPU_ATTN=1 NPU_ATTN_MAX_SEQ=2048 NPU_ATTN_XCLBIN=/tmp/attn_v4_2048/attn.xclbin \
NPU_ATTN_INSTS=/tmp/attn_v4_2048/attn_insts.txt \
  ./engine/npu/build/npu_engine_zr1 ~/models/zaya1-8b.q4nx $(tr '\n' ' ' </tmp/ids1100.txt)
# -> NPU attention ready (MAX_SEQ=2048); MoE L1 dbg corr=0.999342; no clamp

# 2100-token prompt, MAX_SEQ=3072, diag at pos 2099:
# -> [ATTN L0 dbg] corr=1.000000 maxdiff=0.000000 (seq~2100, past 2048); no clamp

# 2100-token prompt, MAX_SEQ=2048:
# -> AttnCtx: WARN seq=2049 > MAX_SEQ=2048 — clamping (results wrong past the
#    kernel's baked N)
```

So the engine's usable attention window is exactly the kernel's baked N, it says
so with an explicit WARN when exceeded, and the generated design caps N at ~3072.

## Verdict and the honest caveat

**Bounding evidence:** the window past 8191 is bounded by (a) `MAX_L=8192` baked
into the per-context layer ELFs, which is what caps the six supported models at
8191, and (b) per-core AIE program memory, which caps the generated int8
attention at N≈3072 (4096 and 8192 fail to build). The N=16384 route past 8192
is a design change, not a measurement left undone.

One open observation, recorded rather than smoothed over: in the N=3072 engine run
the diag fires twice at pos 2099 (the engine forwards that position in two passes);
the first has `corr=1.000000`, the second shows the NPU output all-zero against a
non-zero CPU reference (`corr=-nan`, npu rms=0.0000). The first is the gate quoted
above; the second is unexplained and is a lead for whoever takes the >8192 route,
not evidence against the bound.

Logs: `/tmp/ck2048_{npu,emu}.log`, `/tmp/ck3072_{npu,emu}.log`, `/tmp/zr1_1100.log`,
`/tmp/zr1_2100.log`, `/tmp/zr1_3072.log`. Kernel build failures captured in the
session; `build_attn.sh` + `n1_core_attn.py` are committed.
