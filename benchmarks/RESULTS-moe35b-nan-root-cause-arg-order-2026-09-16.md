# MoE 35B all-NaN root cause: the layer args were bound in the wrong order

**Status: root cause found, fix verified by A/B. Numerical correctness NOT yet established.**

## Result

| arg binding | logits (248,320) | act BO after run |
| --- | --- | --- |
| `idx3=weight, idx4=act` (previous) | **ALL NaN** | 1 page all-NaN, 255 untouched |
| `idx3=act, idx4=weight` (FLM's order) | **argmax=193722, NaN=0, nonzero=248320** | 2038 non-zero values, page 0 |

Run with `moe_smoke` on layer 1, identical BOs, only the two argument bindings swapped.
The old order is still reachable via `MOE_ARG_LEGACY_ORDER=1` and reproduces ALL NaN on demand.

## Why this is the cause, and why every earlier probe failed to find it

The capture log `npu-infer/captures/capture_manifest-2026-09-03.log` records **FLM's own**
bindings:

```
idx=3 size=1048576     <- 1 MB   ACTIVATION
idx=4 size=98566144    <- 94 MB  WEIGHTS
idx=5 size=1048576
idx=6 size=1048576
idx=7 size=33554432    <- 32 MB  KV
```

The vendor's `create_run` binds BOs at `3+i` in **caller order**, with no intrinsic meaning,
and FLM's first BO is its activation — `cap_interposer.cpp:219-226` says so explicitly, and
notes the activation BO is written **in place** (line 279). The harness had been doing the
opposite: `set_arg(3, bo_weight_)` (481,935,360 B) and `set_arg(4, bo_act_)`.

So the ELF read its **activation out of the packed quantized weight bytes**. Roughly 0.4% of
arbitrary 16-bit patterns are bf16 NaN; one NaN in a matmul row makes every output of that
row NaN, and the following norm spreads it across the whole hidden vector. That is why:

- zeroing the act BO changed nothing — **the ELF never read the act BO**;
- zeroing the router, the norms head, the entire 5 MB norms BO, or the whole 481 MB weight BO
  changed nothing — the NaN did not come from any of those *values*;
- both the committed v1.0.x ELF and the regenerated v0.9.46 ELF NaN'd — the defect was in the
  driving, not the ELF, which is the one thing the two runs shared;
- the NaN was invariant to the input, because the "input" was weight bytes either way.

The interposer's own comment had the answer: it was corrected on 2026-09-13 to say idx3 is the
activation, not an instruction BO. The harness's arg order was never updated to match.

## Self-correction in this investigation

An intermediate probe reported "255 pages with finite data -> CHECK THE DUMP OFFSET" and nearly
became a finding that the all-NaN was a dump artifact. It was wrong: the predicate counted
`isfinite(0.0f)`, which is **true**, so the init memset read as data. The pages were 255
byte-identical all-zero blocks. The scan in `moe_smoke.cpp` now counts non-zero separately and
distinguishes all-NaN / all-zero-untouched / real-data pages.

## What is NOT established

- **Numerical correctness.** The logits are finite, which is not the same as correct. The
  single-layer argmax is `193722`; the tool's stored reference for this layer is `76740`, and
  they disagree. That reference was derived with the harness in its broken arg order, so it is
  itself suspect and must be re-derived before it can be used as an oracle. Until a reference
  is re-derived (or the full 40-layer stack is compared against FLM), the only verified claim
  is **NaN eliminated**, not **output correct**.
- That layer 1 is the only affected binding. args 5, 6 and 7 were **not** re-derived against
  FLM's order here — only idx3/idx4 were swapped. FLM's sizes (idx5=1 MB, idx6=1 MB,
  idx7=32 MB) should be checked against the harness's router/norms/kv BOs
  (1,060,864 / 5,242,880 / 134,217,728) before this path is trusted end to end.
- Anything about the v0.9.46 ELF. It was tested only under the broken order.
