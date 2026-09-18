# Configuration sweep for concurrent NPU engines: what actually raises aggregate throughput — 2026-09-18

Follow-on to `RESULTS-dual-engine-concurrency-2026-09-18.md`. Same method (concurrent legs pass
`NPU_NO_DEVICE_LOCK=1` deliberately; rates are each instance's own `ms/tok`, so process startup is
excluded; every concurrent stream is compared byte-for-byte with its serial run), now sweeping the
knobs: instance count, model mix, host threads per instance, CPU pinning, OpenMP placement.

**First, the noise floor.** The same single-0.6B configuration measured 61, 66, 67, 77 tok/s in this
session, so anything under ~15% is not a finding. Every number below is a median or is labelled
with its range.

## 1k context, ng=128, Qwen3 models

| configuration | aggregate decode | note |
|---|---:|---|
| single 0.6B (8 threads) | 61–77 (median 67) | baseline |
| 2 × 0.6B, default | 67–77 | two copies of one model do not add throughput |
| 2 × 0.6B, pinned 0–7 / 8–15 + `OMP_PROC_BIND=close` | **77** | +15% over the same pair unpinned (67) |
| 2 × 0.6B, 16 threads each, no pinning | **77** | same effect via over-subscription |
| 2 × 0.6B, 4 threads each | 67–68 | 4 threads starves the host conversions |
| **0.6B + 1.7B** | **91–113 (median 95)** | 1.7B keeps its solo rate (34–39); 0.6B drops to 56 |
| 0.6B + 1.7B + 4B | 103 | 4B keeps 17–18 |
| 0.6B ×2 + 1.7B + 4B | **113** | |
| 0.6B ×2 + 1.7B ×2 + 4B | 112 | five instances, no better than two |
| 4 × 0.6B, pinned 4 threads each | 68–69 | pinning 4 instances to 4 cores each **hurts** (vs 84 unpinned) |
| mixed set, 4 threads each | 56 | worst configuration measured |

## The findings that matter

1. **~95–113 tok/s is this box's aggregate decode ceiling** for these models, i.e. **1.4–1.6x** a
   single engine instance, and **two instances of different sizes already get most of it**
   (0.6B+1.7B, median 95). Loading five instances adds latency, not throughput.
2. **Mix sizes, don't duplicate.** Same-model pairs do not aggregate (2×0.6B = 67–77, no better
   than one); a small model alongside a large one does, and the large model keeps its full solo
   rate. The large model's per-op gaps are what the small one fills.
3. **Pinning helps exactly two instances, and hurts four.** Two engines: pin to disjoint core sets
   (0–7 / 8–15) with `OMP_PROC_BIND=close`, or equivalently use 16 threads each — either is worth
   +15% over the default. Four engines: **do not pin**; 8 threads each on all 32 hardware threads
   beats 4 threads pinned to 4 cores (84 vs 69), because the host-side conversions scale with
   threads, not with cores.
4. **Four host threads per instance is a mistake** in every configuration measured (56–69).
5. **The in-process alternative is not reachable.** `NPU_BS=B` (B copies of one sequence decoded per
   step, shared prompt prefill, `M=B` batched GEMMs) would be the efficient way to serve identical
   replicas, but on the dense Qwen3 path the automatic path selection routes to the bf16-prefill +
   runlist decode, and the "M=B Batch Decode" branch is never taken (checked at 13- and 1024-token
   prompts). Implementing a batched *serving* path is engine work, not configuration.

## Long context is the opposite regime

At 8160 tokens (where the prefill is host-bound — `RESULTS-0_6b-prefill-hostbound-2026-09-18.md`):

| configuration | 0.6B prefill / decode | 1.7B prefill / decode | pair wall |
|---|---|---|---|
| serial, one after the other | 3897 ms / 35 | 5523 ms / 24 | 11.7 s |
| concurrent, default | 8676 ms / 27 | 9911 ms / 24 | **18.3 s (0.64x loss)** |
| concurrent, pinned 0–7 / 8–15 | 13773 ms / 21 | 14025 ms / 24 | **21.9 s (worse)** |

Pinning does not fix it and makes it slightly worse: the prefill is ~4 s of per-element host work
per instance, and two instances on one 16-core box simply queue for the same cores. **Add a second
engine for chat-shaped traffic; do not add one for long-document traffic.**

Correctness in every concurrent configuration stayed **byte-identical to the serial run**, with no
ERT/TDR/timeout line — 30+ instances across these sweeps.

## Recommended configurations

| workload | configuration |
|---|---|
| throughput, short/medium context | 2 instances, **mixed sizes** (e.g. 0.6B + 1.7B), 8 host threads each, no pinning — or pin disjoint core sets if the pair is same-size |
| max throughput, capacity to spare | the same 2 instances plus one more large model; expect ~1.6x single, not 2x |
| latency for one request | one instance only — concurrency costs the small model ~20% and a same-size pair ~45% each |
| long context (≥8k prompt) | serial; concurrency is a 0.6x loss |
| never | 4 host threads per instance; pinning 4+ instances to 4 cores each |

## Reproduce

```bash
bash ~/sweep.sh    # 2x0.6B: pinning / thread / OMP placement sweep, plus mixed pairs
bash ~/sweep2.sh   # 4 instances, and the 8k pinned-vs-default comparison
bash ~/sweep3.sh   # aggregate vs instance count, mixed candidates, 4-thread comparison
bash ~/repeat.sh   # the two headline configs, 3 repetitions (the noise floor)
bash ~/bsweep.sh   # NPU_BS (in-process batch) probe — shows the batched path is not reachable
```
