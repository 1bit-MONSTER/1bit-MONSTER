# This lane's engine as slots: 26+ resident, 1 hwctx each, and RAM is the wall — 2026-09-18

Step 4 of goal `mu7nwtjv`. The production MoE is stopped (`RESULTS-slot-ceiling-idle-2026-09-18.md`);
this asks how many of **our own** `npu_engine_qwen3_0_6b` instances fit as agent slots before the
driver refuses a hardware context.

## Method

`benchmarks/engine_slot_stagger_probe.sh` (engine analogue of the flm slot probe): launch one slot,
wait until it is **serving** (first decoded token), then launch the next — every slot stays alive, so
the first one the driver refuses is the ceiling. A tiny token-ID prompt (`/tmp/tiny_ids.txt`, 8 ids)
is deliberate: prefill is what contends on the host, and a residency ceiling is about holding
contexts, not about a long prompt. Counts are `xrt-smi examine -r aie-partitions` rows whose first
field is a numeric PID (one row per HW context), which was verified against one instance.

Raw outputs: `evidence-engine-slots/eng_stagger{16,20,24,26}.out`.

## Result: 26 slots resident, each holding exactly 1 hwctx; no driver refusal

| slots launched | served | live hwctx | total engine RSS |
|---:|---:|---:|---:|
| 16 | 16 | 16 | 70.28 GB |
| 20 | 20 | 20 | 87.87 GB |
| 24 | 24 | 24 | 105.53 GB |
| **26** | **26** | **26** | **114.38 GB** |

Every slot served (`Prefill: 85–2461 ms`), no `CREATE_HWCTX`, no ERT/TDR, and the device returned to
0 hwctx after cleanup. **One engine instance holds one hwctx** — four times cheaper than an flm
`qwen3:0.6b` slot, which held four. That is why the engine reaches 26 where flm stopped at 16.

**The binding constraint is host RAM, not the driver.** ~4.4 GB per slot; 26 slots is 114 GB of a
122 GB box. The driver's context wall is not reached at all in the range RAM allows: the engine
would need ~53 slots to hit the ~53-context wall flm hit, but RAM runs out at ~27 first. So the
honest answer to "how many before the driver refuses" is: **it does not refuse before RAM does**.

## The long-prompt simultaneous launch DOES get refused — a bf16 prefill-init race

Launching many engines *at once* with the 1024-token prompt (`/tmp/p_1k.txt`) is different, and
fails:

| run (N=10, `p_1k`, ng=64, all at once) | served | refused | failure |
|---|---:|---:|---|
| first | 4 | 6 | `Bf16Mm::init failed: DRM_IOCTL_AMDXDNA_CREATE_HWCTX IOCTL failed (err=-2)` |
| rerun | 7 | 3 | same |
| N=10, **tiny** prompt, all at once | **10** | 0 | — |

The refusal is in **`Bf16Mm::init`** — the bf16 prefill path — not in the base model context, and it
is **probabilistic** (4/10 then 7/10). With a tiny prompt the same simultaneous launch serves 10/10.
So this is a simultaneous-large-prefill init race, not a slot-count ceiling. Staggered joins (the
real "agents come online over time" pattern) never hit it in any of the 16/20/24/26-slot runs.

Evidence: `evidence-engine-slots/eng_ceil_n10.out` (4/10), `eng_all10_p1k_rerun.out` (7/10),
`eng_all10_tiny.out` (10/10).

## Reproduce

```bash
# on strixhalo, from the repo root:
N=26 P=/tmp/tiny_ids.txt bash benchmarks/engine_slot_stagger_probe.sh    # residency ceiling
N=10 NG=200 P=/tmp/tiny_ids.txt bash benchmarks/engine_slot_ceiling_probe.sh  # all-at-once, tiny
N=10 NG=64  P=/tmp/p_1k.txt  bash benchmarks/engine_slot_ceiling_probe.sh  # all-at-once, long: race
# /tmp/tiny_ids.txt = the first 8 ids of /tmp/p_1k.txt, one line
```
