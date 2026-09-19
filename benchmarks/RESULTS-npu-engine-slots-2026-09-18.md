# Slot-per-agent engine swap: verdict, ceiling, and how to restore the MoE — 2026-09-18

Closing record for goal `mu7nwtjv`. This file answers the two questions the goal ends on: **is the
swap validated, and what is the engine's slot ceiling** — plus the restore command for production.
Detail lives in the two results docs beside it:

* `RESULTS-slot-ceiling-idle-2026-09-18.md` — steps 1–3: baseline, MoE stop, flm idle ceiling.
* `RESULTS-engine-slot-ceiling-2026-09-18.md` — step 4: our engine's slot ceiling.
* `RESULTS-engine-slot-validation-2026-09-18.md` — step 5: byte-identity validation.

## Where production is right now

The production FastFlowLM MoE (`flm serve qwen3.6-moe:35b-a3b --port 8098`, systemd user unit
`flm-35b.service`) is **stopped**: unit `enabled` but `inactive (dead)` since 2026-09-18 21:42:03 ADT,
device at **0 hardware contexts**. It was stopped by exact PID (`pgrep -x flm`), never by a
`pkill -f` pattern. Nothing has been restored by this work.

## Restore command (production MoE)

```bash
systemctl --user start flm-35b.service
# equivalent, directly:
/opt/fastflowlm/bin/flm serve qwen3.6-moe:35b-a3b --port 8098
```

## Ceilings, side by side

| layout | simultaneous slots that serve | per-slot hwctx | binding constraint |
|---|---:|---:|---|
| flm `qwen3:0.6b` service, NPU idle | **16** | 4 | driver `CREATE_HWCTX` on slot 17 |
| our `npu_engine_qwen3_0_6b`, staggered joins | **26** (tested; no refusal) | **1** | host RAM (~4.4 GB/slot) |

The engine holds 1 hwctx per instance against flm's 4, so at the same driver budget it goes ~4×
further — and it never reached the driver wall: RAM (114 GB at 26 slots on a 122 GB box) runs out
first, around ~27 slots.

**Caveat that must not be lost:** launching many engine slots *simultaneously* with a 1024-token
prompt gets refused by a probabilistic **`Bf16Mm::init`** race (4/10, then 7/10 served; the same
simultaneous launch with a tiny prompt served 10/10). Staggered joins never hit it.

## Is the swap validated?

**Partially — validated for staggered joins, not yet for simultaneous long-prompt starts.**

* Validated: staggered slots hold contexts (26 resident, 1 hwctx each) and **8/8 token streams are
  byte-identical** to the serial reference, at 71–74 tok/s per slot; 4 simultaneous long-prompt slots
  and 8 simultaneous tiny-prompt slots are also 4/4 and 8/8 identical. No device errors.
* Validated: the engine's slot capacity exceeds flm's (26+ vs 16), so the slot-per-agent layout is
  viable on capacity.
* **Not** validated as a drop-in replacement for production: a workload that starts many long-prompt
  slots at once can still be refused (bf16 prefill-init race). Do not restore the MoE, but also do
  not call the swap production-ready until that race is fixed or the launcher staggers starts.

## Reproduce

All on strixhalo, from `~/1bit-MONSTER-goal`:

```bash
MAXN=22 bash benchmarks/flm_slot_ceiling_idle_probe.sh              # flm idle ceiling (16)
N=26 P=/tmp/tiny_ids.txt bash benchmarks/engine_slot_stagger_probe.sh   # engine ceiling (26)
N=8 P=/tmp/p_1k.txt NG=64 bash benchmarks/engine_slot_identity_bench.sh # identity
N=10 NG=64 P=/tmp/p_1k.txt bash benchmarks/engine_slot_ceiling_probe.sh # the bf16 race
```

Raw outputs for every number above are in `benchmarks/evidence-engine-slots/`.
