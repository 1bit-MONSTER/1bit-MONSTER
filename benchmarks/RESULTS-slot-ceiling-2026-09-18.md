# How many model slots can be resident at once: 13, then the driver refuses — 2026-09-18

The question was slots, not aggregate: how many agents can each hold their own loaded model on this
NPU. Answer measured by loading `flm serve qwen3:0.6b` servers one at a time (each server = one agent
slot) with the production 35B MoE already resident, and at every step checking the live hwctx count,
the new server's **served** rate (not merely that it started), and its RSS.

## Result: 13 slots serve; slot 14 is refused by the driver

| slot | port | probe | hwctx live | RSS |
|---:|---|---|---:|---:|
| 1 | 8200 | ok 65.4 tok/s | 13 | 4.46 GB |
| 2 | 8201 | ok 61.7 | 17 | 4.46 |
| 3 | 8202 | ok 60.6 | 21 | 4.46 |
| 4 | 8203 | ok 65.1 | 25 | 4.46 |
| 5 | 8204 | ok 65.2 | 29 | 4.46 |
| 6 | 8205 | ok 61.0 | 28 | 4.06 |
| 7 | 8206 | ok 64.7 | 28 | 4.46 |
| 8 | 8207 | ok 65.4 | 37 | 4.04 |
| 9 | 8208 | ok 65.4 | 33 | 4.43 |
| 10 | 8209 | ok 65.2 | 36 | 4.06 |
| 11 | 8210 | ok 65.7 | 42 | 4.46 |
| 12 | 8211 | ok 65.7 | 46 | 4.46 |
| 13 | 8212 | ok 65.5 | (table garbled) | 4.46 |
| **14** | 8213 | **FAIL — could not serve (0.0 tok/s)** | 46 | 0.33 |

```
:8213 [ERROR] Failed to load model: DRM_IOCTL_AMDXDNA_CREATE_HWCTX IOCTL failed (err=-2):
              No such file or directory
```

**So: 13 resident model slots with the 35B MoE also loaded.** The wall is the **driver's
context-creation ioctl**, not clocks, not RAM, not the model: slot 14's process starts and then
cannot get a hardware context.

## What that means for a per-agent-slot design

* **13 agents can each hold a loaded model**, and because idle slots are ~free (every probe returned
  61–66 tok/s with twelve peers resident), a fleet where one agent thinks at a time runs at full
  solo speed. That is the configuration the "each agent gets its own slot" idea wants.
* **Budget ~4.4 GB per slot** (55 GB for 13). RAM is not the binding constraint here — the box has
  125 GB — but it would be around ~25 slots if contexts allowed it, and the failed 14th process
  still held 0.33 GB.
* **Concurrency is still shared compute**: with several slots working at once the aggregate is
  ~57 tok/s total (`RESULTS-eight-models-and-hwctx-2026-09-18.md`). Slots buy *addressability and
  isolation of state*, not parallel FLOPs.
* **`hwctx_limit=16` is not the wall.** It is documented as `[Debug] Maximum number of hwctx`, it is
  per partition, the device reports 2 partitions, and the live count passed 46 before failing. The
  per-process breakdown of `aie-partitions` also garbles past ~35 entries (nonsense PIDs like
  `2793472`), so trust the total, not the attribution.
* Untested and worth knowing: whether the ceiling rises without the production MoE resident (it
  holds ~9 contexts). Testing that means stopping production, so it was not done.

## Reproduce

```bash
MAXN=14 bash ~/slot_ceiling.sh     # loads slots one at a time; stops at the first that cannot serve
# the script leaves the slots running by design and prints their PIDs; stop them by PID
# (a `pkill -f "flm serve qwen3:0.6b"` also matches your own ssh command line and kills the shell)
```

Raw log: `/tmp/slot_ceiling.log` on strixhalo.
