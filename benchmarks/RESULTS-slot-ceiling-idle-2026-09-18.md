# Idle slot ceiling and the production MoE stop — 2026-09-18

Goal `mu7nwtjv`: stop the production FastFlowLM MoE (`flm serve qwen3.6-moe:35b-a3b --port 8098`),
measure what the slot ceiling becomes with the NPU idle, then swap in this lane's own engine as
concurrent slots. This doc records steps 1–3. Steps 4–5 (engine slots + token-identity) and the
restore note follow in `RESULTS-npu-engine-slots-2026-09-18.md`.

## 1. Pre-change baseline

| item | value |
|---|---|
| production server | `flm serve qwen3.6-moe:35b-a3b --port 8098` (systemd user unit `flm-35b.service`, Q4_K) |
| unit state | `enabled`, `inactive (dead)` since **2026-09-18 21:42:03 ADT** — killed by `TERM` after 8h 42m |
| live hwctx | **0** — `xrt-smi examine -r aie-partitions` → "No hardware contexts running on device" |
| `hwctx_limit` | 16 (per partition; the device reports 2) |
| RAM | 122 GB total, ~13 GB used, ~92 free, 17 buff/cache, **109 available** |
| known ceiling with the MoE resident | **13** `qwen3:0.6b` slots; slot 14's first inference refused with `DRM_IOCTL_AMDXDNA_CREATE_HWCTX err=-2` (`RESULTS-slot-ceiling-2026-09-18.md`) |

## 2. Production MoE stopped

The MoE was already down at the start of this session (unit `inactive (dead)`, killed `TERM` at
21:42:03). Re-verified after stopping it and cleaning the idle probe: `pgrep -x flm` empty, device
returns **0 hardware contexts**, RAM back to ~13 GB used. No `pkill -f` pattern was used — the flm
servers were stopped by exact PID (`pgrep -x flm`).

Restore command (the unit is still `enabled`):

```bash
systemctl --user start flm-35b.service
# or directly: /opt/fastflowlm/bin/flm serve qwen3.6-moe:35b-a3b --port 8098
```

## 3. Idle ceiling: 16 slots serve, then the same ioctl fails

Harness: `flm_slot_ceiling_idle_probe.sh` (this directory), run as
`MAXN=22 bash flm_slot_ceiling_idle_probe.sh` on strixhalo; raw log in `slot_ceiling_idle.raw.log`.
Baseline was 0 hwctx, MoE stopped. Ports 8220–8238.

| slot | port | serves? | hwctx | RSS |
|---:|---|---|---:|---:|
| 1 | 8220 | ok 61.9 | 4 | 4.46 GB |
| 2 | 8221 | ok 59.0 | 8 | 4.46 |
| 3 | 8222 | ok 59.3 | 12 | 4.46 |
| 4 | 8223 | ok 62.0 | 16 | 4.46 |
| 5 | 8224 | ok 62.2 | 20 | 4.46 |
| 6 | 8225 | ok 58.3 | 24 | 4.46 |
| 7 | 8226 | ok 58.5 | 28 | 4.46 |
| 8 | 8227 | ok 62.1 | 32 | 4.46 |
| 9 | 8228 | ok 62.2 | 29 | 4.03 |
| 10 | 8229 | ok 63.3 | 39 | 4.46 |
| 11 | 8230 | ok 56.4 | 39 | 4.42 |
| 12 | 8231 | ok 63.2 | 38 | 4.45 |
| 13 | 8232 | ok 63.3 | 38 | 4.46 |
| 14 | 8233 | ok 60.9 | 45 | 4.46 |
| 15 | 8234 | ok 58.2 | 52 | 4.46 |
| 16 | 8235 | ok 63.3 | 53 | 4.46 |
| **17** | 8236 | **starts, first inference FAILS (0.0)** | 48 | 0.33 |
| 18 | 8237 | starts, fails (0.0) | 53 | 0.34 |
| 19 | 8238 | starts, fails (0.0) | 53 | 0.34 |

**Result: 16 resident `qwen3:0.6b` slots serve with the NPU idle; slot 17 cannot.** The failure mode
is byte-for-byte the one seen with the MoE resident — the process starts and answers `GET /v1/models`
with 200, then the first inference dies on:

```
[ERROR] Failed to load model: DRM_IOCTL_AMDXDNA_CREATE_HWCTX IOCTL failed (err=-2):
              No such file or directory
```

Waiting on the MoE's ~9–12 contexts buys **+3 slots** (13 → 16). That is the only capacity the MoE
was costing the slot fleet.

### Caveats worth carrying forward

* **A failed slot does not fail at startup.** It passes the readiness curl and returns a JSON body
  with `decoding_speed_tps = 0`. The harness above treats `0.0` as a failure; the earlier resident
  probe (and the raw log here) printed `ok 0.0`, which reads like success unless you look at the
  server log. The authoritative signal is the `CREATE_HWCTX err=-2` line, not the RSS and not the
  HTTP status.
* **The hwctx count from `xrt-smi examine -r aie-partitions` is not monotonic** past ~35 (here:
  32 → 29 → 39 → 39 → 38 → 38 → 45 → 52 → 53). Use the first refused slot as the ceiling; the live
  count is useful only as a floor/corroboration.
* **Per-slot RSS is ~4.46 GB** while serving (0.33 GB for a refused slot that never loads weights).
  16 slots ≈ 71 GB of model RSS — well inside the 122 GB box, so RAM is still not the wall.
* **Idle slots stay free**: every serving slot returned 56–63 tok/s with 15 peers resident. Slots buy
  addressability/state isolation, not aggregate FLOPs.
