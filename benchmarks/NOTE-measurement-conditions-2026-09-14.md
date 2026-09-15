# Measurement conditions for the FLM-parity numbers (2026-09-14)

Two conditions were found while trying to extend the dense-Qwen3 table to
decode @2k. Both affect any native-vs-FLM number and are recorded here so the
tables can be read correctly.

## 1. NPU power mode was Default, not Performance

`sudo xrt-smi examine -r all` reported `Power Mode: Default` on this box.

- FLM's CLI defaults `--pmode performance`
  (`amd-oss/fastflowlm/src/include/utils/vm_args.hpp:73`,
  `("pmode", ..., default_value("performance"))`), and the published tables say
  results are "Under FLM's default NPU power mode (Performance)"
  (`.../benchmarks/qwen3_results.md`).
- The native engine and `benchmarks/flm_parity.sh` never set the power mode, so
  every on-box native number in this series was taken with the NPU in Default
  mode while the published bar is a Performance-mode number.
- `sudo xrt-smi configure --pmode performance` sets it (needs root:
  `DRM_IOCTL_AMDXDNA_SET_STATE` fails with EACCES otherwise). It is now set to
  performance, matching FLM; the native engine does not manage this itself.
- The size of the effect is **not** quantified — the runs attempted after
  setting it were dominated by (2). Note only that the committed dense-Qwen3
  prefill wins were achieved in Default mode, i.e. they are not propped up by a
  power-mode advantage over the published Performance numbers.

## 2. Concurrent NPU users cause a ~2x decode variance

4B decode @1024 ms/tok across repeated runs:

    52.3, 52.5, 53.0, 54.2    (fast)
    95.2, 105.4, 106.1, 107.5, 112.5, 112.6, 120.1   (slow)

Bimodal, roughly 2x, and unrelated to context length (@256 50.3 / @512 50.7 /
@1024 52.5 when uncontended) or to system load-average (fast and slow runs both
occurred at load <1). Instrumentation pins the cause on the device side, not the
host:

- `NPU_RUNLIST_STATS=1` on a 1-token prefill: `build=1.5–1.8 ms`,
  `exec=53.5–54.4 ms` — the host runlist build is negligible, so the slow mode
  is the device exec doubling, not build/exec overlap failing.
- `fuser -v /dev/accel/accel0` showed the device held by two `llama-server`
  processes plus a concurrent `npu_engine_qwen3_8b` verification loop launched
  by a second (non-mesh) agent in this worktree.

So decode parity numbers are only valid with the NPU exclusively held. The fast
mode (~52 ms/tok = 19.0 tok/s @1k for 4B) is the uncontended value and matches
the two-pass figure in `RESULTS-native-vs-flm-dense-qwen3-REMEASURED-2026-09-14.md`.

## Bearing on the recorded tables

- The 1k and 2k **prefill** tables were measured before the concurrent 8B loop
  started (22:38) and reproduced to <2% across two passes, so they are not
  affected by (2). The power-mode caveat (1) applies to all of them.
- **Decode** was subsequently re-measured while the concurrent agent was
  between jobs; the uncontended values are now in
  `RESULTS-native-vs-flm-dense-qwen3-2k-2026-09-14.md` (1k and 2k, all four
  sizes). Each was reproduced within 0.5% and the ~2x slow mode was absent.

## Recommendation

Pin `--pmode performance` in the harness (or document it as a precondition) and
quiesce other `/dev/accel/accel0` users before recording any decode table.
