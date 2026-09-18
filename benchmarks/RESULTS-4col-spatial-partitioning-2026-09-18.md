# 4-column kernels co-schedule at 2.06x: the spatial-partitioning answer — 2026-09-18

The question this thread started with was whether two NPU engines can run *genuinely in parallel* on
disjoint columns, or whether they can only time-share. Measured answer: **disjoint 4-column
partitions give true parallelism — two processes ran at full speed simultaneously (2.06x aggregate,
each process inflated 0.97x)** — and the 8-column stack cannot, because one context already wants
the whole device (1.79x, each process inflated 1.12x).

## How it was made runnable

1. **A caller-chosen device** — `benchmarks/flm_gemm_4col_build.py` (`/tmp/bounddev.py` does the
   same): a thin shim whose `resolve()` returns the `AIEDevice` enum, whose `__index__`/`__int__`
   let pybind11's `get_target_model(arg: int)` accept it, and which delegates everything else to the
   real iron `Device`. `bound_device_class(name)` returns a *class* pinned to a name, because IRON's
   `Program` instantiates the device class with no arguments. **No file in IRON or the mlir_aie
   package is modified** by the shim — it is a test-scaffold, not a patch.
2. **A parametrisation for the narrow device** — `flm.GEMM`'s test suite gated its shape table on
   `dev.resolve().name in ("npu1","npu2")`, so a 4-column device collected nothing. It now has an
   `npu2_4col` branch with 128-multiple widths (aie2p keeps `tile_n = 128`, so a full sweep is
   128×4 = 512 and the npu1 table's 320/64 shapes are rejected by the op). Test-only.

## The build

```python
Device(AIEDevice.npu2_4col)  →  cols=4 rows=6 arch=AIE2p
build/FLM_GEMM_tn128_ck32_ma64_mc1_emf_conv_even_npu2_4col.xclbin   (139,671 B)
   16 compute cores (4 columns × 4 rows) against 32 for the 8-column build
   AIE_PARTITION: column_width=4  start_columns=['0','1','2','3','4']
```

`start_columns` enumerating 0–4 is what makes two of them fit: the driver may place one 4-wide
partition at columns 0–3 and the other at 4–7. (For the full-width build the descriptor is
`column_width=8, start_columns=['0']` — so the earlier "it is a toolchain default" was half right:
it is *device-derived*, and a narrower device yields a narrower partition.)

## Correctness on device (4-column build)

```
COLDEV=npu2_4col PYTHONPATH=/tmp pytest iron/operators/flm/gemm/test.py -p colplug \
    -k "test_gemm and iter0" -q
→ 12 passed, 1 skipped, 72 deselected in 9.95s
```
Full sweep (N=512), remainder (384 = 3 of 4 columns), single-column (128), plus silu / gelu /
clamp / floor-rounding variants, all against the golden reference, on the real NPU.

## The measurement

Two `pytest` processes running the same matrix at once, against one running alone. The build cache
is warmed first, so this is device time plus a fixed per-process overhead; 20 iterations make the
device dominate.

| build | matrix | single | two at once | ratio | implied aggregate | per-process inflation |
|---|---|---:|---:|---:|---:|---:|
| **npu2_4col** | N=512, 100 tests | **33.1 s** | 32.1 s (A 32.0 / B 32.1) | **0.97** | **2.06x** | 0.97x / 0.97x |
| npu2 (control) | N=1024, 160 tests | 47.3 s | 52.8 s (A 52.8 / B 52.5) | 1.12 | 1.79x | 1.12x / 1.11x |

A 4-column process runs **at full solo speed while another does the same**, which is only possible
if they are on disjoint columns — two 4-wide partitions filling the 8-column array. The full-width
control slows each process 12%, the signature of sharing one partition.

At 6 iterations the same experiment read 1.06 and 1.17; the direction was already there and is
stable.

## What this is worth for the engine

- **The `mm` prefill GEMM can be built 4-wide today** from this tree, verified correct, and it
  co-schedules at 2x.
- The engine already takes an xclbin directory override (`NPU_FLM_XCLBIN_DIR` / `NPU_XCLBIN_DIR`),
  so substituting a 4-column `mm` is a wiring step, not a rewrite.
- The other kernels still need narrow builds: our attention generator takes `-c 4` already
  (`RESULTS-complete-rebuild-scope-2026-09-18.md`), IRON has `dequant`/`mha`/`rms_norm`/`rope`/
  `silu` for the rest, and FLM's fused `layer.xclbin` (whole-layer decode) has no IRON equivalent —
  that one is the remaining design work.
- Toolchains available on the box: **Peano** (what the shipped xclbins and all of the above use) and
  **xchesscc, licensed** — relevant because the shipped stack's PDI differs by compiler and the
  attention kernel is a Chess build.

## Reproduce

```bash
# build a 4-column FLM mm (writes to ~/amd-oss/iron/build/)
~/amd-oss/iron-venv/bin/python ~/1bit-MONSTER-goal/benchmarks/flm_gemm_4col_build.py npu2_4col 256 512 1024
# correctness on device
cd ~/amd-oss/iron && COLDEV=npu2_4col PYTHONPATH=/tmp ~/amd-oss/iron-venv/bin/python -m pytest \
    iron/operators/flm/gemm/test.py -p colplug -k "test_gemm and iter0" -q
# co-scheduling (single vs two-at-once, both builds)
ITERS=20 bash ~/cosched2.sh
```
