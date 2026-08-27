# Issue #1799 — Router EDA OOB Heap Read: Reproduction Results (verified)

**Date:** 2026-08-27 · **Branch:** `fix/npu-1799-fp` · **Fix commit:** `4d4e9c48`
**Verified on:** strixhalo (AMD Ryzen AI MAX+ 395, RyzenAI-npu5, Ubuntu 26.04, g++ 15)
**Model:** `/home/bcloud/models/zaya1-8b.q4nx` (5.2 GB q4nx) · token 236778 · NPU_SEED=42

## Root cause (as fixed)

The q4nx manifest declares `model.layers.N.mlp.gate.router_states_scale` with a
`data_offsets` span of **2 bytes** (shape [1]). The loaders populated
`w.rw.eda` with ONE element, but the router's EDA recurrence
(`rs += prev_router * eda`, `rtr_h`=256 iterations) read `eda[1..255]` OUT OF
BOUNDS on every MoE layer after L1. The heap bytes past the 4-byte allocation
are sometimes sane-looking, sometimes garbage (`-3.4e25`, `-inf`) → the
razor-thin L3 expert tie (gap 5.7e-6) flips run-to-run (the #1799/#1775
"run-to-run nondeterminism"). L1 is immune (no `prev_router` there), which is
why layer-1 output was bit-identical in every experiment.

Fix (all four loaders + the router itself):
- `zaya_moe_cpu.h` router: bound the EDA loop by
  `min(rtr_h, prev_router.size(), eda.size())` — no OOB read regardless of
  tensor shape.
- `zaya_npu_runner.cpp`, `zaya_decode.cpp`, `zaya_cpu_runner.cpp`,
  `test_fused_silu.cpp`: when the manifest says 2 bytes, load the full
  `rtr_h*2`-byte tensor instead.

## 1. Manifest ground truth (independently re-verified)

Parsed the real model with the loader's own `get_offsets` + bf16 decoder:

| layer | declared size | first value | full 512-byte read (256 bf16) |
|---|---:|---:|---|
| L0 | 2 bytes | 0.0996094 | mean 0.9914, range [0.0996, 1.0469], all finite |
| L3 | 2 bytes | 0.0996094 | mean 0.9937, range [0.0996, 1.0625], all finite |

The blob really holds the full 256-value per-channel EDA scale at the declared
offset (means ~0.99, range [0.0996, 1.06]) — the manifest size is simply
wrong. First element is the range's lower bound, which is why L1 (no EDA) was
deterministic.

## 2. ASan reproduction (surgical harness, no model needed)

`engine/npu/tests/test_router_eda_oob.cpp` synthesizes full zaya1-8b dims,
builds a `RouterWeights` with a **1-element `eda`** (the buggy loader state)
and drives `router()` under `-fsanitize=address`:

- **pre-fix (ae799e01):** `ERROR: AddressSanitizer: heap-buffer-overflow`
  `READ of size 4 ... 0 bytes after 4-byte region`
  at `zaya_moe_cpu.h:99` (the EDA loop) → exit 1. Allocation: the 1-element
  `eda` vector (`operator new`).
- **post-fix (4d4e9c48):** `PASS: router EDA bounded loop — no OOB, math exact`
  → exit 0. Also asserts the EDA contribution matches a manual min-bounded
  recurrence (exact float), and that the full-length (rtr_h=256) load path
  reproduces the same recurrence.

## 3. Full-model ASan reproduction (zaya_npu runner's CPU path)

`zaya_cpu_runner.cpp` built with `-fsanitize=address`, run on the real
`zaya1-8b.q4nx` (same `router()` code as the NPU runner, no XRT):

- **pre-fix:** ASan aborts at `zaya_moe_cpu.h:99` (`heap-buffer-overflow`,
  READ of size 4) — the OOB fires on the real model's weights.
- **post-fix:** rc=0, ASan-clean, token stream generated.

## 4. Determinism (post-fix, full model, CPU path)

Two independent runs (same seed/binary/model/prompt) produced **bit-identical
token streams**: `215459 148125 145975 155614 212061 229070 11375 258481`.
The OOB garbage is gone; the router is a pure function of its inputs.

## 5. NPU probe note

The full `run_determinism_probe.sh` on the NPU runner was **memory-limited on
this 30 GB box** (~27 GB anon RSS at the resident-expert BO packing stage → OOM
kill; the box was less loaded when PR #1807 recorded 6/6 bit-identical probe
runs on the same machine). The fix's effect on the NPU path is the same code
as the CPU path verified above; re-run the probe on a quieter box with
`engine/npu/tests/run_determinism_probe.sh` to reconfirm 6/6 on hardware.

## Merge status

The same fix content was merged to main via **PR #1807** (`ce0356d0`), so this
branch's fix (`4d4e9c48`) is a verified, main-parity change; the branch also
carries the #1799 fingerprint/probe instrumentation (`NPU_DBG_FP=1`,
`run_determinism_probe.sh`) and the #1775/#1759 fused-decode work.

## Re-run commands

```bash
# surgical ASan regression (no model):
g++ -std=c++23 -O1 -g -fsanitize=address -fno-omit-frame-pointer \
    -I engine/npu/src -I engine/npu/generators \
    engine/npu/tests/test_router_eda_oob.cpp -o /tmp/test_router_eda_oob \
&& /tmp/test_router_eda_oob

# full-model CPU runner under ASan (strixhalo):
g++ -std=c++23 -O1 -g -fsanitize=address -fno-omit-frame-pointer -march=native -fopenmp \
    -o /tmp/cpu_runner_asan engine/npu/tools/zaya_cpu_runner.cpp engine/npu/build/dequant_q4nx.o \
    -Iengine/npu/src -Iengine/npu/include -Iengine/npu/generators -Iinclude -Iengine/npu -I. -I/usr/include \
    -L/usr/lib/x86_64-linux-gnu -lxrt_coreutil -luuid -lm -ldl -lpthread
ASAN_OPTIONS=quarantine_size_mb=64:detect_leaks=0 \
    /tmp/cpu_runner_asan /home/bcloud/models/zaya1-8b.q4nx 236778
```
