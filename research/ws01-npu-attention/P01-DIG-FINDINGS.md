# P0.1 Dig — NPU "hang" root-caused: it's not a hang, it's a 1000×-slow faulting exec

**Date:** 2026-07-31 · **Subject:** `npu_engine_overlap_fd` (and original) — prefill "hang" at boot-to-decode
**Status:** ✅ Diagnosed. Engine works e2e (real tokens); NPU execs fault on every run and take ~10 s each.

## TL;DR

The documented "~1/3-1/2 of runs hang at boot-to-decode transition" is **not a hang** — it is a **~1000×-slow NPU exec path** where every `AMDXDNA_EXEC_CMD` triggers IO_PAGE_FAULTs and takes ~9.76 s (vs ~7 ms on 07-25), hidden by block-buffered stdout + short timeouts. The engine **does** complete prefill (90 s) and **does** generate real tokens at ~0.1 tok/s.

## Evidence chain

1. **"Hang" reproduced** on both original (attn_stub) and FD-swapped binaries — identical at 60-100 s timeouts. Proved NOT my swap (P0.5 is clean).

2. **dmesg**: `amdxdna 0000:c6:00.1: AMD-Vi: IO_PAGE_FAULT domain=0xNN address=0x7xxx flags=0x0027` — sustained bursts (~17k callbacks suppressed per 5 s in fast-churn mode). Pattern per burst: **24 KB stride** (0x6000) with 512 B sub-offset (0x200) — a descriptor/Buffer-Descriptor chain walk. Fault region ~6 MB, base varies per run (0x74da…, 0x70af…, 0x781d2a8…, 0x7f0b8f6…).

3. **strace** (attached to hung engine): fd 6 = `/dev/accel/accel0` (NPU). Pattern: `SYNCOBJ_TIMELINE_WAIT` (116-232 ms) → `AMDXDNA_CREATE_BO` → `AMDXDNA_GET_BO_INFO` → `AMDXDNA_EXEC_CMD` → … with **CREATE_BO + GEM_CLOSE churn per exec** (68 each in 12 s). Syncobj waits **return 0** — the NPU completes; it's just catastrophically slow.

4. **Long run (240 s timeout)** — the payoff:
   ```
   Prefill: 89913ms (9990 ms/tok)
   Decode: [0] tok=63806 9926.5ms (gpu=65.9ms/l npu=9762ms/l)
           [1] tok=31934 ...
   ```
   Per-layer: GPU 66 ms (fine-ish) · **NPU 9762 ms** (was 7.2 ms on 07-25).

5. **Not the 10:15 xclbin rebuild**: A/B with the pre-rebuild xclbins/insts (git `ff203d1b7~1`) faults identically. Both old and new GU/D pairs hang-slow.

6. **Not the model file** (qwen3_0_6b.q4nx, Jul 28, untouched).

## ✅ ROOT CAUSE CONFIRMED (2026-07-31 14:20) — IOMMU mode regression from grub edit

**The NPU exec slowdown is caused by the AMD IOMMU being re-enabled on 2026-07-30, breaking the amdxdna BO/DMA path.**

| Date | `/etc/default/grub` cmdline | NPU behavior |
|---|---|---|
| Jul 16 | `... amdxdna.aie2_max_col=40` (experiment) | — |
| **Jul 24/25** (5 tok/s benchmark era) | **`amd_iommu=off`** + amdgpu tuning | ✅ 7.2 ms/layer, no faults |
| **Jul 30 18:32** (grub edited, 4 reboots since) | `amd_iommu=off` **removed** | ❌ IO_PAGE_FAULT per exec, ~350 ms/layer |

Evidence:
- `grub.bak-20260724-203711` (day before the bench) contains `amd_iommu=off`; current grub (edited Jul 30 18:32) does not — diffs verified.
- The NPU's IOMMU group (`/sys/kernel/iommu_groups/26`) is the **only `identity`-type group** on the box (GPU groups are `DMA-FQ`) — identity domain + the amdxdna driver's BO/IOVA mapping path = BDs land on unmapped addresses → every exec faults → fault-recovery ~50× slowdown.
- Everything else eliminated: driver pristine (md5 == package), firmware stock 1.1.2.65 (md5 == decompressed .zst), XRT unchanged (Mar 20 build), 3 xclbin generations, engine code (07-25 era), model file, unified_server contention, module reload (sysfs rebind).

**Fix applied (2026-07-31 14:18):** `amd_iommu=off` restored to `GRUB_CMDLINE_LINUX_DEFAULT` (backup: `/etc/default/grub.bak-20260731-1418`), `update-grub` done. **Requires reboot to take effect.**

**Post-reboot validation:** `sudo dmesg | grep -c IO_PAGE_FAULT` before/after a run + `./build/npu_engine_overlap_fd models/qwen3_0_6b.q4nx 32` — expect prefill ~2 s, decode in the 5-40 tok/s class, zero faults.

Note: `unified-router.service` (systemd) also holds NPU-adjacent state; it was not implicated in this regression (faults identical with/without unified_server).

## Root-cause hypothesis (superseded — see above)

1. **BD addresses reference unmapped IOMMU IOVAs.** The per-exec `CREATE_BO`/`GEM_CLOSE` churn means a BD from a previous exec can reference a closed BO's IOVA → unmapped → fault on every exec. XRT's `ext::kernel` path creating/freeing temp BOs (instruction buffer or partials) per submit, with the AIE still touching the old IOVA.
2. **Generator/ABI mismatch**: today's xclbins come from `n1_core_i8_v25/v26.py` (insts 4× smaller, 509 KB→128 KB). If the BD template layout doesn't match XRT's arg-patching convention (`k->operator()(3, 0, 0, *bA, *bB, *bC)`), some BDs keep generator-default addresses → faults. (But pre-rebuild xclbins fault too, so this can't be the whole story.)
3. **Driver-level**: 7.0.0-28 amdxdna + XRT 2.21.75 host-BO mapping path (userptr pages mapped into the IOMMU domain; pages faulted-in after map, or BO freed while exec in flight). The `xdna-driver` tree in `~/` is a newer generation — not the loaded module.

## What this means for the plan

- **P0.1 is now scoped**: fix the NPU exec fault path, not "the hang". Expected win: 9.76 s → ~7 ms per layer ≈ 1400× → back to the documented 5-40 tok/s class.
- **#1207 is the numeric twin**: even when fast, NPU FFN output is wrong (cosine 0.0099) — the faults may or may not be related; fix separately.
- **WS-01 e2e unblocked**: with a 240 s+ timeout the engine reaches decode; my FD attention swap is in the running path (gpu=65.9 ms/layer includes it).

## Next actions (in order)

1. **Isolate generator vs BO-lifecycle**: run the FLM-native path (`backend_npu_flm.cpp`, stock FLM xclbins) — if it's fast and fault-free, the custom v25/v26 generator xclbins are the problem; if it faults too, it's the driver/BO path.
2. **Check BD patching**: dump the insts transaction's BD entries and compare against XRT's patched addresses (`DRM_IOCTL_AMDXDNA_GET_BO_INFO` returns the BO IOVA; faults should land on the BO IOVA ± offsets if patching is correct).
3. **Per-exec BO churn**: check XRT's `ext::kernel`/run lifecycle in this engine (one `xrt::run` reused — the `run_` member is overwritten per launch without waiting for prior completion in some paths; `partials_cache()` in the FD kernel is GPU-side, but the NPU path's `run_` reuse deserves a `run_.wait()` audit).
4. Update the bench doc: prefill 90 s / decode 0.1 tok/s as the honest current baseline (tagged `broken`).

## Files/tools

- `build/npu_engine_overlap_fd` — FD-swapped engine (240 s run: works e2e)
- Evidence: dmesg IO_PAGE_FAULT patterns; strace of /dev/accel/accel0; `/tmp` logs cleaned
- Cross-refs: issue #1207 (NPU FFN numeric), #939/#940 (old crash reports), `docs/research/npu/benchmarks-2026-07-25.md`
