# AMD IOMMU PerfOpt on strixhalo — hardware verification & register experiment

**Date:** 2026-08-31
**Machine:** strixhalo (Ryzen AI Max+ 395, Radeon 8060S, XDNA 2 NPU)
**Kernel:** 7.0.0-30-generic (Ubuntu), `CONFIG_AMD_IOMMU=y`

## Verified IOMMU state (2026-08-31)

| Item | State |
|---|---|
| IOMMU | **ON** — AMD-Vi active, 32 IOMMU groups, no `amd_iommu=off` on cmdline |
| Default domain | Translated (`iommu: Default domain type: Translated`), lazy DMA (`CONFIG_IOMMU_DEFAULT_DMA_LAZY=y`) |
| **GPU** `c5:00.0` (amdgpu, Radeon 8060S) | group **20**, type **identity** (AMD IVRS unity-map) |
| **NPU** `c6:00.1` (amdxdna) | group **26**, type **identity** |
| Everything else (30 groups) | DMA / DMA-FQ translated |

Note: `c6:00.0` is a *dummy function*, not the GPU — earlier notes that placed the GPU at `c6:00.0`
were wrong; the amdgpu device is `c5:00.0`.

## Hardware PerfOpt support: CONFIRMED

- IVHD EFR (global): `0x246577efa2254afa` (boot log) — **bit 45 (FEATURE_PERF_OPT / PerfOptSup) = 1**.
- IOMMU MMIO base: **`0xfd200000`** (`/proc/iomem`, "amd_iommu" region; the IVRS IVHD reports base 0 —
  the kernel's actual mapping is authoritative).
- Register map (probed via `/dev/mem`): `CONTROL` @ +0x00 = `0x020001ff` (IOMMU_EN), EFR @ **+0x30**
  (low dword `0xa2254afa` matches the boot-log EFR), PerfOpt control @ **+0x16C** = 0 (disabled).
- The AMD IOMMU spec (Section 3.4.9, MMIO offset 016Ch): `PERF_OPT_EN` = bit 13 of the register at
  +0x16C. PerfOpt lets privileged integrated I/O devices (GPUs) bypass IOMMU translation — the IOMMU
  then only enforces IR/IW, with no GPA→SPA translations. Only valid while the device is untranslated
  (identity domain) with ATS/PRI/PASID off.

## Register experiment (raw MMIO write) — NEGATIVE result

**Hypothesis:** since the GPU is already in the identity domain, arming `PERF_OPT_EN` directly
(`write 0x2000 @ 0xfd20016c`) might be equivalent to the kernel patch's arm step.

**Result: the GPU WEDGED.**
```
amdgpu 0000:c5:00.0: ring gfx_0.0.0 timeout, signaled seq=748913, emitted seq=748915
amdgpu 0000:c5:00.0: Starting gfx_0.0.0 ring reset
amdgpu 0000:c5:00.0: Ring gfx_0.0.0 reset succeeded
amdgpu 0000:c5:00.0: [drm] device wedged, but recovered through reset
```
The bit readback was confirmed armed (0x2000), the ring hung shortly after, and the GPU recovered
via its own ring reset. The bit was then cleared (`write 0x0`, readback 0) and the GPU returned to
stable operation (benchmark runs clean).

**DMA A/B numbers (during the unstable window — NOT trustworthy):**
- D2D 256 MB memcpy: baseline 102.9 GB/s → "armed" 107.2 GB/s → post-clear 107.0 GB/s
  (variance is clock/thermal noise; no credible PerfOpt effect).
- Small D2D (256 B): 2.00–2.01 µs/op in all states.

**Conclusion:** the raw bit is NOT sufficient — **the kernel patch's detach/reattach cycle (clearing
ATS/PRI/PASID/SVA before arming, `skip_caps` reattach) is genuinely required** to arm PerfOpt safely.
This validates the design of the upstream series (Mario Limonciello, 2026-08-31,
`20260831055108.1893285`): `iommu/amd: Add PerfOpt IOMMU performance optimization support` +
`drm/amdgpu: Enable PerfOpt IOMMU perf optimization when GPU in identity domain`.

## Path forward

1. **Required:** the two-patch kernel series (iommu/amd + drm/amdgpu), applied to a kernel build.
   - The installed 7.0.0-30 kernel predates the series (posted to linux-iommu 2026-08-31); no
     prebuilt Ubuntu kernel carries it.
   - `amd_iommu` is built-in (`=y`) → a full kernel rebuild (or a patched mainline 7.x) is needed;
     the amdgpu side adds the `amdgpu.iommu_perfopt` module param (default 1).
   - The amdxdna driver on this box is the DKMS upstream build (0.17.0) for 7.0.0-30 — a kernel
     change requires rebuilding/reinstalling it for the new kernel.
2. Strix Halo satisfies every prerequisite the series checks: integrated GPU (AMD_IS_APU ✓),
   identity domain (group 20 ✓), hardware PerfOptSup (EFR bit 45 ✓).
3. **amdxdna on the new kernel:** the 7.2 in-tree driver already carries the TDR support
   (`tdr_timeout_ms` module param, default 2000 — same as the upstream 0.17.0 DKMS build on
   the running kernel), so **no DKMS rebuild is needed** for the NPU after switching kernels.
   The old 7.0.0-30 kernel stays as a grub fallback.
4. After the patch: `amdgpu.iommu_perfopt=1` (default) arms PerfOpt at probe; verify via the
   `dev_info_once "PerfOpt armed on IOMMU%d"` line; re-run the DMA A/B with a trustworthy
   benchmark (the raw-write wedge shows the measured-state caution needed).

## Post-build verification checklist (kernel 7.2.0-perfopt)

1. `uname -r` → `7.2.0-perfopt`; 32 IOMMU groups intact; GPU group 20 + NPU group 26 still identity.
2. `sudo dmesg | grep -i perfopt` → `PerfOpt armed on IOMMU0` (the patched driver's dev_info_once).
3. Register readback: `sudo /tmp/iommu_regs r 0xfd20016c` → `0x00002000` (PERF_OPT_EN held).
4. A/B: boot once with `amdgpu.iommu_perfopt=0` on the cmdline (baseline), once with default 1
   (armed), run `/tmp/gpu_dma_bench` in each and compare D2D bandwidth + small-op latency.
5. GPU health: vulkaninfo / the benchmark must stay clean — no `ring ... timeout` in dmesg
   (the patched driver's detach/reattach makes arming safe, unlike the raw write).

## Tools used (for the follow-up verification)

- `/tmp/iommu_regs` — minimal `/dev/mem` read/write of a physical register (32-bit).
- `/tmp/gpu_dma_bench` — HIP DMA benchmark (D2D bandwidth, small-op latency, read kernel).

## Build status (2026-08-31, later)

- **Linux 7.2.0-perfopt built and installed**: mainline 7.2 + both PerfOpt patches (patch 1 clean
  via git apply; patch 2 with minor offsets/fuzz via `patch -p1`). Built with the 7.0.0-30 config
  (debug/BTF/module-signing disabled, `LOCALVERSION=-perfopt`). `bzImage` + modules + initramfs
  installed; GRUB entry **"Ubuntu, with Linux 7.2.0-perfopt"** added (7.0.0-30 stays the default
  fallback). 7.2's in-tree amdxdna has TDR (`tdr_timeout_ms=2000`) — no DKMS rebuild needed.
- Patches saved at `patches/perfopt-{1,2}-*.patch` (iommu/amd 5 files, drm/amdgpu 3 files).
- **Pre-patch DMA baseline (current 7.0.0-30 kernel, GPU identity, PerfOpt off):**
  D2D 256 MB memcpy **93.3–96.6 GB/s** (3 runs; earlier single runs 102.9–107.2 — the bench has
  ~±7% thermal/clock variance, so a PerfOpt effect must exceed that to be visible); small D2D
  (256 B) **~2.00 µs/op** (the cleaner metric for a latency optimization).
- **Post-boot verification** (run as root after booting 7.2.0-perfopt):
  `/usr/local/bin/perfopt-boot-verify.sh` → logs kernel, IOMMU groups, PerfOpt dmesg lines,
  register readback (expect `0x2000`), amdgpu param, DMA bench, GPU health.
  One-time boot: `sudo grub-reboot "Advanced options for Ubuntu>Ubuntu, with Linux 7.2.0-perfopt"`.
  A/B: boot once with `amdgpu.iommu_perfopt=0` on the cmdline (baseline), once default (armed).

### Binary-level verification of the built kernel (no boot needed)

- `nm vmlinux`: `amd_iommu_enable_perfopt` / `amd_iommu_disable_perfopt` (T + ksymtab/CRC),
  `amd_iommu_perfopt_clear` / `amd_iommu_perfopt_restore`, static `__perfopt_write` — all present.
- `nm amdgpu.ko`: `amdgpu_iommu_perfopt` + `iommu_perfopt` param + `U amd_iommu_{enable,disable}_perfopt`
  (resolved against the built-in exports at load).
- Disassembly of `__perfopt_write`: `mov 0x16c(%rax),%edx` (read old) / `mov %eax,0x16c(%rsi)`
  (write) / `mov 0x16c(%rdx),%edx` (readback) — the exact MMIO 0x16C RMW+verify pattern.
- **A/B GRUB entries** (both selectable from the boot menu):
  - `Ubuntu, with Linux 7.2.0-perfopt` — PerfOpt armed (default, `amdgpu.iommu_perfopt=1`).
  - `Ubuntu, with Linux 7.2.0-perfopt (perfopt OFF)` — same kernel, `amdgpu.iommu_perfopt=0`
    (baseline half of the A/B), via `/etc/grub.d/40_custom_perfopt`.

### Stable pinned baseline (2026-08-31, clocks forced to "high")

With `power_dpm_force_performance_level=high` the bench noise collapses to ±1.6%:
- D2D 256 MB memcpy: **median ≈ 105.6 GB/s** (103.8–107.2, 5 runs)
- small D2D (256 B): **2.00–2.01 µs/op** (rock stable)
- GPU identity domain, PerfOpt off (7.0.0-30 kernel), i.e. the "perfopt OFF" reference point.
The verify script now pins the clocks before the 5-run median benchmark, so the post-boot
A/B (entry A armed vs entry B `amdgpu.iommu_perfopt=0`) is a like-for-like comparison.
Source-level review of both patches in the 7.2 tree: all hunks semantically correct
(amdgpu init/fini/resume call sites; iommu skip_caps, -EBUSY guard @ iommu.c:3167,
perfopt_get/put/clear/restore/enable/disable, init.c clear/restore hooks).

### POST-REBOOT VERIFICATION (2026-08-31, kernel 7.2.0-perfopt) — DONE

- Booted 7.2.0-perfopt; IOMMU 32 groups, GPU c5:00.0 identity (group 20), NPU identity (group 26).
- **PerfOpt ARMED by the patched driver**: dmesg `amdgpu 0000:c5:00.0: AMD-Vi: PerfOpt armed on IOMMU0`
  (at probe); register `0xfd20016c = 0x00002000` (hardware readback); `amdgpu.iommu_perfopt=1`.
- **GPU stable with PerfOpt armed** — no ring timeout, no wedge (unlike the raw-write experiment):
  the patch's detach/reattach is confirmed necessary AND sufficient on this silicon.
- **DMA A/B on the SAME kernel** (toggle bit 13 at runtime; the driver-prepared ATS-off state makes
  both directions safe):
  | State | D2D 256MB median | small D2D (256B) |
  |---|---|---|
  | PerfOpt ON  | ~106.4 GB/s | 1.94–1.95 us |
  | PerfOpt OFF | ~106.4 GB/s | 1.95 us |
  → **no measurable PerfOpt effect on this benchmark** (the feature is a "soft, optional latency
  optimization"; the hipMemcpy probe likely doesn't exercise the optimized direct-DMA path).
  The 7.0→7.2 kernel itself improved small-op latency ~3% (2.00 → 1.95 us), independent of the bit.
- **NPU verified on the new in-tree amdxdna 0.10.0 (TDR present)**: fused cascade all-ones
  bad=0/8192 for both Zaya (260096) and Qwen3 (390144), state=4, ~5ms.
- No nvme/PCIe/thermal errors in dmesg; memory/disk healthy. (nginx/llama-server units are NOT on
  this machine — those were the Bosgame M5 docs, a different box.)
