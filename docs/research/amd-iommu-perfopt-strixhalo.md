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

## Fused-cascade real-weight calibration — CLOSED (2026-08-31)

The single-launch GU→SiLU→D cascade's real-weight FFN output is now verified
BIT-EXACT vs the CPU reference: `cascade_real_weight_probe` [pad] and [rep]
both report **EXACT MATCH (bad=0/8192, maxrel=0.0000)** on kernel 7.2.0-perfopt.

Root cause of the multi-round calibration failure: the CPU mirror's D-side model
used `ks_max=1` (a stale "A rows 1..7 = 0" assumption). The actual design:
- The A-tile's 8 rows all carry the SAME h2 slice (batch-replicated reading —
  A(row, K) is row-independent by construction).
- The D phase sums ALL 8 k-slices (worker loop `for ks in range(8)`).
The full-ks D model is silicon-exact. Everything else the calibration suspected
was actually correct and is now independently confirmed:
- deriv-inverse/8x8-microtiled B pack: guread one-hot-A probes 32/32 EXACT.
- cg-major AB element order: matches the worker's cg-outer consumption.
- GU h2 contract: h2r per-pair readouts EXACT for col 0..5 × cg 0..5.
- D-side readback: lay one-hot probes 8/8 EXACT (scr model).

## Fused-cascade production wiring — integer chain EXACT, float fold OPEN (2026-08-31)

The single-launch cascade's INTEGER C2 chain is silicon-exact (calibration CLOSED above).
The production NpuCascadeKernel (engine/fusion/zero_copy/npu_cascade_kernel.h) now:
- packs the deriv-inverse B_gu + cg-major elements + row-major B_d (verified),
- launches once per layer (ninstr = ins[2]; C2 BO must be packed into the kernel's
  OWN bBd — a caller-side B_d BO was silently empty -> C2=0),
- reproduces the verified reference C2 bit-exactly (63627 10287 41656 ... on blk.0).

**Open item — the float dequant fold:** ffn_out = C2*S is NOT a single constant
(rel-std ~189 measured vs the two-launch path on [-1,1] input). Root cause: the
on-core q22 silu saturates — the probe's own h2s stats show 99.4% of h2 pairs at
±127 (the raw int dots C1 ~ +/-1e5 are ~4 orders above the q22 LUT's [-4,4]
range). The int8 q22 path has gs_dummy (no per-column fold); the INT4 fused
kernel (silu_quant_i8_fused_i4) has the gs-header fold that would fix this.
Making C1 = real requires ascale'*gs = 1 -> A_q = A*gs -> sub-int8-resolution
(dead end, verified numerically). => the cascade's float output is
sign-approximate, NOT two-launch-equivalent, without a scale-fold mechanism
the current int8 xclbin lacks. Next candidates: (a) an int4-silu cascade
xclbin with the gs fold, (b) per-tile input scaling, (c) accept sign-approx.

## FastFlowLM open-sourced (ROCm/FastFlowLM, MIT, 2026-08-11) — architecture verdict

The production AMD NPU architecture (now open) settles the cascade float-output
question: **FastFlowLM keeps the activation HOST-SIDE** — libgemm.so exports
`Gemm::generate_seq(..., Activation_Type_t, ...)` (per-op GEMM instruction
generators), and libqwen3_6_moe_npu.so has `simd_bias_add_gelu` (host SIMD
activation) + separate `setup_expert_up_gate_q4k`/`setup_expert_down_gate_q4k`
gate/up/down GEMM sequences. No fused on-NPU silu. This is exactly 1bit-MONSTER's
two-launch pattern (GU + host silu + D), which is silicon-verified bit-exact.

=> The fused single-launch cascade's on-core q22 silu saturation (99.4% h2 at
+-127; cascade-vs-two-launch pearson 0.035, sign agreement 52%) is a real
limitation of the project's own fused design, NOT something AMD's production
architecture does differently. The float-valid paths are: the two-launch
(host silu — proven) or a rebuilt fused cascade with an int4-silu + per-column
gs-header fold (the mechanism the int8 q22 kernel lacks). The cascade remains
useful as the single-launch zero-h2-DMA substrate for integer/approx paths.

FastFlowLM repo cloned to /tmp/fflm (628MB, shallow): 219 xclbins +
src/lib/xrt/*.so + the runtime source. (ROCm/FastFlowLM, amd/IRON, MLIR-AIE 1.2,
ROCm 10 / ROCm.AI.)

---

## 2026-08-31 addendum — mid-session NPU SVA breakage (all DPU launches → state=8)

**Symptom**: after the 09:12 cascade calibration (EXACT MATCH on this boot), every
subsequent NPU launch timed out (`ERT_CMD_STATE_TIMEOUT=8`, DPU PC=0xffffffff,
TXN OP ID=0xffffffff, Context PC=0x28b060ad) with `AMD-Vi: IO_PAGE_FAULT` events at
host-VA addresses (e.g. `0x742b52d6c000`, flags=0x7). First faults at 10:29; the
09:48/10:13 module-flow attempts timed out WITHOUT faults (garbage TXN header).

**Mechanism** (from kernel sources):
- The NPU (0000:c6:00.1, group 26) is in the upstream-default IDENTITY domain for
  PASID-capable devices (`amd_iommu_def_domain_type` → IOMMU_DOMAIN_IDENTITY when
  `pdev_pasid_supported && !SME && !SNP`; **not** from the PerfOpt patch — the patch
  only touches attach_device's `skip_caps` for `dev_data->perfopt` devices).
- The AMD identity domain IS SVA-capable (`pdom_is_sva_capable` = v2 page tables OR
  identity/pt), so attach builds the GCR3 table + enables PASID; the amdxdna client
  binds SVA (`iommu_sva_bind_device`, pasid=1) and the FW context is created with
  that PASID. The DPU's shim DMA (host VAs) must translate via the GCR3.
- The default paging domain is PD_MODE_V1 (`amd_iommu_pgtable = PD_MODE_V1` in
  init.c) — v1 is **not** SVA-capable, so a DMA-domain NPU cannot do SVA at all
  (open would fail: no PASID, no carveout). force_iova=1 allocates a v1 paging
  domain too — verified still faulting. Identity is therefore the ONLY working
  SVA config, and it worked at 09:12.
- Something between 09:12 and 10:29 (repeated TDRs from the broken module-flow
  attempts, firmware-side degradation, or an IOMMU state corruption) broke the
  GCR3 translation for the NPU. Fresh driver reloads (12:05+) did NOT restore it.
  Conclusion: boot-time IOMMU/firmware state must be restored → **reboot**.

**Also fixed**: `/tmp/fflm-layer0.seq` header was garbage (`0x1b 0x1b00`) because
`fflm_dump2` default-constructed `npu_sequence nseq;` (no `setup_device`) →
uninitialized header fields. Fix: `npu_sequence nseq(device_npu2);` → header is now
`06040100 00000108 00000378 000070fc` (Major=0 Minor=1 DevGen=4 Rows=6 Cols=8
MemRows=1, NumOps=888, TxnSize=28924) — the SAME format as the working cascade
control code. The corrected seq is saved at `engine/npu/fflm-run/fflm-layer0.seq`.

**Post-reboot plan**: `engine/npu/fflm-run/post-reboot-verify.sh` checks group 26
type, runs the cascade probe (silicon control), then the real layer kernel via
`/usr/local/bin/fflm_run4 layer.xclbin fflm-layer0.seq` (module flow). Fallback if
identity+SVA is still broken post-reboot: `amd_iommu=v2` on the kernel cmdline
(makes paging domains v2 → SVA-capable → the NPU could use a DMA domain instead).

### ✅ 2026-08-31 14:51 — THE REAL FastFlowLM MODEL RUNS ON THE NPU

The actual ROCm/FastFlowLM runtime (built from the vendored
`third_party/FastFlowLM`) + the real Qwen3-0.6B model (683 MB `model.q4nx`,
downloaded from HuggingFace `FastFlowLM/Qwen3-0.6B-NPU2`) generates and runs
the full model on the XDNA2 NPU on **kernel 7.2.0-perfopt with the pristine
driver and the upstream identity IOMMU (SVA) config**:

```
Prompt "Hello" → 146 tokens, full reply
Prefill: 502.8 ms (27.8 tok/s)    Decode: 1.366 s (96.6 tok/s)
Zero IO_PAGE_FAULTs, zero TDR timeouts during the run
```

**Why the hand-rolled launcher deadlocked but the real runtime works**: the
runtime calls `gen_layer_seq` with the real weights loaded and allocates the BOs
to match the generated patch table. My weight-less dump of the sequence and my
guessed BO sizes did not match, so the shim-DMA sync tokens never fired — the
DPU correctly waited forever. The layer.xclbin, the identity-mode SVA, the
module flow, and the firmware were all fine.

Setup (preserved):
- `engine/npu/fflm-run/run_real_runtime.sh` — the run command
- `~/.config/flm/models/Qwen3-0.6B-NPU2/` — model cache
- `/opt/fastflowlm/share/flm/xclbins/Qwen3-0.6B-NPU2` → vendored xclbins
