# Upstream kernel bug report — amdxdna NPU (Strix Halo) driver issues

**Status:** prepared 2026-09-01 · NOT YET SENT (no mailer on strixhalo)
**Filed on OGC:** https://github.com/OpenGamingCollective/kernel-packages/issues/39
**Target:** dri-devel@lists.freedesktop.org (amdxdna is a DRM/accel driver)
**CC:** Min Ma <mamin506@gmail.com>, Lizhi Hou <lizhi.hou@amd.com> (AMD XDNA maintainers), Shyam Sundar S K <Shyam-sundar.S-k@amd.com> (amd-pmf author)

---

## How to send

Plain email to `dri-devel@lists.freedesktop.org` (CC above) with the text below, subject as given.
Or from a git checkout of linux: `git send-email --to dri-devel@lists.freedesktop.org --cc mamin506@gmail.com --cc lizhi.hou@amd.com --cc Shyam-sundar.S-k@amd.com <this-file-as-patch>`.

---

## Subject: [PATCH?/BUG] amdxdna: NPU4 telemetry unconditionally calls amd_pmf_get_npu_data() → -ENODEV spam on platforms without PMF device; plus IOMMU identity-group fault storm on Strix Halo

### Problem A — PMF telemetry returns -ENODEV on every boot (no PMF ACPI device)

**Symptom** (reproduced on every boot, kernel 7.2.0-next-20260821, Strix Halo desktop):

```
amdxdna 0000:c6:00.1: npu4_update_counters: PMF get npu data failed, ret -19
amdxdna 0000:c6:00.1: aie2_query_sensors: PMF get npu data failed, ret -19
```

**Root cause:** `amd_pmf_get_npu_data()` in `drivers/platform/x86/amd/pmf/metrics.c`:

```c
int amd_pmf_get_npu_data(struct amd_pmf_npu_metrics *info)
{
	struct amd_pmf_dev *pdev;

	if (!info)
		return -EINVAL;

	if (!pmf_device)          /* NULL: no PMF ACPI/platform device bound */
		return -ENODEV;
	...
```

On this machine (Strix Halo desktop, Radeon 8060S + XDNA2 NPU `1022:17f0` rev 11) there is **no PMF ACPI device** (`AMDI0102` absent from /sys/bus/acpi/devices; only AMDI000B/0010/0030/0052 present). `amd_pmf` still loads, but `pmf_device` stays NULL, so every call returns `-ENODEV`.

The amdxdna driver calls this unconditionally from its NPU4 telemetry paths (`drivers/accel/amdxdna/npu4_regs.c` ~line 131, `drivers/accel/amdxdna/aie2_pci.c` ~line 761 via `AIE2_GET_PMF_NPU_METRICS`), so NPU clock/busy/power/temp metrics are dead on any platform without the PMF device, and the failure is logged as an error at every boot.

**Suggestion:** guard the PMF telemetry as optional (e.g., treat `-ENODEV` as "not supported" and skip, or only attempt when the PMF device is present) instead of erroring; or have `amd_pmf_get_npu_data()` document that ENODEV means "no PMF on this platform" and let callers handle it silently.

### Problem B — IO_PAGE_FAULT storm on every NPU exec with IOMMU enabled; NPU is the only `identity`-type IOMMU group

**Symptom:** with the IOMMU enabled (default `Translated` domain), every `AMDXDNA_EXEC_CMD` triggers:

```
amdxdna 0000:c6:00.1: AMD-Vi: IO_PAGE_FAULT domain=0xNN address=0x7xxx flags=0x0027
```

in sustained bursts (~17k callbacks suppressed per 5 s during churn). Pattern: 24 KB stride (0x6000) with 512 B sub-offset (0x200) — descriptor/BD chain walk; fault region ~6 MB, base varies per run.

**Impact:** NPU exec wall time 9762 ms/layer vs 7.2 ms with `amd_iommu=off` (~1000×). Syncobj timeline waits still return 0 (the NPU completes; it is just catastrophically slow via fault recovery).

**Observation:** `/sys/kernel/iommu_groups/26/type` (the NPU's group) is the **only `identity`-type group** on the box; every other group (both GPU groups included) is `DMA-FQ`. Suspected interaction: identity domain + amdxdna's BO/IOVA mapping path → BDs reference unmapped IOVAs → fault per exec.

**Questions for the driver maintainers:**
1. Why does the NPU land in an `identity` domain while the rest of the SoC is `DMA-FQ`? Is the amdxdna device supposed to opt into a translated domain?
2. Is amdxdna's BO/IOVA mapping (userptr pages into the IOMMU domain) known to be compatible with translated domains on Strix Halo / XDNA2?

### Environment

- Host: AMD Strix Halo desktop, Radeon 8060S (gfx1151) + XDNA2 NPU
- NPU: `0000:c6:00.1` `[1022:17f0]` rev 11, driver `amdxdna` (accel driver 1.0.0), `/dev/accel/accel0`, PASID mode on
- Kernel: `7.2.0-next-20260821` (linux-next 2026-08-21; OGC linux-unstable build `7.2.0-next-20260821-unstable-ogc-g2a559b27-1`, clang)
- Firmware: `/lib/firmware/amdnpu/{1502_00,17f0_10,17f0_11}.sbin`
- `iommu: Default domain type: Translated`; cmdline has no `amd_iommu=off` (problem B workaround is `amd_iommu=off`)

### Workload context

We run 1-bit LLM inference (W4A16/Q4NX) on the NPU via XRT (xrt_coreutil). Full root-cause write-up and A/B evidence (grub diffs, firmware md5 checks, 3 xclbin generations, module reloads) available on request. Happy to run any requested tests (with/without IOMMU, strace of the exec path, per-group dmesg).
