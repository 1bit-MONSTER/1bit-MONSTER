# 1bit.MONSTER LIVE image (packaging/live)

A lean, package-driven **live boot** image for the 1bit.MONSTER inference
appliance — no installer, no desktop, no 2.8 GB base-ISO download. Baked
from Ubuntu 26.04 packages via `debootstrap` and assembled into a dd-able
GPT image that boots straight to the engine's OpenAI-compatible API
(`:8088`) on any UEFI x86_64 box.

## Build

```bash
bash build.sh --ssh-key ~/.ssh/id_ed25519.pub        # archive kernel (any build box)
bash build.sh --ssh-key ... --kernel og              # + this host's custom 7.2 og kernel
bash build.sh --ssh-key ... --size 16G               # room for an 8B-tier model
bash build.sh --ssh-key ... --bake-model qwen3-0.6b  # offline-first: model on the stick
```

Output: `build/1bit-monster-live-26.04-amd64.img` (GPT: 537 MB ESP + ext4
root) and `build/console-recovery-password.txt`.

```bash
sudo dd if=build/1bit-monster-live-26.04-amd64.img of=/dev/sdX bs=4M status=progress conv=fsync
```

Payload (`build.sh` fetches it via the shared iso payload step):
- engine `.deb` (single 74 MB ELF — NPU/GPU/CPU in one binary)
- **hrx-b66** HRX bundle at `/opt/hrx` (AMD "Hip Runtime Extended",
  self-contained llama-server + libhrx/libggml-hrx + its own libhsa/vulkan,
  35.8 MB sha256-pinned from `ROCm/ggml-staging-automation`)
- Ubuntu ROCm 7.1 runtime libs (libamdhip64-7/hipblas3/rocblas5/rocsolver0/
  hsa-runtime64-1) + libxrt2 (NPU userspace) + mesa-vulkan + bolt — all
  `apt-mark hold`-ed. **No TheRock pip-SDK, no ROCm stack.**
- og kernel modules (amdgpu/amdxdna) + amdgpu firmware (28 MB)
- `1bit-unified.service` (`HRX_ROOT=/opt/hrx`, API on :8088) and
  `1bit-model-fetch.service` (qwen3-0.6b from the org GitHub release,
  sha256-pinned — runs only when the model is absent)
- GTT tuning on the kernel cmdline:
  `ttm.pages_limit=31457280 amdgpu.no_system_mem_limit=1`

## Gate

```bash
bash test-live.sh build/1bit-monster-live-26.04-amd64.img /path/to/key
```

Boots the image in headless QEMU (UEFI/OVMF), waits for SSH, checks:
kernel, engine deb, `1bit-unified` active, GTT cmdline, apt holds, HRX
bundle, HIP lib resolution, `/v1/health`, model-fetch enabled.
Status: **7/8 PASS** on a clean build (health needs a loaded model — the
q4nx NPU path only runs on real hardware, and QEMU has no NPU).

## Secure Boot

**The live stick requires Secure Boot OFF** (UEFI-only is fine). The ESP's
`BOOTX64.EFI` is an unsigned GRUB and the og kernel is a custom unsigned
build — neither can be signed for the Microsoft/DB trust chain, and the
machine is a single-purpose appliance. If the target box boots Windows
with Secure Boot enabled (Windows 11 default), disable it in the firmware
setup before selecting the USB stick; the `--removable` GRUB install then
boots from any UEFI boot menu.

## Model supply note (2026-09-03)

`qwen3-0.6b` registry entry points at the sha256-pinned org GitHub release
(`1bit-MONSTER/1bit-MONSTER` → release `qwen3-0.6b-q4nx`, asset
`model.q4nx`, 652 MB, sha256 `db8dec4d…`). The old HF source
(`bong-water-water-bong/qwen3-0.6b-q4nx`) is private → 401. Q4NX is fully
decoded: `fastflowlm_analysis/q4nx_assemble.py` regenerates the artifact
payload-identically from any public Qwen3-0.6B Q4_K_M GGUF.
