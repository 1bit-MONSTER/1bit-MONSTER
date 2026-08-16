# 1bit.MONSTER appliance ISO — design

Status: approved for implementation planning
Date: 2026-08-16

## Goal

Produce a downloadable, publicly-distributable Ubuntu Server 26.04 LTS
installer ISO that boots straight into a working 1bit.MONSTER inference
appliance on AMD Strix Halo (Ryzen AI Max+ 395) hardware, with zero manual
driver/toolchain setup — the reverse-engineering pain documented in the
project's README (`docs/journey.md`) should not have to be repeated by
every new owner of this hardware.

Audience: anyone with Strix Halo (or, as a side effect of the engine's own
runtime hardware probing, any x86_64 box) who wants to boot this ISO and
get a running OpenAI-compatible inference API with no manual steps beyond
providing an SSH public key.

## Non-goals (v1)

- Unattended NPU (XDNA) driver install — no installable package exists yet
  anywhere (only the source fork on this box). NPU acceleration is a
  documented manual follow-up post-install, not part of the automated flow.
- CUDA support — no NVIDIA hardware exists to validate against, and no
  version is pinned anywhere in the codebase today. The engine's CUDA path
  stays uncompiled/dormant for this ISO.
- Public hosting/distribution of the built ISO artifact itself (GitHub
  Release size limits, CDN, etc.) — a separate decision once the ISO is
  proven to work. Nothing gets pushed anywhere public as part of this work
  without a separate explicit go-ahead.
- Fixing the stale `1bit-systems` branding in `packaging/deb/DEBIAN/control`
  — noted, left untouched, out of scope here.

## Architecture

`packaging/iso/build.sh` is a script, not a from-scratch `live-build`
rootfs. It:

1. Downloads (or takes as a cached local input) the official Ubuntu Server
   26.04 LTS ("resolute") ISO.
2. Mounts/extracts it, injects:
   - `autoinstall.yaml` (Subiquity/curtin autoinstall seed) at the ISO
     root, plus a GRUB/isolinux boot entry that boots straight to
     autoinstall with no menu wait.
   - A `/pool/` directory on the ISO containing:
     - The built `1bit-MONSTER` `.deb` (from `packaging/deb`, current
       `VERSION`).
     - The pinned driver payload (TheRock `7.14.0a20260612` gfx1151 libs
       package; cached `mesa-vulkan-drivers=26.0.3-1ubuntu1` and
       `libvulkan1=1.4.341.0-1` `.deb`s).
     - The new systemd unit files (`1bit-unified.service`,
       `1bit-model-fetch.service`) and an apt-pin preferences file.
3. Repacks with `xorriso`, preserving the hybrid El Torito boot catalog so
   the output is bootable both as an optical image and `dd`'d to USB.
4. Output: `1bit-monster-26.04-amd64.iso`.

This reuses `packaging/deb`, `packaging/npu-install.sh` (for its hardware
detection logic), and `packaging/model-download.sh` as-is rather than
duplicating packaging logic. Re-running the script against a newer Ubuntu
26.04 point-release ISO, or a newer `1bit-MONSTER` `.deb`, is the entire
update story — no separate rootfs to maintain.

## Install-time flow (fully unattended)

1. Boot → autoinstall starts immediately, no interactive menu wait.
2. Guided full-disk install. Default account: `monster`, SSH-key-only
   (password login disabled). The autoinstall seed's `identity`/
   `ssh` section needs a public key supplied at ISO-build time (build
   script takes a `--ssh-key <path>` argument); document prominently in
   the README and first-boot MOTD how this was provisioned, and what to
   do if no key was baked in (documented recovery path via console
   login is an implementation-plan-level detail to work out — flagging
   here so it isn't silently dropped).
3. `late-commands` (curtin, chrooted into the target, network already up
   at this point in the installer):
   - `dpkg -i` the `.deb` from `/pool/` on the ISO — no network required
     for the engine install itself.
   - `dpkg -i` the pinned Vulkan `.deb`s from `/pool/`, install the
     TheRock gfx1151 payload to `/opt/rocm-therock` (matching the path
     `CMakeLists.txt` / `env.sh` already expect).
   - Write `/etc/apt/preferences.d/1bit-monster-drivers` pinning
     `mesa-vulkan-drivers`, `libvulkan1`, and the kernel meta-package
     (whichever tracks `7.0.0-27-generic`) to their installed versions,
     so a later `apt upgrade` can't silently drift the driver stack or
     kernel and break ABI/GTT-tuning compatibility with the engine.
   - `lspci | grep -i xdna` — if a Strix Halo NPU is detected, note it
     for the first-boot MOTD ("NPU detected — see docs for the manual
     driver setup to enable acceleration"). No driver install attempted.
   - Write `ttm.pages_limit=31457280 amdgpu.no_system_mem_limit=1` to
     `/etc/default/grub.d/1bit.cfg`, run `update-grub` — the confirmed
     working GTT config from the reference box.
   - Install and enable two new **system** systemd units (the existing
     `packaging/services/1bit-agent.service` is a **user** unit running
     the CLI chat agent — a separate, pre-existing concern, left as-is):
     - `1bit-unified.service` — runs the OpenAI-compatible API server
       (`unified`/`zaya_server` subcommand of the one-ELF binary) on
       boot, `After=network.target`.
     - `1bit-model-fetch.service` — oneshot,
       `After=network-online.target`,
       `ConditionPathExists=!/var/lib/1bit/models/qwen3-0.6b.q4nx`, runs
       `model-download.sh qwen3-0.6b` (610 MB, smallest entry in the
       existing model registry) once.
4. Reboot → SSH reachable via the baked-in key, `1bit-unified.service`
   already listening (serving nothing yet), model arrives in the
   background over the next few minutes once network is up, API becomes
   fully useful shortly after first boot.

## Fallback behavior

No Strix Halo NPU present → the engine's own runtime hardware probe
(`has_npu`/`has_hip_gpu`/`has_vulkan`/`has_avx512`, etc.) already selects
CPU/GPU backends automatically — this is existing engine behavior, nothing
ISO-specific is needed. The ISO therefore also works as a generic
"1bit.MONSTER appliance" on non-Strix-Halo x86_64 hardware, minus NPU
acceleration specifically.

## Testing plan

1. Boot the built ISO in QEMU/KVM first — fastest iteration loop, no
   hardware risk. Verify: autoinstall completes fully unattended, SSH
   key access works, both new systemd units come up, model download
   completes, `POST /v1/chat/completions` returns a real completion.
2. Real-hardware validation on the Strix Halo box (`192.168.50.69`) —
   **not** by overwriting its current root disk. Requires confirming
   available spare storage (second drive/USB) before this step; if none
   exists, this step needs a separate decision before it can happen.

## Open items carried into the implementation plan

- Exact recovery path if no SSH key is supplied at ISO-build time
  (console login fallback, one-time password, etc.) — needs a concrete
  answer, not left as "SSH-key-only" with no escape hatch.
- Spare storage confirmed: sda (250G USB Disk) available as of 2026-08-16 for real-hardware validation.
- `packaging/iso/build.sh`'s own dependencies (xorriso, an Ubuntu ISO
  extraction tool) need to be confirmed available/installable on the
  strixhalo box, or documented as required on whichever machine runs
  the build.
