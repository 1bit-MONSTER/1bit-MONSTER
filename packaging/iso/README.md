# 1bit.MONSTER appliance ISO

Builds a fully-unattended Ubuntu Server 26.04 installer ISO that boots into
a running 1bit.MONSTER OpenAI-compatible inference API. Design rationale
and scope: `../../docs/superpowers/specs/2026-08-16-ubuntu-iso-design.md`.

This is the "proper inference drivers out of the box" story: the pinned
TheRock `7.14.0a20260612` gfx1151 SDK, the pinned Vulkan stack, and the
engine `.deb` are **baked into the ISO** — nothing is resolved from
Ubuntu's live repos at install time, so the default-picker problem
(Ubuntu selecting classic ROCm 7.2.4, the enterprise-supported line) can
never affect an appliance built from this image. Pin values live in
`../rocm-lane-pin.env` — the single source shared with
`scripts/setup-therock.sh` (see `../../docs/rocm-lanes.md`).

## Build

```bash
bash fetch-payload.sh          # vendors the pinned driver stack (once, or when versions change)
bash build.sh --ssh-key ~/.ssh/id_ed25519.pub
```

Output: `build/1bit-monster-26.04-amd64.iso` and
`build/console-recovery-password.txt` (a randomly generated local-console
login for the `monster` account — SSH itself is key-only; this password
is for physical/JetKVM-style console recovery if the baked-in SSH key
doesn't work, and is never written onto the ISO itself, only kept
alongside the build output).

## What's baked in vs. what happens on first boot

- Baked in (no network needed at install time): the engine `.deb`, pinned
  `mesa-vulkan-drivers`/`libvulkan1`, pinned TheRock gfx1151 libraries,
  the `1bit-unified.service` and `1bit-model-fetch.service` units.
- First boot (needs network): the `1bit-model-fetch` service downloads the
  default `qwen3-0.6b` model in the background; the API is listening on
  `:8088` immediately but has nothing to serve until that finishes.
- **Not included in v1**: NPU (XDNA) acceleration — detected and noted in
  `/etc/1bit-monster-motd`, but the driver isn't installed automatically
  since no installable package exists yet. CUDA is dropped entirely (no
  NVIDIA hardware to validate against).

## Testing

```bash
bash test-qemu.sh build/1bit-monster-26.04-amd64.iso /tmp/1bit-iso-test-key
```

Boots the ISO in headless QEMU/KVM, waits for the unattended install to
finish, and checks over SSH that the engine, driver holds, kernel cmdline,
and API health endpoint all came up correctly.

## Real-hardware validation

<!-- Pick the matching paragraph based on the Task 1 spare-storage finding
     (docs/superpowers/specs/2026-08-16-ubuntu-iso-design.md "Open items") —
     do not leave both. -->

Spare storage was found at `<device path from Task 1>` on the reference
Strix Halo box. To validate on real hardware: `dd` the built ISO to a USB
drive, boot the box from it via JetKVM (do not touch the box's existing
root disk), and install onto `<device path>`, not the live system's disk.

<!-- or, if none was found:

No spare storage was available on the reference box as of the date in
docs/superpowers/specs/2026-08-16-ubuntu-iso-design.md's recorded
finding. Real-hardware validation is deferred until a spare drive is
available; the QEMU boot test above is this project's automated
correctness gate in the meantime. -->
