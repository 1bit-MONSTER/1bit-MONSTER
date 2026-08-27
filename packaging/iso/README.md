# 1bit.MONSTER appliance ISO

Builds a fully-unattended Ubuntu Server 26.04 installer ISO that boots into
a running 1bit.MONSTER OpenAI-compatible inference API. Design rationale and
scope: `../../docs/superpowers/specs/2026-08-16-ubuntu-iso-design.md`.

## Build

```bash
bash fetch-payload.sh          # vendors the pinned driver stack (once, or when versions change)
bash build.sh --ssh-key ~/.ssh/id_ed25519.pub
```

`fetch-payload.sh` needs a TheRock pip-SDK install to vendor from when the
pinned version is no longer in the nightlies index — it probes
`/opt/rocm-therock` then `~/.cache/pip/therock`, or set `THEROCK_PIP_ROOT`.
Both need ~7 GB of scratch space on the build machine's big disk (the
extracted TheRock wheels alone are ~5 GB; `TMPDIR` is pointed at the payload
dir for this reason).

Output: `build/1bit-monster-26.04-amd64.iso` and
`build/console-recovery-password.txt` (a randomly generated local-console
login for the `monster` account — SSH itself is key-only; this password is
for physical/JetKVM-style console recovery if the baked-in SSH key doesn't
work, and is never written onto the ISO itself, only kept alongside the
build output).

## What's baked in vs. what happens on first boot

- Baked in (no network needed at install time): the engine `.deb`, pinned
  `mesa-vulkan-drivers`/`libvulkan1`/`bolt` (Thunderbolt/USB4 — stock
  Ubuntu, frozen at the tested version, same rationale as Vulkan), pinned
  TheRock `10.1.0a20260822` runtime (rocm-sdk-core + rocm-sdk-devel +
  rocm-sdk-libraries — all three ship together because the 10.x devel
  package's `lib/*.so.N` entries are relative symlinks into libraries),
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

Boots the ISO in headless QEMU/KVM (`-enable-kvm -cpu host`, needs the
`kvm` group), waits for the unattended install to finish, then checks over
SSH that the engine, driver holds, kernel cmdline, Thunderbolt daemon, and
API health endpoint all came up correctly. **Status: PASS** (2026-08-27,
`1bit-monster-26.04-amd64.iso` built from `feat/appliance-iso-finish`).
Re-verified 2026-08-27: rebuilt from the cached payload and booted again in
QEMU — PASS, `/v1/health` returned `unified-server-1.0` with the full backend
list (packaging/iso untouched since the original PASS commit `4aca2306`).

## Branch lineage / consolidation (2026-08-27)

- `feat/appliance-iso-finish` — keeper lineage; all ISO/Flatpak/RPM work
  lives here (plan Tasks 1–8 all `- [x]`).
- `rebuild/iso-appliance` — strixhalo's rebuild continuation; its only
  commit (`96578e2c`, `packaging/AppRun` launcher) was a byte-identical
  duplicate of the existing `packaging/appimage/AppDir/AppRun` and was
  self-reverted (`d137a50e`), so the branch nets to **zero** content vs this
  one — close/archive it.
- `feature/appliance-iso` — original generation (TheRock 7.14 era), fully
  superseded; lineage assessment recommending closure:
  `docs/iso-lineage-assessment-2026-08-27.md` (on that branch).

## Real-hardware validation

Spare storage (sda, 250G USB disk) was confirmed available on the reference
Strix Halo box as of 2026-08-16. Real-hardware validation is still pending:
`dd` the built ISO to a USB drive, boot the box from it via JetKVM (do not
touch the box's existing root disk), and install onto the spare device. The
QEMU boot test above is the automated correctness gate in the meantime.
