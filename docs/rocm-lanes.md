# ROCm on Ubuntu: why inference pins TheRock 7.14.x, not the default ROCm 7.2.4

> **TL;DR** — Ubuntu is the main enterprise platform, used for enterprise
> workloads *and* AI inference — one platform, no split. AMD ships **two ROCm
> SDKs** for it: the **classic ROCm 7.2.x** line (7.2.4 on Ubuntu 24.04) that
> Ubuntu's default GPU tooling (`amdgpu-install`, apt) selects — the
> **enterprise-supported** line — and **TheRock 7.14.x**, the modular
> **AI-inference SDK** AMD is steering inference toward (production since
> 7.14). The engine's HIP path is validated on **both**: classic 7.2.4 on the
> reference Ryzen desktop, TheRock 7.14.x on the Strix Halo appliance. The
> only real problem is the **default picker**: it resolves inference to
> classic 7.2.4 even where the box was validated against — and must pin —
> TheRock. 1bit therefore **pins TheRock `7.14.0a20260612`** on the
> appliance/Strix-Halo path and **verifies the pin every day** — a setup that
> resolves to 7.2.4, or a nightly that drifted past the pin, fails loudly
> instead of silently running on the wrong SDK.

## The two SDKs

| | Classic ROCm 7.2.x (Ubuntu default) | TheRock 7.14.x (AI-inference SDK) |
|---|---|---|
| **Version line** | 7.2.x (7.2.4 on Ubuntu 24.04) | 7.14.x (`7.14.0a20260612`) |
| **Install path** | `amdgpu-install` / AMD apt repo / Ubuntu packages — Ubuntu's default | pip multi-arch index `rocm.nightlies.amd.com/whl-multi-arch/` — nothing installs it by default |
| **What it is** | Full ROCm distribution: kernel driver (DKMS) + userland libs | Modular multi-arch ROCm SDK: `rocm[libraries,devel,device-gfxNNN]` |
| **AMD positioning** | Broad platform/driver support — the **enterprise-supported** line | "TheRock goes production" — the AI software platform AMD is moving inference to |
| **Why it keeps winning** | `amdgpu-install` defaults to the latest classic release; every Ubuntu guide links it | Nothing selects it by default — it only exists if you pin it explicitly |

**One platform, two SDKs.** Ubuntu is used mainly for enterprise, and the
same Ubuntu boxes run both enterprise workloads and AI inference — there is
no split between "enterprise Ubuntu" and "AI Ubuntu". The split is only in
which SDK the default installers select (classic 7.2.x) versus which one a
given box was validated against.

**Reference boxes (both validated).** The engine's HIP path runs on:

| Box | Ubuntu | SDK | Status |
|---|---|---|---|
| Strix Halo (gfx1151) + appliance ISO | 26.04 | **TheRock 7.14.x** (pinned `7.14.0a20260612`) | pin contract applies — verified daily |
| Ryzen desktop | 24.04 | classic ROCm 7.2.4 (Ubuntu universe packages) | validated HIP inference (engine `fix(hip)` work); not a TheRock-pinned target |

A classic-validated box that never runs the pinned path is not "wrong" — it
just isn't an appliance/Strix-Halo target, so the TheRock pin (and its daily
verification) does not apply to it. Ubuntu's defaults are correct for the
classic line; they are simply not the SDK the appliance was built around.

**The driver split.** The amdgpu **kernel driver** is shared base
infrastructure (installed from the classic line's `amdgpu-dkms` or the Ubuntu
kernel) and is required on every AMD box. The **userland** SDK is where it
matters: inference links the HIP runtime, GPU device code, and math
libraries. Mixing classic 7.2.x userland with TheRock code objects is exactly
how you get `hipErrorNoBinaryForGpu` or silent library shadowing. **Driver
from the classic line, userland pinned to TheRock.**

## Why Ubuntu always lands on 7.2.4

1. **`amdgpu-install` defaults to the newest classic release.** AMD's own
   Ubuntu installer ([quick-start guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html))
   installs the latest stable classic ROCm — 7.2.4 on Ubuntu 24.04 — with no
   flag steering it anywhere else. Every Ubuntu-focused guide (AMD docs,
   DigitalOcean, community LLM walkthroughs) follows that path.
2. **Ubuntu's own ROCm packages track the classic line.** Anything pulled via
   `apt` resolves to classic 7.2.x.
3. **TheRock has no default anywhere.** It only exists when you explicitly
   `pip install` from the multi-arch index — so any setup that "just follows
   Ubuntu's instructions" never reaches it.
4. **AMD's position is that the classic 7.2.x line is the enterprise-supported
   line**, and the AI-inference SDK is TheRock — which AMD announced as
   *production* with ROCm 7.14 ("TheRock goes production", see
   [references](#references)). On an enterprise Ubuntu box running inference,
   the enterprise line is the driver base; the inference userland must still
   be TheRock.

## The 1bit pin contract

- **Pin:** `THEROCK_VERSION=7.14.0a20260612` — the same TheRock build the
  Ubuntu appliance ISO ships (`7.14.0a20260612`, held, never resolved from a
  live repo at install time).
- [`scripts/setup-therock.sh`](../scripts/setup-therock.sh) installs exactly
  that version and no other. If a box drifted (e.g. to a nightly `10.x`), the
  script realigns to the pin.
- The **daily systemd timer now verifies instead of upgrading.** The old
  `rocm-therock-update` timer ran `pip install --upgrade` every day — that is
  how boxes silently left the pinned SDK. It is replaced by
  `rocm-therock-verify.timer`, which runs
  [`scripts/verify-rocm-lane.sh`](../scripts/verify-rocm-lane.sh) + `rocm-sdk test`
  and records any drift as a visible systemd failure.
- [`env.sh`](../env.sh) and `/etc/profile.d/1bit-rocm-lane.sh` put TheRock's
  devel tree **first** on `PATH` and `LD_LIBRARY_PATH`, so Ubuntu's default
  classic `/opt/rocm` userland can never shadow inference.
- **Bumping the pin is an explicit, reviewed act** — edit the constant in
  `scripts/setup-therock.sh` (and the ISO pin) or pass `--version` once.
  Never `pip install --upgrade`.

## Out of the box: the appliance ISO

The pin contract is also the appliance story: `packaging/iso/` builds a
fully-unattended Ubuntu Server 26.04 ISO that bakes in this exact pinned SDK
(TheRock `7.14.0a20260612` gfx1151 payload, `mesa-vulkan-drivers`,
`libvulkan1` — all held), so "proper inference drivers out of the box" is a
property of the image, not of post-install setup. See
[packaging/iso/README.md](../packaging/iso/README.md) and
[the ISO design spec](../docs/superpowers/specs/2026-08-16-ubuntu-iso-design.md).

The pin values live in `packaging/rocm-lane-pin.env` — one source shared by
the live setup (`scripts/setup-therock.sh`, `install.sh`) and the ISO build
(`packaging/iso/`), and `build.sh` fails loudly if the autoinstall seed ever
drifts from it.

## Setup

```bash
sudo bash scripts/setup-therock.sh [--version 7.14.0a20260612] \
                                   [--gpu gfx1151] \
                                   [--reinstall] [--no-ollama]
```

What it does:

1. **SDK check** — warns if a classic 7.2.x userland (`/opt/rocm`,
   `amdgpu-install`, classic apt packages) exists; the kernel driver is fine,
   a classic *userland* shadowing TheRock is not.
2. **Pin install** — `pip install "rocm[libraries,devel,device-gfx1151]==7.14.0a20260612"`
   from the multi-arch index into `/opt/rocm-therock`, then `rocm-sdk init`
   (expands the devel tree). If an installed version differs from the pin it
   is realigned, not "upgraded".
3. **SDK env** — writes `/etc/profile.d/1bit-rocm-lane.sh` so every shell
   and non-repo inference process resolves HIP/ROCm to TheRock first.
4. **Ollama override** — regenerates `/etc/systemd/system/ollama.service.d/override.conf`
   with the *current* devel paths (they move across python versions) and the
   pinned env, then restarts ollama.
5. **Verify-only timer** — retires `rocm-therock-update.{service,timer}` and
   enables `rocm-therock-verify.{service,timer}` (daily `verify-rocm-lane.sh
   --quiet --version <pin>` + `rocm-sdk test`).
6. **Final verification** — runs the lane check; non-zero exit if the pin is
   not the active SDK.

## Verification

```bash
scripts/verify-rocm-lane.sh [--version 7.14.0a20260612] [--quiet]
```

| Exit | Meaning |
|------|---------|
| 0 | TheRock pin installed **and** active |
| 1 | TheRock install missing at `/opt/rocm-therock` |
| 2 | **Ubuntu's default classic ROCm (7.2.x) userland shadows TheRock** — `hipcc` resolves outside TheRock |
| 3 | TheRock installed but **drifted from the pin** (e.g. nightly 10.x) |
| 4 | Usage error |

Example output:

```
── ROCm SDK check ────────────────────────────────────────────
  TheRock (AI-inference SDK): /opt/rocm-therock
    installed: rocm 7.14.0a20260612   expected: rocm 7.14.0a20260612
  Classic  (Ubuntu default, 7.2.x):
    /opt/rocm
    amdgpu-install present (installs classic 7.2.x by default)
  active hipcc: /opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_devel/bin/hipcc
  ✓ TheRock 7.14.0a20260612 is the active inference SDK.
```

The daily timer surfaces drift as a failed unit:

```bash
systemctl status rocm-therock-verify.timer
journalctl -u rocm-therock-verify.service -e    # exit 2/3 = wrong SDK active
```

## Upgrade policy

- The inference SDK moves **only** when the team bumps the pin — validate on
  the appliance ISO build first, then update the constant here and in the ISO
  seed together.
- Unpinned nightlies are for experimentation only: `sudo bash scripts/setup-therock.sh --version latest`
  (or any explicit version). The daily timer keeps verifying whatever pin you
  last chose — it never upgrades on its own.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `hipcc`/`rocminfo` resolve to `/opt/rocm` (classic) | Ubuntu's default classic userland on `PATH`. `source env.sh` (or relogin for profile.d) puts TheRock first; `verify-rocm-lane.sh` exits 2 until fixed. |
| `hipErrorNoBinaryForGpu` | Wrong `CMAKE_HIP_ARCHITECTURES` (`gfx1151`), or classic 7.2.x userland shadowing TheRock device code. |
| `librocm_cpp.so` version mismatch / wrong lib loaded | `LD_LIBRARY_PATH` order — TheRock devel lib must come **first** (env.sh and the ollama override do this; hand-rolled exports often don't). |
| Devel paths look wrong (python3.14 → 3.15) | `setup-therock.sh`/`env.sh` resolve devel paths dynamically via `rocm-sdk path --root` with a site-packages fallback — do not hardcode `lib/python3.x`. |
| Ollama uses old ROCm | Re-run `sudo bash scripts/setup-therock.sh` to regenerate the override with current paths, then `systemctl restart ollama`. |
| Box silently on nightly 10.x | That's the old upgrade timer's work — the pin + verify-only timer prevent it going forward; realign with `--version 7.14.0a20260612 --reinstall`. |

## References

- [ROCm/TheRock — RELEASES.md](https://github.com/ROCm/TheRock/blob/main/RELEASES.md) — multi-arch pip installs, `rocm-sdk`, device extras (gfx1151 = Ryzen AI Max+ PRO 395)
- [AMD blog — ROCm 7.14: TheRock Goes Production and Expands AMD's AI Software Platform](https://rocm.blogs.amd.com/ecosystems-and-partners/rocm-7.14-blog/README.html) (mirror: [AMD Developer Community](https://devcommunity.amd.com/t/rocm-7-14-therock-goes-production-and-expands-amd-s-ai-software-platform/729))
- [AMD — ROCm 7.2 on Radeon and Ryzen for Linux release notes](https://www.amd.com/en/resources/support-articles/release-notes/RN-AMDGPU-LINUX-ROCM-7-2.html) — the classic enterprise-supported line
- [ROCm install quick-start](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html) — `amdgpu-install` defaults to the classic line
