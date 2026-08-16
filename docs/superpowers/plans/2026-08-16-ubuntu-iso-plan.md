# 1bit.MONSTER Appliance ISO Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `packaging/iso/build.sh`, a script that produces a fully-unattended Ubuntu Server 26.04 installer ISO (`1bit-monster-26.04-amd64.iso`) that boots into a working 1bit.MONSTER OpenAI-compatible inference API on Strix Halo (or any x86_64) hardware, with a pinned, non-default driver stack baked in.

**Architecture:** Repack the official Ubuntu Server 26.04 ISO with an injected Subiquity `autoinstall.yaml` seed and a `/pool/` payload directory (the built `.deb`, pinned TheRock/Vulkan driver packages, two new systemd units). `late-commands` in the autoinstall seed installs everything unattended during the normal Ubuntu install; nothing is fetched from Ubuntu's live apt repos for the driver stack — only the exact pinned versions shipped on the ISO.

**Tech Stack:** bash, `xorriso`, Ubuntu Subiquity/curtin autoinstall (YAML), systemd unit files, QEMU/KVM for automated boot testing.

**Spec:** `docs/superpowers/specs/2026-08-16-ubuntu-iso-design.md`

## Global Constraints

- **All work happens on the remote box, not the local machine.** The repo lives at `~/1bit-MONSTER` on `bcloud@192.168.50.69` (hostname `strix`, Ubuntu 26.04 LTS). Every command in this plan is written to run over SSH: `ssh -i ~/.ssh/id_ed25519 bcloud@192.168.50.69 '<command>'`. There is no local checkout — files are drafted locally by the executor's tools, then `scp`'d to the box and moved into place inside the repo, exactly like the spec doc was delivered.
- Ubuntu base: **26.04 LTS "resolute"**, ISO `ubuntu-26.04-live-server-amd64.iso` from `https://releases.ubuntu.com/26.04/`.
- Driver pins (exact, never resolved from Ubuntu's live apt repo at install time):
  - TheRock (ROCm/HIP, gfx1151): `7.14.0a20260612`
  - `mesa-vulkan-drivers`: `26.0.3-1ubuntu1`
  - `libvulkan1`: `1.4.341.0-1`
  - `bolt` (Thunderbolt/USB4 userspace daemon): `0.9.10-1` — this hardware's USB4 Host Router already works via 100% stock Ubuntu (in-kernel `thunderbolt` module + the `bolt` package); no custom driver exists anywhere in this project. Same treatment as Vulkan: freeze the exact tested version rather than let a later apt update drift it.
  - Held via `apt-mark hold` post-install (not an apt-preferences pin file) so `apt upgrade` on the running appliance can't drift them, alongside the kernel meta-packages (`linux-image-generic`, `linux-headers-generic`, `linux-generic`).
- No CUDA in v1 (no NVIDIA hardware to validate against, nothing pinned in the codebase). No unattended NPU/XDNA driver install in v1 (no installable package exists anywhere yet) — NPU presence is detected and only noted in the MOTD.
- API server: binary `unified_server` (symlink to the one-ELF `1bit` binary, dispatch confirmed at `tools/onebin.cpp:66`, `cmd == "unified"`), default port **8088** (confirmed at `tools/unified_server.cpp` header comment), endpoints include `GET /v1/health`, `POST /v1/chat/completions`.
- Default first-boot model: `qwen3-0.6b` (610MB, smallest entry in `packaging/model-download.sh`'s registry), lands at `${HOME}/.local/share/1bit/models/qwen3-0.6b.q4nx` for whichever user runs the download (the `monster` account).
- Existing `packaging/services/1bit-agent.service` is a **user** systemd unit for the CLI chat agent — unrelated, not touched by this plan. The two new units this plan adds (`1bit-unified.service`, `1bit-model-fetch.service`) are **system** units.
- Nothing gets pushed anywhere public (no GitHub release, no upload) as part of this plan — the deliverable is a locally-built, locally-tested ISO file on the box.

---

### Task 1: Confirm build prerequisites and record spare-storage findings

**Files:** none created — this is a recon task whose findings gate later tasks (payload fetch needs `apt-get download`; the QEMU test needs `qemu-system-x86_64`).

**Interfaces:**
- Produces: confirmation that `xorriso`, `qemu-system-x86_64`, `qemu-img` are installed on the box; a recorded answer to "is there spare storage for real-hardware validation" that Task 8's runbook depends on.

- [ ] **Step 1: Check for required tools, install what's missing**

Run:
```bash
ssh -i ~/.ssh/id_ed25519 bcloud@192.168.50.69 '
which xorriso qemu-system-x86_64 qemu-img openssl python3 curl apt-get 2>&1
echo "--- installing anything missing ---"
sudo apt-get update -qq
sudo apt-get install -y -qq xorriso qemu-system-x86 qemu-utils
which xorriso qemu-system-x86_64 qemu-img
'
```
Expected: the final `which` line prints all three paths with no "not found" errors.

- [ ] **Step 2: List block devices to check for spare storage**

Run:
```bash
ssh -i ~/.ssh/id_ed25519 bcloud@192.168.50.69 'lsblk -d -o NAME,SIZE,MODEL,MOUNTPOINT'
```
Read the output. If a device exists that is *not* mounted at `/` or `/home` (i.e. not the live system's disk), note its device path — Task 8's real-hardware validation step targets that device. If no such device exists, Task 8's real-hardware step is deferred (documented, not skipped silently) until spare storage is available; the QEMU test in Task 7 remains the plan's actual automated gate either way.

- [ ] **Step 3: Record the finding in the spec's open items**

Edit `docs/superpowers/specs/2026-08-16-ubuntu-iso-design.md` on the box, replacing the "Confirm spare storage exists..." bullet under "Open items carried into the implementation plan" with the concrete finding (device path found, or "none found as of `<date>` — real-hardware validation deferred").

Run:
```bash
ssh -i ~/.ssh/id_ed25519 bcloud@192.168.50.69 'cd ~/1bit-MONSTER && git diff docs/superpowers/specs/2026-08-16-ubuntu-iso-design.md'
```
Expected: diff shows only that one bullet changed.

- [ ] **Step 4: Commit**

```bash
ssh -i ~/.ssh/id_ed25519 bcloud@192.168.50.69 'cd ~/1bit-MONSTER && git add docs/superpowers/specs/2026-08-16-ubuntu-iso-design.md && git commit -m "docs(specs): record build-prereq and spare-storage findings"'
```

---

### Task 2: Create `packaging/iso/` skeleton and gitignore the build/payload output

**Files:**
- Create: `packaging/iso/.gitkeep` (placeholder so the empty dir tracks in git before other files land)
- Modify: `.gitignore`

**Interfaces:**
- Produces: `packaging/iso/` directory that Tasks 3-7 write real files into; `.gitignore` rules so the multi-GB downloaded ISO, driver payload, and QEMU test disk never get committed.

- [ ] **Step 1: Create the directory and check current .gitignore**

Run:
```bash
ssh -i ~/.ssh/id_ed25519 bcloud@192.168.50.69 '
mkdir -p ~/1bit-MONSTER/packaging/iso
touch ~/1bit-MONSTER/packaging/iso/.gitkeep
tail -20 ~/1bit-MONSTER/.gitignore
'
```

- [ ] **Step 2: Append ISO-build ignore rules**

Run:
```bash
ssh -i ~/.ssh/id_ed25519 bcloud@192.168.50.69 "cat >> ~/1bit-MONSTER/.gitignore << 'EOF'

# packaging/iso build output (multi-GB, never committed)
packaging/iso/build/
*.iso
EOF"
```

- [ ] **Step 3: Verify**

Run:
```bash
ssh -i ~/.ssh/id_ed25519 bcloud@192.168.50.69 'cd ~/1bit-MONSTER && git status --short && tail -5 .gitignore'
```
Expected: `packaging/iso/.gitkeep` and `.gitignore` show as changed; no other files listed.

- [ ] **Step 4: Commit**

```bash
ssh -i ~/.ssh/id_ed25519 bcloud@192.168.50.69 "cd ~/1bit-MONSTER && git add packaging/iso/.gitkeep .gitignore && git commit -m 'chore(iso): scaffold packaging/iso, ignore build output'"
```

---

### Task 3: Write `packaging/iso/fetch-payload.sh` (pinned driver payload fetcher)

**Files:**
- Create: `packaging/iso/fetch-payload.sh`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `packaging/iso/build/payload/mesa-vulkan-drivers_26.0.3-1ubuntu1_amd64.deb`, `packaging/iso/build/payload/libvulkan1_1.4.341.0-1_amd64.deb`, `packaging/iso/build/payload/bolt_0.9.10-1_amd64.deb`, `packaging/iso/build/payload/therock-7.14.0a20260612-gfx1151.tar.gz` — Task 6's `build.sh` copies these four files into the ISO's `/pool/`.

- [ ] **Step 1: Write the script locally**

Write to local scratch path, content:

```bash
#!/usr/bin/env bash
set -euo pipefail
# packaging/iso/fetch-payload.sh — vendors the pinned driver stack into
# packaging/iso/build/payload/. This makes the ISO build deterministic:
# nothing is resolved from Ubuntu's live apt repo at install time, only
# the exact versions fetched here. See:
# docs/superpowers/specs/2026-08-16-ubuntu-iso-design.md

ISO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PAYLOAD="${ISO_DIR}/build/payload"
mkdir -p "$PAYLOAD"

THEROCK_VER="7.14.0a20260612"
MESA_VER="26.0.3-1ubuntu1"
VULKAN1_VER="1.4.341.0-1"
BOLT_VER="0.9.10-1"

echo "-- Vulkan + Thunderbolt (bolt): apt-get download pinned versions --"
( cd "$PAYLOAD" && \
  apt-get download "mesa-vulkan-drivers=${MESA_VER}" "libvulkan1=${VULKAN1_VER}" "bolt=${BOLT_VER}" )
test -f "${PAYLOAD}/mesa-vulkan-drivers_${MESA_VER}_amd64.deb" || {
  echo "FATAL: mesa-vulkan-drivers ${MESA_VER} not available via apt-get download." >&2
  echo "       Try: sudo apt-get update, or check 'apt-cache policy mesa-vulkan-drivers'." >&2
  exit 1
}
test -f "${PAYLOAD}/libvulkan1_${VULKAN1_VER}_amd64.deb" || {
  echo "FATAL: libvulkan1 ${VULKAN1_VER} not available via apt-get download." >&2
  exit 1
}
test -f "${PAYLOAD}/bolt_${BOLT_VER}_amd64.deb" || {
  echo "FATAL: bolt ${BOLT_VER} not available via apt-get download." >&2
  exit 1
}

echo "-- TheRock gfx1151 ${THEROCK_VER}: attempting exact-version pip download --"
TMP_PIP="$(mktemp -d)"
if pip download "rocm-sdk-libraries-gfx1151==${THEROCK_VER}" \
    --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ \
    --no-deps -d "$TMP_PIP" > /tmp/therock-pip.log 2>&1; then
  tar czf "${PAYLOAD}/therock-${THEROCK_VER}-gfx1151.tar.gz" -C "$TMP_PIP" .
  echo "   fetched from nightlies index"
else
  echo "   not available in nightlies index (log: /tmp/therock-pip.log)"
  echo "   falling back to vendoring the matching build already on this box"
  # NOTE: only rocm_sdk_libraries_gfx1151 is vendored here, not any
  # rocm_sdk_device_gfx1151 package. Confirmed against CMakeLists.txt:
  # the engine's build only ever links _rocm_sdk_libraries_gfx1151 (the
  # gfx1151 hipblaslt/Tensile kernels this package installs) — a separate
  # rocm_sdk_device_gfx1151 meta-package, if present locally, belongs to
  # a differently-structured, newer TheRock packaging generation and is
  # not something the engine's build references by name. Do not try to
  # vendor it; chasing a version match for an unused package is wasted
  # effort and risks bundling irrelevant/mismatched files.
  LOCAL="/opt/rocm-therock/lib/python3.14/site-packages"
  DIST_INFO="${LOCAL}/rocm_sdk_libraries_gfx1151-${THEROCK_VER}.dist-info"
  CONTENT_DIR="${LOCAL}/_rocm_sdk_libraries_gfx1151"
  # The real installed files live under the underscore-prefixed content
  # dir, NOT under the versioned *.dist-info dir (which is only a pip
  # metadata manifest — RECORD/METADATA/WHEEL, a few KB, no .so/.hsaco
  # files). A glob anchored on the dist-info's own versioned name can
  # never match the content dir, since the content dir's name carries no
  # version at all.
  if [ ! -d "$DIST_INFO" ] || [ ! -d "$CONTENT_DIR" ]; then
    echo "FATAL: ${THEROCK_VER} not found in the nightlies index, and the local" >&2
    echo "       dist-info/content pair for rocm_sdk_libraries_gfx1151 is incomplete" >&2
    echo "       (dist-info: $([ -d "$DIST_INFO" ] && echo present || echo MISSING)," >&2
    echo "        content dir: $([ -d "$CONTENT_DIR" ] && echo present || echo MISSING))" >&2
    echo "       — cannot vendor a payload for this pinned version." >&2
    exit 1
  fi
  # Correlate the content dir to this exact dist-info/version before
  # trusting it: the dist-info's RECORD manifest lists every file it
  # installed, content-dir-relative. If the content dir's own basename
  # doesn't appear as a path prefix in RECORD, this dist-info does not
  # describe what's currently sitting in the content dir (e.g. a later
  # install overwrote the content dir without updating this dist-info) —
  # refuse rather than silently vendor a possibly-mismatched payload.
  if ! grep -q "^_rocm_sdk_libraries_gfx1151/" "${DIST_INFO}/RECORD"; then
    echo "FATAL: ${DIST_INFO}/RECORD does not reference _rocm_sdk_libraries_gfx1151/" >&2
    echo "       — version correlation failed, refusing to vendor a possibly-stale" >&2
    echo "       or mismatched payload." >&2
    exit 1
  fi
  tar czf "${PAYLOAD}/therock-${THEROCK_VER}-gfx1151.tar.gz" -C "$LOCAL" "_rocm_sdk_libraries_gfx1151"
  echo "   vendored $(du -sh "$CONTENT_DIR" | cut -f1) from ${CONTENT_DIR}"
  echo "   (correlated against ${DIST_INFO}/RECORD)"
fi
rm -rf "$TMP_PIP"

echo ""
echo "Payload ready in ${PAYLOAD}:"
ls -la "$PAYLOAD"
```

- [ ] **Step 2: scp it to the box and place it in the repo**

Run:
```bash
scp -i ~/.ssh/id_ed25519 <local-scratch-path>/fetch-payload.sh bcloud@192.168.50.69:/tmp/fetch-payload.sh
ssh -i ~/.ssh/id_ed25519 bcloud@192.168.50.69 '
cp /tmp/fetch-payload.sh ~/1bit-MONSTER/packaging/iso/fetch-payload.sh
chmod +x ~/1bit-MONSTER/packaging/iso/fetch-payload.sh
'
```

- [ ] **Step 3: Syntax-check**

Run:
```bash
ssh -i ~/.ssh/id_ed25519 bcloud@192.168.50.69 'bash -n ~/1bit-MONSTER/packaging/iso/fetch-payload.sh && echo SYNTAX_OK'
```
Expected: `SYNTAX_OK`.

- [ ] **Step 4: Run it for real and verify the payload lands**

Run:
```bash
ssh -i ~/.ssh/id_ed25519 bcloud@192.168.50.69 'cd ~/1bit-MONSTER && bash packaging/iso/fetch-payload.sh'
```
Expected: script exits 0, final `ls -la` listing shows all four files:
`mesa-vulkan-drivers_26.0.3-1ubuntu1_amd64.deb`, `libvulkan1_1.4.341.0-1_amd64.deb`, `bolt_0.9.10-1_amd64.deb`, `therock-7.14.0a20260612-gfx1151.tar.gz`.

If the TheRock fetch falls into the local-vendoring branch, the script's own existence/correlation checks (dist-info present, content dir present, RECORD cross-reference) are the automated guard against vendoring a metadata-only or mismatched payload — a prior version of this task shipped a fallback that silently tarred up a 16KB `.dist-info` metadata dir instead of the real ~1.9GB library content, which these checks now catch. Additionally verify by hand once: `tar tzvf` the produced tarball and confirm it contains `.so`/`.hsaco` files, not just `RECORD`/`METADATA`/`WHEEL`.

- [ ] **Step 5: Commit**

```bash
ssh -i ~/.ssh/id_ed25519 bcloud@192.168.50.69 "cd ~/1bit-MONSTER && git add packaging/iso/fetch-payload.sh && git commit -m 'feat(iso): add pinned driver payload fetcher'"
```

---

### Task 4: Write the two new systemd system units

**Files:**
- Create: `packaging/services/1bit-unified.service`
- Create: `packaging/services/1bit-model-fetch.service`

**Interfaces:**
- Consumes: nothing.
- Produces: two unit files that Task 6's `build.sh` copies onto the ISO's `/pool/`, and that the autoinstall `late-commands` (Task 5) installs to `/etc/systemd/system/` and enables.

- [ ] **Step 1: Write `1bit-unified.service` locally**

```ini
[Unit]
Description=1bit.MONSTER OpenAI-compatible inference API (unified_server)
Documentation=https://1bit.monster
After=network.target

[Service]
Type=simple
User=monster
ExecStart=/usr/bin/unified_server --port 8088
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

- [ ] **Step 2: Write `1bit-model-fetch.service` locally**

```ini
[Unit]
Description=1bit.MONSTER first-boot default model fetch (qwen3-0.6b)
After=network-online.target
Wants=network-online.target
ConditionPathExists=!/home/monster/.local/share/1bit/models/qwen3-0.6b.q4nx

[Service]
Type=oneshot
User=monster
Environment=HOME=/home/monster
ExecStart=/usr/share/1bit/model-download.sh qwen3-0.6b
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
```

- [ ] **Step 3: scp both to the box and place them in the repo**

```bash
scp -i ~/.ssh/id_ed25519 <local>/1bit-unified.service <local>/1bit-model-fetch.service bcloud@192.168.50.69:/tmp/
ssh -i ~/.ssh/id_ed25519 bcloud@192.168.50.69 '
cp /tmp/1bit-unified.service /tmp/1bit-model-fetch.service ~/1bit-MONSTER/packaging/services/
'
```

- [ ] **Step 4: Verify unit syntax**

Run:
```bash
ssh -i ~/.ssh/id_ed25519 bcloud@192.168.50.69 '
systemd-analyze verify ~/1bit-MONSTER/packaging/services/1bit-unified.service 2>&1
systemd-analyze verify ~/1bit-MONSTER/packaging/services/1bit-model-fetch.service 2>&1
'
```
Expected: warnings about `/usr/bin/unified_server` or the model-download script not existing on *this* box's root filesystem are fine (they're not installed here — this box uses source builds, not the packaged paths) — treat those specific "file does not exist" warnings as expected. Any unit-syntax parse error (`Failed to parse`, unrecognized directive) is a real failure to fix.

- [ ] **Step 5: Commit**

```bash
ssh -i ~/.ssh/id_ed25519 bcloud@192.168.50.69 "cd ~/1bit-MONSTER && git add packaging/services/1bit-unified.service packaging/services/1bit-model-fetch.service && git commit -m 'feat(iso): add system units for the API server and first-boot model fetch'"
```

---

### Task 5: Write `packaging/iso/autoinstall.yaml.tmpl`

**Files:**
- Create: `packaging/iso/autoinstall.yaml.tmpl`

**Interfaces:**
- Consumes: `__SSH_PUBLIC_KEY__` and `__PASSWORD_HASH__` placeholders, substituted by Task 6's `build.sh`.
- Produces: the autoinstall seed that Task 6 copies (post-substitution) to the extracted ISO root as `autoinstall.yaml`; references filenames from Task 3's payload and Task 4's unit files by name (`1bit-systems_*_amd64.deb`, `mesa-vulkan-drivers_26.0.3-1ubuntu1_amd64.deb`, `libvulkan1_1.4.341.0-1_amd64.deb`, `bolt_0.9.10-1_amd64.deb`, `therock-7.14.0a20260612-gfx1151.tar.gz`, `1bit-unified.service`, `1bit-model-fetch.service`, `model-download.sh`).

- [ ] **Step 1: Write the template locally**

```yaml
#cloud-config
autoinstall:
  version: 1
  refresh-installer:
    update: false
  locale: en_US.UTF-8
  keyboard:
    layout: us
  network:
    version: 2
    ethernets:
      alleth:
        match:
          name: "en*"
        dhcp4: true
  apt:
    preserve_sources_list: true
  storage:
    layout:
      name: direct
  identity:
    hostname: monster
    username: monster
    password: "__PASSWORD_HASH__"
  ssh:
    install-server: true
    allow-pw: false
    authorized-keys:
      - "__SSH_PUBLIC_KEY__"
  packages: []
  late-commands:
    - "mkdir -p /target/opt/1bit-iso-pool"
    - "cp -r /cdrom/pool/. /target/opt/1bit-iso-pool/"
    - "curtin in-target --target=/target -- sh -c \"dpkg -i /opt/1bit-iso-pool/1bit-systems_*_amd64.deb\""
    - "curtin in-target --target=/target -- dpkg -i /opt/1bit-iso-pool/mesa-vulkan-drivers_26.0.3-1ubuntu1_amd64.deb /opt/1bit-iso-pool/libvulkan1_1.4.341.0-1_amd64.deb /opt/1bit-iso-pool/bolt_0.9.10-1_amd64.deb"
    - "curtin in-target --target=/target -- apt-mark hold mesa-vulkan-drivers libvulkan1 bolt linux-image-generic linux-headers-generic linux-generic"
    - "curtin in-target --target=/target -- mkdir -p /opt/rocm-therock"
    - "curtin in-target --target=/target -- tar xzf /opt/1bit-iso-pool/therock-7.14.0a20260612-gfx1151.tar.gz -C /opt/rocm-therock"
    - "curtin in-target --target=/target -- mkdir -p /etc/default/grub.d"
    - "curtin in-target --target=/target -- sh -c \"printf 'GRUB_CMDLINE_LINUX_DEFAULT=\\\"\\$GRUB_CMDLINE_LINUX_DEFAULT ttm.pages_limit=31457280 amdgpu.no_system_mem_limit=1\\\"\\n' > /etc/default/grub.d/1bit.cfg\""
    - "curtin in-target --target=/target -- update-grub"
    - "curtin in-target --target=/target -- cp /opt/1bit-iso-pool/1bit-unified.service /etc/systemd/system/1bit-unified.service"
    - "curtin in-target --target=/target -- cp /opt/1bit-iso-pool/1bit-model-fetch.service /etc/systemd/system/1bit-model-fetch.service"
    - "curtin in-target --target=/target -- mkdir -p /usr/share/1bit"
    - "curtin in-target --target=/target -- cp /opt/1bit-iso-pool/model-download.sh /usr/share/1bit/model-download.sh"
    - "curtin in-target --target=/target -- chmod +x /usr/share/1bit/model-download.sh"
    - "curtin in-target --target=/target -- systemctl enable 1bit-unified.service"
    - "curtin in-target --target=/target -- systemctl enable 1bit-model-fetch.service"
    - "sh -c 'lspci | grep -qi xdna && curtin in-target --target=/target -- sh -c \"echo NPU_DETECTED=1 > /etc/1bit-monster-motd\" || curtin in-target --target=/target -- sh -c \"echo NPU_DETECTED=0 > /etc/1bit-monster-motd\"'"
    - "curtin in-target --target=/target -- sh -c \"printf '1bit.MONSTER appliance — API on :8088. NPU status: see /etc/1bit-monster-motd\\n' > /etc/motd\""
  user-data:
    disable_root: true
```

Note for the executor: the `GRUB_CMDLINE_LINUX_DEFAULT` line's shell quoting is the single trickiest part of this file — curtin's `late-commands` entries are plain strings passed to `sh -c`, and nested quoting through `curtin in-target -- sh -c "..."` is easy to get subtly wrong. **Do not trust this by inspection** — Task 7's QEMU boot test is what actually proves this file is correct; if `cat /proc/cmdline` in that test doesn't show `ttm.pages_limit=31457280`, fix the quoting here first.

- [ ] **Step 2: scp it to the box and place it in the repo**

```bash
scp -i ~/.ssh/id_ed25519 <local>/autoinstall.yaml.tmpl bcloud@192.168.50.69:/tmp/autoinstall.yaml.tmpl
ssh -i ~/.ssh/id_ed25519 bcloud@192.168.50.69 'cp /tmp/autoinstall.yaml.tmpl ~/1bit-MONSTER/packaging/iso/autoinstall.yaml.tmpl'
```

- [ ] **Step 3: YAML syntax check**

Run:
```bash
ssh -i ~/.ssh/id_ed25519 bcloud@192.168.50.69 '
sed -e "s/__PASSWORD_HASH__/x/" -e "s/__SSH_PUBLIC_KEY__/ssh-ed25519 AAAAtest/" \
  ~/1bit-MONSTER/packaging/iso/autoinstall.yaml.tmpl | python3 -c "import sys, yaml; yaml.safe_load(sys.stdin); print(\"YAML_OK\")"
'
```
Expected: `YAML_OK`. If `cloud-init` is installed on the box, additionally run `cloud-init schema --config-file <substituted-copy>` for a stricter check and fix any reported errors.

- [ ] **Step 4: Commit**

```bash
ssh -i ~/.ssh/id_ed25519 bcloud@192.168.50.69 "cd ~/1bit-MONSTER && git add packaging/iso/autoinstall.yaml.tmpl && git commit -m 'feat(iso): add autoinstall seed template'"
```

---

### Task 6: Write `packaging/iso/build.sh` and produce the first ISO

**Files:**
- Create: `packaging/iso/build.sh`

**Interfaces:**
- Consumes: `packaging/iso/fetch-payload.sh` (Task 3), `packaging/services/1bit-unified.service` + `1bit-model-fetch.service` (Task 4), `packaging/iso/autoinstall.yaml.tmpl` (Task 5, already wired to the real `1bit-systems_*_amd64.deb` glob emitted by `packaging/Makefile`'s `package-deb` target — confirmed during design, see the spec's non-goals re: stale branding), `packaging/model-download.sh` (existing).
- Produces: `packaging/iso/build/1bit-monster-26.04-amd64.iso`, `packaging/iso/build/console-recovery-password.txt` — both consumed by Task 7's QEMU test.

- [ ] **Step 1: Build the `.deb` and confirm the emitted filename matches what `autoinstall.yaml.tmpl` expects**

Run:
```bash
ssh -i ~/.ssh/id_ed25519 bcloud@192.168.50.69 'cd ~/1bit-MONSTER/packaging && make package-deb && ls -la build/*.deb'
```
Expected: a file matching `build/1bit-systems_*_amd64.deb`. This is the exact glob pattern `autoinstall.yaml.tmpl`'s `late-commands` already `dpkg -i`s — if the Makefile's output ever changes name (e.g. if the stale-branding cleanup mentioned in the spec's non-goals happens later), `autoinstall.yaml.tmpl`'s glob and this step both need updating together.

- [ ] **Step 2: Write `build.sh` locally**:

```bash
#!/usr/bin/env bash
set -euo pipefail
# packaging/iso/build.sh — builds the 1bit.MONSTER appliance ISO.
# See docs/superpowers/specs/2026-08-16-ubuntu-iso-design.md.
#
# Usage: build.sh --ssh-key /path/to/id_ed25519.pub [--out DIR]

ISO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${ISO_DIR}/../.." && pwd)"
WORK="${ISO_DIR}/build"
UBUNTU_VER="26.04"
UBUNTU_ISO_NAME="ubuntu-${UBUNTU_VER}-live-server-amd64.iso"
UBUNTU_ISO_URL="https://releases.ubuntu.com/${UBUNTU_VER}/${UBUNTU_ISO_NAME}"

SSH_KEY_PATH=""
OUT_DIR="$WORK"
while [ $# -gt 0 ]; do
  case "$1" in
    --ssh-key) SSH_KEY_PATH="$2"; shift 2 ;;
    --out) OUT_DIR="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done
[ -z "$SSH_KEY_PATH" ] && { echo "usage: build.sh --ssh-key /path/to/key.pub [--out DIR]" >&2; exit 1; }
[ -f "$SSH_KEY_PATH" ] || { echo "ssh key not found: $SSH_KEY_PATH" >&2; exit 1; }

mkdir -p "$WORK" "$OUT_DIR"

echo "[1/7] Building the .deb..."
( cd "${REPO_ROOT}/packaging" && make package-deb )
DEB_PATH="$(ls "${REPO_ROOT}"/packaging/build/1bit-systems_*_amd64.deb | head -1)"
[ -f "$DEB_PATH" ] || { echo "FATAL: no .deb produced by 'make package-deb'" >&2; exit 1; }

echo "[2/7] Fetching base Ubuntu ${UBUNTU_VER} ISO..."
if [ ! -f "${WORK}/${UBUNTU_ISO_NAME}" ]; then
  curl -fL --progress-bar "${UBUNTU_ISO_URL}" -o "${WORK}/${UBUNTU_ISO_NAME}"
  curl -fL "https://releases.ubuntu.com/${UBUNTU_VER}/SHA256SUMS" -o "${WORK}/SHA256SUMS"
  ( cd "$WORK" && grep "${UBUNTU_ISO_NAME}\$" SHA256SUMS | sha256sum -c - )
fi

echo "[3/7] Fetching pinned driver payload..."
bash "${ISO_DIR}/fetch-payload.sh"

echo "[4/7] Extracting base ISO..."
EXTRACT="${WORK}/extract"
rm -rf "$EXTRACT"
mkdir -p "$EXTRACT"
xorriso -osirrox on -indev "${WORK}/${UBUNTU_ISO_NAME}" -extract / "$EXTRACT"
chmod -R u+w "$EXTRACT"

echo "[5/7] Building autoinstall seed + staging payload pool..."
PASSWORD="$(openssl rand -base64 24)"
PASSWORD_HASH="$(python3 -c "import crypt,sys; print(crypt.crypt(sys.argv[1], crypt.mksalt(crypt.METHOD_SHA512)))" "$PASSWORD")"
SSH_PUBKEY="$(cat "$SSH_KEY_PATH")"
sed \
  -e "s|__PASSWORD_HASH__|${PASSWORD_HASH}|" \
  -e "s|__SSH_PUBLIC_KEY__|${SSH_PUBKEY}|" \
  "${ISO_DIR}/autoinstall.yaml.tmpl" > "${EXTRACT}/autoinstall.yaml"
echo "$PASSWORD" > "${OUT_DIR}/console-recovery-password.txt"
chmod 600 "${OUT_DIR}/console-recovery-password.txt"
echo "  Console recovery password: ${OUT_DIR}/console-recovery-password.txt (NOT copied onto the ISO)"

mkdir -p "${EXTRACT}/pool"
cp "$DEB_PATH" "${EXTRACT}/pool/"
cp "${WORK}/payload/"*.deb "${EXTRACT}/pool/"
cp "${WORK}/payload/therock-"*.tar.gz "${EXTRACT}/pool/"
cp "${REPO_ROOT}/packaging/services/1bit-unified.service" "${EXTRACT}/pool/"
cp "${REPO_ROOT}/packaging/services/1bit-model-fetch.service" "${EXTRACT}/pool/"
cp "${REPO_ROOT}/packaging/model-download.sh" "${EXTRACT}/pool/"

echo "[6/7] Wiring autoinstall boot entry..."
GRUB_CFG="${EXTRACT}/boot/grub/grub.cfg"
[ -f "$GRUB_CFG" ] || { echo "FATAL: ${GRUB_CFG} not found — inspect ${EXTRACT}/boot/grub/ and fix this path" >&2; exit 1; }
if ! grep -q "autoinstall" "$GRUB_CFG"; then
  sed -i 's|linux\t/casper/vmlinuz|linux\t/casper/vmlinuz autoinstall ds=nocloud\\;s=/cdrom/|' "$GRUB_CFG"
fi

echo "[7/7] Repacking ISO..."
OUT_ISO="${OUT_DIR}/1bit-monster-${UBUNTU_VER}-amd64.iso"
xorriso -as mkisofs \
  -r -V "1bit.MONSTER ${UBUNTU_VER}" \
  -o "$OUT_ISO" \
  -J -joliet-long \
  -b boot/grub/i386-pc/eltorito.img \
  -c boot.catalog -no-emul-boot -boot-load-size 4 -boot-info-table \
  -eltorito-alt-boot -e EFI/boot/bootx64.efi -no-emul-boot \
  -isohybrid-gpt-basdat \
  "$EXTRACT"

echo "Built: ${OUT_ISO}"
```

**Note for the executor:** the exact `-b`/`-e` El Torito paths in the final `xorriso -as mkisofs` call are Ubuntu-ISO-layout-specific and may not match reality — Step 4's extraction is the way to find the real paths (`find "$EXTRACT" -iname "eltorito.img" -o -iname "bootx64.efi"`) if the repack step fails or produces a non-bootable image. Fix the paths here based on what's actually found before moving to Task 7; don't guess a second time; verify by inspection of the extracted tree.

- [ ] **Step 3: scp `build.sh` to the box and place it in the repo**

```bash
scp -i ~/.ssh/id_ed25519 <local>/build.sh bcloud@192.168.50.69:/tmp/build.sh
ssh -i ~/.ssh/id_ed25519 bcloud@192.168.50.69 '
cp /tmp/build.sh ~/1bit-MONSTER/packaging/iso/build.sh
chmod +x ~/1bit-MONSTER/packaging/iso/build.sh
'
```

- [ ] **Step 4: Syntax-check**

```bash
ssh -i ~/.ssh/id_ed25519 bcloud@192.168.50.69 'bash -n ~/1bit-MONSTER/packaging/iso/build.sh && echo SYNTAX_OK'
```

- [ ] **Step 5: Generate a disposable test SSH key and run the real build**

Run (this downloads a multi-GB ISO the first time — expect it to take a while depending on the box's link):
```bash
ssh -i ~/.ssh/id_ed25519 bcloud@192.168.50.69 '
ssh-keygen -t ed25519 -N "" -f /tmp/1bit-iso-test-key
cd ~/1bit-MONSTER && bash packaging/iso/build.sh --ssh-key /tmp/1bit-iso-test-key.pub
'
```
Expected: script exits 0, prints `Built: .../1bit-monster-26.04-amd64.iso`. If the El Torito boot-path guess from Step 2's note is wrong, fix `build.sh` per that note and re-run — this step isn't done until it exits 0 with a real ISO on disk.

- [ ] **Step 6: Commit**

```bash
ssh -i ~/.ssh/id_ed25519 bcloud@192.168.50.69 "cd ~/1bit-MONSTER && git add packaging/iso/build.sh && git commit -m 'feat(iso): add ISO build script'"
```
(If Step 1 required fixing the `.deb` filename reference in `autoinstall.yaml.tmpl`, include that fix in this commit or a preceding small one — don't leave it uncommitted.)

---

### Task 7: Write `packaging/iso/test-qemu.sh` and run the automated boot test

**Files:**
- Create: `packaging/iso/test-qemu.sh`

**Interfaces:**
- Consumes: the ISO from Task 6 Step 5, `/tmp/1bit-iso-test-key` (private half of the test key baked into that ISO).
- Produces: pass/fail verdict on the whole pipeline — this is the plan's real correctness gate for Tasks 3-6.

- [ ] **Step 1: Write the script locally**

```bash
#!/usr/bin/env bash
set -euo pipefail
# packaging/iso/test-qemu.sh — boots the built ISO in headless QEMU, waits
# for unattended autoinstall to finish, then verifies the appliance came up
# correctly over SSH.
#
# Usage: test-qemu.sh /path/to/1bit-monster-26.04-amd64.iso /path/to/test_key

ISO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISO="${1:?usage: test-qemu.sh /path/to/1bit-monster-26.04-amd64.iso /path/to/test_key}"
TEST_KEY="${2:?usage: test-qemu.sh /path/to/1bit-monster-26.04-amd64.iso /path/to/test_key}"
DISK="${ISO_DIR}/build/test-disk.qcow2"
PIDFILE="${ISO_DIR}/build/qemu.pid"
SSH_PORT=2222

rm -f "$DISK"
qemu-img create -f qcow2 "$DISK" 40G

echo "Booting installer — unattended autoinstall runs now, expect 10-20 min..."
# -boot once=d (not a persistent -boot d): the CD must only be the boot
# device for the FIRST boot. After autoinstall finishes, curtin reboots
# the guest — with a persistent -boot d that reboot would boot the ISO
# again instead of the freshly-installed disk, sending the VM back into
# the installer (or worse, re-triggering autoinstall against an
# already-partitioned disk) instead of ever reaching the installed
# target system this script is trying to test.
qemu-system-x86_64 \
  -m 8G -smp 4 -enable-kvm \
  -drive file="$DISK",if=virtio \
  -cdrom "$ISO" \
  -boot once=d \
  -netdev user,id=net0,hostfwd=tcp::${SSH_PORT}-:22 -device virtio-net,netdev=net0 \
  -display none -daemonize -pidfile "$PIDFILE"

cleanup() { [ -f "$PIDFILE" ] && kill "$(cat "$PIDFILE")" 2>/dev/null || true; }
trap cleanup EXIT

echo "Waiting for SSH (install + first boot)..."
UP=0
for i in $(seq 1 90); do
  if ssh -i "$TEST_KEY" -p "$SSH_PORT" -o StrictHostKeyChecking=no \
      -o UserKnownHostsFile=/dev/null -o ConnectTimeout=5 \
      monster@localhost true 2>/dev/null; then
    echo "SSH is up after ~$((i*20))s"
    UP=1
    break
  fi
  sleep 20
done
[ "$UP" -eq 1 ] || { echo "FAIL: SSH never came up after 30 minutes"; exit 1; }

RUN() { ssh -i "$TEST_KEY" -p "$SSH_PORT" -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/dev/null monster@localhost "$@"; }

FAIL=0
echo "-- engine installed --"
RUN "dpkg -l | grep -i 1bit" || { echo "FAIL: engine .deb not installed"; FAIL=1; }
echo "-- unified service active --"
RUN "systemctl is-active 1bit-unified.service" || { echo "FAIL: 1bit-unified.service not active"; FAIL=1; }
echo "-- GTT kernel params present --"
RUN "cat /proc/cmdline | grep -o 'ttm.pages_limit=[0-9]*'" || { echo "FAIL: GTT kernel params missing"; FAIL=1; }
echo "-- driver packages held --"
RUN "apt-mark showhold | grep -q mesa-vulkan-drivers" || { echo "FAIL: mesa-vulkan-drivers not held"; FAIL=1; }
RUN "apt-mark showhold | grep -q bolt" || { echo "FAIL: bolt not held"; FAIL=1; }
echo "-- Thunderbolt daemon active --"
RUN "systemctl is-active bolt.service" || { echo "FAIL: bolt.service not active"; FAIL=1; }
echo "-- API health --"
RUN "curl -sf localhost:8088/v1/health" || { echo "FAIL: /v1/health not responding"; FAIL=1; }

if [ "$FAIL" -eq 0 ]; then
  echo "PASS"
else
  echo "One or more checks FAILED — see output above"
  exit 1
fi
```

- [ ] **Step 2: scp it to the box and place it in the repo**

```bash
scp -i ~/.ssh/id_ed25519 <local>/test-qemu.sh bcloud@192.168.50.69:/tmp/test-qemu.sh
ssh -i ~/.ssh/id_ed25519 bcloud@192.168.50.69 '
cp /tmp/test-qemu.sh ~/1bit-MONSTER/packaging/iso/test-qemu.sh
chmod +x ~/1bit-MONSTER/packaging/iso/test-qemu.sh
'
```

- [ ] **Step 3: Syntax-check**

```bash
ssh -i ~/.ssh/id_ed25519 bcloud@192.168.50.69 'bash -n ~/1bit-MONSTER/packaging/iso/test-qemu.sh && echo SYNTAX_OK'
```

- [ ] **Step 4: Run the full boot test**

Run (this is the plan's longest step — real unattended install inside QEMU):
```bash
ssh -i ~/.ssh/id_ed25519 bcloud@192.168.50.69 '
cd ~/1bit-MONSTER
bash packaging/iso/test-qemu.sh packaging/iso/build/1bit-monster-26.04-amd64.iso /tmp/1bit-iso-test-key
'
```
Expected: final line `PASS`. If any check fails, the fix belongs in whichever task produced the broken piece (`autoinstall.yaml.tmpl` for install-time steps, the `.service` files for unit issues, `build.sh` for boot/repack issues) — go fix it there, re-run Task 6 Step 5 to rebuild the ISO, then re-run this step. Don't consider this task done until `PASS` prints from a build produced by the final, committed versions of every upstream file.

- [ ] **Step 5: Commit**

```bash
ssh -i ~/.ssh/id_ed25519 bcloud@192.168.50.69 "cd ~/1bit-MONSTER && git add packaging/iso/test-qemu.sh && git commit -m 'test(iso): add automated QEMU boot smoke test'"
```

---

### Task 8: Write `packaging/iso/README.md`, link it from the top-level README, document real-hardware validation

**Files:**
- Create: `packaging/iso/README.md`
- Modify: `README.md` (repo root)

**Interfaces:**
- Consumes: Task 1's spare-storage finding.
- Produces: nothing further downstream — this is the plan's terminal documentation task.

- [ ] **Step 1: Write `packaging/iso/README.md` locally**

```markdown
# 1bit.MONSTER appliance ISO

Builds a fully-unattended Ubuntu Server 26.04 installer ISO that boots into
a running 1bit.MONSTER OpenAI-compatible inference API. Design rationale
and scope: `../../docs/superpowers/specs/2026-08-16-ubuntu-iso-design.md`.

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
  `mesa-vulkan-drivers`/`libvulkan1`/`bolt` (Thunderbolt/USB4 — stock
  Ubuntu, frozen at the tested version, same rationale as Vulkan), pinned
  TheRock gfx1151 libraries, the `1bit-unified.service` and
  `1bit-model-fetch.service` units.
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

<one of the following two paragraphs, chosen based on Task 1's finding —
write whichever is actually true, don't leave both>

Spare storage was found at `<device path from Task 1>` on the reference
Strix Halo box. To validate on real hardware: `dd` the built ISO to a USB
drive, boot the box from it via JetKVM (see
`../../MEMORY.md`-referenced access notes — do not touch the box's
existing root disk), and install onto `<device path>`, not the live
system's disk.

<or, if none was found:>

No spare storage was available on the reference box as of the date in
`docs/superpowers/specs/2026-08-16-ubuntu-iso-design.md`'s recorded
finding. Real-hardware validation is deferred until a spare drive is
available; the QEMU boot test above is this project's automated
correctness gate in the meantime.
```

- [ ] **Step 2: scp it to the box and place it in the repo**

```bash
scp -i ~/.ssh/id_ed25519 <local>/iso-README.md bcloud@192.168.50.69:/tmp/iso-README.md
ssh -i ~/.ssh/id_ed25519 bcloud@192.168.50.69 'cp /tmp/iso-README.md ~/1bit-MONSTER/packaging/iso/README.md'
```

- [ ] **Step 3: Add a link from the top-level README**

In `README.md` at the repo root, find the line block containing the existing `**[Website]... · **[JARVIS]... · **[The story]...` link row (per the file read during design) and add one more link: `· **[Appliance ISO](packaging/iso/README.md)**`.

Run:
```bash
ssh -i ~/.ssh/id_ed25519 bcloud@192.168.50.69 'grep -n "\[JARVIS\]" ~/1bit-MONSTER/README.md'
```
Take the returned line, add the new link segment to it via `sed` or a direct edit, matching the existing markdown link style exactly.

- [ ] **Step 4: Verify**

```bash
ssh -i ~/.ssh/id_ed25519 bcloud@192.168.50.69 'cd ~/1bit-MONSTER && git diff README.md'
```
Expected: diff shows exactly one new link segment added, nothing else changed.

- [ ] **Step 5: Commit**

```bash
ssh -i ~/.ssh/id_ed25519 bcloud@192.168.50.69 "cd ~/1bit-MONSTER && git add packaging/iso/README.md README.md && git commit -m 'docs(iso): add ISO build/test/validation docs, link from top-level README'"
```
