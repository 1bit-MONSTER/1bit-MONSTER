# VEK385 Offline AI Box — Design (Rev 2)

> Revision 2 of the offline box design. Compute moves from the Versal AI Edge
> Gen 1 **VEK280** to the Gen 2 **VEK385** eval kit, and the box gains a
> first-class requirement: **create and sync datasets between devices, with
> LoRA fine-tuning** (§5). The VEK280 draft remains at
> [vek280-offline-box.md](vek280-offline-box.md) for the interconnect/power
> research that carries over unchanged.
>
> **Positioning: the VEK385 box is the future home of 1bit.MONSTER.** It is
> the canonical deployment target for the whole stack — engine, JARVIS, mesh,
> datasets, LoRA — not a peripheral. Laptops and phones become thin clients
> that dock (USB4/TB) or join (Mesh/LAN) a self-contained box that owns the
> data and the models. Dataset sync between devices is **behind the scenes**:
> a background daemon reconciles continuously; nobody presses "sync".
>
> **The pitch: full offline inference for everyone — no hardware upgrade.**
> Dock it to any laptop over Thunderbolt or MagSafe it to an iPhone: the
> machine you already own gains a self-powered 184-TOPS NPU running the
> entire 1bit.MONSTER stack, fully offline. **The stick is the upgrade — no
> new laptop, no new phone, no new desktop** (R-08).

## 0. Requirement summary

| ID | Requirement |
|----|-------------|
| R-01 | **Independent, battery-powered AI stick.** Self-contained and standalone — boots Linux, runs the full 1bit.MONSTER stack, and serves inference with no host attached. Three radios: Thunderbolt (dock/host), WiFi (mesh + sync), Bluetooth (phone control/presence). Internal battery keeps it alive undocked. |
| R-02 | **On-device dataset creation.** The box can capture, curate, and version datasets locally (JSONL), without a host. |
| R-03 | **Dataset sync between devices — behind the scenes.** Automatic, continuous, versioned sync of datasets and trained artifacts between the box, its host, and sibling boxes (LAN), plus phone via JARVIS mobile. Runs as a background daemon; no manual pull/push. |
| R-04 | **LoRA fine-tuning on synced datasets.** Train LoRA adapters against any synced dataset; adapters export to GGUF and run in the 1bit engine on-box. |
| R-05 | **The box is 1bit.MONSTER's home.** The full stack (engine, JARVIS, mesh, dataset hub, LoRA) runs on the box; laptop/phone are thin clients. |
| R-06 | **Battery-powered + standalone.** Internal pack, USB-C PD charging, always-on sync daemon and mesh node even when undocked; no wall power, no host required. |
| R-07 | **MagSafe — attaches to an iPhone.** The stick carries a MagSafe magnet ring so it can ride on the back of an iPhone; the phone (JARVIS mobile) is the thin-client UI over Bluetooth/WiFi. |
| R-08 | **No-hardware-upgrade value proposition.** Any existing laptop, desktop, tablet, or phone gets full offline AI inference by docking or attaching the stick — **no new laptop, no new phone, no new desktop**. The stick is the upgrade; the user keeps the hardware. |

## 1. Architecture

**Two builds, one target.** The eval kit is the dev platform only; the
shipping product is a **stick** on a custom carrier around the XC2VE3858 —
the eval kit (241 × 203 mm PCIe card) physically cannot be a stick. Both
builds share the software stack; only the carrier, power, and I/O differ.

```
[ Any laptop / tablet (TB4/5/USB4) ]      ← thin client, hot-plug, ~6-7 GB/s
        │ TB4/5 or USB4 cable
        ▼
[ 1bit.MONSTER STICK — custom carrier, XC2VE3858 ]
        │ 8× Cortex-A78AE (self-hosting Linux) · 144 AI Engine-ML v2 (184 TOPS)
        │ 20 GB LPDDR5X · M.2 NVMe (datasets + adapters) · integrated TB controller
        │ internal battery (USB-C PD) · WiFi 6E · Bluetooth LE · MagSafe ring
        │
        ├──[ sync daemon ]  watch + incremental reconcile, runs on battery
        ├──[ faces ]        TB/USB4 (dock) · WiFi 6E (mesh) · BT LE (phone)
        └──[ Mesh ]         /v1/mesh/*                      ← discovery + datasets API

  [ iPhone ]  ◂── MagSafe magnet ring ── [ STICK ]
     JARVIS mobile = thin-client UI over Bluetooth/WiFi
```

- **Build A — dev/prototype (Rev 2a):** VEK385 eval kit in a portable,
  battery-powered enclosure. Proves the stack, power budget, and USB4/PCIe
  plumbing with off-the-shelf parts. Not the product form factor.
- **Build B — the stick (Rev 2b, the goal):** custom carrier around the
  XC2VE3858 with an integrated Thunderbolt controller (JHL9580-class),
  USB-C PD charging, internal pack, WiFi 6E + Bluetooth LE, a MagSafe ring
  (rides on an iPhone back), and a stick enclosure with vapor-chamber
  cooling. This is the **independent AI stick with its battery** (R-01/R-06/
  R-07).

Key deltas vs the VEK280 draft:

- **VEK385 is self-hosting.** 8× Cortex-A78AE app cores run Linux directly
  (Versal AI Edge Gen 2 boots Petalinux/Yocto). The separate ARM SBC
  (Jetson Orin) is **no longer required** — the box is the host.
- **The box is the future home of 1bit.MONSTER.** R-05: the whole stack lives
  on the box; laptop/phone are thin clients. R-02/R-03 make local dataset
  capture + **behind-the-scenes** sync first-class; §5 is the design.
- **Form factor honest note:** the VEK385 *eval kit* is a full-size PCIe card
  (241 × 203 mm) — the dev platform only. The shipping **stick** is a custom
  carrier around the XC2VE3858 (Build B, §1): integrated TB controller,
  internal battery, roughly 2-3× the size of a big USB stick. §6 has both
  BOMs.
- **MagSafe honesty:** a 184-TOPS Versal + 20 GB LPDDR5X + battery cannot fit
  inside an Apple MagSafe battery-pack puck (≈ 64 × 64 × 11 mm). The stick is
  a **slab about the size of an iPhone-back MagSafe charger area**
  (~110 × 70 × 20-25 mm, ~250-400 g), held by a MagSafe ring for
  carry/handheld use. Sustained full-tilt AI on the phone's back is thermally
  bounded (see §7); the dock or a desk is where it runs hard.
- **Saves the hardware, not just the cloud bill.** R-08: users never buy a
  new machine — the stick upgrades whatever they own (laptop, desktop, or
  iPhone). This is the product's headline value proposition.

## 2. Platform — AMD VEK385 (EK-VEK385-G)

| Item | Spec |
|------|------|
| Adaptive SoC | Versal AI Edge Gen 2 **XC2VE3858**-2MSESSVA2112 |
| CPU | 8× Arm Cortex-A78AE application cores |
| Real-time | 10× Arm Cortex-R52 |
| GPU | 4-core Arm Mali-G78AE |
| AI / DSP | **144 AI Engine-ML v2 tiles (up to 184 INT8 TOPS)**, 2,064 DSPs |
| FPGA fabric | 1,188K system logic cells, 543,104 LUTs |
| Memory | 20 GB 160-bit LPDDR5X (5× 4 GB soldered) |
| Storage | M.2 Gen5 M-Key socket, 64 GB UFS, OSPI + microSD |
| PCIe | x8 edge connector — Gen5 x4 / Gen3/4 x8 (EP + RP support) |
| Networking | 1× QSFP28 (100G), 1× SFP28 (25G), 3× GbE RJ45 |
| USB | USB 3.2 Gen 2×1 Type-C (DP 1.4 alt), USB 3.0, 2× USB 2.0 |
| Video | HDMI 2.1 in + out, VCU (H.264/H.265), 3× ISP tiles |
| Power | 12 V via 2× ATX or 3-pin connector |
| Price / lead | **$15,995** (AMD store), ~10-week lead |

Sources: [AMD VEK385 product page](https://www.amd.com/en/products/adaptive-socs-and-fpgas/evaluation-boards/vek385.html),
[CNX Software announcement](https://www.cnx-software.com/2026/02/23/amd-vek385-versal-ai-edge-gen-2-fpga-evaluation-kit-plugs-directly-into-a-pcie-gen5-gen4-slot/).

> **Eval kit vs part:** the specs above describe the *kit* (EK-VEK385-G).
> The stick uses the **XC2VE3858** device on a custom carrier — same SoC,
> new power/thermal/IO design (§6 Build B).

## 3. What changed vs the VEK280 draft

| Aspect | VEK280 draft (Rev 1) | VEK385 (this doc) |
|--------|----------------------|-------------------|
| Compute | Versal AI Edge Gen 1 VE2802 (~$4,900) | Gen 2 XC2VE3858 — AI Engine-ML v2, 184 INT8 TOPS |
| Host | Separate ARM SBC (Jetson Orin Nano) | Self-hosted on 8× A78AE (SBC dropped, −$249, −power) |
| Memory | kit DDR | 20 GB LPDDR5X (5× 4 GB) — enough for 8-14B Q4/Q8 models resident |
| PCIe face | Gen4 x16 card edge | Gen5 x4 / Gen4 x8 edge — pairs with UT4G/TB5 bridges |
| LAN face | none specified | GbE + SFP28 25G + QSFP28 100G — dataset sync backbone |
| Datasets | n/a | **R-02/R-03: create + sync, LoRA (§5)** |
| Cost (compute) | ~$4,900 | ~$15,995 — the box is now ~86% compute |

## 4. Interconnect

| Link | Effective bandwidth | Notes |
|---|---|---|
| **Integrated TB4/5 (stick)** — JHL9580-class on the carrier | ~3-7 GB/s | Build B: no external bridge. TB5 ≈ 6-7 GB/s, TB4 ≈ 2.5-3 GB/s |
| PCIe Gen5 x4 (VEK385 edge, native) | ~6.5-7 GB/s | Inside the device; carrier routes to the TB controller |
| USB4/TB → PCIe bridge (Build A only) | ~2.5-7 GB/s | ADT-Link UT4G (40 Gbps) or TB5 enclosure harvest (80 Gbps) |
| WiFi 6E (stick) | 1-2 Gbps | box↔box / box↔phone mesh when undocked |
| Bluetooth 5.3 LE (stick) | ~2 Mbps / LE | control plane: pairing, discovery, JARVIS handoff, presence — not bulk data |
| 25/100 GbE (Build A only) | 3.1 / 12.5 GB/s | SFP28/QSFP28 on the dev box |

## 5. NEW — Dataset creation & sync between devices (LoRA)

This is the new requirement (R-02/R-03/R-04). It makes the box a
**self-contained dataset node**: capture → curate → version → sync → train.

### 5.1 Dataset creation (on-device, R-02)

- Canonical dataset format is **JSONL** (same input the existing training
  stack consumes: `tools/train/*.py` and `build/lora_train`).
- Capture sources, all local/offline-first:
  - **Conversation logs** — the box's own OpenAI-compatible server session
    logs (chat/completions history) → instruction pairs.
  - **JARVIS voice capture** — mic → VAD → STT transcripts (the existing
    voice pipeline) → text dataset entries; audio `.voice` packs retained
    for voice cloning.
  - **Corpus ingestion** — files/documents dropped on the box (USB, LAN,
    M.2 storage) chunked + normalized to JSONL.
  - **Host/phone push** — datasets written on a laptop or JARVIS mobile
    phone, synced to the box (§5.2).
- Every dataset carries a **manifest**: schema version, entry count, hash
  (content-addressed), created-by, license/source tags.
- Tooling: `1bit dataset` subcommand (create / validate / dedup / stats) in
  the single-binary engine; minimal curation UI on the box's web port.

### 5.2 Dataset sync between devices — behind the scenes (R-03)

**The sync runs itself.** A background **sync daemon** on every device
watches the local dataset store and reconciles continuously whenever a link
is present (dock USB4/TB, LAN, or phone-on-mesh). There is no "sync" button:
create or edit a dataset anywhere and it lands everywhere the mesh reaches.

- **Daemon model.** Local change → manifest update → notify peers
  (`/mesh/datasets/subscribe`) → incremental blob transfer → merge. The
  daemon is part of the 1bit binary (`1bit sync` subcommand), collides with
  nothing, survives reboots (systemd/user service on the box; in-process on
  the JARVIS mobile Flutter client).
- **Runs on battery.** The daemon is always on, docked or not: on the stick
  it runs from the internal pack over WiFi mesh, so datasets reconcile even
  when the stick is unplugged and riding in a bag (low-power duty cycling;
  full flush the moment any link appears). Bluetooth LE presence wakes the
  stick for a reconcile when a paired phone comes in range or MagSafe-snaps
  to the back.
- **Offline-first + bidirectional.** Every device keeps a full local copy;
  no device requires the cloud; when offline, the daemon queues changes and
  flushes on reconnect.
- Transports, in priority order:
  1. **Docked (host ↔ box):** USB4/TB link — the box's external face already
     exists; manifests + blobs ride it over the box's HTTP server. The
     daemon treats the dock as "link present" and reconciles.
  2. **LAN (box ↔ box, box ↔ phone):** GbE/SFP28/QSFP28 + the existing
     **Mesh** discovery (`docs/mesh-protocol.md`, mesh/1.0). Sibling boxes
     and JARVIS mobile (Flutter) join the same mesh; the daemon reconciles
     with every peer it can see.
  3. **Sneakernet fallback:** USB drive — export/import a dataset bundle for
     devices that are never on the same network.
- **Design: extend Mesh with a datasets API** — new endpoints under
  `/v1/mesh/datasets/*` (mesh/1.1):
  - `GET /mesh/datasets` — list manifests (id, schema, count, hash)
  - `GET /mesh/datasets/{id}/manifest` — fetch one manifest
  - `POST /mesh/datasets/{id}/sync` — reconcile with this peer (daemon call)
  - `POST /mesh/datasets/{id}/blobs` — transfer missing blobs (incremental)
  - `POST /mesh/datasets/subscribe` — receive change notifications
- **Versioning & conflicts — resolved automatically where possible.**
  Content-addressed blobs + append-only entries with tombstones; the daemon
  merges per-dataset on reconcile. Two devices editing the *same* entry is
  the only case surfaced to a human (keep-both / prefer-newer / manual) —
  everything else merges silently.
- **Security:** datasets are personal data — encrypt at rest on the box
  (LUKS on the M.2/SSD), TLS between peers; mesh handshake already carries
  identity (persistent UUID node cards).

### 5.3 LoRA fine-tuning (R-04)

LoRA is the training path for synced datasets. The stack already exists:

| Piece | Where | State |
|-------|-------|-------|
| `tools/train/` (Docker, Unsloth) | host-side SFT/RL/FFT + GGUF export | ships today |
| `build/lora_train` (pure C++23, ~400 KB) | zero-Python training: JSONL → tokenizer → packed batches → Q4NX in-place LoRA → AdamW → checkpoint | hackathon track-1 binary |
| Q4NX in-place fine-tune | LoRA A/B in FP32, base stays Q4NX — 8B in ~8 GB, no merge drift | shipped |
| GGUF export | checkpoint → `model.q4_k_m.gguf` → engine loads it | shipped |
| Video/image LoRA | `tools/video-lora` (SD1.5/AnimateDiff, Vulkan) | shipped |

**Proposed on-box flow:**

1. Dataset lands on the box (created locally or synced — §5.1/5.2).
2. Fine-tune: host GPU for big jobs (`tools/train` on Strix Halo / MI300X);
   on-box training is a roadmap item — the A78AE cores + AI Engine-ML v2
   GEMMs (Vitis AI flow) are the target, `lora_train` currently targets
   ROCm HIP (gfx1151), not ARM.
3. Adapter exports to **GGUF**, syncs back through the same §5.2 channels,
   and runs in the 1bit engine (any device in the mesh).
4. The 1BP/GGUF artifacts and the datasets use the same content-addressed
   manifest scheme — one artifact store, two kinds of objects.

### 5.4 What exists today vs what's new

| Capability | Exists today | New work |
|------------|--------------|----------|
| JSONL dataset handling | `tools/train`, `lora_train` | `1bit dataset` CLI (create/validate/dedup) |
| Mesh discovery + peer API | mesh/1.0 (`/mesh/me, peers, ask, answer`) | `/v1/mesh/datasets/*` (mesh/1.1) |
| USB4/TB external link | Rev 1 design (UT4G / TB5) | — |
| LoRA training | `tools/train`, `lora_train`, video-lora | on-box port (A78AE/AI Engine) |
| Artifact sync | manual (scp/rsync style) | background `1bit sync` daemon + manifest + reconcile/subscribe |

## 6. BOM & costing (US street, mid-2026)

### Build A — dev/prototype (VEK385 eval kit + portable battery box)

| Item | Unit $ |
|---|---|
| AMD VEK385 eval kit (EK-VEK385-G, incl. PSU, Vivado voucher) | 15,995* |
| USB4/TB4 → PCIe x4 bridge (ADT-Link UT4G-BK7, ASM2464PD) | 128-168 |
| PCIe riser (x4) + bracket + USB4 cable (40 / 80 Gbps) | 55-120 |
| 2 TB NVMe SSD (M.2 Gen5 — datasets + adapters) | 130-200 |
| Portable enclosure (fits 241 × 203 mm card) + fans | 100-180 |
| Battery + power-path (§7) | 180-280 |
| **Build A total** | **~$16,600-16,950** |

\* AMD store price, ~10-week lead. Verify on quote.

### Build B — the stick (custom carrier, production target)

Per-unit estimates; **NRE excluded** (board design, bring-up, thermal +
battery validation, tooling — a 4-6 month engineering program, low six
figures). Volume pricing on the XC2VE3858 TBD — the range below is the
feasibility envelope until quoted.

| Item | Unit $ (est.) |
|---|---|
| XC2VE3858 BGA package (volume pricing TBD) | 2,000-4,000 |
| Carrier PCB (12-16 layer, HDI, + assembly) | 400-800 |
| 20 GB LPDDR5X (5× 4 GB) | 120-180 |
| Integrated TB4/5 controller (JHL9580 / ASM2464PDX) + retimers | 40-90 |
| USB-C PD 100 W sink + power-path (ORing, chargers, BMS) | 30-60 |
| Battery pack (3S2P 21700 ≈ 100 Wh, BMS) | 60-90 |
| NVMe SSD 2 TB (M.2) | 130-200 |
| WiFi 6E module + antennas | 25-50 |
| Stick enclosure (aluminum) + vapor-chamber/heat-pipe + fan | 60-120 |
| **Build B unit total (est.)** | **~$2,900-5,600** |

## 7. Power & runtime — battery stick

- **Load envelope.** XC2VE3858 SoC: ~10-15 W idle / A78AE-only, ~30-50 W
  light AI (a few AI-Engine tiles), ~60-90 W sustained multi-tile inference.
  The stick's thermal + battery design targets the light-AI envelope for
  sustained use; full tilt is burst (thermal throttle).
- **Battery.** 3S2P 21700 (≈ 100 Wh, 11.1 V nominal) is the baseline — at the
  100 Wh airline carry-on limit: ~1.5-2 h light AI, ~6 h idle. Slimmer 4S1P
  (≈ 70 Wh) option for a thinner stick: ~1-1.5 h light AI. USB-C PD 100 W
  charges over the same connector family as TB; power-path ORing runs the
  load from wall power when docked, battery otherwise.
- **Thermal.** 40-90 W in a stick means a vapor chamber + heat-pipe to an
  aluminum shell plus a small fan; sustained budget realistically 30-50 W
  without throttling. Operating 0-45 °C ambient.
- **Standalone behavior.** Undocked on battery: the sync daemon duty-cycles
  (wake for mesh reconcile), engine/JARVIS stay up for WiFi clients. Docking
  adds wall power + TB link; the daemon flushes queued changes immediately.
- **Safety.** BMS (over/under-voltage, temp cut), fuse, charging limits;
  stay at or under 100 Wh to fly.

## 8. Roadmap

1. **Phase 1 (now): dev box on the eval kit (Build A)** — VEK385 eval kit,
   self-hosted Linux, USB4 face, datasets hub, battery box. Proves the stack
   and the power budget.
2. **Phase 1.5: behind-the-scenes sync** — `1bit sync` daemon + mesh/1.1
   `/v1/mesh/datasets/*` + artifact manifest store; reconcile over USB4,
   LAN, and WiFi, zero user action.
3. **Phase 2: on-box LoRA** — port `lora_train` to A78AE (NEON) /
   AI Engine-ML v2 GEMMs; dataset → adapter → GGUF entirely on the device.
4. **Phase 3: the stick (Build B)** — custom carrier around the XC2VE3858:
   integrated TB controller, internal battery + USB-C PD, WiFi 6E, stick
   enclosure + vapor-chamber cooling. The independent Thunderbolt stick is
   the shipping product (R-01/R-06). NRE program: board design → bring-up →
   thermal/battery validation.
5. **Phase 4: Zyphra MoE on a standalone ASIC** — 1-bit quantization keeps
   weights small enough for on-chip SRAM (see Rev 1 validation: Bonsai-1.7B
   ≈ 239 MB on-die; DRAM-free regime at the ≤2B class).

## 9. Status / next steps

**Immediate (Build A — dev platform):**
- [ ] Quote VEK385 (AMD store / Mouser / DigiKey) — confirm lead time
- [ ] Decide: USB4 bridge at 40 Gbps (UT4G) vs 80 Gbps (TB5 JHL9580)
- [ ] Confirm self-hosted Linux on A78AE (Petalinux/Yocto) — no separate SBC
- [ ] Spec mesh/1.1 `/v1/mesh/datasets/*` endpoints + manifest schema
- [ ] Land `1bit dataset` CLI (create / validate / dedup / stats)
- [ ] Land `1bit sync` daemon (watch → notify → reconcile → merge)
- [ ] Feasibility: `lora_train` on A78AE / AI Engine-ML v2 (GEMM path)
- [ ] LUKS encryption + TLS for dataset-at-rest/in-transit

**Build B — stick engineering program:**
- [ ] Carrier board spec around XC2VE3858 (power rails, LPDDR5X, TB controller)
- [ ] Integrated TB4/5 controller choice: Intel JHL9580 vs ASM2464PDX
- [ ] Battery + USB-C PD power-path design (ORing, charging, BMS)
- [ ] Thermal: vapor chamber + heat-pipe + fan CFD at 30-50 W sustained
- [ ] Stick mechanical: enclosure, cell layout, antenna placement
- [ ] NRE estimate + volume unit-cost model (§6 Build B)
