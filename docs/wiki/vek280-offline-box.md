# VEK280 Offline AI Box (Versal + ARM host, battery powered)

> Design research: a self-contained, offline AI inference box — AMD Versal
> VEK280 eval kit as the accelerator, a low-power ARM SBC as host/controller,
> USB4 external port, separate power rail, internal LiPo battery.

## Architecture

```
[ Laptop / tablet (TB4/5/USB4) ]          ← external, hot-plug, ~6-7 GB/s
        │ USB4 v2 cable
        ▼
[ USB4↔PCIe bridge board ]                ← ASM2464PDX today (40 Gbps, Gen4 x4);
        │                                   JHL9580 Barlow Ridge for 80 Gbps
        ▼
[ VEK280 — Versal AI Edge VE2802 ]        ← PCIe Gen4 x16 card edge (native)
        ▲ internal link: plain PCIe / OCuLink (Gen3/4 x4 is enough)
[ ARM SBC host ]                          ← Jetson Orin Nano Super (Gen4 x4)
```

Key decisions (from research, mid-2026):

- **USB4 can't be the internal link.** No low-power ARM SBC (Pi 5, RK3588,
  Jetson) has a USB4/TB host port — no silicon, no drivers. Keep ARM↔Versal
  on plain PCIe; USB4 is only the *external* face for docking a laptop.
- **VEK280 is already a Linux ARM device**: dual-core Cortex-A72 @1.7 GHz +
  dual-core R5F, runs Petalinux/Yocto. The separate SBC is optional — only
  needed for more control-plane CPU/IO or host/FPGA lifecycle separation.
- **No TB/USB link aggregation exists.** Lane bonding is inside a single
  link, not across cables. No standard or OS support for bonding two TB
  ports. (Only historical exception: Promise Pegasus dual-TB RAID arrays,
  vendor-specific, N/A here.)
- **80 Gbps vs 40 Gbps bridge:** USB4 v2 (80 Gbps) standalone bridge boards
  not on the market yet — ASMedia 80 Gbps parts not shipping; Intel JHL9580
  exists only inside TB5 NVMe enclosures (Acasis TB501 ~$184). Either build
  at 40 Gbps (ADT-Link UT4G-BK7) or harvest a TB5 enclosure's bridge board.

## Interconnect facts

| Link | Effective bandwidth | Notes |
|---|---|---|
| PCIe Gen4 x16 card edge (VEK280 native) | ~28 GB/s | It's a card; put it in a host |
| OCuLink (direct PCIe Gen4 x4) | ~6.6 GB/s | No hot-plug, no tunneling overhead |
| TB5 / USB4 v2 → PCIe bridge | ~6-7 GB/s | Hot-plug; 120 Gbps is display-only |
| TB4 / USB4 40Gbps → PCIe bridge | ~2.5-3 GB/s | Workable for control + results |
| 2× TB4 "aggregated" | — | Does not exist |

## BOM & costing (US street, mid-2026)

### A. Compute — ~$5,150

| Qty | Item | Unit $ |
|---|---|---|
| 1 | AMD VEK280 eval kit (VE2802; incl. cooler, PSU, license voucher) | ~4,900* |
| 1 | Jetson Orin Nano Super dev kit (67 TOPS, Gen4 x4, 7-25 W) | 249 |

\* AMD publishes no list price; ~$4.9k via Mouser/DigiKey — verify on quote.

### B. Interconnect — ~$270

| Qty | Item | Unit $ |
|---|---|---|
| 1 | ADT-Link UT4G-BK7 (USB4/TB4 → PCIe x4, ASM2464PD) | 128-168 |
| 1 | M.2 Key M → PCIe x16 open-slot riser (x4-wired) | 30-50 |
| 1 | PCIe x16 riser cable (x4) + bracket | 25-40 |
| 1 | USB4 certified cable (40 Gbps, or 80 Gbps) | 25-80 |

80 Gbps option: +~$185 (TB5 enclosure, harvest JHL9580 board).

### C. Power & battery — ~$225

| Qty | Item | Unit $ |
|---|---|---|
| 1 | 6S LiPo 22.2V 5000 mAh (NHX / Admiral) — 111 Wh | 100-110 |
| 1 | 6S BMS 40A w/ balance | 15-25 |
| 1 | USB-PD 100 W sink module | 15-25 |
| 1 | 22.2V→12V 20 A buck (VEK280 rail, ~75 W max) | 25-40 |
| 1 | 22.2V→5V 5 A buck (SBC + bridge rail) | 10-15 |
| 1 | Power-path board (ORing / load switch) | 10-15 |
| 1 | XT60 + fuse + switch + wiring | 20-30 |

### D. Enclosure & misc — ~$265

| Qty | Item | Unit $ |
|---|---|---|
| 1 | Aluminum chassis / extrusion frame (VEK280 is full-size ATX card; plan ~45×30×15 cm) | 80-150 |
| 2 | 120 mm chassis fans + grills | 20-30 |
| 1 | 1 TB NVMe SSD (Orin boot + datasets) | 70-90 |
| — | Standoffs, brackets, thermal pads, misc | 30-50 |

### Totals

| Scenario | Cost |
|---|---|
| Baseline (40 Gbps USB4, LiPo, Orin host) | ~$5,900 |
| + TB5 enclosure harvest for 80 Gbps | ~$6,100 |
| Budget swap: Pi 5 8GB (Gen3 x1 internal link) | −$170 |

VEK280 is ~83% of the bill. If this is a prototype before a custom carrier
board, sections B/C/D (~$900) are fully reusable — the carrier replaces the
$4.9k kit, not the supporting parts.

## Power & runtime notes

- USB-C connector runs **data-only** (no PD negotiation); system fed from a
  dedicated 12 V rail. Battery sits between charger and load.
- Budget: VEK280 20-75 W (workload-dependent) + SBC 5-15 W + bridge 5-10 W
  ≈ 30-100 W total.
- 111 Wh pack (6S 5000 mAh) ≈ 1-3 h at realistic loads, ~40 min at full tilt.
- Airline carry-on limit is 100 Wh — use the 4500 mAh variant if it must fly.
- Embedded vs external charger: internal PD sink keeps it self-contained;
  external balance charger (ToolkitRC M6 ~$60) is simpler but needs a wall unit.

## Roadmap (3 years)

> Draft — from the research session that produced this page.

1. **Phase 1 (now): VEK280-based offline box** — this page's build as the
   dev/prototype platform: Versal AI Edge runs inference (DPU / Vitis AI),
   ARM host controls, battery-powered. Proves the offline form factor,
   power budget, and USB4/PCIe plumbing with off-the-shelf parts.
2. **Phase 2: custom Versal carrier** — drop the $4.9k eval kit for a
   carrier board sized to the real power/IO budget; reuse B/C/D sections.
3. **Phase 3: Zyphra MoE on a standalone ASIC** — the model itself becomes
   the chip. Enabler: 1-bit quantization keeps weights small enough for
   on-chip SRAM, which removes the external-DRAM bandwidth wall that makes
   MoE inference memory-bound. ~3-year target. **Validated — see below.**

## Validation (2026-07)

Claim: 1-bit MoE weights in on-chip SRAM removes the DRAM bandwidth wall.

**Evidence:**

- Zyphra ships natively-1-bit models (Bonsai 27B, Apache 2.0): 1.125
  bits/weight binary ≈ 4 GB (phone-runnable), 1.58-bit ternary at ~95% of
  FP16 quality. Trained in 1-bit, not PTQ. Bonsai-1.7B/4B variants exist.
- MoE decode is memory-bound (multiple independent papers: expert routing
  fragments batches, decode is bandwidth-bound not compute-bound).
- No-DRAM SRAM inference is proven silicon: Groq LPU = 230 MB on-chip SRAM,
  ~80 TB/s, 25× H100 bandwidth, deterministic.

**The constraint — model class:**

| Model @ 1.125 bits/param | Footprint | Fits on-die in 2028? |
|---|---|---|
| Bonsai-1.7B | ~239 MB | Yes — ~50-65 mm² SRAM on N3-class (30-40 Mb/mm²), realistic on a ~200-300 mm² edge die |
| Bonsai-4B | ~563 MB | No — ~150-190 mm² SRAM, Groq-class not edge |
| BitNet 2.4B | ~300-470 MB | Marginal (1-bit) / no (1.58-bit) |
| Bonsai-27B | ~3.8 GB | No — needs DRAM or wafer-scale |

**Consequence:**

- ≤2B class at 1.125 bits/param: DRAM wall removed. 239 MB SRAM read per
  token at 2-4 TB/s → ~8-16k tok/s ceiling, zero DRAM. New regime.
- 27B class + LPDDR: still bandwidth-bound (~30-100 tok/s), ~8-10× better
  than FP16 — better, not a new regime.
- So the roadmap choice is: **quality (27B + DRAM) vs regime change
  (1.7B, DRAM-free, tok/s in the thousands)**. The Phase 3 ASIC only
  delivers its thesis at the 1.7B class.

## Status / next steps

- [ ] Decide: build at 40 Gbps (UT4G) or harvest TB5 enclosure for 80 Gbps
- [ ] Decide: Jetson Orin host vs no separate SBC (A72 may suffice)
- [ ] Quote VEK280 (Mouser/DigiKey)
- [ ] Custom carrier board vs eval kit for the production version
