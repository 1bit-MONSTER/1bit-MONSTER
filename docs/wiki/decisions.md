# Decisions Ledger — Through-Line, Security, and Expansion Policy

> Standing decisions for the project (engine → JARVIS, one binary, commodity
> AMD silicon). Written 2026-08-12, in the UPDATE 34 "burn" era. The point of
> this page: these questions were already answered — don't re-litigate them in
> future sessions.

## The through-line (why we burn branches)

Engine (NPU + GPU + CPU, one binary) → JARVIS (voice assistant, reference app).
Criterion for cutting: *anything that isn't the engine or the app that proves
it*. UPDATE 34 (`cbce9630`) deleted ~19k lines on this basis: SaaS, agent
stack, voice cloning, JARVIS v1 side-servers. The same criterion applies to
future ideas, including hardware. Four branches were evaluated and rejected in
one session — see [Rejected Branches](#rejected-branches).

## Language policy (the Mojo fold)

- **Runtime rule**: nothing that ships runs an interpreter. The engine binary
  (`build/1bit`) is C++23 for compute kernels; Mojo 1.0 (`mojo==1.0.0`, pinned,
  no pre-1.0 community packages) is the unified language for servers,
  converters, tooling, control planes.
- **Fold per-tool, not blanket "no new Python"**. A tool earns folding only if
  it (a) ships to users (runtime Python dependency removed), or (b) links the
  engine via `libonebit.so` (the C ABI seam) — a converter that *calls* C++
  deletes duplicate logic instead of reimplementing it.
- **Twins must be wrappers, not reimplementations.** The first folded twins
  (`tools/qwen3_to_onebp.mojo`, `tools/tokenizer_json_to_htok.mojo`) are
  standalone reimplementations with hand-rolled JSON (`tools/jsonx.mojo`).
  Accepted as proof-of-concept; the real milestone is a converter whose `.so`
  call does the work. Two sources of truth for a format is a maintenance trap.
- **Never fold dev-time reference/logit-checkers** (e.g. `zaya_logit_check_ref.py`)
  — they exist to compare against torch/HF references and must stay Python.
- **Ecosystem gaps are a cost, not a virtue**: no `std.net` (libc sockets), no
  regex (fixed-pattern parsers), no SIGTERM in `Process` (lazy-reap). Hand-rolled
  JSON/HTTP is only acceptable for fixed-shape payloads on one machine — e.g.
  the Adrenalin control-plane rewrite.

## Physical security model (cold boot and firmware tampering)

Threat model: **physical attacker with the device**. The answer is the
industry stack, nothing custom:

1. **AMD SME on** (`mem_encrypt=on`) — DRAM encrypted with a CPU-held key;
   cold-boot RAM dumps yield ciphertext. ⚠️ Verify SME against the raw-ioctl
   NPU DMA path (UPDATE 33 IOMMU-fault wedge precedent) — SME forces device
   DMA through IOMMU encryption. Test before trusting.
2. **UEFI Secure Boot + TPM2 measured boot** — stages hash into PCRs;
   secrets (disk key, firmware signing key) are PCR-sealed. Swapped firmware
   → PCR mismatch → key never unseals. This is the firmware-integrity answer.
3. **MCU keybox** — long-term keys (host-link auth, weight-signing verify key)
   live in the expansion MCU's protected OTP: readout protection locked, debug
   port fused off in production, signed boot on the MCU itself. The host never
   holds long-term keys, so host compromise or cold boot yields nothing to
   escalate with.
4. **No custom crypto, no anti-forensics layers** (no memory-scrub hacks, no
   suspend/reboot disabling). SME + measured boot + keybox is the whole answer.

## Home Assistant expansion (GPIO header — APPROVED direction)

A future hardware revision may add a GPIO expansion header (covered by a
rubber pad) for Home Assistant integration. Design constraints, decided:

- **IO co-processor pattern**: a "dumb" MCU between the header and the host.
- **The security is one decision: no radio.** Choose a no-wireless MCU
  (RP2040-class). If anyone proposes an ESP32 for "expandability", that ends
  the security story — the MCU's only link is a private UART/USB to the host,
  narrow fixed-schema protocol, fixed-key auth. Nothing to attack over the air.
- **Isolation is the safety**: opto/galvanic isolation + ESD + current limiting
  on every header pin. The pad is cosmetic; isolation is "no compromises".
- **v1 uses ESPHome, not custom firmware**: an ESP32-C3 with stock ESPHome on
  USB gives HA GPIO integration in a weekend and teaches us the protocol before
  we design the production daughterboard (signed boot, no radio).
- **Software first, hardware when the box exists**: JARVIS↔HA integration
  (MQTT / WebSocket / REST over localhost, from Mojo/C++ — keeps our side
  interpreter-free) needs no GPIO and is the actual killer app: voice → intent
  → home action.

## Weight security (signed weights — APPROVED direction)

Model weights are **signed blobs**: hash in the 1bp header, signature verified
by the engine at load, verification key held in the MCU keybox. Tampered weight
= refused at load. New model = new signed blob, zero silicon.

This is strictly better than etched/fused weights (see Rejected Branches):
tamper-evidence **and** free swapping — the converter ladder stays the product.

## Rejected branches

| Idea | Why rejected |
|---|---|
| **Industrial/PLC via GPIO** | PLCs speak fieldbuses (Modbus/Profinet/EtherCAT/OPC-UA), not GPIO. Certifications (IEC 61508 SIL, IEC 61131-3), liability, decade sales cycles. Not the through-line. |
| **Etched weights** (Etched-style ASIC) | Immutability ≠ security: no patches, weights extractable by anyone who buys the chip (die analysis). Kills model-swapping flexibility — the converter ladder IS the product. Signed blobs cover the real need. |
| **Custom ASIC / "alternative to Speedcore"** (Arteris/foundry-time notion) | $1–30M masks, years, architecture fixed in silicon. Arteris is NoC IP, not a foundry broker; foundry-time sharing is MPW shuttles (Efabless ChipIgnite etc.). Commodity AMD + engine + converter ladder already is the affordable alternative to custom silicon. |

Rejection test for any future idea: *does it run on commodity AMD silicon and
serve engine → JARVIS?* If not, it's a branch, and branches get burned.
