# WS-04 — AMD XDNA NPU Firmware RE (amdxdna driver + npu.sbin)

**Status:** ✅ complete (2026-08-09) — full map of driver + both firmware families
**Owner:** npu
**Inputs:** linux 7.2-rc6 `drivers/accel/amdxdna/` (in-tree), `/lib/firmware/amdnpu/*.sbin` (linux-firmware), PSPTool, SMU RE community repos

## Goal

"Decrypt the NPU firmware" → established what the firmware actually is, its ISA, its structure, and how it maps to the in-tree driver. Original premise (encryption) disproven; complete architecture documented.

## Key results (see FINDINGS.md for full detail)

1. **NOT encrypted.** `amdnpu/*.sbin` are AMD PSP `$PS1` key-store containers: 0x100 header (magic @0x10, body_size @0x14, certifying_id @0x38) + plaintext body + RSA signature (0x100 AIE2 / 0x200 AIE4). Exact arithmetic: `file = 0x100 + body + sig`.
2. **ISA: Xtensa LE** (AMD SMU-family core) with a **Tensilica FFT/DSP TIE extension** — vector MAC (VMULAS), FFT radix-2 butterfly (AMS/R2BF), 128-bit Q-accumulator, 64/128-bit vector loads. Encodings shared with ESP32 (documented in ESP32-S3 TRM + Espressif binutils match tables). ~55–70% of `ee.*` decodes are real ops; ~30–45% are literal-pool data.
3. **Functions**: NPU4 (AIE2) = 964, NPU3 (AIE4) = 1,319 (ENTRY prologues). Mixed windowed (call8/call4) + call0 ABI.
4. **MERT dispatch decoded for both families**:
   - AIE4: single 156-entry pointer table @0x2D140 indexed by driver msgid (0x2→CREATE_CONTEXT, 0xC→EXEC_BUFFER_CF, 0x10→EXEC_DPU, 0x18→CHAIN_EXEC_NPU, … — verified against `aie2_msg_priv.h`)
   - AIE2: compare-chain function @0x9D84 (`beqi a3,0xc` = EXEC_BUFFER_CF, `bnei a3,0x10` = EXEC_DPU) — per-category dispatch, no single table
5. **Control-plane**: mailbox-info block @0x2DF0–0x2E10 mirrors the driver's `mgmt_mbox_chann_info` byte-for-byte (`_NPU` magic 0x55504E5F, register pairs 0x030DC000/0x030DD000, ring values).
6. **Command architecture**: handlers are thin wrappers (load global @0x4598 → load windowed args → tail-call shared executor). Context struct: +8 = ctx pointer, +0x80 = per-context SP field. DSP kernels = FFT-stage loops with Xtensa hardware `loop` + vector MAC.

## Reproduction

- Firmware: decompress `npu.sbin.*.zst`, carve body at 0x100 (skip sig tail: 0x100 AIE2 / 0x200 AIE4)
- Tools in `tools/` (capstone 6.0.0a10, `pip install capstone==6.0.0a10`):
  - `xtensa_walker.py` — pool-aware disassembler (skips literal pools; reads functions cleanly)
  - `find_dispatch.py` — locate MERT dispatch (pointer tables + compare-chains)
  - `analyze_firmware.py` — full pipeline: carve, entropy map, ISA check, entry/function inventory
- Ghidra 12.1.2 + Temurin JDK 24 (Ghidra 12 rejects 17/21/25): import carved body as Xtensa:LE:32:default. Gotchas: new scripts fail OSGi in used dirs (fresh dir per script); OSGi error masks javac errors (compile manually); Listing API renamed (`getDefinedData(true)`).

## Remaining

- AIE2 per-msgid mapping for context-mgmt ops (GUI task)
- Deep per-function RE (mechanical; tooling in place)

## Validation

Honesty tags: every major claim was independently verified (entry-prologue checks, exact size arithmetic, driver-source cross-references). Four intermediate conclusions were corrected during the session (ISA, "no code", ee.* real-vs-data) — the FINDINGS.md reflects the final verified state.
