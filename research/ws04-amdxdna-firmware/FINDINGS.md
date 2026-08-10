# WS-04 FINDINGS — AMD XDNA NPU firmware + driver reverse engineering

Full technical record. Companion analysis pages in the LLM wiki ([[amd-npu-firmware-and-driver-complete-re-map]]).

## 1. Format (verified)

```
+0x000  AMD PSP key-store header (0x100 bytes), magic "$PS1" @0x10
        0x14 body_size | 0x38 certifying_id (16B key fingerprint) | 0x58 format word
+0x100  body = Xtensa firmware (plaintext)
+tail   RSA signature (0x100 = RSA-1024: AIE2 family; 0x200 = RSA-2048: AIE4 family)
```

Exact check: `file_size = body_size + 0x100 + sig_len` holds for NPU1, NPU4, NPU3 npu.dev, NPU3 cert. Key IDs: AIE2 `db74e212 39479e36 17b05cad ff21485e`, AIE4 `2849f833 504f70c2 96f45faf c8af972a`. `npu.dev.sbin` is md5-identical to decompressed `npu.sbin.zst`. Firmware loaded by PSP via VALIDATE→START_COPY_FW (driver `aie_psp.c`) — signature check, no decryption.

## 2. ISA (verified)

**Xtensa LE** + custom **Tensilica FFT/DSP TIE extension**:
- VMULAS families (vector multiply-accumulate-shift, s8/s16/u8/u16 × qacc/accx accumulators × ld.ip/ld.xp) — dominant op
- FFT ops: `fft.ams` (radix-2 add-multiply-subtract), r2bf, bitrev, cmul
- 128-bit Q-accumulator loads (`ld.qacc_h/l.128.ip`), 64/128-bit vector loads/stores (`ldf`, `stf`, `vld`, `ld.128.usar`)
- Encodings shared with ESP32 — capstone 6's Xtensa decoder IS the Espressif fork; match tables public (`espressif/binutils-gdb/opcodes/esp/match_opcode_xespv2p1.c`, 360 `esp.*` ops). Semantics: ESP32-S3 TRM.
- **~30–45% of `ee.*` decodes are 4-byte literal-pool data** following l32r/call/branch (Xtensa pools); mid-stream ~55–70% are real. NPU3: ~6.2k real mid-stream ops, 104 distinct.
- Functions: NPU4 = 964, NPU3 = 1,319 ENTRY prologues. Windowed (call8/call4) + call0 mixed.

## 3. MERT dispatch

**AIE4 (NPU3)** — single 156-entry pointer table @0x2D140, indexed by driver msgid:

| msgid | handler | driver op |
|---|---|---|
| 0x02 | 0x114DC | CREATE_CONTEXT |
| 0x03 | 0x23B6C | DESTROY_CONTEXT |
| 0x04 | 0x23D24 | GET_TELEMETRY |
| 0x07 | 0x23C74 | SYNC_BO |
| 0x0C | 0x23CCC | EXEC_BUFFER_CF |
| 0x0D | 0x1D894 | QUERY_COL_STATUS |
| 0x0E | 0x2598C | QUERY_AIE_TILE |
| 0x10 | 0x1F674 | EXEC_DPU |
| 0x11 | 0x259C0 | CONFIG_CU |
| 0x12 | 0x17D94 | CHAIN_EXEC_CF |
| 0x18 | 0x17F78 | CHAIN_EXEC_NPU |

(0x13/0x14/0x15/0x1E/0x1F → 0x25884 = not-supported stub.) All verified as Xtensa ENTRY functions.

**AIE2 (NPU4)** — compare-chain @0x9D84 (per-category dispatch, not a table): `beq a3,a9` (l32r-loaded case), `beqi a3,0xc` = EXEC_BUFFER_CF, `beqi a3,8`, `beqi a3,0xa`, `bnei a3,0x10` = EXEC_DPU → per-case call8 handlers (EXEC_DPU → 0xFAB0). Multiple dispatch functions by category — explains absence of a single table. AIE2 addressing: mixed bases (0x100xxxxx code, 0x200xxxxx data-seg, raw offsets in handler tables).

## 4. Control-plane / mailbox

NPU4 block @0x2DF0–0x2E10 = firmware's mailbox-info struct, mirrors driver `mgmt_mbox_chann_info` byte-for-byte: register pairs 0x030DC000/4 + 0x030DD000/4 (DLDO/ONO power regs per firmware strings), `_NPU` magic 0x55504E5F @0x2E0C (driver MGMT_MBOX_MAGIC), ring values 0x26000/0x26800. Mailbox/SRAM ring buffers 0x030A0000–0x030BF000 (page-aligned; = driver NPU1 SRAM X2I_MAILBOX_0 0x30A0000).

## 5. Command architecture (from function reads)

- Handlers are thin wrappers: `entry; l32r global(0x4598); load windowed args; l32i obj[0x14c]; add.n a2,a5,a12; j shared_executor` (tail-call).
- FUN_0002BF54 = stride-4 table lookup ("find object by ID").
- Context struct: +8 = ctx pointer, +0x80 = per-context SP back-ref (set at create, zeroed on exit; read at 50+ sites).
- CREATE_CONTEXT (0x114DC): records caller SP, 4 helper calls, error returns -16/-22.
- DSP kernels (0x77E0/0x7840/0x78A0 — 60-byte-stride trio): stack-aligned prologue, Xtensa hardware `loop`, body = vmulas + loads + helpers — FFT-stage structure.

## 6. Tooling

- Ghidra 12.1.2 + Temurin JDK 24 (`JAVA_HOME_OVERRIDE`), project with both firmwares (base 0).
- capstone 6.0.0a10 venv (`CS_ARCH_XTENSA`).
- `tools/` scripts: pool-aware walker, dispatch finder, full-analysis pipeline.
- Ghidra gotchas: OSGi error masks javac errors (compile manually with `javac -cp $(find Ghidra -name '*.jar')`); Listing API renamed (`getDefinedData(true)` iterator); fresh script dir per script.

## 7. Corrected conclusions (methodology note)

1. "pure Thumb-1 / Cortex-M" → wrong; Xtensa.
2. "no CPU code in blob" → wrong; Xtensa (llvm-mc's xtensa lacks ENTRY 0x36; sampling missed code regions).
3. "ee.* all real DSP ops" → partially wrong; ~40% literal-pool data (found by reading actual functions).
4. "string xrefs prove code" → coincidences against real pointer tables.
Lesson: verify ISA with strict decoders + real function reads (pool-aware walker) before pattern claims; capstone's Thumb-1 decodes nearly anything.
