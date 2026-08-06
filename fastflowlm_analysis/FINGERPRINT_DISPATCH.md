# Fingerprint Dispatch — The VAIP/ONNX-EP Kernel Selection Formula

Reverse-engineered from `onnxruntime_vitisai_ep.dll` (Windows, EP 1.8.60) and
`libonnxruntime_vitisai_ep.so` (Linux, ORT 1.24.3). Same code both platforms.
This is the dispatch mechanism of the **VAIP-generation** EP (the ONNX Runtime
Execution Provider path) — complementary to the FLM direct-load path in
`FLM_SECRETS.md`.

## The Formula

```
config_key = std::_Hash_bytes("target", 6, 0xc70f6907)  XOR  dpu_fingerprint
           = 0x49c07fb326fecfb2                          XOR  PassDpuParamProto.dpu_fingerprint
```

- `std::_Hash_bytes` = libstdc++ MurmurHash64A (m = 0xc6a4a7935bd1e995, r = 47)
- `"target"` = static 6-byte constant (string literal at VA 0x1f6819e in the .so)
- `seed = 0xc70f6907` (the libstdc++ default seed)
- `dpu_fingerprint` = field 4 of `vaip_core.PassDpuParamProto` (C++ offset 0x88)
- Result formatted as a decimal string and used as the lookup key into the
  driver's `config_xclbin.txt` (fingerprint → xclbin filename).

### Verification (disassembly, `vaip_core::get_xcompiler_fingerprint` @ 0xf3efa0)

```asm
lea  0x102906b(%rip), %rdi      # "target"          → 0x1f6819e
mov  $0x6, %esi                 # len = 6
mov  $0xc70f6907, %edx          # seed
call _ZSt11_Hash_bytesPKvmm@plt # MurmurHash64A
xor  0x88(%r12), %rax           # ^= dpu_fingerprint (PassDpuParamProto)
```

### Key structure (verified against all 28 entries of config_xclbin.txt)

```
key = (version_marker << 32) | low32
  gen3 (v2.5.0.0 toolchain):   marker 0x08000205  →  dpu_fingerprint hi32 = 0x41c07db6
  gen4 (v3.5.0.0 toolchain):   marker 0x0a000205  →  dpu_fingerprint hi32 = 0x43c07db6
```

The markers are exactly `Hash("target")_hi32 ^ marker` — i.e. the version
generation is baked into `dpu_fingerprint`'s high 32 bits (values 24.06f /
384.98f as f32, 2^4 apart). The low 32 bits carry the per-subgraph hash.

## The xclbin side

`vaip_level1_dpu::get_xclbin_fingerprint` (@ 0xf55190) is **not a hash** — it
is a parser: it validates the xclbin magic, extracts the AIE_PARTITION section,
and **reads** the stored `inference_fingerprint` field. The runtime then
compares it against the computed subgraph key
("Fingerprint of xclbin does not match subgraph's fingerprint").

```
42/42 fingerprints in config_xclbin.txt resolve against the xclbins'
AIE_PARTITION.inference_fingerprint (extracted from the driver store).
```

### What this means for reuse

- **Lookup is fully open**: any kernel's dispatch key is readable from its own
  AIE_PARTITION section — a custom runtime can select any of the 28+ shipped
  kernels without the compiler.
- **The compiler-side 32-bit subgraph hash** (inside `dpu_fingerprint`) is
  computed by the VAI export toolchain, which is not shipped in the runtime
  binaries — it remains the one closed piece.

## The embedded kernel inventory (Windows EP 1.8.60)

`onnxruntime_vitisai_ep.dll` embeds **46 kernel PDI blobs** (~6 MB) — no
external kernel loading:

| Set | Blobs | Fingerprint | Notes |
|-----|-------|-------------|-------|
| FP kernels | 8 × 2 | unique per kernel | DPU PDI + companion PDI (same fp in-body) |
| Shape variants | 28 | shared: 576318747933928537 | 2 sets × 14 (`DPU_PDI_0..13`, ids 0xc0-0xcd); per-shape recompiled microcode (~65 KB / ~250 KB) |
| Extra pair | 2 | 576462972803149505 | 4x4_2352 family |

Kernel identity: IP `"vadd:vadd_1"` (AP_CTRL_CHAIN @ 0x80000) + PS kernel
`"DPU_1x4:IPUV1CNN"` (kernel id 0x100). Context/preemption kernel id 0x104.
Preemption xclbins: PSO → `AMD_AIE2P_2x4x1_Overlay`, PSV → `AMD_AIE2P_4x4_Overlay`.
Next-gen (Linux EP 1.24.3 strings): `AMD_AIE2P_4x8_CMC_Overlay`,
`AMD_AIE4A/4B_4x3_CMC_Overlay` (CMC = context management controller).

### PDI blob layout (all 46)

```
+0x00  dd000000 44332211 88776655 ccbbaa99   # PDI magic
+0x30  "IDPP"                                 # partition format marker
+0xd4  u32 ×3 equal component sizes           # paired CDO/ELF parts ≈ blob/4
       "CDO" parts: {reg, count, value} triples → AIE tile registers
       (0x60402300 pattern = col/row/module encoding)
```

## Driver contract (shared by both OSes — same firmware)

Mailbox opcodes (identical on Windows `ipustack.sys` and Linux `amdxdna`):

```
0x10 EXEC_DPU            0x11 CONFIG_CU
0x12 CHAIN_EXEC_BUFFER_CF 0x13 CHAIN_EXEC_DPU   ← the one submission call
0x18 CHAIN_EXEC_NPU (preempt/ELF)
0x101-0x10C: SUSPEND, RESUME, ASSIGN_MGMT_PASID, INVOKE_SELF_TEST,
             MAP_HOST_BUFFER, GET_FIRMWARE_VERSION, SET/GET_RUNTIME_CONFIG,
             REGISTER_ASYNC_EVENT_MSG
```

Submission path: cmd BO (`ERT_START_NPU` = opcode 20: `{u64 inst_buf,
u32 inst_size, u32 prop_count, u32 prop_args[]}`) → driver fills
`cmd_chain_slot_dpu {inst_buf_addr, inst_size, inst_prop_cnt, cu_idx,
arg_cnt, args[≤34]}` → ONE `MSG_OP_CHAIN_EXEC_DPU` → firmware walks chain.

The single heap: firmware-defined `AIE2_DEVM_BASE 0x4000000 / SZ_64M`
(`aie2_pci.h`) = the xclbin HOST region. One heap, one chain submit.

## Live verification (Strix Halo, NPU firmware 1.0.0.196)

- `xrt-smi validate` 11/11: gemm **51.3 TOPS**, latency 74.7 µs, cmd-chain
  27.6 µs, df-bw 67.7 GB/s per shim DMA, TCT 776K/s (all columns),
  array reconfig 3.5 ms (PDI load)
- FLM Qwen3-0.6B: 89.4 tok/s decode; aie-partitions report shows HW contexts,
  live submission counters (305 → 1,140 over two prompts), migrations 0→5,
  preemption suspensions observed; 4303 MB heap usage, 1376 KB instruction BOs
- Windows driver version gate: FLM requires kipudrv ≥ 32.0.203.304 (silent
  exit otherwise); 32.0.20101.3760 = current. FLM needs `xrt_coreutil.dll`
  from the CURRENT driver store (entry-point mismatch otherwise).
- xrt-smi `telemetry` report exists only in the older (.251) xrt-smi binary.

## Source of evidence

- Full session notes with all 18 sections, extraction scripts and raw
  artifacts archived alongside the 1bit-systems work (see repo PR description
  for the archive location).
- The 46 kernel blobs and 48 xclbin section dumps were extracted with
  `xclbinutil` (Vitis 2026.1) from the Windows driver store + EP DLL.
