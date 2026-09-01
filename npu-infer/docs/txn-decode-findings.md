# Runtime layer-TXN weight-BD decode (#2006/#2015) — findings (2026-09-01)

Decoder tool: `npu-infer/tools/decode_txn.cpp`. It generates the FastFlowLM
runtime's ACTUAL control code through the same generators the runtime links
against (`libqwen3_npu.so` / `libdequant.so` / `libgemm.so`) and decodes each
TXN into structured JSON (full command decode + BD -> (arg_idx, arg_offset)
patch table + DMA direction). Decode JSONs: `npu-infer/decodes/qwen3-0.6b/`.

## TXN binary format (source of truth: mlir-aie include/aie/Runtime/TxnEncoding.h)

- Header (4 words): `w0 = (rows<<24)|(devGen<<16)|(minor<<8)|major`,
  `w1 = (memTileRows<<8)|cols`, `w2 = op_count`, `w3 = total_bytes`.
  For NPU2: w0 = 0x06040100, w1 = 0x00000108.
- WRITE        op 0x00, 6 words: [0][0][addr][0][val][0x18]
- BLOCKWRITE   op 0x01, 12 words (shim-DMA BD): addr word = (row<<20)|(col<<25)|0x1d000;
  w3 op_size=48; w4 buffer_length; w5 buffer_offset(patched); w6 packet;
  w7 D0 (size<<20 | stride-1); w8 D1 | 0xc0000000; w9 D2 | cache<<24;
  w10 iter (size-1<<20 | stride-1); w11 next_bd<<27 | valid<<25 | locks.
  VARIANT: op 0x01 with w1==24 = 6-word MEMTILE BD-register write
  (e.g. memtile BD init, addr 0x06202400 family, value 1).
- MASKWRITE    op 0x03, 7 words: [3][0][addr][0][val][mask][0x1c]
- TCT (sync)   op 0x80, 4 words: [0x80][0x10][(dir)|(row<<8)|(col<<16)][(chan<<24)]
- DDR_PATCH    op 0x81, 12 words: [0x81][0x30][0][0][0][0][addr 0x1d004][0][arg_idx][0][arg_offset][0]
- DMA direction comes from the paired queue WRITE: reg 0x1d204/0x1d20c = S2MM
  (tile->BO), 0x1d214/0x1d21c = MM2S (BO->tile); bd_id in the write's value & 0xF.

FastFlowLM's generators emit HEADERLESS op runs; the layer TXN embeds the
real TXN header mid-stream (parser detects headers anywhere).

## Layer TXN — qwen3_npu_sequence::gen_layer_seq(seq, L)  (decode)

Buffer = TWO byte-identical TXN copies (double-buffered ping-pong), each
8543 words (34172 B), header claims op_count=1048. Per copy:
- 41 x WRITE val=1 to memtile BD regs (0x06202400, 0x06308400, ...)
- 204 BLOCKWRITE/DDR_PATCH pairs, 197 MASKWRITE (issue tokens), 197 TCT, 246 WRITE
- Data flow (arg_idx = BO slot in the layer kernel's host-BO list):
  - arg1 (weight BO) MM2S reads: 192 BDs, lens {10240,20480,30720}, offsets at
    40960-B strides, 4 blocks of 1310720 B (== rc_dst_0_1310720.bin size, the
    reorder_cpy output capture) -> the layer kernel READS the reordered weights.
  - arg4 (kv) MM2S reads {256,4096}; arg0/arg2/arg3 tiny (RMS/norm weights).
  - L only shifts kv offsets (arg4 @0 for L=1 vs @130048 for L=128).
  - The GEMMs themselves are NOT in this TXN — separate mm.xclbin TXNs.

## mm TXN — runtime's shipped mm.bin (xclbins/Qwen3-0.6B-NPU2/mm.bin)

8-stage chained GEMM, 20 BD/patch pairs, stages on NPU cols 0..7; even cols
carry a weight BD, odd cols chain (next_bd). Per stage:
- arg1 (weight): BD(0,col,0) len=32768, D0=256/1, D1=64/512, D2s=256 — reads a
  [256,256] bf16 K-chunk (32768 B = 131072/4) ; chunks at 131072-B strides
  (0, 131072, 262144, 393216) = [256,1024] weight in 4 K-chunks.
- arg0 (act): BD(0,col,14) len=16384 at 256-B strides (token blocks).
- arg2 (out): linear len=65536 at 262144-B strides = [256,128] bf16 output.
- arg0/arg1/arg2 = first/second/third host BO in the kernel call.

**INSTS MISMATCH (#2015):** the hand-rolled per-shape streams
(gen_mm_insts_batch -> mm_256_1024_128_0.bin) contain only 6 BD pairs (ONE
GEMM: 4 weight K-chunks + act + out, single column), while the runtime submits
the 8-stage chained mm.bin. The engine also binds BOs with the ERT slot
numbers (group_id 3..7) while the TXN arg_idx counts host BOs from 0.

## Dequant TXN — Dequant::generate_dequant_q80_packed_in_q4nx_seq
(and q4_1; same layout) for q_proj block 0 (D_in=1024, D_out=256, woff=0)

Per TXN copy (modes 0..3 identical layout):
- arg1 MM2S (reads Q4NX input): BD len=5120 (one 5120-B Q4NX tile), D0=128/1,
  D1=20/128, D2s=20480, iter=8/2560; tile offsets 0 and 163840.
- arg0 S2MM (writes dequant output): BD len=32768, D0=128/1, D1=128/256,
  D2s=128, iter=2/32768; output offsets 0 and 262144.
  Output region per write: [32,256]-ish tile(s), f32 (1MB total for the block
  == captured B0 size 1048576 B). B0 = the runtime's dequant.xclbin output.

## Next steps

1. Correlate captured B0 (npu-infer/captures/bo_from_000_1048576.bin) bytes
   against the Q4NX tiles dequantized with the decoded layout/offsets:
   (q-zp)*scale (bit-exact formula, round-27) placed per the dequant S2MM BD
   geometry; resolve the "B0 does not match host dequant" mystery (#2015).
2. Derive the exact weight BO layout from the dequant S2MM BD geometry
   (D0=128/1 D1=128/256 D2s=128 iter=2/32768) -> implement in
   npu_pack_weight_bo (src/model.c); byte-verify hand-packed == captured B0.
3. Match insts: engine should submit the runtime's 8-stage chained mm format
   (or at least bind BOs so TXN arg_idx maps to the right host BO).
4. On-device validation loop: hand-rolled path vs runtime path on identical
   activations -> byte-identical output.

## Round-30 (2026-09-01, post-reboot) — REAL runtime running + BO capture

1. **The real FastFlowLM runtime now runs on this NPU** via a minimal harness
   (`npu-infer/tools/capture/run_qwen3_npu.cpp`): links ONLY libqwen3_npu.so +
   libq4_npu_eXpress.so + xrt, drives qwen3_npu::load_weights + forward
   (no AutoModel/tokenizer). load_weights + 4 decode steps verified on device.
2. **LD_PRELOAD interposer** (`npu-infer/tools/capture/cap_interposer.cpp`)
   hooks the C++ xrt::bo::sync (mangled _ZN3xrt2bo4syncE18xclBOSyncDirectionmm,
   defined in libxrt_coreutil) and captures every BO sync. A real run captured
   290 syncs: 28x32MB kv init, 28x10MB workspace, 233x1MB uploads + 1x94MB.
3. **The 1MB uploads are the NORMS / RTP weights** (2-4 KB of bf16 data in a
   1MB BO): layer-0 input_layernorm matched EXACTLY ([0.1357, 0.7070, ...]).
   The projection weights live in the **94MB BO** (CAP 0141) — dense bf16
   values (~0.01-scale) in the runtime's REAL dequant layout.
4. **B0 (captures/bo_from_000_1048576.bin) is NOT a weight BO** — it matches
   no dequant/formula/projection; the 1MB BO_FROMs in the old capture were
   likely activation read-backs (mislabeled). The runtime's real weight BO is
   the 94MB upload.
5. **mm.bin arg binding (the #2015 insts mismatch root cause)**: decoding the
   DMA directions from the queue-write registers shows mm.bin's arg0 = OUTPUT
   (S2MM writes), arg1 = WEIGHT (MM2S), arg2 = INPUT (MM2S), bo3/bo4 = ws/kv.
   The engine's run_gemm bound (act, ws, w1, w2, kv) — weight at the wrong
   slot → silent zeros. The fixed test call `(bo_out, wt, act, ws, kv)` now
   produces real GEMM output on the NPU.
6. **mm.bin is aiebu-format TXN**: header [magic 0x535f544e "NTS_"][0xd00]
   [op_count][bytes], and the buffer = TWO identical TXN copies (like the
   layer TXN). The decode tool now handles both header formats.
7. **dequant.xclbin runs with opcode 3** + the decoded dequant TXN and is
   input-sensitive; its output is **BF16 [256,1024] of W = q*scale + zp**
   (the dequant KERNEL formula differs from the host lib's (q-zp)*scale!).

## Next steps (round-31)

- Decode the 94MB weight BO layout: correlate against the q4nx tiles
  (reorder permutation + q*s+zp formula known) → the exact weight-BD layout
  the mm kernel consumes → implement in npu_pack_weight_bo.
- Validate: engine packer output == runtime 94MB slice byte-for-byte; then
  the fixed mm call produces the reference GEMM (validation loop closed).
- Re-run capture with the FULL runtime (chat flow) to label every BO.
