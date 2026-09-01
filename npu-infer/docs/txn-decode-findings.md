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

## Round-30c — THE RUNTIME WEIGHT LAYOUT IS DECODED (byte-exact)

The captured BO traffic reveals the complete weight layout:

1. **Per-layer weight BOs (10 MB) = the REORDERED Q4NX TILES** (verified
   byte-exact for 513+ leading tiles): layers in REVERSED order (27 first,
   layer 0 last), each layer = q,k,v,o,gate,up,down projections in that
   order, each projection's tiles in the reorder permutation
   out[o] = in[8*(o//8) + (o//2)%4 + 4*(o%2)] (per 8-tile group), raw
   5120-B Q4NX tile bytes (within-tile layout unchanged).
2. Per layer: 1920 tiles = q(256)+k(128)+v(128)+o(256)+gate(384)+up(384)+
   down(384) = 9830400 B, in a 10485760-B BO.
3. The 94 MB BO = the reordered lm_head tiles (18992 tiles, verified).
4. The 1 MB uploads = per-layer small tensors (input/post_attn layernorms,
   q/k norms, ones) — the layer TXN's arg0/2/3 RTP reads.
5. The mm/layer kernels DEQUANTIZE IN-KERNEL from the reordered tiles
   (round-28's conclusion, now confirmed by the captured layout): the
   hand-rolled packer must emit the reordered tiles (NOT a dequant).
6. Layers are uploaded reversed; the engine's packer and per-layer BO layout
   must match (layer 27 first).

## Next steps (round-31)

- Implement npu_pack_weight_bo as the tile reorder (permutation + per-layer
  BO layout, layers reversed); byte-verify the packer output == captured BO.
- Map mm.bin's arg1 BDs (128 KB strides) onto the reordered-tile BO — the
  on-device dequant's tile-window arithmetic.
- Fix run_gemm arg binding (arg0=out, arg1=weight, arg2=input) in the engine.
- On-device validation: engine path output == runtime path output.

## Round-30d — THE PACKER IS BYTE-EXACT (28/28 layers)

Implemented `npu_pack_layer_bo` (src/model.c): packs a full layer's 7
projections into the runtime's per-layer 10 MB weight BO — reordered raw
Q4NX tiles (no dequant!). Verified **28/28 layers byte-identical** against
the captured runtime BOs (tests/verify_packer.cpp, 0 bytes different).
Layout: layers in model order; per layer q(G=8,256 tiles), k(G=8,128),
v(G=8,128), o(G=16,256), up/gate alternating 64-tile chunks (up0, gate0,
up1, gate1, ..., G=8 each), down(G=24,384). Tile reorder:
out[o] = in[G*(o/G) + (o/2)%(G/2) + (G/2)*(o%2)].

## Next steps (round-31)

- Engine integration: WeightPacker → npu_pack_layer_bo (per-layer 10 MB BOs,
  layers in model order); run_gemm arg binding fix (arg0=out, arg1=weight,
  arg2=input); per-shape insts → the runtime's chained 8-stage mm TXN.
- On-device validation loop: engine mm output vs CPU reference (q*s+zp
  in-kernel dequant), then vs the real runtime's output on identical inputs.

## Round-31 — runtime kernel structure + CPU-math validation status

1. **The runtime submits ONE runlist per forward** (28 layers + lm_head in a
   single xrt::runlist; 4 executes for 4 forward steps). Individual runs:
   ~169 xrt::run objects (reused), args set via set_arg_at_index:
   idx3=1MB act, idx4=10MB per-layer weight (or 94MB lm_head), idx5/6=1MB
   (out/act), idx7=32MB kv. The TXN arg_idx = BO-list position (arg0..arg4)
   consistent with the layer TXN decode (arg1=weight, arg4=kv).
2. **The weight BO is NEVER modified on-device** (post-runlist dumps == the
   uploaded reordered tiles, 0 bytes changed) — the layer/mm kernels read the
   tiles and dequantize in-kernel; the hand-rolled packer's tile layout is
   the exact device layout.
3. **CPU reference vs the real runtime's logits** (token 1000): corr 0.34,
   top-50 0. The dequant formula q*scale+zp is confirmed (host lib's
   q4nx_dequantize = (q-zp)*scale gives corr ~0; q*s+zp is the runtime's
   formula). The residual gap = per-layer accumulation divergence (the
   hidden state grows outliers through the RMS norms; layer-27 explodes in
   the f32 ref while the runtime's bf16 kernels stay bounded). Status: the
   transformer-math validation needs the runtime's per-layer hidden states
   (or a bf16-accurate ref) — next round.
4. Layer TXN arg1 tile-window reads: 2 tiles per 8-tile stride — the
   mm/layer in-kernel dequant window arithmetic (next round: map onto the
   packed layer BO).

## Round-31b — runtime hidden state captured (the validation target)

1. **The runtime's ACT BO (the final hidden state) is now captured** (the
   interposer tags post-exec dumps with BO addresses; the set_arg idx=3 BO =
   the act). The runtime's final hidden (token 1000, layer-28 output) has
   std ~27 with outliers to ±464 — the RMS-norm outlier amplification is
   REAL in the runtime (my CPU ref shows the same phenomenon, growing to
   ±136 at the post-norm / ±3290 pre-norm).
2. **The runtime's kv-cache BO** (32MB, reused per layer): 8192 nonzero bf16
   = 4 forwards x 2048 (one layer's k+v each). Layer k/v comparisons vs my
   CPU ref: corr ~0 at every layer; my k values are ~1.5-50x smaller than
   the runtime's per layer — the divergence starts at layer 0 and is
   systematic (my dequant or attention differs subtly from the runtime's
   in-kernel math).
3. **Status**: the CPU-reference validation of the transformer math is
   blocked on the exact in-kernel dequant/attention semantics. The LAYOUT
   half (byte-exact packer, arg binding, kernel structure) is complete; the
   MATH half has the full capture toolchain + CPU ref, with the divergence
   pinpointed to the layer-0+ per-layer math (next round: capture the
   runtime's per-layer act or compare layer-0 k/v via a single-layer probe).

## Round-31c — layer TXN = prep phase; compute runs reuse run objects

1. **The layer TXN (gen_layer_seq) is the PREP phase, not the compute**: its
   only S2MM write is arg0 len=512 (RTP). All other traffic = reads (arg1
   weight tiles 192x, arg4 kv 8x, arg0/2/3 RTP). A layer.xclbin submission
   with the decoded layer TXN + packed layer BO executes (ERT completes) but
   writes no layer output at any opcode (2/3/4/5) — consistent with prep.
2. **The per-forward run structure**: ~42 unique xrt::run objects per forward
   (reused across the 4 forwards: 169 ctor calls / 4); args set via
   set_arg_at_index idx3-7 only (act 1MB, weight 10MB/94MB, out 1MB, out
   1MB, kv 32MB) — opcode/instr/ninstr (idx 0-2) bound via the kernel
   operator() internals (not observable via set_arg hooks).
3. The runtime's per-call compute TXNs (mm/attn) are generated at runtime
   (Gemm::generate_seq with per-projection params) and their insts are not
   directly capturable with the current interposer. mm.bin is one shipped
   example; its weight BDs read 32KB bf16 windows at 128KB strides — feeding
   the packed TILE layout yields NaN (the tiles are not bf16 weights), so
   the runtime's actual per-projection mm TXNs must use tile-aware BD
   geometry (next round: capture via the xrt::ext path or ERT submit hook).

## Open items (next round)

- Capture the runtime's per-call compute TXNs (ext::kernel/ERT hook).
- Derive the mm kernel's tile-window dequant geometry from those TXNs.
- CPU-ref validation: per-layer divergence root cause (in-kernel dequant
  semantics).
- Engine integration: per-layer packer (done) + layer/mm TXN submission +
  arg binding (done in the test) + on-device validation loop.

## Round-31d — THE KERNEL ABI: insts are a BO ARG, not the ERT instr slot

Disassembly of npu_app::create_run (libqwen3_npu.so @0x4b9c0) — the runtime
builds EVERY kernel run as:

    run = xrt::run(kernel)
    set_arg(0, 3)                       ; opcode
    set_arg(1, 0)                       ; instr slot = 0
    set_arg(2, 0)                       ; ninstr = 0
    set_arg(3, data_buffer.bo)          ; bo0 = the data/insts buffer's BO
    set_arg(4, weight_buffer.bo)        ; bo1 = the weight BO (10MB tiles / 94MB lm_head)

The per-call TXNs are passed as a **BO ARGUMENT (bo0)**, NOT in the ERT
(opcode, instr_bo, ninstr) slots — the firmware takes the control code from
the bo0 buffer (coherent memory, written post-arg, no sync — so the BO sync
hooks never see it). The runtime's 5-BO runs add bo2/bo3/bo4 (out/out/kv)
via the caller.

This explains the engine's behavior: the engine's ABI (3, instrs_bo,
ninstr, ...) ALSO works (the ERT accepts the insts from the instr slot) —
the engine's mm test DID execute with it. The remaining gap is the runtime's
per-call mm insts CONTENT (generated per call; mm.bin is the shipped
example) and the mm tile-window dequant geometry.

## Next steps (round-32)

- Fix the interposer's insts-BO capture (the buffer<uint8_t> BO is created
  via the npu_app buffer path — hook that allocation) to obtain the
  runtime's actual per-call TXNs.
- Decode those TXNs (the decode_txn tool handles the format) -> the exact
  per-projection mm geometry -> validate the mm path on device.
- Engine integration + validation loop.

## Round-31e — insts-BO structure + engine-vs-runtime ABI tests

1. The runtime's 1MB ext::bo objects (npu_app buffers) are the bf16 data
   buffers (norms/act) — the TXN insts are written into the bo0 buffer after
   set_arg (coherent, no sync). The xclbins' AIE_PARTITION sections hold
   only the config metadata (no control code; the PDI binaries are small
   config blobs).
2. Test of the runtime's create_run ABI form (3, 0, 0, insts, weight, ...)
   crashes in xrt (the instr slot must be a real BO/pointer) — the engine's
   (3, instrs_bo, ninstr, ...) form is the working submission path.
3. Engine ABI + packed-tile weight BO: the mm kernel writes ~228K nonzero
   outputs (vs 16K with dequant-bf16 weights) — the tile bytes ARE being
   consumed (structured reads), confirming the mm's in-kernel tile handling;
   the exact per-call mm insts remain the key open item.

## Next (round-32)
- Extract the runtime's per-call TXNs from the bo0/insts buffer (hook the
  npu_app buffer fill or the run submit) OR decode the aiebu PDI control
  code from the xclbin's AIE_PARTITION (proper PDI extraction).
- Derive the mm tile-window dequant geometry; validate on device.
