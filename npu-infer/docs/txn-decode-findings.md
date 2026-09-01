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

## Round-32 — THE PER-CALL TXNs ARE THE ELF/MODULE: runtime TXN capture CLOSED

### The discovery: TXNs are embedded in the ELF, not passed as a BO

Reading npu_utils_xrt.hpp (`npu_app`), the runtime's per-call TXN flow is:

    operator()/create_run:
      if (!module_valid || !seq_valid ||
          module_version != ctrl_seq->sequence_version()) {
          _setup_kernel();       // <-- regenerates the ELF+module+kernel
      }
      kernel->operator()(3, 0, 0, args.bo()...);   // NO TXN BO argument!
      run.wait() / runlist.execute()

    _setup_kernel():
      data = ctrl_seq->dump();              // the raw TXN words
      _gen_elf(&elf_buf, data);             // wraps TXN in an ELF
      elf   = new xrt::elf(elf_buf, size);  // <-- the TXN is IN this ELF
      module = new xrt::module(*elf);
      kernel = new xrt::ext::kernel(ctx, *module, name);

The `(3, 0, 0, ...)` args are opcode=3, instr_slot=0, ninstr=0 — the
firmware takes the control code from the MODULE (embedded at kernel
creation), not from a BO arg. The "data/insts buffer" bo0 is just the
bf16 data buffer. The ELF's `.ctrltext` section IS the per-call TXN.

### The hook that finally worked: xrt::elf ctor

cap_interposer now interposes `_ZN3xrt3elfC1EPKvm` (xrt::elf(const char*,
size_t)) — called by _setup_kernel with the freshly generated TXN. Every
kernel re-setup dumps the ELF. The runtime re-setups the kernel whenever
set_context_length() changes the sequence version — i.e. on EVERY forward
(the kv-cache offsets move with the token position).

    set_context_length(ctx):
      gen_layer_seq(seq, ctx+1);            // L = context length + 1
      runlist.reset();
      if (version changed) _setup_kernel(); // new ELF with new kv offsets
      run = xrt::run(kernel); set args; runlist.add(run);
      ... runlist.execute() (via forward)

### Captured: 7 ELFs from a 4-token run (run_qwen3_npu)

    elf_0001 (37360 B)  = layer TXN @ ctx=1   -> .ctrltext 34172 B (8543 words)
    elf_0002 (421536 B) = lm_head TXN         -> .ctrltext 392212 B (98053 words)
    elf_0003 (37360 B)  = layer TXN @ ctx=1   (identical to 0001)
    elf_0004 (37360 B)  = layer TXN @ ctx=2   (kv offsets +0x400/token)
    elf_0005 (37360 B)  = layer TXN @ ctx=3
    elf_0006 (37360 B)  = layer TXN @ ctx=4
    elf_0007 (37360 B)  = layer TXN @ ctx=5

### Byte-exact verification (0 word diffs)

    captured layer @ctx=1 == gen_layer_seq(seq, 1) with MAX_L=8192   [0 diffs]
    captured layer @ctx=2 == gen_layer_seq(seq, 2) with MAX_L=8192   [0 diffs]
    captured layer @ctx=3 == gen_layer_seq(seq, 3) with MAX_L=8192   [0 diffs]
    captured lm_head     == gen_lm_head_seq() (single copy)          [0 diffs]

IMPORTANT: MAX_L matters! With MAX_L=4096 the generated kv offsets are
4 MB apart; the runtime's are 8 MB apart (32 MB kv BO = 4 regions x 8 MB,
8192 tokens x 1024 B/token per region). gen_layer_seq(seq, L) must be
called on a sequence constructed/set with MAX_L=8192 to match the
runtime byte-for-byte. (The harness constructs qwen3_npu(config, npu,
4096) but set_max_length is applied from the model config; the captured
offsets prove MAX_L=8192.)

### What the decoded layer TXN says (arg1 = the weight BO)

The layer TXN (8543 words, 1048 ops) has 204 BLOCKWRITE + 204 DDR_PATCH
+ 197 MASKWRITE + 197 TCT + 246 WRITE. The patch table (BD -> arg@offset):

    arg0 @0        BD(0,2,0)   len=512    MM2S   (norm/RTP read)
    arg2 @0        BD(0,2,1)   len=1024   MM2S
    arg3 @0        BD(0,2,2)   len=192    MM2S
    arg0 @0        BD(0,2,10)  len=512    S2MM   (RTP write)
    arg4 @0/8M/16M/24M         len=256/4096 MM2S (kv cache, 4 regions)
    arg1 @0..9.7MB (192 BDs)   len=10240/20480/30720  MM2S  <-- THE WEIGHTS

Weight BD stream (arg1): 192 MM2S BDs reading 10240-byte windows at
40960-byte stride across the full 10 MB weight BO:
    - 64 BDs len=10240 @ 0..2.58 MB (q/k/v/o region? every 8th window)
    - 16 BDs len=20480 @ 2.62..3.85 MB
    - 96 BDs len=10240 @ 3.93..7.82 MB
    - 16 BDs len=30720 @ 7.86..9.7 MB
16 unique BDs (rows 0,1,6,7 x cols 0,1 x bd_id 1,2,9,10) reused 12x each.
The 40960-byte stride = 8 Q4NX tiles (5120 B each); len 10240 = 2 tiles.
This IS the mm kernel's tile-window read geometry over the reordered-tile
BO that npu_pack_layer_bo already produces byte-identically (28/28
layers) — the weight half of the decode is now closed end-to-end:
generator output == captured runtime TXN == packer layout.

### lm_head TXN (elf_0002, 98053 words)

arg1 = 2384 BDs, all len=10240, at 40960 stride spanning 0..97.6 MB —
the full 94 MB lm_head weight BO (98566144 B). arg0: BD(0,3,15) len=76288
S2MM; arg2/arg3: 512 B MM2S each. Matches gen_lm_head_seq byte-exact.

### Corrected ABI understanding (supersedes round-31d/31e)

- The runtime does NOT pass TXNs as a BO arg. `(3, 0, 0, ...)` + module-
  embedded control code is the runtime's real submission form.
- The engine's `(3, instrs_bo, ninstr, ...)` form ALSO works (xrt accepts
  insts from the instr slot) — the engine path is valid, but the runtime
  path is module-embedded.
- The pre-exec BO dumps (preinsts_*) at runlist::execute were all bf16
  data (activations/logits), NOT TXNs — because the TXNs never touch a
  BO. Only the xrt::elf hook sees them.

### Repo evidence

    npu-infer/captures/txn-elfs/elf_0001_layer.bin, elf_0002_lmhead.bin,
        layer_ctx1/2/4_ctrl.bin, lmhead_ctrl.bin, gen_layer_MAXL8192_L1.bin
    npu-infer/decodes/qwen3-0.6b/captured/layer_ctx1.json, lmhead.json

### Next steps (round-33)

- Wire the MAX_L=8192 discovery into decode_txn / gen_layer_seq usage
  (the tool currently uses MAX_L=4096 -> doubles the kv offsets; fix the
  driver to call set_max_length(8192) or construct with the runtime value).
- On-device validation: submit the captured layer TXN (ELF ctrltext) via
  the engine with the packed 10 MB weight BO + 1 MB act + 32 MB kv and
  compare against the runtime's logits (validation loop).
- Engine integration: npu_pack_layer_bo (done, byte-verified) + the layer
  TXN (now byte-identical to the runtime's) + the (3,0,0,...) ABI.

## Round-32b — on-device submission of the captured layer TXN (module ABI)

### Replicating the runtime's submission: xrt::elf -> module -> ext::kernel

test_layer_elf.cpp builds the runtime's EXACT path:

    xrt::elf elf((const char*)elfbuf, esz);   // the captured ELF
    xrt::module mod(elf);
    xrt::ext::kernel kern(hwctx, mod, "MLIR_AIE");
    xrt::run run(kern);
    set_arg(0, 3); set_arg(1, 0); set_arg(2, 0);
    set_arg(3, act 1MB); set_arg(4, weight 10MB);
    set_arg(5, out1); set_arg(6, out2); set_arg(7, kv 32MB);
    run.start(); run.wait();  -> state 4 (completed)

With ext::bo buffers (the runtime's buffer type) the run completes (state 4)
but writes nothing visible to act/out1/out2/kv. Why: the layer TXN is the
PREP phase — its only S2MM writes are 512B RTP to arg0 + 4x256B kv writes
to arg4. The hidden-state output (2048B) is NOT in this TXN; it is produced
by the mm/dequant kernels (their ELFs are loaded via load_elf from the
xclbin PDI at construction — not via ctrl_seq dump, hence not captured by
the xrt::elf ctor hook; the mm.bin/mm_256_*.bin files in the xclbins dir
are those static instruction streams).

### Key validation facts from the runtime's own buffer traffic

- The runtime's act buffer (idx3 of the layer run, 1MB) holds the TOKEN
  EMBEDDING before the forward and the layer output hidden state after —
  the layer compute happens in-place in arg0 (confirmed: preinsts vs post
  dumps differ by exactly the hidden-state region [0:2048]).
- The lm_head run binds idx5 = the layer's act buffer (reads the final
  hidden state as its input) and writes logits to its own idx3 (1MB).
- So the hand-rolled path must run: layer prep TXN (captured) + mm/dequant
  kernels (mm.bin etc.) per layer, then lm_head TXN (captured) — matching
  the runtime's runlist (28 layer-prep + 1 lm_head + 28 compute runs per
  forward in the capture).

### Remaining for the on-device loop (round-33+)

- Wire the mm/dequant ELFs (from the xclbin PDI or the .bin files) into the
  engine's per-layer submission so the full layer computes (prep -> dequant
  -> mm), then compare the act buffer against the runtime's captured post-
  exec act (post_002_156_59c84ec1fc60 style) and the logits vs
  logits_1000.bin. That is the byte-level validation loop closure.
- The two identical layer ELFs (0001 == 0003) are the layer app and a second
  app instance sharing gen_layer_seq content; per forward the runtime
  re-setups the layer kernel once (new kv offsets) and arms 28 runs per
  kernel instance.

## Round-32c — runlist structure: the layer kernel IS the compute (no separate mm kernel in the forward)

### The per-forward runlist (captured via interposer)

RUNLIST 1 (forward 1000): 57 runs = 28× layer-kernel-A + 1× lm_head + 28×
layer-kernel-C. Kernels A and C have IDENTICAL ELF content (md5
2e50c6ea...) = gen_layer_seq(ctx=1) — the runtime arms the layer prep
twice per forward. RUNLIST 3/5/7: 28 runs of a fresh layer kernel (ctx=2/3/4).
NO separate mm/dequant/mha kernel appears in the forward runlist — the
runtime's forward is layer-prep + lm_head only (the layer kernel's AIE
execution performs the dequant+mm in-tile).

### Which BO is what (from the captured arm bindings)

Layer run (28 arms): idx3=1MB act (FIXED for all layers), idx4=10MB weight
(rotates per layer), idx5=1MB + idx6=1MB (norm params, rotate per layer),
idx7=32MB kv (rotates per layer).
lm_head run: idx3=1MB logits-out, idx4=94MB weight, idx5=1MB (= the layer's
act buffer! the final hidden state), idx6=1MB.

### Verified data flow from the captured buffers

- act (idx3) before forward = token embedding; after the runlist = the
  layer-27 output hidden state (std ~3.4, values +-4 — RMSNorm-amplified).
  The lm_head READS it as idx5 and writes logits to idx3 (0x59c84ec20450).
- The lm_head's logits buffer (preinsts_003 dump) == logits_1000.bin
  EXACTLY (151936/151936 u16 match) — the capture chain is end-to-end valid.
- kv BO (idx7, 32MB) after forward: 4 regions (0/8M/16M/24M) each with
  1024B of k/v — the layer TXN's 4x256B S2MM kv writes per region.

### The layer TXN's act write: 512B S2MM BD(0,2,10) vs 2048B hidden state

The decoded layer TXN has only 5 S2MM queue writes: 512B to arg0 (col2
bd10) + 4x256B kv (col3/4 bd0). The hidden-state write is NOT visible as a
separate BD — it is produced by the layer kernel's in-tile compute writing
back to arg0 via the same BD machinery (the firmware's shim-DMA BD
buffer_length semantics for S2MM writes: w4=0x200 with w8/w9 dim/iteration
fields may expand the effective write; the mm.bin OUT BD uses the same
pattern with w4=0x4000=16384 for a known 16384B output). The BD format
detail (buffer_length unit / iteration) is the last open sub-item; the TXN
CONTENT itself is byte-verified against the runtime generator, which is the
deliverable.

### On-device test status

- test_npu_layer_elf: builds the runtime's exact submission (elf->module->
  ext::kernel, (3,0,0,act,weight,out1,out2,kv), ext::bo) — run completes
  (state 4) with no visible writes: expected, since ONE layer run is the
  PREP; the full forward needs the 56-run + lm_head sequence the runtime
  arms.
- The captured lm_head TXN (elf_0002) decodes: 2384 arg1 weight BDs
  (10240B each, 40960 stride, spanning 0..97.6MB of the 94MB lm_head BO)
  + 76288B S2MM logits write + 2x512B MM2S reads. Byte-exact vs
  gen_lm_head_seq.

## Round-32d — on-device layer validation: prep-vs-compute and BD iteration semantics

### test_npu_forward_full: replicated the runtime's per-layer submission

Submits the captured layer ELF (module ABI) per layer with hand-packed
weight BOs (npu_pack_layer_bo, byte-verified), per-layer norm buffers
(idx5/6) and per-layer kv BOs (idx7) from the runtime's captures:

  - 1 layer: act output std=2.7, range +-3.4, finite — plausible hidden
    state (vs CPU ref layer-0 std 0.56: different dequant formula / stage).
  - 28 layers chained: EXPLODES (std 284, range +-4288) — the chain does
    NOT reproduce the runtime's bounded final act (std 3.4, +-4.6).

### Conclusion: the layer TXN is the PREP, not the full matmul

The decoded layer TXN writes only 512B to arg0 (act) + 4x256B kv — it
cannot produce the 2048B hidden-state output by itself. The runlist has
56 layer runs + 1 lm_head per forward, but the layer TXN content is
identical for all 56. The ACTUAL matmul must happen in-kernel via BD
iteration: the S2MM write BDs carry dim/iteration fields (w6/w8/w9 in the
12-word BD) that the decode tool reads as iter=1/1 but which expand the
effective write (the lm_head's 76288B S2MM BD produces the full 151936-bf16
logits — 4x expansion via dims). The layer's arg0 512B write similarly
expands to the 2048B hidden state via the same mechanism.

### The blocker for the compute half

The 12-word shim-DMA BLOCKWRITE's w6/w8/w9 iteration semantics (D0/D1
dims + iteration count) are not decoded. mm.bin OUT BD: w6=0, w7=0x4000000,
w8=0xd00001ff; lm_head col2 BD: w6=0x40000000, w8=0xc0000000, w9=0xe000000;
layer col2 BD: w6=0, w8=0xc0000000, w9=0x2000000. Decoding these fields
(from mlir-aie / amdxdna BD docs or the firmware) unlocks the effective
I/O sizes and closes the on-device loop.

### What IS closed (byte-verified, committed)

1. Per-call TXN capture (xrt::elf ctor hook) — 7 ELFs from a 4-token run.
2. layer TXN == gen_layer_seq(ctx+1) @ MAX_L=8192 (0 word diffs).
3. lm_head TXN == gen_lm_head_seq (0 word diffs).
4. Weight BD geometry: 192 arg1 BDs, 10240/20480/30720B at 40960 stride
   over the 10MB BO — satisfies npu_pack_layer_bo's layout (28/28 layers).
5. Runtime structure: set_context_length -> gen_layer_seq -> _setup_kernel
   (new ELF per forward, kv offsets +1024B/token); runlist = 56 layer runs
   + 1 lm_head; per-layer kv BOs (idx7), shared act (idx3).
6. lm_head logits buffer == logits_1000.bin EXACTLY (151936/151936 u16) —
   capture chain end-to-end valid.

### Next step (round-33)
Decode the 12-word BD iteration fields (w6/w7/w8/w9) — either from
mlir-aie/amdxdna sources or by correlating known I/O sizes (lm_head
76288B -> 303872B logits = x4; layer 512B -> 2048B hidden = x4) — then
re-run test_npu_forward_full and compare act/logits vs the runtime
captures. That closes the validation loop.

## Round-33 — weight-BO byte-identity PROVEN + the runlist flow decoded

### Byte-identity: packed weight BO == runtime's weight BO (0 diffs in 10MB)

/tmp/pack_check regenerates npu_pack_layer_bo layer-0 output and compares
against the runtime's captured layer-0 weight BO (post_002_59_...capR):

    weight BO diff positions: 0 of 10485760  -> IDENTICAL

The hand-rolled packer produces the EXACT bytes the runtime's layer kernel
consumes. This closes the layout half with a direct byte-for-byte proof
(the earlier 28/28 layer verification used the packer-vs-runtime BO compare;
this round re-confirms on a fresh capture).

### The 4x S2MM write expansion (76288B -> 303872B logits; 512B -> 2048B hidden)

The lm_head TXN writes 76288B to arg0 (idx3=logits), yet the logits buffer
holds 151936 bf16 = 303872B (verified 151936/151936 against logits_1000.bin
AND logits_1001.bin on a fresh capture). The layer TXN writes 512B to arg0
yet the act buffer holds 2048B. The ratio is ~4x for both. The queue-write
repeat_count=0 and the BD dims are linear (D0=D1=0, iter=1), so the
expansion is NOT visible in the TXN ops — it is firmware behavior in
npu.dev.sbin (the amdxdna firmware parses TXNs; the host kernel defers
parsing to firmware). mlir-aie's WriteBdToBlockWritePattern shows
buffer_length = sizes[0..2] product in granularity/8 units (granularity=32
bits -> bf16 count/2), but 76288 vs the expected 75968 (303872/4) differs
by 320 — the exact unit is not yet pinned down.

### Runlist flow decoded (forward disassembly + interposer)

The runtime's forward (libqwen3_npu.so @0x401d0) runs a pipeline:
  40742-4077d: run.start() (lm_head run) -> set_context_length (regenerates
  the layer TXN for the NEXT context: gen_layer_seq + _setup_kernel + new
  ELF) -> run.wait() -> sync_from_device
  407ba: runlist.execute() (the 57-armed runlist: 28 layer prep + 1 lm_head
  + 28 layer prep, all with module-embedded control code)

The runtime registers layer.xclbin, mm.xclbin, dequant.xclbin, attn.xclbin
(rodata strings), but the forward runlist only uses the layer + lm_head
kernels (3 EXTKERNELs observed). The layer kernel's in-tile AIE execution
performs the dequant+matmul (the mm/dequant/attn apps are registered but
not invoked in this decode-forward path).

### On-device loop status (honest)

- Single layer: produces a plausible hidden state (std 2.7, +-3.4) with
  packed weights (byte-identical) + captured norms + empty kv.
- 28-layer chain: EXPLODES (std 284) with identical numbers regardless of
  kv/norm/runlist-vs-sequential submission. The layer kernel's compute
  diverges from the runtime — the root cause is in the module's in-tile
  execution semantics (how the AIE program consumes the TXN-setup BDs and
  streams the 4x-expanded act I/O). Closing this needs either the
  npu.dev.sbin BD semantics or replicating the runtime's exact
  lm_head-first pipeline order.
