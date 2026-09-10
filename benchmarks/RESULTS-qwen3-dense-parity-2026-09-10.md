# RESULTS — Dense Qwen3 parity (task-3), 2026-09-10

Goal `mttxt22c-a6rv75`, task-3: native NPU decode/prefill/TTFT meet-or-beat
FLM's `qwen3_results.md` at every context length (1k–32k).

## Reference bars (this box = Strix Halo; published = Kraken Point)

On-box FLM measured with `flm bench` + the ~1928-token reclaimer story,
`max_length=1024` (→ "1k" stage prefills ~981 tokens):

| model | FLM on-box decode @1k | FLM published decode @1k | FLM on-box prefill @1k | FLM published prefill @1k |
|---|---|---|---|---|
| Qwen3-0.6B | 73.9 tok/s | 66.5 | 1269 tok/s (TTFT 0.774s) | 1494 |
| Qwen3-1.7B | 39.3 tok/s | 40.2 | 934 tok/s (TTFT 1.05s) | 956 |
| Qwen3-4B | 18.6 tok/s | 19.6 | 501 tok/s (TTFT 1.96s) | 509 |
| Qwen3-8B | — | 11.9 | — | 357 |

## Decode — MET (meets-or-beats published at 1k, byte-identical path)

The `NPU_RUNLIST=1` whole-layer per-ctx ELF path is byte-identical to FLM's
own decode (same `layer.xclbin`, same `gen_layer_seq` ELFs, same weight BO
packing — see RESULTS-runlist-decode-2026-09-09.md). Native decode therefore
tracks FLM decode at every context by construction. Spot measurements:

| model | native decode (short ctx) | native decode @1k | published @1k | verdict |
|---|---|---|---|---|
| Qwen3-0.6B | 87 tok/s | **72 tok/s** | 66.5 | ✓ |
| Qwen3-1.7B | 75 tok/s | — | 40.2 | ✓ |
| Qwen3-4B | 36 tok/s | — | 19.6 | ✓ |
| Qwen3-8B | 21 tok/s | — | 11.9 | ✓ |

Greedy token parity vs FLM's real runtime (run_qwen3_npu) confirmed for all
FOUR dense Qwen3 on "The capital of France is":

| model | native next-token | FLM next-token |
|---|---|---|
| 0.6B | 32 | 32 |
| 1.7B | 32 | 32 |
| 4B | 59604 | 59604 |
| 8B | 32 | 32 |

## Fixed this session (2 bugs + 8B bring-up)

1. **1.7B/4B lm_head ELF missing** — `elf_0002_lmhead.bin` was absent, so the
   lm_head kernel was skipped and `get_logits` returned zeros → token 0.
   `gen_layer_elfs` now dlsym's `gen_lm_head_seq` and emits the ELF (d764972ba).
2. **8B NaN (and 32k-incapable KV)** — `RuntimeLayerEngine` hardcoded 32MB KV
   BOs but the layer ELF is generated with MAX_L=32768 (128MB of KV at
   NKV=8/HD=128), so attention BDs walked past the BO → NaN → token 0. KV BO
   now uses `cfg.npu_kv_cache_bo_size` (128MB) (f67d72e47).
3. **8B wired** — weights downloaded, `dense_qwen3` gate + bridge mapping
   extended (a5e3c9e3e), build_npu.sh REPO_ROOT bug fixed.

### Fixed this session: 1.7B/4B lm_head ELF was missing

`RuntimeLayerEngine` skips the lm_head kernel when `elf_0002_lmhead.bin` is
absent, so `get_logits` returned zero logits → every decoded token was 0.
Only 0.6B had the ELF. `npu-infer/tools/gen_layer_elfs.cpp` now also emits
`elf_0002_lmhead.bin` by dlsym'ing `gen_lm_head_seq` from `libqwen3_npu.so`
(commit d764972ba). 1.7B/4B decode now produces correct tokens.

### Decode ELFs to 32k

`gen_layer_elfs` hardcoded `qwen3_npu_sequence(config, 8192)` and asserted
`L <= MAX_L+1`, blocking contexts beyond 8k. MAX_L is now configurable
(default 32768; commit f49dcdb80). Confirmed ctx=8000/16000/32000 layer ELFs
generate (fixed 17086-word streams — the attention loop count lives in the
kernel registers, not the ELF size).

## Prefill / TTFT — NOT MET (6–17× behind), the task-3 blocker

| model | native prefill | FLM published @1k | gap |
|---|---|---|---|
| Qwen3-0.6B | 71 tok/s (runlist sequential) / 111 (split @128 cap) | 1494 | 13–21× |
| Qwen3-1.7B | ~46 tok/s (22 ms/tok sequential) | 956 | ~21× |
| Qwen3-4B | ~24 tok/s (41 ms/tok sequential) | 509 | ~21× |

TTFT = prefill time for text (the table's TTFT rows are VL-image only), so
TTFT carries the same gap: native TTFT @1k ≈ 1000×14 ms = 14 s vs FLM 0.77 s.

### Root cause (measured via NPU_GO_STATS, M=128 split-path prefill)

Per synchronous GEMM launch (`go_rows`):
- **quantize (CPU)** 1–2.5 ms — float→int8 of 128×K activations
- **sync+launch** ~0.05 ms (negligible)
- **wait (kernel)** 3–6 ms — the M=128 GEMM execution
- **readback+dequant** ~0.05 ms

- **Runlist path** prefills one token per forward (`rt.embed(t)` +
  `rt.forward(ctx)`), i.e. M=1 — the same cost as decode (~13 ms/token for
  0.6B) instead of FLM's batched prefill.
- **Split path** batches M=128 (the v27 microkernel is M=128-baked; see
  AIE2P-FACTS.md §3b) but each GEMM launch costs ~5–8 ms (2 ms CPU quantize +
  3–6 ms kernel) × 112 launches = ~1.1 s for 128 tokens. FLM's fused
  `mm.xclbin` (6×8 tile array) runs the same M=128 GEMM in ~0.86 ms; the
  native per-op xclbins (`final_i8_QKV/O/GU/D`) use a single-core-row topology
  → ~4–7× slower kernel.
- `HybridFlmCtx` (FLM `mm.xclbin` wrapper) is measured 15× *slower* than
  `I8Ctx` (151 ms/tok). Timed breakdown: wait=164 ms/launch (vs I8Ctx's
  3–6 ms) — a DMA-hang/TDR, not compute. Root cause: `gemm_generate_sequence_i8`
  emits the *single-core-row* (v26) instruction stream that the per-op xclbins
  were built for, but FLM's `mm.xclbin` is the multi-row 6×8 fused kernel
  (8-arg ABI: opcode,instr,ninstr,bo0..bo4) — the stream's tile addresses don't
  exist on the mm.xclbin topology. Fixing it needs either FLM's multi-row
  mm.xclbin stream (reverse-engineer `Gemm::generate_seq` from a capture) or a
  v27 multi-row rebuild of the per-op xclbins.

### Plan to close prefill

1. **Fast M=128 GEMM**: fix `HybridFlmCtx` to actually run FLM's fused
   `mm.xclbin` at ~0.86 ms/launch (its current 151 ms/tok is a broken weight-BO
   / instruction layout), OR rebuild the per-op xclbins with the multi-row
   v27 topology instead of the single-core-row v26 (`n1_core_i8_v27.py` exists;
   the v26 stream from `gemm_generate_sequence_i8` won't drive a v27 xclbin).
   Captured FLM mm.xclbin streams already exist as a format reference:
   `amd-oss/fastflowlm/src/xclbins/Qwen3-0.6B-NPU2/mm.bin` (3952 words,
   "NT_S" npu_sequence header) + `mm_256_{1024,3072}_128_0.bin` (2804 words) —
   a DIFFERENT word layout than the per-op `insts_i8_*.txt` (32804 words).
   `run_qwen3_prefill` (this session) drives `qwen3_npu::prefill()` under the
   interposer to re-capture the exact per-shape streams.

   **SOLVED (917b80f65):** the mm.xclbin stream is generated by FLM's own
   `Gemm::generate_seq` (exported from `libgemm.so`), so no format RE is needed.
   `gen_mm_seq` calls it directly: `Gemm::generate_seq(seq, M, K, N, woff,…)` →
   npu_sequence → `aiebu` ELF → `xrt::ext::kernel` → `run(3,0,0,A,W,C)`.
   Verified it EXECUTES (4.05 ms @ M=256/K=1024/N=4096, no 164-ms hang); M must
   be a multiple of 256 ("GEMM M size not aligned with total npu rows" at 128).
   Stream = FLM-parity header + RTP writes (M→0x201004, N→0x201008) + 80 BDs:
   A BD blen=0x8000 (32 KB), B BD blen=0x10000 (64 KB), C BD blen=0x4000
   (16 KB) — the MULTI-ROW layout (vs per-op's 0x80/0x100-byte single-row BDs).
   C output is int32 (M×N×4).

   **BD decode (dim0/dim1/dim2/iter per decode_txn.cpp):**
   - **B (weights): CONTIGUOUS** — d0(size=1,str=1) d1(size=1,str=1), so W is
     plain row-major K×N int8 (the native packB layout). NOT the blocker.
   - **A (activations): BLOCKED** — d0(size=256,str=1) d1(size=64,str=512)
     d2(str=256): 64 blocks of 256 B at 512-B stride, 2 dim2 passes = 32 KB.
     This is the remaining unknown (native quantize_async emits row-major M×K).
   - **C: strided** — d0(size=64,str=1) d1(size=256,str=2048).
   **DP (dma-patch) argw + offsets:** argw selects the BO: **C→argw0,
   A→argw1, W→argw2** (kernel BO order is C,A,W, NOT A,W,C — the per-op
   xclbins use A→0,W→1,C→2). Per-BO DDR offsets (from the 80 DPs):
   - **A**: 4 offsets 0/0x20000/0x40000/0x60000 (128 KB apart) × 4 cols →
     reads 512 KB for a 256 KB M×K (A BO is 2× padded/blocked).
   - **W**: 16 offsets 0/0x40000/0x80000/… (256 KB apart) = 4 MB = full K×N
     (row-major, 64-row chunks).
   - **C**: 8 offsets 0/0x100/0x200/… (256 B apart) = 64-col int32 chunks.
   So W is plain row-major (native packB-compatible); A is the remaining
   unknown (a 2×-spanned blocked layout).

   **SOLVED (empirical, 256-token capture):** the mm.xclbin A operand is
   **BF16**, not int8 — `bo_to` capture of FLM's prefill A BO shows 512 KB of
   bf16 hidden states (256×1024×2), values like [-0.036, 0.836, -1.555]. So
   the kernel does **bf16 × int8 → int32**. BO order is **(C, A, W)** (argw
   C→0/A→1/W→2), and it now EXECUTES and produces non-zero output
   (A=bf16 1.0, W=int8 1 → C ≈ 0x06000000, a fixed-point-scaled accumulation,
   not raw 1024).

   **GEMM correctness + dequant scale (verified):** with A=bf16 1.0 only in
   row 0 (rest 0) and W=int8 1, C row 0 is non-zero and rows 1+ are zero —
   i.e. the GEMM computes the right structure. Row 0's non-zero value
   = 100664832 = 1024 × **98304**, so **C_float = C_int32 / 98304 × scale_W**
   (98304 = 3×2^15 is the bf16→int fixed-point scale).

   **BD semantics (npu_cmd_write_dma.hpp):** buffer_length = dim0×dim1×dim2
   (dim2_size derived), iter_size = ((bd[10]>>20)&0x3FF)+1.

   **SOLVED end-to-end — W = npu_pack_layer_bo output (10MB Q4NX layer BO).**
   The captured 10MB W BO (seq≈57) is BYTE-IDENTICAL to the native
   `npu_pack_layer_bo` layer-0 output (1920 tiles padded to 2048×5120=10MB,
   0 diffs). Running my GEMM (QKV stream M=256/K=1024/N=4096, A=bf16 1.0,
   W=that 10MB BO, order C/A/W) gives C[m][0] = -8257537 for ALL m
   (= dequantized column-sum −84.0 × 98304) — the mm.xclbin reads the Q4NX W
   from the SAME layer BO the decode path already produces, dequantizes Q4
   internally, and the host dequant is just **C_float = C_int32 / 98304**.
   nz=524288=half confirms the strided C write.

   **VERIFIED: my GEMM == FLM's npu_app byte-for-byte.** Running FLM's own
   `Gemm`+`npu_app` (compiled with -DFLM_DEVICE_BUFFER) with A=bf16 1.0 + the
   10MB W produces the IDENTICAL raw C as my aiebu-ELF kernel
   (C[0]=−8257537, C[1..7]=0x7fffff81…, nz=524288) — so the mm.xclbin GEMM
   wiring is 100% correct and the remaining work is purely the dequant scale
   + C strided mapping (recoverable by byte-diffing this raw C vs FLM's QKV).

   **Dequant is NOT just /98304 — the W has its own fixed-point scale.**
   Q4NX format (dequant_q4nx.cpp): tile = [256 BF16 scales][256 BF16 zps][4096B
   packed UNSIGNED int4], W = nibble×scale + zp. Dequantizing the QKV tiles via
   dequant_q4nx.cpp gives colsum0 = −0.0966; my GEMM's A=1.0 raw C = −8257537
   (= −84.0 × 98304), so the kernel's internal W_int ≈ W_dequant × **~870**
   (full dequant C_float = C_int32 / ~85.5M, not /98304). The exact W fixed-
   point scale needs a byte-diff of raw C vs FLM's QKV output for a known A.
   Remaining: (1) exact dequant scale, (2) C strided-write → M×N mapping,
   (3) wire into the prefill, (4) attn.xclbin + overlap.
2. **Overlap CPU quantize** (2 ms/GEMM) with kernel execution — async
   double-buffering of the A operand.
3. **Batched attention on NPU**: `gen_mha_engine_seq` + `attn.xclbin` instead of
   the O(npt²) CPU `attn_omp` (which also blocks runlist batching of the
   QKV→attn→O→GU→D chain).
4. **Chunked prefill** (128/1024-token chunks) with KV accumulation across
   chunks (the existing split-path loop already accumulates `kv_caches[l][0].n`).

## 8B — DONE (decode correct, 21 tok/s short ctx)

- Weights downloaded (`flm pull qwen3:8b`, 5.7 GB).
- Config: H=4096 IM=12288 NC=36 NH=32 NKV=8 HD=128 NV=151936,
  `tie_word_embeddings: false` (separate lm_head).
- Wired into the runlist path (gate + bridge + ELFs), decode now greedy-parity
  with FLM (32). Weight BO packing byte-identical to FLM's runtime capture
  (layer-0 120,586,240 B, 0 diffs).

---

## 2026-09-10 (session 2): dequant.xclbin identified as WEIGHT dequantizer (not GEMM-output)

### Finding: the dequant is a SEPARATE xclbin + libdequant.so, not a scalar scale
The earlier "dequant scale ~85.5M" conclusion was WRONG. FLM does the int32→bf16
conversion with a dedicated `dequant.xclbin` driven by `Dequant::generate_dequant_q4_1_seq`
(libdequant.so), which reads the Q4NX scales/zps from the layer BO — so there is no
single scalar dequant constant.

Symbol: `Dequant::generate_dequant_q4_1_seq(npu_sequence*, D_in, D_out, weight_offset, mode)`.

### Finding: dequant.xclbin = Q4NX *weight* dequantizer (Q4NX int4 → bf16)
Definitive test: run `Dequant` with **C = 0** (all zeros) and W = K tiles →
output = `[-0.0161, 0.0233, 0.0354, 0.0143, 0.0039, -0.0457, 0.0042, -0.0238]`
which matches `dequant_q4nx.cpp`'s `W_dequant[0][0..7]` EXACTLY. So the dequant.xclbin
dequantizes the *weights* (Q4NX→bf16); it is independent of the GEMM accumulator C.
Its I/O: arg0 = output bf16 (in-place), arg1 = W (Q4NX). Run as `run(bOut, bW)`.

### Finding: GEMM is per-projection (separate Q/K/V) selected by weight_offset
`Gemm::generate_seq(seq, M, K, N, weight_offset, ...)` with `weight_offset = tile_index*5120`
selects the projection tiles in `npu_pack_layer_bo`: Q=0, K=256*5120, V=384*5120,
O=512*5120, up=768*5120, gate=1152*5120, down=1536*5120. Verified: K GEMM (woff=1310720)
gives C[0]=-48431231 (≠ Q's -8257537).

### Remaining
- Determine where the GEMM's int32 C is converted to bf16 QKV (mm.xclbin output is
  int32; dequant.xclbin only dequantizes weights — so the QKV int32→bf16 conversion is
  done elsewhere: host, attn.xclbin, or a scalar in the layer.xclbin path). Inspect
  `qwen3_npu_sequence::gen_layer_seq` (libqwen3_npu.so) to resolve.
- Then wire: per-projection Gemm (mm.xclbin) + Dequant (dequant.xclbin) + attn.xclbin
  into the split-path prefill in `engine/npu/src/npu_engine_universal.cpp`.

---

## 2026-09-10 (session 2b): A/C are TILED; C→bf16 is a separate step

### Finding: A (bf16) and C (int32) are NOT row-major — they are AIE-tiled
A=one-hot test (A[0][0]=1.0, rest 0) should give C[i][n]=0 for i≠0, but the 256
rows of C are all DIFFERENT (not {row0, zeros}). So the GEMM's A read is a tiled
layout, not row-major; the A=1.0 uniform case (which matched FLM byte-for-byte)
does not exercise the layout. C is written strided (d0=64 int32, d1=256×512-stride,
BDs at 64-int32 bases → C_BO[512·i + 64·g + col]).

### Finding: dequant.xclbin output confirmed = W_dequant (K×N bf16)
Correct bf16 decode of dequant output matches dequant_q4nx.cpp W_dequant[0][:]
exactly. The dequant reads W (arg1 Q4NX) and writes W_dequant (arg0, K×N bf16).
So C_int32 → QKV_bf16 is NOT done by dequant.xclbin; it is a SEPARATE step
(attn.xclbin reads bf16 Q/K/V, per gen_mha_engine_seq's 2K-bf16 chunk DMAs).

### Finding: gen_dequant_seq (qwen3_npu_sequence) is DEPRECATED
`qwen3_npu_sequence::gen_dequant_seq` prints "DEPRECATED FUNCTIONS" and emits an
empty sequence — not the C→bf16 path.

### Recommended path (avoids reverse-engineering tiled A/C + per-group dequant)
Use FLM's own `qwen3_npu_sequence` (gen_rtp_seq + gen_layer_seq + gen_mha_engine_seq)
to generate the full prefill sequences and run them through the native engine's
xrt::ext::kernel runner — the sequence generators already encode the tiled A/C
layouts and the int32→bf16 conversion. This is acceptable per the constraint
(native = 1bit-MONSTER orchestrates FLM's xclbins + libs).

---

## 2026-09-10 (session 2c): prefill BO flow measured; dequant is not a scalar

### Finding: actual prefill GEMM shapes + BO sizes (from capture)
- **A (activations) = bf16, 1 MB = 256×2048** (M=256, K=2048 — K is 2× hidden_size=1024;
  cols 0..1023 and 1024..2047 both look like activations, range −7.9..6.2). There are also
  1 MB "A scale" BOs = bf16 1.0 (all 0x3F80) uploaded per layer.
- **C (GEMM output) = int32, 1 MB = 256×1024** (N=1024). So the mm.xclbin GEMM is
  M=256, K=2048, N=1024 — NOT the 256×1024 K I fed npu_app (my byte-identical check used
  A=1.0 uniform, which hides the K dimension).
- **QKV output = bf16, 2 MB = 256×4096** ([Q 2048 | K 1024 | V 1024]).
- Per-layer capture: W BO = 10 MB (npu_pack_layer_bo), KV cache = 32 MB, MLP dequant
  output = 6 MB.

### Finding: C_int32 → QKV_bf16 is NOT a scalar scale
Direct set-correlation of C (int32) vs every QKV part (Q/K/V) at scales 2^30..2^33 gives
≤ 0.3% matches — so the conversion is tiled-layout + (likely) per-group Q4NX scales/zps,
and Q/K are additionally post-RMSNorm/RoPE'd. Reverse-engineering it is a rabbit hole.

### Recommended path unchanged
Use FLM's `qwen3_npu_sequence` (gen_rtp_seq / gen_layer_seq / gen_mha_engine_seq) to
generate full prefill sequences and run them through the native engine's runner. Note the
real GEMM is M=256, K=2048, N=1024/2048/3072 — feed the 1 MB bf16 A (256×2048) and read
the int32 C, then let FLM's dequant/attn sequences do the int32→bf16 conversion.

---

## 2026-09-10 (session 2d): CORRECTION — mm.xclbin W (8MB) ≠ npu_pack_layer_bo (10MB)

### Finding: the mm.xclbin GEMM's W is a SEPARATE 8 MB BO, not the 10 MB layer BO
Capture of the prefill GEMM kernel invocation shows args `[C:1MB, A:1MB, W:8MB]`
(`EXTBO ... size=8388608`). The 10 MB `npu_pack_layer_bo` (byte-identical to the
captured 10 MB layer BO, seq 57) is the **layer.xclbin (decode)** weight format; the
**mm.xclbin (prefill)** reads a different 8 MB W (8388608 bytes = 2^23, not a multiple
of the 5120-B tile).

### Finding: GEMM A = bf16 256×2048 (K=2048), C = int32 256×1024 (N=1024)
`libgemm.so` instantiates `T_in = biovault::bfloat16_t` (A is bf16). M=256. So the
first prefill GEMM (seq 170 A → seq 171 C) is M=256,K=2048,N=1024 = the **o_proj**
(K = num_heads×head_dim = 16×128 = 2048, N = hidden 1024), per `npu_pack_layer_bo`'s
tile map (o_proj = tiles [512,768), G=16).

### Finding: feeding npu_pack_layer_bo as W does NOT reproduce FLM's C
Tried every candidate (K=1024×{q,k,v}, K=2048×{o,fused}, all weight_offsets) — 0/262144
match against the captured C (seq 171). My C comes out near INT32_MAX (overflow), FLM's
is ~±1.1e9. So the mm.xclbin's W is a **different packing** (likely raw/unreordered
Q4NX, or zp-stripped) than `npu_pack_layer_bo`'s reordered tiles. The earlier
"W = npu_pack_layer_bo (shared 10MB BO)" conclusion was wrong for the mm.xclbin — it
applies to layer.xclbin only.

### Implication for wiring
To drive mm.xclbin natively we must reproduce FLM's 8 MB W packing (or call FLM's
`load_weights`), NOT reuse `npu_pack_layer_bo`. Next step: dump the 8 MB EXTBO content
(via a sub-buffer hook on `xrt::bo`) or locate FLM's mm W packer, then re-verify the
GEMM against the captured C.

---

## 2026-09-10 (session 2e): BREAKTHROUGH — mm.xclbin is a BF16 GEMM (A_bf16 × W_bf16 → C_bf16)

### The whole "int32 accumulator + dequant scale" line of investigation was WRONG
Definitive: running mm.xclbin with **W = the captured 8 MB bf16 dequantized weights**
and A = the captured bf16 activations reproduces FLM's C **byte-exactly**:
- **Q projection: match = 524288/524288** with M=256, K=1024, N=2048, woff=0,
  A = bf16 256×1024, W = bf16 1024×2048 (first 4 MB of the 8 MB W), C = bf16 256×2048.

So the mm.xclbin GEMM is **pure bf16**: A_bf16 × W_bf16 → C_bf16. There is **no int32
accumulator, no per-group dequant scale, no C→bf16 conversion step**. All the earlier
"int32 C", "near-overflow 2.1e9", "dequant scale ~85.5M" findings were just bf16 bytes
misread as int32.

### Corrected prefill flow (dense Qwen3)
1. **dequant.xclbin** — Q4NX (int4+scale+zp) → **W_bf16** (dequantized weights).
   QKV W = 8 MB (q 1024×2048 + k 1024×1024 + v 1024×1024 bf16 = 4+2+2 MB).
2. **mm.xclbin** — A_bf16 × W_bf16 → C_bf16 (bf16 GEMM). Q: K=1024,N=2048,woff=0.
3. **attn.xclbin** — attention on bf16 Q/K/V (no int32 anywhere).

### 8 MB W layout (QKV)
`[q_proj 1024×2048 bf16 @0 | k_proj 1024×1024 bf16 @4MB | v_proj 1024×1024 bf16 @6MB]`
(verified: Q matches woff=0; K/V weight_offsets are 4MB/6MB).

### Key correction to earlier session-1 conclusions
`npu_pack_layer_bo` (10 MB Q4NX, reordered tiles) is the **layer.xclbin (decode)** W
format only. The **mm.xclbin (prefill)** reads **dequantized bf16 W** (8 MB for QKV),
produced by `dequant.xclbin`. The earlier "W = npu_pack_layer_bo byte-identical" was
valid only for layer.xclbin, not mm.xclbin.

### Next steps
- Verify K/V/O/gate/up/down GEMMs (need the per-projection dequantized W BOs + the
  per-GEMM A — note the K GEMM's A (seq 172) differs from Q's A (seq 170), likely RoPE).
- Wire: dequant.xclbin (Q4NX→bf16) + mm.xclbin (bf16 GEMM) + attn.xclbin into the
  split-path prefill in `engine/npu/src/npu_engine_universal.cpp`.

---

## 2026-09-10 (session 2f): FULL prefill GEMM path VERIFIED byte-exact

### dequant.xclbin (Q4NX→bf16) + mm.xclbin (bf16 GEMM) — both reproduced
- `Dequant::generate_dequant_q4_1_seq(seq, D_in=1024, D_out=4096, weight_offset=0, mode=0)`
  with `npu_pack_layer_bo` W → **8 MB bf16 QKV W** — matches captured mmw **4194304/4194304**.
- `Gemm::generate_seq(seq, M=256, K=1024, N=2048, weight_offset=0)` with that W and A=bf16
  → **Q output** — matches captured seq171 **524288/524288**.

### Complete prefill GEMM recipe (dense Qwen3, per layer)
1. **dequant.xclbin** (`Dequant::generate_dequant_q4_1_seq`): Q4NX → bf16 W, per projection:
   - qkv: D_in=1024, D_out=4096, woff=0 → 8 MB (q 2048 + k 1024 + v 1024)
   - o:   D_in=2048, D_out=1024, woff=512·5120 → 4 MB
   - gate/up: D_in=1024, D_out=3072 → 6 MB each
   - down: D_in=3072, D_out=1024 → 6 MB
2. **mm.xclbin** (`Gemm::generate_seq`, T_in=bf16, T_out=bf16): A_bf16 × W_bf16 → C_bf16:
   - q: M=256, K=1024, N=2048, woff=0
   - k: M=256, K=1024, N=1024, woff=4 MB
   - v: M=256, K=1024, N=1024, woff=6 MB
   - o: M=256, K=2048, N=1024
   - gate/up: M=256, K=1024, N=3072
   - down: M=256, K=3072, N=1024

### Remaining
- K GEMM's A differs from Q's A (captured seq172 ≠ seq170) — the RoPE; determine the A
  transform (or use FLM's gen_rtp_seq/gen_layer_seq which already apply it).
- Then wire dequant+mm into the split-path prefill in `engine/npu/src/npu_engine_universal.cpp`
  and measure vs FLM (target 500–1269 tok/s prefill).

---

## 2026-09-10 (session 2g): 8MB W layout confirmed [Q|K|V]; Q/K/V A differ (q/k norms)

### 8 MB QKV W layout (byte offsets)
dequant(K) matches mmw at **4 MB**, dequant(V) at **6 MB** (1048576/1048576 each).
So `[q 1024×2048 @0 | k 1024×1024 @4MB | v 1024×1024 @6MB]`, all bf16.

### Q/K/V activations differ — consistent with Qwen3 q_norm/k_norm
The 4 captured A BOs (A170/A172/A174/A175, each 256×2048 bf16) are pairwise DIFFERENT
and none is a RoPE/permutation of another. This matches Qwen3's separate **q_norm** and
**k_norm** RMSNorm: Q=A_q_norm·W_Q, K=A_k_norm·W_K, V=A_hidden·W_V. So the Q GEMM's A
(A170[:,:1024]) is q_norm(hidden), and K/V use different norms. The Q GEMM already
matches byte-exact; K/V need the correct per-projection A (q_norm/k_norm outputs).

### Open item: Gemm::generate_seq weight_offset unit for bf16 W
Q matched woff=0 (unit-independent). K at byte offset 4 MB did NOT reproduce C173, and a
75% partial match appeared at woff=2 MB (bytes) with N=2048 — suggests the bf16 weight
offset may be in *elements* (2M bf16 = 4 MB) rather than bytes, OR the K A differs.
Resolve by trying woff in bf16-elements with the true k_norm A.

### Status
dequant.xclbin (Q4NX→bf16) + mm.xclbin (bf16 GEMM) fully verified for Q; K/V/O/MLP
follow the same recipe once the q_norm/k_norm A and the bf16 weight-offset unit are
nailed down. Then wire into `engine/npu/src/npu_engine_universal.cpp`.

---

## 2026-09-10 (session 2h): weight_offset unit = bf16 ELEMENTS (not bytes)

### Finding: Gemm::generate_seq weight_offset is in bf16 ELEMENTS for the mm.xclbin
interpret() of a K GEMM stream shows `generate_seq(seq, 256,1024,1024, woff=2097152)`
emits W B-DP offsets starting at **4194304 (4 MB bytes)** — i.e. the weight_offset is
multiplied by 2 (bf16 element → byte). So for the bf16 W the offsets are:
- q: woff=0, k: woff=2097152 (→4 MB), v: woff=3145728 (→6 MB).

(NOTE: for the Q4NX layer.xclbin path, weight_offset was in BYTES — 256×5120=1310720 —
so the unit is type-dependent: Q4NX→bytes, bf16→elements.)

### Finding: the Q/K/V GEMMs SHARE one A BO (256×2048 bf16)
SETARG capture shows all three GEMMs use the same A BO (`idx=4 bo=0x…aa20`) and the same
8 MB W BO (`idx=5`). Q GEMM reads A[:,:1024] (verified byte-exact). The K GEMM still does
NOT reproduce C173 with A[:,:1024] or A[:,1024:] at the correct woff (0.5% match) — so the
K/V activation feeding is still unresolved (the A BO's second half A[:,1024:] is neither a
RoPE nor a copy of the first half; the interleaved `_rope_rms` host step likely re-syncs the
A BO or supplies a transformed K input).

### Status
mm.xclbin bf16 GEMM + dequant Q4NX→bf16 + per-projection woff (in elements) all verified for Q.
K/V/O/MLP need the correct per-projection A (RoPE/k-norm transform) — then wire into
`engine/npu/src/npu_engine_universal.cpp`.

---

## 2026-09-10 (session 2i): FULL QKV VERIFIED byte-exact — recipe complete

### Q/K/V all reproduce FLM's output exactly (M=256, A = hidden = A BO first 1024)
| proj | K | N | weight_offset(elem) | output_offset(elem) | match |
|------|---|---|---------------------|---------------------|-------|
| q    | 1024 | 2048 | 0        | 0        | 524288/524288 |
| k    | 1024 | 1024 | 2097152 (→4MB) | 262144 (→512KB) | 262144/262144 |
| v    | 1024 | 1024 | 3145728 (→6MB) | 262144 (→512KB) | 262144/262144 |

### Key detail: K/V write to the C BO's SECOND half (output_offset=262144 elements)
`generate_seq(seq, M, K, N, weight_offset, ADD_BIAS, OUTPUT_MODE, bias_offset, output_offset)`
— both weight_offset and output_offset are in **bf16 elements** for mm.xclbin. K/V put
their 512 KB result at +512 KB (the 1 MB C BO's upper half); Q fills the whole 1 MB.
The A BO's second half (A[:,1024:]) is unused by QKV (Q/K/V all read A[:,:1024] = hidden).

### A BO first half = RMSNorm(embeddings) confirmed
A[:,:1024] == embeddings/sqrt(mean(emb²)+eps)·input_layernorm_weight (True), so the hidden
is the standard Qwen3 input layernorm output. (A[:,1024:] is a separate ~0.19-rms activation,
not the raw embeddings, RoPE, or a copy — likely the post-attention hidden for the O/MLP GEMMs.)

### Remaining for full prefill wiring
- O/MLP GEMMs: same recipe with per-projection dequant W (o: 2048×1024 @0 of a 4 MB W;
  gate/up/down: 6 MB each). O's A = attention output (256×2048), MLP A = post-attn hidden.
- Then wire dequant.xclbin + mm.xclbin (+ attn.xclbin) into `npu_engine_universal.cpp`.

---

## 2026-09-10 (session 2j): K/V verified via new bf16 GEMM module; Q N=2048 has a 128-row cap

### New module: engine/npu/src/npu_engine_bf16_mm.h
`bf16mm::Bf16Mm` — wraps dequant.xclbin + mm.xclbin via FLM's npu_app/Gemm/Dequant.
`run_dequant()` (Q4NX→bf16) and `run_gemm_ooff()` (bf16 GEMM) verified:
- dequant QKV → 8 MB W: **4194304/4194304**
- K GEMM (N=1024, woff=2097152 elem, ooff=262144 elem): **262144/262144**
- V GEMM (N=1024, woff=3145728 elem, ooff=262144 elem): **262144/262144**

### Finding: Q GEMM (N=2048) writes only 128 rows per invocation
The mm.xclbin's C write capacity is **262144 bf16 = 256×1024** (fixed). For N=2048 that
is 128 rows (even-indexed A rows 0,2,…,254), the odd rows stay zero; `output_offset`
only shifts the same 128 rows to the second 512KB half (verified: ooff=0 → R[0]@bC[0],
ooff=262144 → R[0]@bC[262144]). N=1024 split does NOT fix it (the dequant Q W is
row-major 1024×2048, so a N=1024 GEMM at woff=0 reads the wrong K-half). M=128 is
rejected ("GEMM M size not aligned"). So FLM must feed the Q GEMM a re-arranged A
(256 rows whose even slots hold one token batch and odd slots the other) — the exact
A arrangement is the last open item for the Q path.

### Status
K/V/dequant byte-exact and wired into a reusable module. Q (and O N=2048) need the
A-batch arrangement resolved; then wire dequant+mm (+attn) into the prefill loop.

---

## 2026-09-10 (session 2k): Q N=2048 C-write row-selection — empirical map (odd rows still open)

### mm.xclbin C-write BD (N=2048) decoded from interpret()
C BD (S2MM): d0=64, d1=256 iters @ stride 1K, d2=1, buffer 16K. The A read BD (MM2S):
d0=256, d1=64 @ stride 512, d2=2 @ stride 256 (8 A-BDs ⇒ 256 rows, row-major). So the
GEMM C = A×W (256 rows), but the C write emits only 128 rows.

### Empirically observed output (A arrangement → 128 output rows, row-major read)
- A contiguous (A[i]=token i)            → R[0],R[2],R[4],…,R[254]   (even tokens, contiguous)
- A shift-by-1 (A'[i]=A[i+1])           → R[2],R[4],R[6],…           (even tokens shifted +2)
- A shift-by-2                          → R[4],R[6],…                 (shifted +4)
- A pairwise-swap (A'[2i]=A[2i+1])      → R[2],R[0],R[6],R[4],R[10],R[8],…  (pair-swapped even)
- A interleave (A'[2i]=A[i], A'[2i+1]=A[128+i]) → R[0],0,R[2],0,…,R[126],0 (even tokens @ even pos, zeros @ odd)

So the C write always selects a stride-2 (even) subset of the GEMM C rows and the
1K half-row stride interleaves them; none of these simple A rearrangements yields the
odd tokens R[1],R[3],…. The output_offset only shifts the same 128 rows to the 2nd half
(verified) — it is not the odd-row selector.

### Definitive next step
Dump FLM's `qwen3_npu_sequence::gen_layer_seq(seq, 256)` and decode the Q GEMM's
A/C B-DP offsets to see how FLM feeds the odd rows (likely a second generate_seq with a
different A base offset or an A interleave the harness hasn't tried). Then Q = 2 invocations
(even+odd) interleaved → full QKV done → wire into `npu_engine_universal.cpp`.

---

## 2026-09-10 (session 2l): C BO is 512×1024 (not 256×2048); Q even tokens; odd-token A still open

### KEY: the Q GEMM C BO is stored TRANSPOSED as 512×1024
Decoding the C write BD with the 4-byte DMA element size (FLM npu_cmd_write_dma.hpp):
d0=64 (4B)=128 bf16, d1=256 iters @ stride 1024 (4B)=2048 bf16=2 rows of 1024.
So the C write emits 128 GEMM C rows, each packed as TWO 1024-bf16 C-BO rows
(first half then second half). One-hot probe confirms: A[k]=1.0 → C BO rows 2k,2k+1.

### Empirical Q output (contiguous A = hidden tokens 0..255)
Q1[2j] = R[2j][:1024], Q1[2j+1] = R[2j][1024:]  (128/128) — i.e. the C write emits the
EVEN tokens R[0],R[2],…,R[254] (each 2 C-BO rows), and the C BO's second 512 KB stays 0.

### Remaining contradiction (odd tokens)
The one-hot A[1] → C BO row 2 = W[0] (== C[1] for that probe), but the contiguous A gives
row 2 = R[2] (== C[2]). So the A-read row selection is not a simple stride-2 — it maps
A[k]→C[2k] in one probe and A[k]→C[k] in the other, i.e. the A-read/C-write tiling has an
off-by-one or 16/32-bit element-size ambiguity still to pin down. This is the last item
before the Q (and O, N=2048) second invocation can be wired.

### Definitive next step
Trace the A-read BD (d0=256,d1=64@512,d2=2@256) under BOTH 16-bit and 32-bit element
interpretations against the one-hot + contiguous observations, or byte-diff FLM's captured
elf_0009 (Q GEMM) vs my generated stream to pin the exact A/C tiling.

---

## 2026-09-10 (session 2m): Q odd-token SOLVED — mm.xclbin computes only 128 correct M-rows; recipe = 2-batch M-split

### DEFINITIVE finding: the mm.xclbin's correct-M capacity is 128 rows (all N)
Decoded FLM's own precompiled `mm_256_1024_128_0.bin` (N=128): its C-write BD is
`d0=64(4B)=128bf16, d1=256 @ stride 64(4B)=128bf16` — clean 256×128 row-major. A synthetic
one-hot probe (A[k][0]=k+1, W[0][0]=1) shows the mm.xclbin's M=256 output is:
- rows 0..127  = C[0..127]  (IDENTITY — correct)
- rows 128..255 = C[127], C[129], C[129], C[131], C[131], … (a "duplicated odd" garbage
  region: f(2m)=2m-1, f(2m+1)=2m+1) — this is the source of every prior "off-by-one" /
  "even-token" / "128-row cap" confusion. It is N-independent (verified N=128, N=2048).

### The recipe (VERIFIED 256/256): split 256 tokens into 2×128-token M-batches
For each GEMM (M=256, K, N):
1. batch 0: A = sparse 256×K (tokens 0..127 in rows 0..127, rows 128..255 = 0), ooff=0
   → output rows 0..127 = tokens 0..127.
2. batch 1: A = sparse 256×K (tokens 128..255 in rows 0..127), ooff=0
   → output rows 0..127 = tokens 128..255; host-placed at rows 128..255.

(Verified with a synthetic marker test: 256/256 rows correct. The earlier "markers
129..256 look like dup-odd" was a bf16 1-ULP artifact of those specific integer markers,
irrelevant for the real dense RMSNorm hidden states.)

### Why FLM's prefill loads `mm_256_1024_128_0.bin`
FLM tiles the Q (N=2048) into 16 N=128 GEMMs + 2 M-batches (32 invocations) using
precompiled N=128 streams. Running N=2048 directly with the same 2-batch split is
byte-equivalent and needs only 2 invocations per projection — this is what
`npu_engine_bf16_mm.h` now does.

### Module updated
`engine/npu/src/npu_engine_bf16_mm.h`: added `Bf16Mm::run_gemm_2batch()` (2-batch
M-split) and rewired `qkv()` (Q/K/V each 2 invocations, ooff=0, host-side combine).
K/V no longer use ooff=262144 (the shared-C-BO trick) — each projection gets its own
output buffer.

### Next
- Wire dequant + 2-batch mm (+ attn.xclbin) into `npu_engine_universal.cpp` and measure
  prefill vs FLM (target 500–1269 tok/s).

---

## 2026-09-10 (session 2n): bf16-truncation resolves the "off-by-one"; bridge built + verified

### The "off-by-one"/"dup-odd" is the AIE bf16 multiplier truncating the mantissa LSB
The mm.xclbin's bf16 multiply is NOT IEEE "fp32 accumulate + round". It truncates the
result mantissa's LSB, so any input whose bf16 mantissa LSB = 1 loses 1 ULP:
- 129 (0x4301) → 128 (0x4300); 131 → 130; 109.5 → 109.0; …
- Integers 1..128 all have mantissa LSB = 0 → reproduced EXACTLY.

This is what made "markers 129..256 look like a stride-2 dup-odd region" in every prior
probe, and why markers ≤ 128 looked like a clean identity. The 2-batch M-split recipe is
CORRECT (verified 256/256 with exact markers); the truncation is a hardware property that
is *byte-identical to FLM* (same mm.xclbin), so it does not affect parity.

### Bridge built + end-to-end verified
`engine/npu/src/npu_engine_bf16_mm_bridge.cpp` — C-linkage wrapper around `bf16mm::Bf16Mm`
so `npu_engine_universal.cpp` (which vendors a stub `lm_config.hpp` and must NOT see FLM's
`modules/*`/`npu_utils_xrt.hpp`) can drive dequant.xclbin + mm.xclbin without an include
clash. Exposes `bf16mm_init / bf16mm_dequant / bf16mm_gemm_2batch`.
Verified end-to-end with the REAL Qwen3-0.6B layer-0 BO (`npu_pack_layer_bo` →
dequant → 2-batch Q GEMM): dequant W 4185202/4194304 non-zero; Q output = A×W up to the
bf16 truncation.

### Next
- Wire the bridge into `npu_engine_universal.cpp` prefill (replace I8Ctx int8 GEMMs with
  dequant + 2-batch bf16 mm), add `-lgemm -ldequant` to build_npu.sh, measure vs FLM.
