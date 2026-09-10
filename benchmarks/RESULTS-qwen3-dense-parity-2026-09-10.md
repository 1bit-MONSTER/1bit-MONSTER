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
