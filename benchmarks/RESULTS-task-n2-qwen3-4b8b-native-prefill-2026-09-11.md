# RESULTS — task-n2: dense Qwen3 4B/8B native bf16 prefill

Goal `mttxt22c-a6rv75`, task-n2. Native bf16 prefill (`NPU_PREFILL_BF16=1` +
`NPU_RUNLIST=0`) for Qwen3-4B/8B, using the already-embedded `attn_mha_256_nh32.elf`
(NH=32) + the task-n1 optimizations (GEMM fusion, GU concat, async 2-batch, pipeline).

## Status: 4B/8B native prefill RUNS, but 4B/8B token mismatch — nh32 ELF bug (open)

| model | H / IM | 256-tok prefill | tok/s | FLM published @1k | gap |
|---|---|---|---|---|---|
| Qwen3-0.6B | 1024 / 3072 | ~450 ms | ~570 | 1494 | 2.6× |
| Qwen3-4B | 2560 / 9728 | 1763 ms | 145 | 509 | 3.5× |
| Qwen3-8B | 4096 / 12288 | 1841 ms | 139 | 357 | 2.6× |

- **FLM prefill reference (NPU_FLM_PREFILL) boot = 151667** for 0.6B, 1.7B, 4B alike.
- 0.6B/1.7B (NH=16, nh16 ELF) → boot 151667 ✓ (byte-correct).
- 4B/8B (NH=32, nh32 ELF) → boot 115230/30955 ✗ — **wrong attention output**.
- **Root cause:** two bugs found — (1) host `run_dequant` hardcoded a 10 MB layer-BO
  copy, so the GU dequant read garbage for 4B (63 MB) / 8B (82 MB) — FIXED (copy
  per-projection tiles). (2) the captured `attn_mha_256_nh32.elf` itself produces
  wrong attention (commit 6dce200e8 mis-attributed the 21894-vs-220 divergence to
  "prefill-vs-decode mismatch"; the real bar is FLM prefill 151667). Needs re-capture
  of the nh32 ELF from FLM's 4B prefill + verification.
- 1.7B (H=2048) reuses the NH=16 ELF and the 0.6B recipe; not re-measured here.

## Notes

- The NH=32 attention ELF was already embedded (`engine/npu/xclbins/attn_mha_256_nh32.elf`)
  and `run_attn` already selects it via `attn_qout==4096` — no new ELF capture was needed.
- The 4B/8B gap is the same structural finding as task-n1 (host f32↔bf16 conversions
  + RMSNorm/RoPE/SiLU); the per-op native prefill is bounded ~2.6–3.5× short of FLM.
- **Not done:** position-shifted attention ELFs for >256-token chunks (the fixed
  `attn_mha_256_*.elf` bakes RoPE positions [0,256), so chunked prefill >256 tokens
  needs a shifted ELF per chunk). The current path caps npt at 256.


## nh32 ELF re-capture status (2026-09-11)

Attempted re-capture via `run_qwen3_prefill ~/.config/flm/models/Qwen3-4B-NPU2`
under `cap_interposer.so` — **fails**: `MHA parameter check failed: L_begin: 0,
L_end: 0` (the 4B prefill's attention is called with an empty context; the tool
was validated only for 0.6B). The engine's own `NPU_FLM_PREFILL=1` path runs the
4B prefill fine (boot=151667), so the correct next step is to LD_PRELOAD the
interposer on the ENGINE's FLM-prefill path for 4B and capture the nh32 ELF +
act/kv/out there, then replay byte-exact via `replay_attn`.


## nh32 ELF: raw capture swapped in, but 4B boot still wrong (2026-09-11)

- The embedded `attn_mha_256_nh32.elf` (46640 B) was a 32-byte **mis-trim** of the
  raw FLM capture; `elf_0012` (46672 B) from the engine's `NPU_FLM_PREFILL=1` 4B
  capture is the raw FLM MHA ELF (boot=151667 source). Swapped in.
- 4B boot = 2561 (vs FLM 151667), 8B = 30955. Ruled out the **GU large-N** hypothesis:
  splitting gate/up into two N=IM GEMMs (with woff=H·IM) gives the SAME 4B boot 2561,
  and the combined N=2·IM GU is byte-identical on 0.6B/1.7B — so N=19456 is fine.
  Remaining candidates: (a) **large-K** O GEMM (K=qout=4096) / D GEMM (K=IM=9728,
  vs 1.7B's verified K=6144), (b) nh32 attention act/kv layout, (c) NH=32 q_norm/RoPE.
  Needs byte-exact isolation (targeted re-capture of layer-0 QKV/O/D + replay).


## ISOLATED: the 4B bug is the NH=32 attention, not the GEMMs (2026-09-11)

- Forced CPU attention (`NPU_ATTN_CPU=1`) → 4B boot **151667 = FLM** ✓. So the
  QKV/O/GU/D GEMMs (incl. N=2·IM=19456 GU and K=IM=9728 D) and the q_norm/k_norm/RoPE
  are all **correct** for NH=32. The wrong token comes solely from the **nh32 attention
  ELF** (`attn_mha_256_nh32.elf`) or the act/kv layout it expects.
- Next: byte-exact `replay_attn` of the captured nh32 ELF against FLM's captured
  act/kv/out (256×4096 Q), to decide ELF-vs-act/kv-layout.


## Final: nh32 attention output = 0.04% match vs CPU (2026-09-11)

- Compared layer-0 attention output (bf16) between CPU (`attn_omp`) and NPU (`attn_mha_256_nh32.elf`)
  paths: **449/1048576 = 0.04% match** — the nh32 ELF produces a completely different
  attention result (CPU path is correct, boot=151667). Engine's act (Q) dump is a correct
  [token][head][dim] 256×4096 layout.
- Likely cause: the captured `elf_0012` (via the engine's `NPU_FLM_PREFILL=1`, which runs
  FLM's prefill as a **runlist**) is a runlist-format ELF, not a single attention-kernel ELF
  the engine's per-op path expects — or its Q/GQA read geometry mismatches the input.
- Resolution needs a byte-exact `replay_attn` (256×4096) against FLM's captured act/kv/out,
  or re-capturing the MHA ELF from `run_qwen3_prefill` (which failed for 4B with
  "MHA parameter check L_begin 0, L_end 0"). Left open.


## KV region stride bug FOUND + FIXED (2026-09-11)

- The attention KV cache region stride is **model-dependent** (measured from FLM's
  prefill captures): 8MB (MAX_L=8192) for H≤2048, 12MB (12288) for H=2560, 24MB
  (24576) for H=4096. The engine hardcoded 8MB — so 4B/8B had the KV in the wrong
  layout. Fixed via `bf16mm_set_attn_kv_region` + H-based `kv_region` (commit ffeabbeb4).
- 0.6B still boot=151667 ✓ (8MB unchanged). 4B boot 115230→116941, 8B→41119 — the KV
  fix moved the output but 4B/8B are **still wrong**, so a **second** attention bug
  remains: candidates are (a) the nh32 ELF's Q-read geometry (reads 16 not 32 heads —
  46640/26928 = 1.73×, not the ~2× a full Q-doubling predicts), (b) act layout.
  Needs byte-exact `replay_attn` (256×4096) against FLM's captured act/kv/out.


## nh32 ELF diagnosed: uniform attention (Q-read wrong) (2026-09-11)

- Byte-exact replay of `attn_mha_256_nh32.elf` with FLM's captured act/kv (via an
  adapted cap_attnio for 48MB KV + 2MB act): output is **uniform** — rms 0.0038,
  all heads ≈ 0x3c48 (0.0122) = the softmax degenerates to 1/256, i.e. the ELF reads
  ~zero Q. The GEMMs/norm/RoPE/KV/act-layout are all verified correct (CPU attn → 151667).
- Conclusion: the nh32 ELF's **Q-read BD geometry is wrong** (reads 16 heads / wrong
  strides instead of 32 × 4096-wide). The `run_qwen3_prefill` capture yields the same
  ELF, so either the capture tool missed the true 32-head MHA ELF, or the ELF must be
  regenerated from the 16-head ELF by doubling the Q-read BDs + token-stride (8192 B).
  This is the last dense-Qwen3 4B/8B item; it needs ELF decode/regeneration.


## Latest: arg-order ruled out; engine Q ≠ FLM Q (2026-09-11)

- Tested all 6 arg-order permutations (NPU_ATTN_ARGS) → none gives 151667; (out,act,kv)
  is correct. Reverted.
- Compared engine bActQ vs FLM's captured act (same 256-token prompt): **completely
  different** (0x3dfa… vs 0xb9ee…), yet the CPU-attention path with the same bqo gives
  the right token for the default prompt. So the remaining 4B bug is upstream of the
  attention ELF — a subtle QKV/q_norm/RoPE difference for NH=32 (256-token batch) —
  not the ELF/arg-order/KV. Needs a layer-0 QKV dump comparison vs FLM.


## Status (2026-09-11): 4B/8B NH=32 Q divergence — bisection pending

- engine Q (`bActQ`) ≠ FLM's captured act for the same 256-token prompt. All structural
  candidates ruled out by inspection: QKV N=6144, q_norm/k_norm weights (128-dim, per-head),
  RoPE (full HD rotary, position sp+pi), qkvn/qkv offsets, A-reuse cache (K-change + 64-sample
  guard). Next is a layer-0 dump of the engine's raw QKV GEMM output (`bC`, pre-norm) vs FLM's
  to bisect QKV-GEMM vs q_norm/RoPE — bounded but tedious.


## CORRECTION: QKV/norm/RoPE are CORRECT — bug is the ELF Q-read (2026-09-11)

- Ran engine 4B CPU-attention with the same 256-token prompt: boot=**738 = FLM** (GREEDY_NEXT 738).
  So the QKV GEMM, q_norm/k_norm, RoPE, and embedding are all **correct for NH=32**. The earlier
  "engine Q ≠ FLM Q" was an invalid comparison (the cap_attnio "act" idx4 was the wrong BO).
- The bug is therefore back to the **nh32 attention ELF's Q-read geometry**: the ELF reads ~zero Q
  (uniform softmax, rms 0.0038), so its Q-read BD token-stride/head-count is wrong. Fix = regenerate
  the nh32 ELF from the 16-head ELF (double Q-read BDs 64→128, token-stride 4096→8192 B), or decode
  + patch the ELF's BDs.


## nh32 ELF capture is the issue (2026-09-11)

- Decoded the two ELFs' .ctrltext TXN: nh16 = 6084 words, nh32 = 10628 words (1.747×).
  The BD opcode groups grow: opcode-12 160→288 (+128), opcode-8 64→129 (≈2×) — the Q/out
  BDs roughly double but the total is short of the ~1.89× a clean Q(64→128)+out(64→128)+KV
  (16 same) doubling predicts. Also, the `run_qwen3_prefill` capture shows the prefill runs a
  **5-BO fused-layer ABI**, not the 3-BO attention — so `elf_0012` may be a fused-layer/GEMM
  ELF, not the 3-BO MHA attention ELF the engine's `run_attn` expects.
- Root cause candidates: (a) the nh32 ELF is mis-captured (16-head or fused-layer), (b) its
  Q-read BD token-stride/head-count is wrong. Fix = capture the 3-BO nh32 MHA ELF correctly,
  or decode+patch the Q-read BDs.

## 2026-09-12: BD-level comparison — Q-read BDs doubled in COUNT but .0 length field NOT doubled

Re-confirmed the 4B bug: NPU attn (nh32 ELF) boot=116941, CPU attn boot=151667=FLM.
Decoded both ELFs' .ctrltext via aiebu-dump and diffed the DMA BD fields:

| BD field | nh16 (16 heads) | nh32 (32 heads) | verdict |
|---|---|---|---|
| BLOCKWRITE count | 160 | 288 | +128 ✓ doubled |
| .4 control | 128×0xc40003ff + 32×0xc80000ff | 256×0xc40007ff + 32×0xc80000ff | 0x3ff→0x7ff ✓ (len 1024→2048) |
| .0 (addr/low len) | 128×0x1000 + 16×0x2000 + 16×0x4000 | 256×0x1000 + 16×0x2000 + 16×0x4000 | **0x1000 NOT doubled** ⚠ |
| .5/.7 stride | 0x2000000 (32MB) | 0x2000000 (32MB) | unchanged |
| .3 | 0x4000000 (64MB) | 0x4000000 (64MB) | unchanged |

The 128→256 Q/out BDs doubled in count and the .4 control length (0x3ff→0x7ff),
but the .0 field (0x1000=4096B) was NOT doubled. If .0 is the per-BD buffer
length/stride for the Q read, the nh32 ELF still reads Q at the 16-head (4096B)
granularity — consistent with "reads ~zero Q → uniform softmax". This is a
concrete new lead for the decode+patch path, but confirming it needs the AIE2
DMA BD field semantics (not yet decoded) and a correct reference. Still
multi-day; not attempted this session.

## 2026-09-12 (cont.): empirical ELF-patch experiments — buffer_length matters but is not sufficient

Decoded the full BLOCKWRITE/DDR_PATCH field structure of both ELFs and ran two
binary patches on attn_mha_256_nh32.elf (rebuilt + tested 4B, reverted after):

1. **buffer_length (w4) 0x1000→0x2000** (256 Q/out BDs): boot 116941 → **32595**.
   Field is part of the Q-read geometry (output changes) but NOT sufficient.
2. **D1 stride 0xc40007ff→0xc40003ff** (nh16 value): boot unchanged (116941).
   D1 is not the binding field for the default prompt.

DDR_PATCH field analysis (the runtime buffer-address patches) shows the offsets
ARE correct: arg_idx histogram {0:64,1:64,2:32} (nh16) → {0:128,1:128,2:32} (nh32)
(Q/out BDs doubled, KV unchanged), and arg_off values double 262144→524288
(= 64 tokens × token_width, matching the 2× wider 32-head Q). So the addressing
is right; the residual bug is in the per-BD transfer geometry (buffer_length +
the D0/D1/D2 stride encodings), which needs the AIE2 DMA BD register spec to
decode. Not fixed; still multi-day.

## 2026-09-12 (cont.2): found the BD field spec + compared vs FLM's generated NH=32 sequence — patch still unresolved

Located the source of truth for the BD field layout:
`mlir-aie-main/include/aie/Runtime/TxnEncoding.h` (BLOCKWRITE = 4-word header +
8 data words) and `xaie2pgbl_params.h` (BD0_0 buffer_length[13:0] @0x1D000,
D0/D1/D2 stepsize/wrap @0x1D008..0x1D014). NPU2 = devGen 4 (STX/KRK).

Ran FLM's own `gen_mha_engine_seq(0,256)` for Qwen3-4B (NH=32) and compared its
Q/out BD fields to the captured ELF:

| field | nh16 | captured nh32 | FLM-generated nh32 |
|---|---|---|---|
| buffer_length | 0x1000 (4096) | 0x1000 | 0x1000 |
| w7 (D0 size<<20) | 0x4000000 (64) | 0x4000000 (64) | 0x10000000 (256) |
| w8 (D1) | 0xc40003ff (sz64,str1024) | 0xc40007ff (sz64,str2048) | 0xc10007ff (sz16,str2048) |

Three binary patches tested on the captured ELF (all reverted):
1. buffer_length 4096→8192 → boot 116941→32595 (changes, but not fixed)
2. D1 stride 0x7ff→0x3ff → no change
3. D0 size 64→256 + D1 size 64→16 (to match FLM-generated) → no change (116941)

The FLM-generated sequence has a DIFFERENT structure (320 BDs, 22792 words, vs
the captured 288 BDs / 10628 words) — consistent with the prior note that
gen_mha_engine_seq uses a different granularity than the runtime's fixed
256-token ELF — so its field values are a hint, not a drop-in. The captured
nh32 ELF's Q-read geometry remains the unresolved bug; needs the correct 3-BO
nh32 attention ELF (re-capture) or full BD decode. Still multi-day.

## 2026-09-12 (cont.3): correct NOC-module DMA BD layout located; captured nh32 differs from nh16 only in D1_STEPSIZE

The attention ELF uses the **NOC (shim) module** DMA BD, not the memory-module
layout I decoded earlier. Correct layout (`xaie2pgbl_params.h`, NOC_MODULE_DMA_BD0_*):

| BLOCKWRITE data word | register | fields |
|---|---|---|
| data[0] | 0x1D000 | BUFFER_LENGTH[31:0] |
| data[1] | 0x1D004 | BASE_ADDRESS_LOW (DDR_PATCH target) |
| data[2] | 0x1D008 | flags + BASE_ADDRESS_HIGH[15:0] |
| data[3] | 0x1D00C | D0_WRAP[29:20] \| D0_STEPSIZE[19:0] |
| data[4] | 0x1D010 | D1_WRAP[29:20] \| D1_STEPSIZE[19:0] (+burst[31:30]) |
| data[5] | 0x1D014 | SMID/cache \| D2_STEPSIZE[19:0] |
| data[6] | 0x1D018 | ITER_curr[31:26] \| ITER_wrap[25:20] \| ITER_stepsize[19:0] |

Decoded Q/out BD fields (correct layout):

| field | nh16 | captured nh32 | FLM-generated nh32 |
|---|---|---|---|
| BUFFER_LENGTH | 4096 | 4096 | 4096 |
| D0_WRAP | 4 | 4 | **256** |
| D1_WRAP | 64 | 64 | **16** |
| D1_STEPSIZE | 1024 | 2048 | 2048 |

So the captured nh32 changed ONLY D1_STEPSIZE (1024→2048) from nh16; the
FLM-generated sequence additionally changes D0_WRAP (4→256) and D1_WRAP
(64→16) — but it has a different BD count (320 vs 288), so those values are a
different ABI. Applying the FLM-generated D0_WRAP/D1_WRAP values to the captured
ELF (patch #3) produced no change, confirming the ABI mismatch rather than a
single-field bug. The captured nh32's Q-read geometry remains unresolved;
re-generating it correctly needs the exact NOC-DMA multi-dim semantics for the
3-BO attention ABI. Multi-day; not completed.
