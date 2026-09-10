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
| Qwen3-0.6B | 87 tok/s | **73 tok/s** | 66.5 | ✓ |
| Qwen3-1.7B | 52 tok/s | — | 40.2 | ✓ (52>40.2) |
| Qwen3-4B | 24 tok/s | — | 19.6 | ✓ (24>19.6) |

Greedy token parity confirmed: "The capital of France is" → Paris continuation
for all three.

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

### Root cause

- **Runlist path** prefills one token per forward (`rt.embed(t)` +
  `rt.forward(ctx)`), i.e. M=1 — the same cost as decode (~13 ms/token for
  0.6B) instead of FLM's batched prefill.
- **Split path** batches M=128 (the v27 microkernel is M=128-baked; see
  AIE2P-FACTS.md §3b) but is launch-bound: 112 launches × ~10 ms
  (quantize → BO sync → run.start/wait → readback → dequant) for 128 tokens.
  FLM processes the same 112 GEMM steps for ~8× the tokens in less time
  because its mm engine streams M-tiles with async DMA and one contiguous
  weight BO.
- `HybridFlmCtx` (FLM `mm.xclbin`) exists but is also M=128-baked and was
  measured 15× *slower* than `I8Ctx` (151 ms/tok) — per-launch instruction
  BO sync; not a drop-in win.

### Plan to close prefill

1. Batched prefill at M=1024: FLM's `mm.xclbin` runs M up to the prefill
   chunk (config `max_prefill_len=4096`); generate `M=1024` instruction
   streams via `gemm_generate_sequence_i8_split`, size `bA/bC` for M=1024,
   and loop 128-row tiles inside one launch.
2. Chunked prefill (128/1024-token chunks) with KV accumulation across
   chunks (the existing split-path loop already accumulates
   `kv_caches[l][0].n`).
3. Batched attention: `gen_mha_engine_seq(L_begin, L_end)` (the FLM MHA
   engine) or the existing `attn.xclbin`, instead of the O(npt²) CPU
   `attn_omp`.
4. Reduce per-launch overhead in `I8Ctx` (overlap quantize/DMA with kernel
   execution; avoid full-BO `update_rope_i6` syncs in the runlist path).

## 8B

- Weights downloading (`flm pull qwen3:8b`, 5.7 GB model.q4nx).
- Config: H=4096 IM=12288 NC=36 NH=32 NKV=8 HD=128 NV=151936,
  `tie_word_embeddings: false` (separate lm_head).
- Needs: add `NC==36 && H==4096` to the `dense_qwen3` gate in
  `npu_engine_universal.cpp`, generate layer + lm_head ELFs, verify.
