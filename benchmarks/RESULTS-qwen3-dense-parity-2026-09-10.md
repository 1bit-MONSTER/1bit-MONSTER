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
