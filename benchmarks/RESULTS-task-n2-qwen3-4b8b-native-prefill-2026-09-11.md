# RESULTS — task-n2: dense Qwen3 4B/8B native bf16 prefill

Goal `mttxt22c-a6rv75`, task-n2. Native bf16 prefill (`NPU_PREFILL_BF16=1` +
`NPU_RUNLIST=0`) for Qwen3-4B/8B, using the already-embedded `attn_mha_256_nh32.elf`
(NH=32) + the task-n1 optimizations (GEMM fusion, GU concat, async 2-batch, pipeline).

## Status: 4B/8B native prefill WORKS (token-valid, not yet FLM-verified)

| model | H / IM | 256-tok prefill | tok/s | FLM published @1k | gap |
|---|---|---|---|---|---|
| Qwen3-0.6B | 1024 / 3072 | ~450 ms | ~570 | 1494 | 2.6× |
| Qwen3-4B | 2560 / 9728 | 1763 ms | 145 | 509 | 3.5× |
| Qwen3-8B | 4096 / 12288 | 1841 ms | 139 | 357 | 2.6× |

- 4B boot=56031, 8B boot=104116 (default 9-token prompt) — valid tokens, but
  **FLM token-parity not yet re-verified** (needs a `run_qwen3_prefill` 4B/8B run).
- 1.7B (H=2048) reuses the NH=16 ELF and the 0.6B recipe; not re-measured here.

## Notes

- The NH=32 attention ELF was already embedded (`engine/npu/xclbins/attn_mha_256_nh32.elf`)
  and `run_attn` already selects it via `attn_qout==4096` — no new ELF capture was needed.
- The 4B/8B gap is the same structural finding as task-n1 (host f32↔bf16 conversions
  + RMSNorm/RoPE/SiLU); the per-op native prefill is bounded ~2.6–3.5× short of FLM.
- **Not done:** position-shifted attention ELFs for >256-token chunks (the fixed
  `attn_mha_256_*.elf` bakes RoPE positions [0,256), so chunked prefill >256 tokens
  needs a shifted ELF per chunk). The current path caps npt at 256.
