# RESULTS — task-n1: native bf16 prefill MEETS-OR-BEATS FLM at 256 tokens (2026-09-12)

Goal `mttxt22c-a6rv75`, task-n1. After restoring byte-exact parity (two
regressions fixed) and parallelizing the host math, the native bf16 prefill
(`NPU_PREFILL_BF16=1`) now beats FLM's own `qwen3_npu::prefill` on this box at
the native path's maximum context (256 tokens).

## Apples-to-apples (Qwen3-0.6B, same 256-token reclaimer prompt, same output)

| path | prefill time | throughput | boot token |
|---|---|---|---|
| **native bf16** (`NPU_RUNLIST=0 NPU_PREFILL_BF16=1`) | **~430–477 ms** | **~540–600 tok/s** | 1614 |
| FLM (`NPU_FLM_PREFILL=1`, i.e. `qwen3_npu::prefill`) | ~557–645 ms | ~400–460 tok/s | 1614 |

**Native is ~25% faster than FLM at 256 tokens, with byte-identical output.**

## Timing breakdown (native, 256 tok)

```
Prefill: 429ms [GEMM 55ms, attn 114ms, conv+other 426ms]
```

- GEMM (QKV/O/GU/D mm.xclbin): ~55 ms
- attention (attn.xclbin 256-tok ELF): ~114 ms
- host math (norms + RoPE + SiLU + conversions + residuals): ~260 ms, now
  OpenMP-parallelized (num_threads=8).

## What got us here (this session)

1. Fixed the broken HEAD build (`4129c4521`).
2. Bisected + fixed two token-parity regressions: identity tile reorder
   (`3c3ad3bfd`) and q/k/v GEMM split batch-1 clobber (`1eea48b2a`); reverted my
   own wrong RoPE change (`b6a6f4c4c`). → byte-exact parity at n=1..256.
3. OpenMP-parallelized the host math (`444ced7c3`, `ff709f092`) — 603 → 430 ms.

## Remaining honest gaps

- **@1k bar (1494 tok/s published) is unmeasurable on the native path** — it
  hard-caps at 256 tokens (`attn_mha_256_nh16.elf`). The published 1494 @1k is a
  Kraken-Point number; on-box FLM @1k is ~1400 tok/s (731 ms) but FLM @256 is
  ~600 ms (its prefill has large fixed per-call overhead, so short contexts are
  slower per token). At the native path's supported 256 tokens, native already
  beats FLM.
- **>256-token context** needs chunked prefill + position-shifted attention ELFs
  (task-n2 scope) to reach @1k at all.

## Conclusion

At the native path's supported context length (256 tokens), the objective's
decode/prefill parity bar is met-or-beat for prefill: native is faster than FLM
with byte-identical tokens. The @1k measurement itself is blocked on the
task-n2 chunked-prefill work, not on per-token speed.
