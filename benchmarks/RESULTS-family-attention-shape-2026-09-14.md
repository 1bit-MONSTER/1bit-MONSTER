# Family attention-shape status (2026-09-14)

Follow-on to the dense-Qwen3 decode work: a check of what the *other* families
need, and one bounded engine bug fixed.

## The `>1024` attention slot lent the nh32 kernel to any shape

`Bf16Mm::run_attn` selects the long-context kernel by context alone past 1024
keys. The branch was

    attn_tokens > 1024 ? (nh16 ? attn_kernel2k : attn_kernel2k32)
                       : ...

so any shape that is neither nh16 (qout 2048) nor nh32 (qout 4096) — Nanbeige
(nh20, qout 2560), Phi4 (nh24, qout 3072) — was handed the **nh32 2048** kernel.
The `attn_shape_ok` guard did not catch it, because `attn_shaped_ok` is already
set true by the shape's own ≤1024 ELF, so the guard passes and the wrong 2k
kernel runs. Fixed in `npu_engine_bf16_mm.h`: only nh16/nh32 may use a 2k
capture; every other shape returns nullptr and the caller runs the CPU
reference (slow but correct). Dense Qwen3 and Llama-3.1-8B are unaffected (they
are nh16/nh32); re-checked after the rebuild: 4B @2048 2693 ms, 8B @2048
3999 ms, boot 220 both.

## Nanbeige4.1-3B: the bf16 path is correct; only its attention kernel is bad

1000-token prompt, gate = FLM's own prefill boot (`NPU_FLM_PREFILL=1`):

| path | boot | prefill |
|---|---|---|
| FLM reference | 163569 | 1859 ms (538 tok/s) |
| native bf16 + shape ELF (`attn_mha_1024_nh20_hd128.elf`) | **90724** ✗ | 1416 ms (706 tok/s) |
| native bf16 + `NPU_ATTN_CPU=1` | **163569** ✓ | 10819 ms (92 tok/s) |

So the Gemm/host-math half of the Nanbeige bf16 path is right, and the nh20
attention ELF is the sole cause of the divergence — the same conclusion
reachable from `RESULTS-coverage-multifamily-2026-09-13.md` §9/§11, now with a
matched-length gate.

At 1087 tokens (>1024) the same prompt gives, with the selection fix:

| path | boot | prefill |
|---|---|---|
| FLM reference | 163569 | 2140 ms (508 tok/s) |
| native bf16 (CPU fallback) | **163569** ✓ | 12722 ms (85 tok/s) |

So Nanbeige is now *correct at every context* on the native bf16 path, but at
85–92 tok/s prefill it is ~6x under FLM's bar until a correct nh20 attention
kernel exists (capture or binding fix). The same shape gap is why Phi4-mini
(nh24) and Gemma3 (nh4/nh8) do not gate.

## Bearing on coverage

Six models still gate natively at 1k (dense Qwen3 0.6B/1.7B/4B/8B,
Qwen3-VL-4B, Llama-3.1-8B). Nanbeige and Phi4 now fail *correctly* (CPU
fallback) instead of silently with a wrong-shape kernel, but neither meets the
performance bar. Closing them is per-shape attention-kernel work (capture the
family's own kernel at the needed context), not a bounded engine change.
