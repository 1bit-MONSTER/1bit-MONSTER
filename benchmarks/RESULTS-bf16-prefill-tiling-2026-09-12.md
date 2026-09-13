# bf16 prefill: row tiling + chunked attention (2026-09-12, round 30)

Follow-up to `RESULTS-bf16-prefill-CORRECTION-2026-09-12.md`, which withdrew the
"native @1k beats FLM by 40%" claim because the bf16 prefill path hardcoded
exactly two 128-row GEMM batches (= 256 rows) and never computed tokens >= 256.

## What was fixed

1. **All four GEMM stages tile the whole prompt.** `Bf16Mm::ensure_a()` stages
   two 128-row halves from the A pointer it is handed, so the engine now walks
   the prompt in 256-row blocks (`batch 0` = rows b..b+127, `batch 1` =
   b+128..b+255, both from the same shifted base — batch 1 is an `ensure_a`
   cache hit, so it costs no extra staging) for QKV, O, GU and D. Buffers are
   sized `NPAD = round_up(npt,256)+256` so the last block's staging stays in
   bounds.
2. **SiLU output moved to its own buffer** (`bGu`). The old code wrote SiLU back
   into `bA`, which is only safe when every block's kernel has already been
   launched; a block loop cannot guarantee that (`bA[pi*IM]` overlaps unread
   rows of `bA[pi*H]` for `H < IM`).
3. **Attention is called once per 256-query block** (`bf16mm_set_attn_rows` /
   `bf16mm_set_attn_tokens`), with the Q/out pointers shifted to the block and
   `attn_tokens` = keys in the prefix `[0, b+rows)`. The captured kernel computes
   at most 256 query rows; one call for the whole prompt left rows >= 256 stale.
4. **`NPU_PREFILL_MAX` is capped to the attention ELF's verified key range**
   (see below) with an explicit warning, instead of silently returning a wrong
   token.

## Verification (boot token vs the two trusted paths)

The trusted references are `NPU_RUNLIST=1` (int8 whole-layer path, byte-exact vs
FLM) and `NPU_FLM_PREFILL=1` (FLM's own prefill). They agree at every length
tested.

| npt | trusted | bf16 native | |
|---|---|---|---|
| 256 | 1614 | 1614 | ✅ |
| 512 | 220 | 220 | ✅ |
| 1024 | 25 | 220 | ❌ |
| 1024 + `NPU_ATTN_CPU=1` | 25 | **25** | ✅ |

The 1024-with-CPU-attention row is the important one: it proves the **GEMM
tiling is correct at 1024** and isolates the remaining error to attention. The
captured embedded attention ELF is correct up to **512 keys** and silently wrong
beyond (1024 returns 220, the 512 answer, instead of 25). The engine now caps
`NPU_PREFILL_MAX` at 512 for that reason; `NPU_ATTN_CPU=1` restores correctness
at any length (~15 s attention over 1024 tokens).

## Performance (device free)

| npt | prefill | tok/s |
|---|---|---|
| 256 | 377-384 ms | 667-680 |
| 512 | 652-813 ms | 630-785 |
| 1024 | 932 ms | 1099 (incorrect) |

FLM on-box at ctx 1k, same harness: **prefill 1387-1430 tok/s, decode 75.2-75.4,
TTFT 0.687-0.708 s**.

Native decode on the byte-exact runlist path: **80 tok/s** (+6% vs FLM).
A 1024-token npt=1024 prefill at 1099 tok/s is still ~23% behind FLM's 1430.

## ⚠️ Measurement-integrity caveat

Later timings in this session are polluted by **NPU contention**: a
`flm serve qwen3.6-moe:35b-a3b` process (up since 16:00) plus several concurrent
`npu_engine_qwen3_0_6b` processes (not started by this session) were holding the
device. The same npt=512 run measured `attn 179 ms` on a free device and
`attn 4527 ms` under contention — a 25x swing. Only the boot-token gates are
contention-independent; **re-measure all timings on an otherwise idle NPU**
before quoting them.

## To go past 512 keys

The generated per-position-range ELFs are the path (`gen_mha_engine_seq(L0,L1)`
+ aiebu; `~/npu-build/mha/gen_attn_chunk`). One chunk ELF per 256-query block,
selected by block index. The single-call `gen(0,1024)` ELF was tested and does
**not** fix it (it reproduces the embedded ELF's wrong 1024 answer, and is ~1500x
slower), so the per-chunk variant needs to be generated and gated before use.
