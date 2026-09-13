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

`gen_mha_engine_seq(L0,L1)` + aiebu is the available generator
(`~/npu-build/mha/gen_attn_chunk`). Measured ELF sizes:

| range | txn words | ELF bytes |
|---|---|---|
| `gen(0,256)` | 22792 | 95936 |
| `gen(256,512)` | 22792 | 95936 |
| `gen(512,768)` | 22792 | 95936 |
| `gen(768,1024)` | 22792 | 95936 |
| `gen(0,512)` | 44808 | 188128 |
| `gen(0,1024)` | 88840 | 372512 |

Size depends **only on `L1-L0`**, not on `L1`. So the key range of a generated
ELF is `[L0,L1)` — the same width as its query range — i.e. a *diagonal tile*,
not a growing prefix `[0,L1)`. A chunk-wide key prefix would need progressively
more DMA descriptors and could not come out byte-identical in size.

That means the generated ELFs are not a drop-in long-context kernel: covering
`npt` tokens needs either a tiled scheme (all `(query chunk, key chunk)` pairs,
with the online-softmax combine done somewhere — the ABI exposes only
`out/act/kv`, no accumulator BO, so the combine would have to live in the kernel
or be added on the host) or a different kernel entirely.

Two further data points:

- The **single-call `gen(0,1024)` ELF does not fix npt=1024** — with the GEMM
  tiling corrected it still returns boot=220 (the same wrong answer as the
  embedded ELF), while `NPU_ATTN_CPU=1` returns the trusted 25. So its 4 query
  blocks are not being applied the way a prefix-causal kernel would.
- The embedded captured ELF (26928 B) is **not** a `gen()` product at all — it is
  ~3.5x *smaller* than `gen(0,256)`, consistent with the ABI note that it is
  FLM's own captured runtime stream. It is correct up to 512 keys.

Recommended next step: treat the generated ELFs as diagonal tiles and determine
empirically (one 256x256 case against `NPU_ATTN_CPU=1` at npt=256) whether a
single tile call reproduces block-local causal attention; if it does, the tiling
+ combine scheme is viable.

