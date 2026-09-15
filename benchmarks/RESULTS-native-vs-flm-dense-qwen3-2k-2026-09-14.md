# Native vs FLM across the dense Qwen3 family — 2k context (2026-09-14)

Extends `RESULTS-native-vs-flm-dense-qwen3-REMEASURED-2026-09-14.md` (1k) to the
2048-token context — the longest context the native engine can serve today.

## What changed

`engine/npu/xclbins/attn_mha_2048_nh32.elf` (352432 B): a 2048-context, NH=32,
head_dim=128 attention capture wired through the new `NPU_ATTN_ELF_2048_NH32`
slot in `npu_engine_bf16_mm.h`. Before it, the `(1024, 2048]` range for the
NH=32 models had no shape-and-context-matching ELF, so `run_attn()` returned
false and the caller fell back to host attention — correct but ~152 s per 2k
prefill (4B). With the capture, layer attention over 2048 tokens costs ~0.6 s
(4B 612 ms, 8B 599 ms).

The NH=16 models (0.6B/1.7B) already had their 2048 capture
(`attn_mha_2048_nh16.elf`); they are measured here to complete the family row.

## Method

- 2048-token prompt: `~/npu-build/parity/ids2048.txt` (the 1024-token prompt
  repeated twice). ids1024 / ids1280 / ids1536 / ids2048 are all prefixes of a
  single token sequence (verified).
- Native prefill: `NPU_RUNLIST=0 NPU_PREFILL_BF16=1 NPU_PREFILL_MAX=2048` — the
  bf16 per-op path, one attention call with 2048 query rows and 2048 keys.
- FLM on-box reference: `NPU_FLM_PREFILL=1` = FLM's own `qwen3_npu::prefill`.
- Published bar: `amd-oss/fastflowlm/docs/docs/benchmarks/qwen3_results.md`
  (Kraken Point / Ryzen AI 7 350). This box is Strix Halo (Ryzen AI MAX+ 395),
  as for all prior entries in this series.

## Results — prefill (tok/s): native vs published vs FLM on-box

| model | @1k native | pub @1k | FLM on-box @1k | @2k native | pub @2k | FLM on-box @2k |
|---|---|---|---|---|---|---|
| 0.6B  | 1905 | 1494 (+28%) | 1101 | **2232** | 2003 (+11%) | 1724 (+29%) |
| 1.7B  | 1319 |  956 (+38%) |  767 | **1562** | 1263 (+24%) | 1282 (+22%) |
| 4B    |  674 |  509 (+32%) |  409 |  **765** |  582 (+31%) |  606 (+26%) |
| 8B    |  470 |  357 (+32%) |  295 |  **514** |  435 (+18%) |  433 (+19%) |

Native meets-or-beats the published prefill table at every size and both
contexts, and beats FLM's own on-box prefill at every point.

### Decode @2k where the captures allow it

The runlist decode needs `layer_ctx<N>.elf` for every step's context, and the
per-model captures are: 0.6B up to ctx 2200, 1.7B/4B/8B up to ctx 2048. So a
2048-token prompt followed by 32 decode steps (ctx 2049..2080) is only runnable
for 0.6B:

| model | decode @2k native | published @2k | verdict |
|---|---|---|---|
| 0.6B | **67.0 tok/s** (14.9 ms/tok) | 57.5 | +16.5% beats |
| 1.7B | — | — | blocked: captures stop at ctx 2048 |
| 4B   | — | 18.1 | blocked: needs layer_ctx2049+ |
| 8B   | — | 11.5 | blocked: needs layer_ctx2049+ |

1.7B/4B/8B also fall back to the split i8 path when the runlist build fails,
and that path has no `final_i8_G_K2560_N9728.xclbin` (4B) — so there is no
second route. The exact published @2k decode columns for those three remain
unmeasured, not merely unrecorded.

@2k native is two passes where run twice (4B 758.2 → 765.1 tok/s, 8B 524.9 →
513.9 tok/s; reported as the pass pair, <2% spread). 0.6B/1.7B @2k are single
passes (0.448 / 0.640 ms/tok). The two-pass 1k table is in the REMEASURED doc.

## Correctness gates @2048 (greedy boot token)

Every path that can run at 2048 agrees on boot = **220**:

| model | native bf16 | FLM-ref | runlist (byte-exact) | CPU attn |
|---|---|---|---|---|
| 0.6B | 220 | 220 | — | 220 |
| 1.7B | 220 | 220 | — | — |
| 4B   | 220 | 220 | 220 | 220 |
| 8B   | 220 | 220 | — | — |

For 4B the same four-way agreement also holds at 1536 (220/220/220/220).

### Caveat: greedy instability at 1280

At npt=1280 the four independent implementations disagree on the boot token:
FLM-ref 15626, byte-exact runlist 46111, bf16+NPU-attn 77087, bf16+CPU-attn
3519. This is a property of the prompt/position, **not** the new ELF: the
CPU-attention path never touches the 2048 ELF and diverges too, and the
divergence vanishes at 1536/2048 where all four agree. Boot-token gates at
numerically unstable positions carry no evidence; the 1536/2048 gates are the
reliable ones.

## Still open

- **Decode @2k for 1.7B/4B/8B**: blocked on layer-ELF captures beyond ctx 2048
  (they exist to 2048; 0.6B reaches 2200). `NPU_RUNLIST=1` decode at 2048 falls
  through to the split i8 path and fails to init
  `final_i8_G_K2560_N9728.xclbin`, which does not exist in
  `engine/npu/xclbins/` — so there is no fallback route either. Capture
  layer_ctx2049+ (or build the missing i8 G xclbin) is the bounded follow-up.
- **Context > 2048**: no attention capture exists, so the published table's
  4k/8k/16k/32k columns are unreachable today.
- Decode vs the published table remains the tightest gap for this family
  (4B −3.1%, 8B −7.6% @1k); prefill/TTFT are native wins.

## Measurement conditions

The NPU was in **Default** power mode for all of the above (FLM defaults to
`--pmode performance` and the published bar is a Performance-mode number), and
decode numbers are contaminated ~2× by concurrent `/dev/accel/accel0` users.
Both are recorded in `NOTE-measurement-conditions-2026-09-14.md`; the prefill
wins here were achieved in Default mode (i.e. not power-mode-assisted) and were
taken before the concurrent 8B loop started.
