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

### Decode (tok/s): native vs published

The runlist decode needs `layer_ctx<N>.elf` for every step's context. The
per-model captures stopped at ctx 2048 for 1.7B/4B/8B, so the 32 decode steps
from a 2048-token prompt (ctx 2049..2080) had to be generated. **Do not use the
engine's lazy build (`RT_ELF_GEN`) for that** — it calls `gen_layer_elfs` with
the tool default `max_l=32768`, while every committed ELF in these dirs is
`max_l=8192` (`SESSION-FINDINGS-2026-09-14.md` §4b). The mismatch yields
byte-size-identical files with a different KV stride that run at the **same
speed** but return wrong tokens (0.6B @2k, 8192 → 220,504,220,1614…; 32768 →
220,32613,950,2527…; both 14.9 ms/tok). The ranges were regenerated with
`max_l=8192` and sha-verified; the table is measured on those.

| model | decode @1k native | pub @1k | Δ | decode @2k native | pub @2k | Δ |
|---|---|---|---|---|---|---|
| 0.6B | 79.4 | 66.5 | **+19.4%** | **67.0** | 57.5 | **+16.5%** |
| 1.7B | 40.0 | 40.2 | −0.5% | **36.8** | 35.8 | **+2.8%** |
| 4B   | 19.2 | 19.6 | −2.0% | **18.1** | 18.1 | ≈0% |
| 8B   | 11.0 | 11.9 | −7.6% | 10.6 | 11.5 | −7.8% |

Native's decode degrades more slowly with context than FLM's, so 1.7B and 4B
cross from just-behind at 1k to level/ahead at 2k. Only 8B trails at both.

The @1k column was re-measured under `pmode=performance`, within 1% of the
earlier Default-mode figures. The intermittent ~2x slow mode is NPU contention,
not a native setting: the values above are the uncontended mode, each reproduced
within 0.5% (4B @2k 55.0/55.1/55.2, 1.7B 27.2/27.2, 8B 94.1/94.1 ms/tok).
Correctness: decode [1] = 220 = the prefill boot for all four models, but [1]
alone does not catch the `max_l` error (it also returns 220); the continuation
is stable across runs on the sha-verified ELFs — for 0.6B native[2..5] =
504,220,1614,220 equals FLM's own decode, which starts one token later.

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

- **Context > 2048**: **RESOLVED for 4k (2026-09-15)** — nh16 and nh32
  4096-context attention captures now exist and are wired into a `(2048, 4096]`
  slot, so no dense-Qwen3 size falls back to CPU attention below 4096 any more.
  See `RESULTS-native-4096-context-2026-09-15.md`. The published table's
  8k/16k/32k columns are still unreachable (the longest capture is 4096, and the
  CLI caps a prompt at 4095).
- **8B decode**: the only dense metric still behind published (−7.6% @1k,
  −7.8% @2k). 0.6B/1.7B/4B decode and every size's prefill now meet-or-beat
  the published table at 1k and 2k.
- The per-ctx layer ELFs for 1.7B/4B/8B now exist to ctx 2200, all
  `max_l=8192` (sha-verified). The engine's `RT_ELF_GEN` lazy build still
  defaults to `max_l=32768` and must not be used until fixed — it returns wrong
  tokens at the same speed.

## Measurement conditions

The NPU was in **Default** power mode for all of the above (FLM defaults to
`--pmode performance` and the published bar is a Performance-mode number), and
decode numbers are contaminated ~2× by concurrent `/dev/accel/accel0` users.
Both are recorded in `NOTE-measurement-conditions-2026-09-14.md`; the prefill
wins here were achieved in Default mode (i.e. not power-mode-assisted) and were
taken before the concurrent 8B loop started.
