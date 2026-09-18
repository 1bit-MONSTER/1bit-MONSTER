# The 8-decode-token window measures engine warm-up — and the decode clause fails for 0.6B

Goal `mu35shsg-i3hlyi`, criterion (c). Box: strixhalo, `accel0` idle except the production
`flm serve`.

Every decode number in `RESULTS-yardstick-defaultpath-2026-09-16.md` was taken with
**8 decode tokens** (`--decode-tokens 8`). Re-measuring the same rows with **32** decode
tokens, everything else identical, shows that the first few decode steps carry engine
warm-up: the 8-token average is inflated, sometimes by a lot, and in a way that differs per
model.

## The direct evidence — same command, only the decode count changed

| model (bf16 default path, 1k) | 8 decode tokens | 32 decode tokens |
|---|---:|---:|
| Qwen3-0.6B | 80 tok/s | **69.9 tok/s** |
| Qwen3-4B | 20.0 tok/s | **18.5 tok/s** |
| Qwen3-VL-4B | 13.9 tok/s | **18.5 tok/s** |
| Qwen3-0.6B @8k | 33 tok/s | **33.4 tok/s** (unchanged) |

The VL-4B row is the clearest: the 8-token window measured 13.9 tok/s, which I first read as
a 0.75x deficit against FLM; at 32 tokens it is 18.5 — identical to Qwen3-4B's. The transient
was 1.42x slower than the steady state, and for 8 tokens that is most of the sample.

## Matched 32-token comparisons (both arms, same passage and token count)

Native: `NPU_PREFILL_BF16=1 NPU_BF16=1 NPU_GREEDY=1 NPU_PREFILL_MAX=16384 <engine> model.q4nx 32 /tmp/p_<ctx>.txt`.
FLM: `npu_ab.sh --flm-tag <tag> --prompt /tmp/p_<ctx>.txt --ctx-k <k> --decode-tokens 32 --reps 1 --skip-native`
(FLM v1.0.4; the harness warns the production `flm serve` is present).

| model | ctx | lane | prefill t/s | TTFT | decode t/s | prefill | TTFT | decode |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B | 1k | native | **1869** | **0.548 s** | 69.9 | **1.33x** | faster | **0.93x** |
| Qwen3-0.6B | 1k | FLM | 1405.85 | 0.699 s | **75.25** | | | |
| Qwen3-0.6B | 8k | native | **1989** | 4.102 s | 33.4 | **1.01x** | +3.9% | **0.99x** |
| Qwen3-0.6B | 8k | FLM | 1968.14 | **3.947 s** | **33.63** | | | |
| Qwen3-4B | 1k | native | **646** | **1.585 s** | 18.5 | **1.33x** | faster | **0.99x** |
| Qwen3-4B | 1k | FLM | 486.87 | 2.016 s | **18.70** | | | |
| Qwen3-VL-4B | 1k | native | **645** | **1.587 s** | 18.5 | **1.21x** | faster | **0.99x** |
| Qwen3-VL-4B | 1k | FLM | 533.87 | 1.829 s | **18.63** | | | |

## What this changes for criterion (c)

**The objective's premise — "decode at/above FLM, 1..8191 tokens, default path" — does not
hold at a warm-up-free window, for the very model it came from.** Qwen3-0.6B's decode is
**0.93x FLM at 1k** and **0.99x at 8k**; the recorded "80 vs 73.72" was an 8-token artefact.
The build-side story is unchanged and still good — prefill is ahead at both contexts (1.33x,
1.01x) and TTFT is faster at 1k — but the decode clause is not met, and it was the clause
the premise leaned on.

Current state of the three criterion-(c) clauses at a 32-token window:

| model | prefill >= FLM | TTFT | decode >= FLM |
|---|---|---|---|
| Qwen3-0.6B | **yes** 1.33x @1k, 1.01x @8k | faster @1k, +3.9% @8k | **no** 0.93x / 0.99x |
| Qwen3-1.7B | yes (8-token record; re-measure pending) | — | **no** (0.64–0.98x, 8-token record) |
| Qwen3-4B | **yes** 1.33x @1k | **faster** | **no** 0.99x |
| Qwen3-VL-4B | **yes** 1.21x @1k | **faster** | **no** 0.99x |
| Qwen3-8B | yes @1k, no @8k | 3.9x faster @1k, slower @8k | not re-measured at 32 tok |
| Llama-3.1-8B | yes @1k, no @8k | 3.8x faster @1k, slower @8k | not re-measured at 32 tok |

**Criterion (c) is not met, and now for a precise reason: the native decode is ~1–7% behind
FLM at equal windows, while prefill is ahead.** That is a different and more actionable
statement than "0.6B satisfies it" — it says the remaining work is decode throughput, not
prefill.


## CORRECTION (2026-09-18, later the same session): the 1k decode gap is parity within variance

The 1k native decode figures above (69.9 tok/s for 0.6B) were taken in a window where the
unified path's device exec measured 15.5–15.9 ms/token. That does not reproduce: after a
rebuild the same configuration measures 12.5 ms exec, and the decode is faster:

| Qwen3-0.6B, 1k, 32 decode tokens, current binary | decode |
|---|---:|
| native unified (bf16 prefill + runlist decode), 3 runs | 72, 72, 71 tok/s |
| native pure runlist, 2 runs | 78, 77 tok/s |
| FLM on-box | 72.33 tok/s (75.25 in the earlier window) |

So the 0.6B 1k decode clause is **0.98–1.00x FLM — parity within measurement variance — not
the 0.93x deficit recorded above**, and FLM's own figure moved 4% between windows. The 8k row
(33.4 vs 33.63 ≈ 0.99x) is unchanged in substance.

The 8-vs-32-token warm-up finding is unaffected: it is a within-window, within-binary
comparison (80 → 69.9 on the same binary), and the 32-token figures are the ones to cite —
but they must be **repeated**, because runlist exec time varies up to ~27% between
builds/sessions. See `RESULTS-unified-decode-penalty-2026-09-18.md` for the mechanism
investigation that was withdrawn with it.

Consequence for criterion (c): the decode clause is **not decisively failed** for Qwen3-0.6B
at 1k on this evidence; the honest reading is parity within variance, with prefill ahead
(1.33x) and TTFT faster. A decisive verdict needs repeated runs on a quiet device.

## Bounds and open items

- **The 8k native rows are clamped**: Qwen3-0.6B prefilled **8161** of 8192 tokens and
  Qwen3-4B **8185** of 8192 (`max_seq_len 4096 … raise with NPU_PROMPT_MAX`). Both 8k native
  rows are therefore slightly favourable and are not exactly the same computation as FLM's.
- **The 4B/VL-4B 8k rows still have no gate** (`RESULTS-bf16-defaultpath-4b-2026-09-18.md`).
- **Qwen3-1.7B, Qwen3-8B and Llama-3.1-8B decode must be re-measured at >=32 tokens** before
  their decode clauses are cited either way. Their 8-token numbers are in the same
  `--decode-tokens 8` family as the rows above and the warm-up content is unknown per model.
- `RESULTS-yardstick-defaultpath-2026-09-16.md` still presents its 8-token decode column
  without this qualifier; the marker belongs on it, and the co-lane agent (who is re-running
  its gate arm) is the one editing that file.
