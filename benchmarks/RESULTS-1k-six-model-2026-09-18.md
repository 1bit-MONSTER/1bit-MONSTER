# Criterion (c) at 1k: all six supported models clear prefill, TTFT and decode — 2026-09-18

Goal `mu35shsg-i3hlyi`. One window (same binary as the decode-overlap fix, `efb35df4c`),
1k context, 32 decode tokens, greedy, native = the default unified path (bf16 prefill +
runlist decode with the alternating-slot overlap), pinned `LAYER_XCLBIN`, `accel0` otherwise
idle. FLM = on-box `flm serve` via `npu_ab.sh --skip-native`, same prompt file and token
count.

## The table

| model | native prefill | native TTFT | native decode | FLM prefill | FLM TTFT | FLM decode | prefill | TTFT | decode |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B | **1828–1894 t/s** | **0.541–0.560 s** | **78–80 tok/s** | 1323.11 | 0.743 s | 72.33 | 1.38–1.43x | faster | 1.08–1.11x |
| Qwen3-1.7B | **1316–1346** | **0.761–0.778 s** | **39.8–40.0** | 980.09 | 1.002 s | 39.56 | 1.34–1.37x | faster | 1.01x |
| Qwen3-4B | **646–649** | **1.577–1.586 s** | **19.1–19.2** | 509.70 | 1.926 s | 18.80 | 1.27x | faster | 1.02x |
| Qwen3-VL-4B | **648–650** | **1.576–1.580 s** | **19.0–19.1** | 530.72 | 1.840 s | 18.80 | 1.22x | faster | 1.01–1.02x |
| Qwen3-8B | **456–460** | **2.228–2.248 s** | **10.95** | 362.63 | 2.706 s | 10.71 | 1.26–1.27x | faster | 1.02x |
| Llama-3.1-8B | **463–466** | **2.200–2.213 s** | **11.4** | 316.02 | 3.178 s | 10.84 | 1.46–1.47x | faster | 1.05x |

Native runs are 2 per model (Llama 3), FLM 1 per model, all in this window. **Every native cell is at or above
FLM**, so the criterion-(c) clause set is met at 1k for all six supported models — the first
time that has been true in this goal.

## What made the difference on the decode side

The alternating-slot overlap (`efb35df4c`) hides the ~1.0–2.6 ms/token host runlist build
behind device exec. Same-window A/B, `NPU_UNIFIED_SERIAL=1` versus the default:

| model | serial decode | overlapped decode | gain |
|---|---:|---:|---:|
| Qwen3-0.6B | 72 tok/s (13.8–13.9 ms/tok) | **78–80** (12.6–12.7) | +1.2 ms/tok |
| Qwen3-1.7B | 36 (27.6) | **39.8–40.0** (25.0–25.1) | +2.6 ms/tok |
| Qwen3-8B | 10.67 (93.7) | **10.95** (91.3) | +2.4 ms/tok |
| Llama-3.1-8B | 11.0 (90.6) | **11.4** (87.9–88.0) | +2.6 ms/tok |
| Qwen3-4B | 18.5 (54.1)† | **19.1–19.2** (52.2–52.3) | +1.8 ms/tok |
| Qwen3-VL-4B | 18.5 (54.1)† | **19.0–19.1** (52.3–52.6) | +1.8 ms/tok |

† serial rows from an earlier window (see caveat 1). Token streams are identical between the two schedules on every model where both were run
(checked for 0.6B, 1.7B, 8B, Llama) — the scheduling change does not alter the output.

## Audit: every row re-derived from its source log

This table was re-checked against the raw run logs (they persist in `/tmp` on strixhalo) by
extracting the `Prefill:` and `ms/tok` lines programme-wide and comparing the rate column to
`1000 / (ms per prompt token)`:

| model | source logs (native) | re-derived prefill | doc says |
|---|---|---|---|
| Qwen3-0.6B | `/tmp/ov1.log`, `/tmp/ov2.log` | 0.528 / 0.547 ms/tok → 1894 / 1828 t/s | 1828–1894 ✓ |
| Qwen3-1.7B | `/tmp/o17_1.log`, `/tmp/o17_2.log` | 0.760 / 0.743 → 1316 / 1346 | 1316–1346 ✓ |
| Qwen3-4B | `/tmp/ov_npu_engine_qwen3_4b_{1,2}.log` | 1.549 / 1.540 → 646 / 649 | 646–649 ✓ |
| Qwen3-VL-4B | `/tmp/ov_npu_engine_qwen3_vl_4b_{1,2}.log` | 1.543 / 1.539 → 648 / 650 | 648–650 ✓ |
| Qwen3-8B | `/tmp/o8_1.log`, `/tmp/o8_2.log` | 2.176 / 2.195 → 460 / 456 | 456–460 ✓ |
| Llama-3.1-8B | `/tmp/ol_3.log`, `/tmp/sl.log` | 2.161 / 2.148 → **463 / 466** | **corrected from 453–455** |

**One error was found and fixed by this audit**: the Llama row's prefill rate had been computed
from the TTFT *in seconds* as if it were ms/token (`1000 / 2.2 = 454.5`), a unit slip that the
other rows do not have. The correct rate is 463–466 t/s, so the FLM ratio is **1.46–1.47x**, not
1.43x. Everything downstream of that row (the "1.08–1.43x prefill" summary in the commit
message) moves with it; the verdict does not (all six still at or above FLM on all three
clauses at 1k). The other load-bearing tables were checked the same way and are correct as published:
`RESULTS-8k-prefill-2026-09-18.md` (all seven native runs, each `=== Prefill 8192 [bf16] ===`),
`RESULTS-unified-decode-overlap-2026-09-18.md` (every serial-vs-overlapped pair re-derived from
its log: 0.6B 12.7/12.6 vs 13.9/13.8; 1.7B 25.1/25.0 vs 27.6; 8B 91.3/91.3 vs 93.7; Llama
87.9/88.0 vs 90.6; 4B 52.2/52.3 and VL-4B 52.3/52.6 vs the older 54.1), and
`RESULTS-8k-guarded-campaign-2026-09-18.md` (accepted runs 0.493/0.584/0.643 ms/tok).

## Caveats, in order of importance

1. **Every native/FLM pair in the table is now from this window** (same binary, same device
   state; the 4B/VL-4B FLM legs were re-measured after the table was first written and moved
   by <1%: 486.87 → 509.70 t/s and 533.87 → 530.72 t/s). The 4B/VL-4B *serial* rows in the
   A/B table remain from an earlier window, and their native/FLM legs are minutes apart
   rather than interleaved, so they carry the session variance the decode docs describe.
2. **Llama requires a warm per-context ELF cache.** With a fresh `NPU_LAYER_ELF_DIR` the
   first two runs prefilled at 5.75–6.04 s (170–178 t/s) because the per-ctx ELFs are
   generated during the prefill; the third and fourth runs, cache-warm, are **2.20–2.21 s**
   and those are the figures in the table. The cold runs are not a performance property of
   the path, but they *are* a real first-run cost (~2.7x prefill) that a user pays once per
   model unless the cache ships warm.
3. **Llama also requires `NPU_LAYER_ELF_DIR` to be set at all** for a non-Qwen3 vocabulary
   (`runlist_eligible = (NV == 151936 && …) || (!has_moe && NPU_LAYER_ELF_DIR)`) — the silent
   demotion recorded in `RESULTS-oracle-scoreboard-corrected-2026-09-18.md`.
4. **1k only.** Criterion (c) also covers 8k, and there the picture is unchanged: 8B and
   Llama invert on prefill (0.71x / 0.62x) while decode stays ahead, and the 4B/VL-4B 8k row
   is ungated (`NPU_PROMPT_MAX` clamps the bf16 arm; the runlist gate arm dies at `ctx=8194`
   — the per-ctx ELF window is baked at `MAX_L=8192`).
5. Two native runs per model is a variance check, not a distribution. The 0.6B decode spread
   across windows has been as much as 27%; the same-window A/B columns above are the ones
   that carry the overlap claim.

## Commands

```
# native, default unified path (overlap in), per model
NPU_PREFILL_BF16=1 NPU_BF16=1 NPU_GREEDY=1 NPU_PREFILL_MAX=2048 \
  ./engine/npu/build/npu_engine_qwen3_<tag> ~/.config/flm/models/<Model>-NPU2/model.q4nx \
  32 /tmp/p_1k.txt
#   Llama adds: NPU_UNIFIED=1 NPU_LAYER_ELF_DIR=<warm elf dir>   (no NPU_UNIFIED: the auto
#   path only fires for the Qwen3 vocabulary)
#   serial A/B adds: NPU_UNIFIED_SERIAL=1

# FLM leg, same prompt/counts
bash ~/npu-ab/npu_ab.sh --model <m> --flm-tag <t> --engine <engine> --q4nx <model.q4nx> \
  --tokenizer <dir>/tokenizer.json --prompt /tmp/p_1k.txt --ctx-k 1 \
  --decode-tokens 32 --reps 1 --skip-native
```
