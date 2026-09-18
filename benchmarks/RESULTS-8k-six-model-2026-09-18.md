# Criterion (c) at 8k, all six models, guard-accepted: native is AHEAD of FLM — 2026-09-18

Goal `mu35shsg-i3hlyi`. The 2026-09-16 yardstick recorded native **losing badly** at 8k
(Qwen3-8B 0.71x prefill, 35.2 s TTFT vs FLM's 23.5 s; Llama-3.1-8B 0.62x, 33.2 s vs 19.6 s).
A single-run re-measurement then claimed native ahead, but the repeated campaign showed only
"parity" with ±35% noise. With the measurement pipeline fixed, the six-model answer is now
**native ahead on prefill for every model, at 1.04–1.10x**.

## Method

`benchmarks/c8k_guarded.sh`, whose acceptance rules were built from the four defects this
criterion's measurement turned out to have:

- **device axis**: every `accel0` holder sampled before/after each run, excluding only the
  production `flm serve`; a foreign holder discards the run;
- **host axis**: the top foreign CPU consumer sampled **during** each run (excluding this
  lane's own engine and ELF generator) — because the 1-minute load average lags by ~1 minute in
  *both* directions, which had both accepted a `pf`-at-3131% run and rejected a clean one;
- **`C8K_WAIT_QUIET=1`**: wait for both axes to clear, then take one sample;
- prompt = 8192 tokens exactly (`NPU_PROMPT_MAX=8192 NPU_PREFILL_MAX=8192`, log shows
  `=== Prefill 8192 [bf16] ===` for every run); native `ng=1` (the second decode forward needs
  `ctx=8194`, past the baked `MAX_L=8192` per-ctx ELF window); FLM via `npu_ab.sh
  --skip-native --ctx-k 8 --decode-tokens 1`, interleaved per run.

## The result

Medians over the guard-accepted runs; the ratio column is per-run native/FLM prefill rate.

| model | accepted | native prefill | FLM prefill | native TTFT | FLM TTFT | prefill ratio |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B | 3/3 | **0.505 ms/tok (1980 t/s)** | 1909.69 t/s | 4.133–4.151 s | 4.039–4.113 s | **1.03–1.05x** |
| Qwen3-1.7B | 3/3 | **0.691 (1447)** | 1330.58 | 5.627–5.677 s | 5.816–5.840 s | **1.07–1.11x** |
| Qwen3-4B | 1/3 | **1.605 (623)** | 567.47 | 13.152 s | 13.674 s | **1.10x** |
| Qwen3-VL-4B | 2/3 | **1.5675 (638)** | 586.11 | 12.831–12.856 s | 13.218–13.242 s | **1.09x** |
| Qwen3-8B | 2/3 | **2.163 (462)** | 424.36 | 17.684–17.754 s | 18.237–18.331 s | **1.08–1.09x** |
| Llama-3.1-8B | 3/3 | **2.162 (463)** | 445.28 | 17.175–17.936 s | 17.336–17.965 s | **1.04–1.06x** |

**All six models are at or above FLM on 8k prefill.** TTFT is at parity or faster for five of
six (1.7B +3.4%, 4B +3.8%, VL-4B +2.9%, 8B +3.0%, Llama ≈parity); 0.6B is ~1–2% slower
(4.14 s vs 4.07 s) and is the one clause that does not clearly pass.

The accepted sets are tight: per-run spreads of **0.4% (0.6B), 0.9% (1.7B), 1.5% (8B), 2.5%
(Llama), 0.2% (VL-4B)** — not the ±35% of the load-uncontrolled campaign, because the runs that
would have produced that spread are now rejected by the during-run foreign-CPU sample.

## What this refutes

The 2026-09-16 inversion is **not reproduced for any model**. The two headline rows:

- **Qwen3-8B**: recorded 0.71x (35.2 s vs 23.5 s) → measured **1.08–1.09x** (17.7 s vs 18.3 s).
  The native 8k prefill is *twice as fast in absolute terms* as the old record, and FLM's own
  leg also moved (23.5 s → 18.3 s), so the old pair was an environment reading on both sides.
- **Llama-3.1-8B**: recorded 0.62x (33.2 s vs 19.6 s) → measured **1.04–1.06x** (17.4 s vs
  17.8 s), with its per-ctx ELF cache warm and `NPU_UNIFIED=1 NPU_LAYER_ELF_DIR` set.

The earlier attempts in this session failed for three separable reasons, all now instruments:
the unpinned `layer.xclbin` (fixed in the engine), the foreign `accel0` holders (guarded), and
the host CPU floor of another lane's test suite (capped at the source by its owner after the
guard attributed it — `OMP_NUM_THREADS=4`, and their suite now declares itself in
`/tmp/1bit-npu-device.lock`).

## Bounds

- **8k decode still does not exist.** `ng=1` is not a choice: the second forward needs
  `ctx=8194`, the per-ctx ELFs bake `MAX_L=8192`, and the split-path fallback then dies on the
  missing i8 `G_K2560_N9728` tile. The criterion's decode clause cannot be evaluated at 8k by
  any current path — that is the phase-2 boundary, not a regression.
- **4B has only one accepted run** (the other two were rejected for a Prism device holder and a
  load rise); its 1.10x is a single sample, consistent with its quiet single-run 1.24x and its
  two *rejected* pairs (1.19x, 1.19x) but not itself repeated.
- FLM legs run with the production `flm serve` present, as in every reference in this lane.
  Each campaign ran while the Prism suite was active with a 4-thread CPU floor
  (`foreign=` 176–399% in the accepted runs).
- These are 1-run-per-arm pairs, repeated 2–3 times, not interleaved repeats of a single arm.

## Commands

```
# per model (Llama adds NPU_UNIFIED=1 NPU_LAYER_ELF_DIR=<warm dir>)
C8K_WAIT_QUIET=1 bash benchmarks/c8k_guarded.sh <Model-NPU2> <flm_tag> <engine> 3 \
    Qwen3-0.6B-NPU2 qwen3:0.6b npu_engine_qwen3_0_6b
# logs: benchmarks/c8k-guarded-<Model>-<utc>/{native-N.log,flm-N.log,foreign-cpu-N.txt}
```
