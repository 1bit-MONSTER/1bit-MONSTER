# Criterion (c) at ~8k — final 18/18 table (2026-09-19)

Goal `mu7ikqtg-nage9h`. Completes the 8k TTFT parity matrix: **18/18 cells accepted** across the
six dense models (3 runs each). A cell is accepted only when (1) native TTFT ≤ FLM TTFT
(ratio ≤ 1.0) and (2) the run is a valid measurement (guard-accepted + sane first token).

Supersedes `RESULTS-8k-criterion-c-reconstitution-2026-09-18.md` (which recorded 13 accepted +
1 rejected Qwen3-8B + 4 unrun). The five remaining cells were completed 2026-09-19 after fixing
a missing-xclbin blocker (below).

## Method (unchanged from the pinned accept gate)

- Harness: `benchmarks/c8k_guarded.sh <Model-NPU2> <flm_tag> <engine> <runs>` — both contamination
  axes gated (foreign `accel0` holders + during-run foreign-CPU sample), `C8K_WAIT_QUIET=1`.
- Prompt: 8192 tokens (`NPU_PROMPT_MAX=8192 NPU_PREFILL_MAX=8192`, `=== Prefill 8192 [bf16] ===`).
- Native: `ng=1`, `NPU_PREFILL_BF16=1 NPU_BF16=1 NPU_GREEDY=1`, with `NPU_XCLBIN_DIR` pinned to
  `$PWD/engine/npu/xclbins`.
- FLM reference leg: `npu_ab.sh --skip-native --ctx-k 8 --decode-tokens 1`, FLM **v1.0.4**
  (`/opt/fastflowlm/bin/flm`) throughout — same reference as every earlier accepted cell.
- Llama adds `NPU_UNIFIED=1 NPU_LAYER_ELF_DIR=/tmp/llama-elfs`.
- First token sanity: Qwen3 family → `[1] 576`; Llama-3.1-8B → `[1] 785`. All 18 cells emit it.

## The 18-cell table

| model | run | native TTFT (ms) | FLM TTFT (s) | ratio (native/FLM) | valid |
|---|---:|---:|---:|---:|:---:|
| Qwen3-0.6B | 1 | 3807 | 3.868363 | **0.984** | ✅ |
| Qwen3-0.6B | 2 | 3840 | 3.922206 | **0.979** | ✅ |
| Qwen3-0.6B | 3 | 3852 | 3.863247 | **0.997** | ✅ |
| Qwen3-1.7B | 1 | 5441 | 5.666079 | **0.960** | ✅ |
| Qwen3-1.7B | 2 | 5498 | 5.814714 | **0.946** | ✅ |
| Qwen3-1.7B | 3 | 5478 | 5.801051 | **0.944** | ✅ |
| Qwen3-4B | 1 | 13013 | 13.759027 | **0.946** | ✅ |
| Qwen3-4B | 2 | 13045 | 13.795332 | **0.946** | ✅ |
| Qwen3-4B | 3 | 13399 | 13.757898 | **0.974** | ✅ |
| Qwen3-VL-4B | 1 | 13020 | 13.627117 | **0.955** | ✅ |
| Qwen3-VL-4B | 2 | 12998 | 13.571691 | **0.958** | ✅ |
| Qwen3-VL-4B | 3 | 12863 | 13.482731 | **0.954** | ✅ |
| Qwen3-8B | 1 | 17713 | 18.569485 | **0.954** | ✅ |
| Qwen3-8B | 2 | 18106 | 18.431244 | **0.982** | ✅ |
| Qwen3-8B | 3 | 17896 | 18.256824 | **0.980** | ✅ |
| Llama-3.1-8B | 1 | 16966 | 17.580656 | **0.965** | ✅ |
| Llama-3.1-8B | 2 | 16969 | 17.531263 | **0.968** | ✅ |
| Llama-3.1-8B | 3 | 17199 | 18.873436 | **0.911** | ✅ |

**18/18 accepted.** Every cell has ratio ≤ 1.0 (native TTFT meets-or-beats FLM), a guard-accepted
run, and a sane first token. Per-model ratio range: 0.911–0.997 (native ahead by 0.3%–8.9%).

## Per-cell run-log extracts

Cell identity = `<logdir> / native-<n>.log` (+ `flm-<n>.log`).

| model | run | logdir | native Prefill line | FLM rep-1 line |
|---|---|---|---|---|
| Qwen3-0.6B | 1 | c8k-guarded-Qwen3-0.6B-NPU2-20260918T213223Z | `Prefill: 3807ms (0.465 ms/tok)` `[1] 576` | `ttft=3.868363s prefill=2007.47t/s` |
| Qwen3-0.6B | 2 | (same) | `Prefill: 3840ms (0.469 ms/tok)` `[1] 576` | `ttft=3.922206s prefill=1979.92t/s` |
| Qwen3-0.6B | 3 | (same) | `Prefill: 3852ms (0.470 ms/tok)` `[1] 576` | `ttft=3.863247s prefill=2010.14t/s` |
| Qwen3-1.7B | 1 | c8k-guarded-Qwen3-1.7B-NPU2-20260918T213415Z | `Prefill: 5441ms (0.664 ms/tok)` `[1] 576` | `ttft=5.666079s prefill=1370.03t/s` |
| Qwen3-1.7B | 2 | (same) | `Prefill: 5498ms (0.671 ms/tok)` `[1] 576` | `ttft=5.814714s prefill=1335.04t/s` |
| Qwen3-1.7B | 3 | (same) | `Prefill: 5478ms (0.669 ms/tok)` `[1] 576` | `ttft=5.801051s prefill=1338.42t/s` |
| Qwen3-4B | 1 | c8k-guarded-Qwen3-4B-NPU2-20260918T213641Z | `Prefill: 13013ms (1.588 ms/tok)` `[1] 576` | `ttft=13.759027s prefill=563.95t/s` |
| Qwen3-4B | 2 | (same) | `Prefill: 13045ms (1.592 ms/tok)` `[1] 576` | `ttft=13.795332s prefill=562.47t/s` |
| Qwen3-4B | 3 | c8k-guarded-Qwen3-4B-NPU2-20260919T120556Z | `Prefill: 13399ms (1.636 ms/tok)` `[1] 576` | `ttft=13.757898s prefill=564.04t/s` |
| Qwen3-VL-4B | 1 | c8k-guarded-Qwen3-VL-4B-Instruct-NPU2-20260918T213929Z | `Prefill: 13020ms (1.589 ms/tok)` `[1] 576` | `ttft=13.627117s prefill=569.06t/s` |
| Qwen3-VL-4B | 2 | (same) | `Prefill: 12998ms (1.587 ms/tok)` `[1] 576` | `ttft=13.571691s prefill=571.36t/s` |
| Qwen3-VL-4B | 3 | c8k-guarded-Qwen3-VL-4B-Instruct-NPU2-20260919T120848Z | `Prefill: 12863ms (1.570 ms/tok)` `[1] 576` | `ttft=13.482731s prefill=575.23t/s` |
| Qwen3-8B | 1 | c8k-guarded-Qwen3-8B-NPU2-20260918T214216Z | `Prefill: 17713ms (2.162 ms/tok)` `[1] 576` | `ttft=18.569485s prefill=417.82t/s` |
| Qwen3-8B | 2 | c8k-guarded-Qwen3-8B-NPU2-20260919T121134Z | `Prefill: 18106ms (2.210 ms/tok)` `[1] 576` | `ttft=18.431244s prefill=420.99t/s` |
| Qwen3-8B | 3 | (same) | `Prefill: 17896ms (2.185 ms/tok)` `[1] 576` | `ttft=18.256824s prefill=425.01t/s` |
| Llama-3.1-8B | 1 | c8k-guarded-Llama-3.1-8B-NPU2-20260918T214600Z | `Prefill: 16966ms (2.071 ms/tok)` `[1] 785` | `ttft=17.580656s prefill=442.59t/s` |
| Llama-3.1-8B | 2 | (same) | `Prefill: 16969ms (2.071 ms/tok)` `[1] 785` | `ttft=17.531263s prefill=443.84t/s` |
| Llama-3.1-8B | 3 | c8k-guarded-Llama-3.1-8B-NPU2-20260919T121716Z | `Prefill: 17199ms (2.099 ms/tok)` `[1] 785` | `ttft=18.873436s prefill=412.30t/s` |

## Blocker fixed to complete the matrix (Qwen3-4B / Qwen3-VL-4B native legs)

The 4B/VL native runs initially failed `Bf16Ctx: xclbin init failed: No such file
'…/final_bf16_G_K2560_N9728.xclbin'` — four bf16 G/U 9728 artifacts had been moved out of
`engine/npu/xclbins/` (found in `/tmp/xclbins-aside/`). Restored:

- `engine/npu/xclbins/final_bf16_G_K2560_N9728.xclbin`
- `engine/npu/xclbins/final_bf16_U_K2560_N9728.xclbin`
- `engine/npu/xclbins/insts_bf16_G_K2560_N9728.txt`
- `engine/npu/xclbins/insts_bf16_U_K2560_N9728.txt`

These were never git-tracked (their `N10752` and i8 `9728` siblings are), so they are added here
to make the 4B/VL native prefill reproducible on a fresh checkout.

## Bounds (unchanged)

- **8k decode still does not exist** — `ng=1` is forced: the second decode forward needs
  `ctx=8194`, past the baked `MAX_L=8192` per-ctx ELF window. This table is TTFT/prefill only.
- FLM reference is v1.0.4 for every cell (baseline and 2026-09-19 runs); no FLM version change
  occurred mid-campaign.
