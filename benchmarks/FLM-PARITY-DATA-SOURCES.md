# FLM-Parity Data Sources

Reference index for `benchmarks/flm_parity.sh` (goal `mttxt22c-a6rv75`, task-1 harness).

## 1. Published reference bars (the objective's bar)

FastFlowLM's published benchmark tables live in `amd-oss/fastflowlm/docs/docs/benchmarks/`
(Kraken Point hardware). These are the **official reporting bar**:

| Family | File | Notes |
|---|---|---|
| Dense Qwen3 | `qwen3_results.md` | 0.6B/1.7B/4B/8B decode+prefill 1k–32k |
| Qwen3.6 MoE | `qwen3.6_results.md` | 35B-A3B decode+prefill 1k–32k |
| Llama 3.x | `llama3_results.md` | 3.2-1B/3B, 3.1-8B |
| Gemma 4 | `gemma4_results.md` | E2B/E4B/12B |
| Phi-4 | `phi4_results.md` | mini-instruct |
| Nanbeige 4.1 | `nanbeige4.1_results.md` | 3B |
| LFM2 | `lfm2_results.md` | 1.2B/2.6B |
| Others | `gemma3_results.md`, `gpt-oss_results.md`, `qwen2.5_results.md`, `qwen3.5_results.md`, `smolvla_results.md` | not yet covered by the harness |

> **Cross-hardware caveat.** These tables are Kraken-Point numbers. This box is
> Strix Halo (Ryzen AI MAX+ 395 / XDNA 2). FLM's own on-box re-runs (e.g. the
> `RESULTS-qwen3.6-35b-a3b-npu-flm-2026-07-30.md` v0.9.46 run: 13.65 decode /
> 78.98 prefill @1k) trail the published table by ~20% on decode — the gap is
> hardware, not software.

## 2. On-box FLM (the comparison baseline)

- Binary: `/opt/fastflowlm/bin/flm` (env `FLM`), hidden `flm bench <tag> -i <config.json>`
  writes `bench_<tag>_<yyyymmdd>.csv` with ttft/prefill/decode per context.
- Dense Qwen3 / remaining families: v1.0.4 libs (system install).
- **MoE (Qwen3.6-35B-A3B): v0.9.46 only** — v1.0.x `libqwen3_6_moe_npu.so` NaNs on the
  bf16 GatedDeltaNet recurrence. Staging: `/home/bcloud/.local/flm-v0946/{include,lib/xrt,xclbins,model/Qwen3.6-35B-A3B-NPU2/config.json}`;
  lib md5 `39a6c36a`; config `flm_version=0.9.45`; XRT 2.21.75 or 2.26.0.

## 3. Native engine + weights

- Native engine binaries: `engine/npu/build/npu_engine_<model>` (env `NPU_XCLBIN_DIR`
  points their compiled-in xclbin path at `engine/npu/xclbins`).
- Weights (`.q4nx`) + tokenizers: `~/.config/flm/models/<Model>-NPU2/{model.q4nx,tokenizer.json}`.
- Tokenizer → token IDs: `engine/npu/tokenizer/tokenize` (comma output → `tr ',' ' '`).

## 4. Measurement caveats

- The harness's `NPU_FLM_PREFILL`/`NPU_FLM_DECODE` modes drive **FLM's own NPU libs**
  through the native engine (orchestration) — that is FLM re-run on this box, not the
  native int8/bf16 backend. The native backend is 6–21× behind on prefill/TTFT
  (`RESULTS-qwen3-dense-parity-2026-09-10.md`). Do not present the orchestration numbers
  as "native ≥ FLM".
- TTFT for text = `prompt_tokens / prefill_tok_s`; the published tables' TTFT rows are
  vision-language (image input) only, so text TTFT is derived from prefill throughput.
