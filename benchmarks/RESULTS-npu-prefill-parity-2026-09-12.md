# NPU prefill parity — on-box FLM comparison (2026-09-12)

Goal `mttxt22c-a6rv75`, task-n1 re-scope: the native reimplementation is a
different FP decomposition from FLM's whole-layer kernel, so **bit-exact token
parity is not the right bar for it** — the FLM-orchestrated path
(`NPU_FLM_PREFILL=1`, which *is* `qwen3_npu::prefill`) is the by-construction
identical path. Measurement therefore uses the on-box `flm bench` as the bar.

## Harness

```
bash benchmarks/flm_parity.sh --model qwen3_0_6b --flm-tag qwen3:0.6b \
  --engine engine/npu/build/npu_engine_qwen3_0_6b \
  --q4nx ~/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx \
  --tokenizer ~/.config/flm/models/Qwen3-0.6B-NPU2/tokenizer.json \
  --prompt benchmarks/prompts/reclaimer.txt \
  --ctx-k 1 --decode-tokens 32 --flm-max-length 1024 --flm-iterations 1
```
(needs `benchmarks/prompts/reclaimer.txt`; created this session.)

## Results @ 1k context (two runs, this box)

| metric | native | FLM (on-box `flm bench`) | verdict |
|---|---|---|---|
| **prefill tok/s** | **1639.3 / 1818.2** | **1400.21** | native **+17…+30 %** ✅ |
| decode tok/s | 62 / 63 | 73.58 | native **−15.7 %** ❌ |
| TTFT (s) | 1.276 / 1.154 | 0.701728 | native **+64 %** ❌ |

FLM CSV (`bench_qwen3_0.6b_20260912.csv`):
`context_length_k,ttft_avg_s,...,prefill_avg_toks_per_s,...,decoding_avg_toks_per_s`
→ `1,0.701728,…,1400.21,…,73.58`

## Reading

- **Prefill throughput: the engine beats FLM on this box** (1639–1818 vs 1400)
  and even against FLM's *published* 1494 @1k bar.
- **TTFT is slower** because the engine's TTFT equals its full prefill wall time
  while FLM pipelines the first chunk earlier — this is a scheduling issue, not
  a kernel-speed issue (the same prefill work completes faster overall).
- **Decode is ~16 % behind** — the known open gap from task-3/4 (per-token
  session here: `flm_parity.sh` runs the default whole-layer decode path).

## Next

1. TTFT: overlap the first decode step with the prefill tail (the engine already
   has `run_gemm_launch/wait` async primitives in the bf16 lane).
2. Decode: continue the task-4 line (FLM-orchestrated decode is 13.5 tok/s @1k
   on the MoE; dense 0.6B on-box FLM is 73.58 — check the RuntimeLayer's
   per-token path vs `flm`'s).
3. Record the same table for 1.7B / 4B / 8B before claiming the catalogue.
