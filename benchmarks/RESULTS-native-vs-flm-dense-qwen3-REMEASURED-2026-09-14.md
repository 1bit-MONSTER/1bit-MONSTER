# Native vs FLM across the dense Qwen3 family — RE-MEASURED (2026-09-14)

**This supersedes the prefill/TTFT rows in
`RESULTS-coverage-qwen3-dense-2026-09-13.md`.** That document reports native
*behind* FLM on prefill for 1.7B/4B/8B (gap widening −13.3% → −15.6% → −22.2%)
and calls it a structural "native degrades faster with model size" result. Those
numbers predate the cross-block GEMM pipelining, which lands between the two
measurements. With the current build native **beats** FLM on prefill and TTFT
for every dense Qwen3 size, and the lead is largest where the old doc said it
was most negative.

## Method

`benchmarks/flm_parity.sh` with `FLM_PARITY_TRUE_NATIVE=1`, a tokenizer-exact
**1024-token** prompt (`~/npu-build/parity/prompt1024.txt`), `--decode-tokens 32`,
engine + q4nx + tokenizer per model, and the correct FLM tag per model
(`qwen3:0.6b` / `qwen3:1.7b` / `qwen3:4b` / `qwen3:8b`).

Matching the prompt length matters: the harness's default prompt (the
"reclaimer" story) tokenizes to 2088 tokens while `flm bench` runs with
`max_length=1024`, so the *default* invocation compares 2088-context native
against 1024-context FLM. Use a 1024-token prompt for a 1k comparison.

## Results (native / FLM on-box)

| model | decode tok/s | prefill tok/s | TTFT (s) | verdict |
|---|---|---|---|---|
| Qwen3-0.6B | 79.3 / 77.6 | **1905.2** / 1100.9 | **0.537** / 0.718 | beats on all three |
| Qwen3-1.7B | 40.0 / 40.3 | **1319.3** / 766.6 | **0.776** / 1.030 | prefill +72%, TTFT −25% |
| Qwen3-4B | 19.0 / 19.0 | **673.9** / 408.8 | **1.520** / 1.931 | prefill +65%, TTFT −21% |
| Qwen3-8B | 11.0 / 10.8 | **469.7** / 294.9 | **2.180** / 2.677 | prefill +59%, TTFT −19% |

| model | decode | prefill | TTFT |
|---|---|---|---|
| 0.6B | +2.3% | **+73%** | 25% faster |
| 1.7B | −0.6% (tie) | **+72%** | 25% faster |
| 4B | +0.2% (tie) | **+65%** | 21% faster |
| 8B | +2.2% | **+59%** | 19% faster |

Decode is a wash at 1.7B/4B (within ±0.6%) and a small native win at 0.6B/8B.
Prefill and TTFT are native wins everywhere, by a wide margin.

Note the absolute rates collapse with model size on both sides (FLM prefill
1100.9 → 766.6 → 408.8 → 294.9 tok/s), so the two implementations scale
similarly in absolute terms; the difference is that native keeps a constant
~1.6-1.7x prefill lead rather than losing ground.

## Correctness gates (unchanged, all equal the byte-exact `NPU_RUNLIST=1` path)

Per-model boot tokens with the bf16 prefill, no env overrides:

| model | npt | boot | trusted |
|---|---|---|---|
| 0.6B | 256 / 512 / 1024 / 2048 | 1614 / 220 / 25 / 220 | same |
| 1.7B | 1024 / 2048 | 220 / 220 | same |
| 4B | 1024 | 220 | same |
| 8B | 1024 | 220 | same |

## Still open

- **nh32 (4B/8B) above 1024 context**: there is no nh32 2048-context capture, so
  (1024, 2048] falls back to the CPU attention reference — correct but ~152 s
  (boot=220 verified). Capturing from the 4B model faults the NPU
  (`aie2_dump_ctx: Fatal error task ID: 0`), which recovers on its own but is not
  something to do casually; the capture harness
  (`npu-infer/tools/capture/run_qwen3_prefill`) is only known-safe for
  0.6B/1.7B. A safer harness (it hardcodes `qwen3_npu model(config, &npu, 4096)`)
  is the prerequisite.
- Single-run numbers carry FLM's own run-to-run spread (~5-6% on prefill/TTFT),
  so re-run before quoting a small gap.

## Provenance

The pipelining and the nh32/nh20 long-context captures came from a concurrent
session working in this same worktree; this document re-measures their effect
end-to-end. The 0.6B long-context captures (nh16 1024/2048) and the
shape+context guard on the ELF slot came from this session.
