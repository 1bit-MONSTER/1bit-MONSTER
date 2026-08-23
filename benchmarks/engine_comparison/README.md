# Cross-Engine Inference Benchmark (1bit-monster fork)

A reproducible benchmark harness that compares inference engines on the
**same GGUF files**, the **same host**, through **one uniform OpenAI
`/v1/chat/completions` surface**.

**This directory is vendored from [zhongkaifu/TensorSharp](https://github.com/zhongkaifu/TensorSharp)
(`benchmarks/engine_comparison/`), Copyright (c) 2026 Zhongkai Fu, BSD 3-Clause
License (see upstream `LICENSE`). Upstream's copyright notice is retained here;
modifications are the 1bit-monster additions listed below.

- the **`zaya` engine** (`engines.py` → `ZayaServer`): launches
  `build/zaya_server` against any model the harness can point at,
- **`timings` support in zaya_server itself** (`tests/zaya_server.cpp`): the
  server now emits the llama.cpp-compatible `timings.predicted_per_second`
  block in its `/v1/chat/completions` responses, so zaya's decode rate is
  server-measured (burst-immune) and the upstream harness works **unmodified**
  against it,
- a 1bit-oriented `benchmark_config.json` (models from `models/`, zaya +
  llama.cpp engines, HIP/CPU backends),
- `report.py`'s hero engine is now configurable (`BENCH_HERO_ENGINE`,
  default `zaya`) instead of hard-coded to TensorSharp.

## Engines

| Engine | Launch | Notes |
|---|---|---|
| `zaya` | `build/zaya_server --model <file> --port 8090` | Primary engine under test. Auto-detects its backend (HIP > Vulkan > Zamba2 > GGUF-CPU > CPU). Accepts `.gguf` (Qwen2/Qwen3/Mamba families) and native `.1bp`/`.h1b` files. |
| `llamacpp` | vendored `third_party/llama.cpp` `llama-server` | Reference engine; emits its own server-side `timings`. |

## Honest measurement caveats (zaya)

The harness derives `ttft` / `prefill_tps` from the *stream window* and
`decode_tps` from the engine-reported `timings.predicted_per_second` when
present (burst-immune). For zaya, be aware of two asymmetries:

1. **zaya's `timings` span prompt prefill + decode.** The router's timer
   starts before the prompt forward pass, so `predicted_per_second` is
   end-to-end generation throughput, not a pure decode rate. It is honest
   and server-measured, but systematically lower than llama.cpp's pure-decode
   number on long prompts.
2. **zaya's SSE is replayed post-generation** (`router.infer` is
   all-or-nothing today), so its TTFT / prefill_tps cells are **not
   meaningful**. Exclude zaya TTFT columns from any head-to-head comparison;
   decode_tps is the reliable metric. (Fix = per-token router callback +
   true streaming, see `tests/backends/token_router.h`.)

zaya currently runs **text / multi_turn scenarios only**; `function_call`,
`json_mode` and multimodal cells are gated out for it in
`config.applies()` (llama.cpp still runs them).

## Run

```bash
# 0. Build both engines
cmake --build build --target zaya_server -j8          # zaya
# (llama-server: see third_party/llama.cpp build instructions)

# 1. Full matrix (defaults: zaya + llamacpp, hip backend, small models)
python3 benchmarks/engine_comparison/run_matrix.py

# 2. Report + CSV (results/*.json -> docs/engine_comparison_report.md)
python3 benchmarks/engine_comparison/report.py
```

Key flags (same as upstream): `--engines zaya,llamacpp`, `--backends hip,cpu`,
`--models qwen35-9b`, `--scenarios text_short,multi_turn`,
`--concurrency 1,4` (parallel-request scaling), `--skip-existing`.
Host paths are retargetable via env vars (`BENCH_ZAYA_SERVER`,
`BENCH_LLAMA_SERVER`, `BENCH_MODEL_ROOT`, `BENCH_RESULTS`, ...) — see
`config.py`.

## Results

`run_matrix.py` writes one JSON per cell to `results/` (never overwrites by
default), plus per-group server logs under `results/logs/`. `report.py`
aggregates them into `docs/engine_comparison_report.md` and
`results/results.csv`. Commit the JSON results with each run so every number
in a published report is reproducible.
