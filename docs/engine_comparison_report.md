# Engine comparison benchmark — 1bit-systems zaya_server vs reference engines

Same GGUF files, same host, one uniform OpenAI `/v1/chat/completions` surface, across text / image / audio / video / single-turn / multi-turn / function-call / structured-output scenarios on the selected compute backends (ggml_cuda / ggml_vulkan / ggml_metal / ggml_cpu / cpu / ...).

Numbers are tokens/second (higher is better). `—` = not applicable / skipped, `fail` = errored at runtime, `n/a` = combination never attempted.

## Software / hardware

| Component | Version / detail |
|---|---|
| 1bit-systems zaya_server | git `05ee30a05` (backends: ) |
| llama.cpp | `/home/bcloud/1bit-systems/third_party/llama.cpp/build/bin/Release/llama-server` |
| vLLM | endpoint `http://127.0.0.1:8000` (connect-only) |
| GPU | unknown |


## Methodology

- Each `(engine, backend, model)` group launches its server once; all of that group's scenarios run against it, so per-scenario timings exclude model-load cost.
- Metrics come from the **streamed** response: `ttft` is time-to-first-token (prefill latency proxy), `prefill_tps = prompt_tokens / ttft`, and `decode_tps = (completion_tokens - 1) / (t_last - t_first)`.
- DiffusionGemma denoises whole blocks (no token stream), so it is run non-streaming and its `decode_tps` is wall-clock tokens/second.
- Greedy sampling (`temperature=0`); one warmup request per server is discarded.
- The headline per-engine tables are the **single-stream, MTP-off** baseline. MTP on/off and parallel-request scaling are reported in their own sections below.

## Performance ratio — 1bit-systems zaya_server vs reference engines

Geomean of 1bit-systems zaya_server's per-scenario speedup over each reference engine on the **same backend**, across every scenario both engines ran (single-stream, MTP-off). A value **> 1.0× means the hero engine is faster** (for decode / prefill throughput) or lower-latency (for TTFT); `—` = no overlapping cells. Per-scenario ratios are in each model's section below.

_No overlapping 1bit-systems zaya_server / reference cells to compare._

## qwen3-06b  (`qwen3-06b`)

**Decode throughput (tok/s)**

| Scenario | 1bit-systems zaya_server · gpu | llama.cpp · gpu |
|---|---:|---:|
| text_short | — | 253.4 |
| prefill_128 | 0.2 | 357.6 |
| prefill_512 | — | 335.1 |

**Prefill throughput (tok/s)**

| Scenario | 1bit-systems zaya_server · gpu | llama.cpp · gpu |
|---|---:|---:|
| text_short | 1.8 | 9744.6 |
| prefill_128 | 3.8 | 1972.2 |
| prefill_512 | 3.2 | 7640.7 |

**Time to first token (ms, lower is better)**

| Scenario | 1bit-systems zaya_server · gpu | llama.cpp · gpu |
|---|---:|---:|
| text_short | 1097495.1 | 203.4 |
| prefill_128 | 48217.5 | 91.3 |
| prefill_512 | 172695.1 | 70.8 |

**Performance ratio — 1bit-systems zaya_server vs reference (> 1.0× = {HERO_DISPLAY} faster)**

_Decode throughput_

| Scenario | vs llama.cpp · gpu |
|---|---:|
| text_short | — |
| prefill_128 | 0.00× |
| prefill_512 | — |

_Prefill throughput_

| Scenario | vs llama.cpp · gpu |
|---|---:|
| text_short | 0.00× |
| prefill_128 | 0.00× |
| prefill_512 | 0.00× |

_Time to first token (latency; > 1.0× = 1bit-systems zaya_server lower)_

| Scenario | vs llama.cpp · gpu |
|---|---:|
| text_short | 0.00× |
| prefill_128 | 0.00× |
| prefill_512 | 0.00× |

## Output quality — 1bit-systems zaya_server vs llama.cpp

Both engines decode the **same GGUF greedily** (temperature=0) on the same backend, so their outputs should agree closely. `similarity` is a whitespace-normalized SequenceMatcher ratio between the two outputs (1.00 = identical); low similarity, an invalid JSON object in `json_mode`, or a missing tool call in `function_call` flags an output-quality problem on one side. Prefill scenarios (8-token outputs) are excluded. Side-by-side excerpts follow the table, lowest agreement first.

_No overlapping ok cells with captured output to compare._

## Image editing (stable-diffusion)

Same input image, prompt, resolution, step count, cfg and seed for every engine. Timings are each engine's **own pipeline timers**, so weight-file loading and HTTP/process overhead are excluded on both sides. Lower is better.

_No image-edit cells were run (see the `image_edit` scenario)._

## MTP / NextN speculative decoding (on vs off)

Single-stream decode tok/s with MTP/NextN speculative decoding off vs on (the hero engine only). Speedup `< 1.0×` means speculation cost more than it saved for that cell — expected when the fused full-model decode path is already the fast path.

_No MTP on/off pairs were run (use `--mtp off,on`)._

## Parallel-request scaling (concurrency)

`decode/req` is the mean per-request decode tok/s; `aggregate` is the system-wide decode throughput (total generated tokens / the wall window during which any sequence was decoding) when N identical requests are fired at one server at once.

_No parallel-request cells were run (use `--concurrency 1,4,8`)._

## Function-calling correctness

_No function-call cells were run._
