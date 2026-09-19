# NPU A/B — qwen3_0_6b (ctx=1k, decode=32 tok, reps=3)

| lane | path | decode tok/s | prefill tok/s | TTFT (s) | correctness gate |
|---|---|---:|---:|---:|---|
| FLM (on-box) | flm serve/bench qwen3:0.6b | None | None | None | SKIPPED |
| native NPU | runlist (NPU_RUNLIST=1) | 89.0 | 83.3 | 12.1 | NATIVE-TOKENS-PLAUSIBLE |

- FLM lane skipped (--skip-flm): no parity claim made.

FLM sample: ``
native[runlist] last tokens: `47572,22520,47572,4261,12240,4261,70428,47572`
native[runlist] last tokens: `47572,22520,47572,4261,12240,4261,70428,47572`
native[runlist] last tokens: `47572,22520,47572,4261,12240,4261,70428,47572`
