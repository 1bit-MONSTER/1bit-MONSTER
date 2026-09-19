# NPU A/B — qwen3_0_6b (ctx=1k, decode=32 tok, reps=3)

| lane | path | decode tok/s | prefill tok/s | TTFT (s) | correctness gate |
|---|---|---:|---:|---:|---|
| FLM (on-box) | flm serve/bench qwen3:0.6b | 75.24 | 1427.09 | 0.69 | FLM-TEXT-OK |
| native NPU | runlist (NPU_RUNLIST=1) | 88.0 | 83.3 | 12.3 | NATIVE-TOKENS-PLAUSIBLE |

- `runlist` vs FLM: native is **117.0%** of FLM decode.

FLM sample: `The capital of France is **Paris**.`
native[runlist] last tokens: `47572,22520,47572,4261,12240,4261,70428,47572`
native[runlist] last tokens: `47572,22520,47572,4261,12240,4261,70428,47572`
native[runlist] last tokens: `47572,22520,47572,4261,12240,4261,70428,47572`
