# NPU A/B — qwen3_0_6b (ctx=2k, decode=32 tok, reps=3)

| lane | path | decode tok/s | prefill tok/s | TTFT (s) | correctness gate |
|---|---|---:|---:|---:|---|
| FLM (on-box) | flm serve/bench qwen3:0.6b | 64.0 | 1927.09 | 1.01 | FLM-TEXT-OK |
| native NPU | runlist (NPU_RUNLIST=1) | 73.0 | 76.9 | 26.39 | NATIVE-TOKENS-PLAUSIBLE |

- `runlist` vs FLM: native is **114.1%** of FLM decode.

FLM sample: `The capital of France is **Paris**.`
native[runlist] last tokens: `88173,1835,88173,5191,2113,50338,96309,98341`
native[runlist] last tokens: `88173,1835,88173,5191,2113,50338,96309,98341`
native[runlist] last tokens: `88173,1835,88173,5191,2113,50338,96309,98341`
