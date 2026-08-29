# BENCHMARK — HRX vs HIP llama.cpp on gfx1151 (WS-12 P1)

Measured 2026-08-29 on this machine (Strix Halo, **gfx1151**, 32 cpu, 122 GiB).
Same model both sides: **Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf** (18.6 GB).

| Backend | binary | device | note |
|---|---|---|---|
| HRX | `llama-hrx-b59` (self-contained, `libhrx`/`libloomc`) | `HRX0` (gfx1151) | AMD staging build |
| HIP | llama.cpp `4df29be4f` built with HIP/hipBLAS on gfx1151 | `ROCm0` (gfx1151) | the same GPU |

Both are `llama-server`; identical `/v1/chat/completions`, timings read from
llama-server's own fields (`prompt_per_second` = prefill, `predicted_per_second`
= decode).

## Short prompt (cold first-token prefill — "capital of France")

| Backend | prompt tok/s (cold) | gen tok/s (cold) |
|---|---|---|
| HRX | 143.5 | 36.8 |
| HIP | 171.0 | 30.0 |

**HIP is ~19% faster on cold short-prompt prefill.** Gen (decode) is close.

## Warm short prompt (it2/it3, KV cache warm, incremental)

| Backend | prompt tok/s (warm) | gen tok/s (warm) |
|---|---|---|
| HRX | ~72–75 | ~175 |
| HIP | ~46–62 | ~70 |

**HRX decode is ~2.5x to 3x faster warm** — its fused kernels avoid the
memory-bound row-gather that dominates HIP's per-token decode on this GPU.

## Large prefill (true prefill — the RFC's headline claim)

| Backend | prompt tokens | prefill tok/s | result |
|---|---|---|---|
| HIP | 1815 | **1313** | completed, correct |
| HIP | 7592 | **1227** | completed, correct |
| HRX | 1815 / 7592 | — | **FAILED CLOSED** |

### HRX fail-closed root cause (important)

On any prompt large enough that the decode graph contains a row-gather, HRX
is unsupported:

```
E graph_compute: unsupported HRX node 2989: GET_ROWS
   output=3663:f32[2048,0,1,1] <- GET_ROWS
   inputs=[3661:f32[2048,512,1,1]<-MUL_MAT, 3662:i32[0,1,1,1]] consumers=[2991:ADD]
E graph_compute: ggml_backend_sched_graph_compute_async failed with error -1
E llama_decode: failed to decode, ret = -3
```

This happens at the **first decode batch** (`off = 0, n_batch = 2048`) when the
graph needs a `GET_ROWS` node — regardless of exact sequence length (triggered
at 1815 and 7592 tokens; sequence length just determines *when* the node is
reached). This is the same **fail-closed** contract WS-12 FINDINGS §4 already
documented (`unsupported HRX node 0: GET_ROWS`). Short prompts sailed through
because the whole sequence stayed within the set of fused HRX nodes.

## Conclusion vs the RFC claim

- **RFC claim (30–50% prefill uplift, parity→+15% decode) is NOT reproduced on
  gfx1151 for this model.**
  - Real large prefill: **HIP wins decisively** (1227–1313 tok/s). HRX could not
    complete any large-prefill run — it fails closed on `GET_ROWS`.
  - Cold short-prompt prefill: HIP slightly ahead (171 vs 143 tok/s).
  - **Warm decode: HRX wins big** (~175 vs ~70 tok/s) because its kernel JIT
    produces tightly-fused per-token kernels.
- **Interpretation:** HRX's win is on **decode latency for short, cache-warm
  sequences**, not on general prefill. Its compiler (Loom/IREE, "kernels on the
  fly") specializes the per-token path, but the AMD `ggml-hrx` backend only
  fuses a narrow node set and **is not a general GGUF backend** yet. The RFC
  preface itself said "doesn't currently qualify as a GGML backend... fused
  Qwen3-specific hardcoded compute" — this measurement confirms that.
- **Caveat:** this is one model, one quant (Q4_K_M), one GPU. The RFC's 30–50%
  prefill number likely comes from a different target/configuration (and/or a
  model fully inside the fused set). It was not reproducible here.

## Artifacts

- `bench_hrx_vs_hip.sh` — short-prompt chat bench (HRX + HIP concurrently).
- `bench_prefill.sh` — large cold-prompt prefill bench.
- Logs: `/tmp/bench-hrx/{hrx,hip}.log`.
- HIP llama.cpp baseline: `third_party/llama.cpp/build-hip/bin/{llama-server,llama-bench}`.
