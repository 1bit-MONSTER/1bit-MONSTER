# ZAYA1-74B-preview — ROCm HIP Benchmark (2026-08-05)

## Summary
Measured live on the current Strix Halo host after a cold reboot, to replace the
stale/blank performance figure in the README catalog.

- **Model:** `ZAYA1PREVIEW-74B-A4B-Q4_K_M.gguf` (45.76 GiB, 74.79B params, 4.89 BPW, GGUF V3)
- **Arch:** `zaya` (MoE, 24 experts, 1 expert/token, 120 layers, 16 heads, 2 KV heads)
- **Source:** `SUPEROXIDES/ZAYA1_PREVIEW_74B-A4B_-_GGUF` (compat: Juste-Leo2 Zyphra llama.cpp layout)
- **Runner:** `zaya-llama.cpp/build-hip/bin/llama-server` (Juste-Leo2 Zaya branch, `-DGGML_HIP=ON`, ggml version 9097)
- **Runtime:** ROCm TheRock 7.15a (`/opt/rocm-therock`), `HSA_OVERRIDE_GFX_VERSION=11.5.1`
- **Hardware:** AMD Ryzen AI MAX+ 395 (Strix Halo), Radeon 8060S (gfx1151), 122880 MiB unified VRAM
- **Offload:** `-ngl 99` (full GPU), `-c 1024`, flash attention on

## Decode (generation) — tok/s
| Run | Tok/s |
|-----|-------|
| gen100 | 16.99 |
| gen256 | 16.81 |
| gen512 | 16.78 |
| gen100 (repeat) | 16.18 |
| gen256 (repeat) | 16.55 / 16.64 |
| 3× sustained 512 | 16.74 / 16.65 / 16.61 |
| **Mean decode** | **≈ 16.7 tok/s** (stdev 0.06) |

## Prefill — tok/s (scales with batch)
| Prompt size | Prefill tok/s |
|-------------|---------------|
| 102 tok | 271.4 |
| 242 tok | 477.2 |
| 262 tok | 555.7 |

## Interpretation
- **Decode ≈ 16.7 tok/s** — confirms and slightly revises the archived preliminary
  figure of **17.9 tok/s** (same methodology, current weights/build/ROCm stack).
- Prefill is 271–556 tok/s and scales with prompt length (batch efficiency).
- Model + KV fully fit in the 122 GiB unified VRAM; ~84 GiB free after load.
- Usable for interactive chat / document Q&A / coding (16.7 tok/s).

## Reproduction
```bash
export LD_LIBRARY_PATH=/opt/rocm-therock/lib/python3.14/site-packages/_rocm_sdk_libraries/lib
export HSA_OVERRIDE_GFX_VERSION=11.5.1 HSA_ENABLE_SDMA=0
zaya-llama.cpp/build-hip/bin/llama-server \
  -m models/ZAYA1PREVIEW-74B-A4B-Q4_K_M.gguf -ngl 99 -c 1024 --port 13308
# POST /v1/completions for timing
```

## ROCm version A/B — 7.2.4 vs TheRock 7.15a (same binary, same model, same prompt)

| Runtime | Decode mean (512-tok, ×4) | Prefill @242 tok |
|---------|---------------------------|------------------|
| **TheRock 7.15a** | **16.84 tok/s** (stdev 0.12) | 502.6 tok/s |
| **ROCm 7.2.4** (ollama-bundled) | **16.97 tok/s** (stdev 0.08) | 496.8 tok/s |

- Difference: **0.13 tok/s (< 1%)** — statistically marginal, both runtimes equal for this workload.
- 7.2.4 stack at `/usr/local/lib/ollama/rocm_v7_2`; TheRock at `/opt/rocm-therock`. The `build-hip` llama-server was itself compiled against 7.2.4, so 7.2.4 is the ABI-native context; running it on TheRock 7.15a loses nothing measurable.
- Conclusion: decode is **~16.8–17.0 tok/s** on either runtime. Runtime choice does not move the number.
