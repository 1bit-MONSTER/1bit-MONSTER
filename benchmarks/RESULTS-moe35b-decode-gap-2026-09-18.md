# Qwen3.6-35B-A3B MoE decode gap — fresh measurement (moe-wiring landed)

_Captured 2026-09-18 on strixhalo (Ryzen AI MAX+ 395 / Radeon 8060S, XDNA 2, XRT 2.26.0 runlist stack)._

Supersedes the stale `0.32–0.57 tok/s` native figures in
`RESULTS-runlist-decode-35b-moe-2026-09-10.md` (CPU MoE 0.57 tok/s, NPU-fused
M=1 0.32 tok/s). Those were the engine's *old* 80/160-launch split MoE path.
This records the **MoERuntimeLayerEngine** single-launch whole-layer path now
that its build-wiring has been recovered and landed (PR #2608).

## Wiring status (steps 1–2)

- `engine/npu/build_npu.sh` now compiles `npu-infer/src/runtime_layer_moe.cpp`
  into `npu_runlist_moe.o` and adds it to `ENGINE_OBJS` (PR #2608).
- The rebuilt 35B binary (`npu_engine_qwen3_6_moe_35b`) exports all 17
  `MoERuntimeLayerEngine::` symbols (`nm -C`), alongside the dense
  `RuntimeLayerEngine`.

## Smoke correctness (step 3)

`npu-infer/tools/moe_smoke` (layer 1, token 151644):

```
forward(1): EXECUTED
lm_head: HOST path (3-D/Q8_0 source, device lm_head skipped)
logits: argmax=193722 max=0.0242 NaN=0 nonzero=248320 (of 248320)
```

This **supersedes the earlier "NaN=0 because lm_head never ran"** result: the
host lm_head (3-D Q8_0 `[7760,8,8704]`) now actually runs and produces finite,
non-zero logits across the whole 248320 vocab. (The single-layer argmax is
informational only, per the harness.)

## Fresh per-phase timing (step 4)

Measured with a per-phase timing driver (model layer 1, warm device, N=3):

| Phase | Time |
|---|---|
| model_load | ~4 ms |
| init: pack layer-1 weight BO (460 MB) | 210–354 ms |
| embed | 0.02 ms |
| **forward(1) — single layer, single `xrt::runlist` submit** | **5.8–6.4 ms** |
| **lm_head — HOST path (3-D Q8_0, 248320 vocab)** | **18–27 ms** (OpenMP, 32 threads) |

### Fresh decode estimate vs FLM

- 40 layers × ~6 ms = ~240 ms/token for the whole-layer forwards.
- host lm_head ≈ **~20 ms/token** after `#pragma omp parallel for` (was ~470 ms single-threaded).
- Naive full-decode: ~260 ms/token ≈ **3.8 tok/s** — still ~4.5× below FLM's
  17.48 tok/s @1k, but ~7–12× above the stale native 0.32–0.57 figures and
  ~2.7× above the pre-OMP ~1.4 tok/s measurement.

## Bottleneck (step 5 direction)

Two walls, in order:

1. **The layer ELF does not consume expert weights.** `forward()`'s own comment
   (and `RESULTS-moe35b-flm-real-arg-bindings-2026-09-16.md`) records that the
   per-ctx `moe_layer_ctxN.elf` runs the shared-expert/attention/norm/router
   portion but the routed expert FFN is carried by *other* kernels (FLM's
   `mm.xclbin` + `dequant_mm.xclbin`) — it is not yet a complete MoE layer. The
   5.8 ms forward therefore measures the non-expert slice, not full MoE decode.
   Fix direction: wire the expert GEMM/dequant kernels into the runlist (or the
   engine's existing `NPU_MOE` expert path) after the layer ELF.
2. **Host lm_head ≈ 470 ms/token was the dominant cost — now FIXED.** The 3-D
   Q8_0 lm_head cannot be expressed by the device lm_head kernel
   (`ndim==3 → n_tiles==0`, #2434), so it dequantizes + matmuls on the CPU.
   Landed two focused perf PRs: (a) accumulate (not assign) across the 8
   tile_col slices (the old `out[v] = acc` kept only the last 256 of 2048 hidden
   dims — finite but wrong logits) and (b) `#pragma omp parallel for` over
   tile_row. Result: 470 ms → ~20 ms/token (~20×). The layer forward is now the
   dominant term.

## Verdict

Build-wiring landed (PR #2608), symbols exported, smoke produces finite non-zero
logits end-to-end (host lm_head path, now full-width after the accumulate fix).
The fresh decode is ~3.8 tok/s (naive), up from ~1.4 tok/s after the lm_head
parallelization. Remaining parity gap is the layer forward (240 ms/token) and,
more fundamentally, the expert-FFN integration — the concrete next optimization
target.
