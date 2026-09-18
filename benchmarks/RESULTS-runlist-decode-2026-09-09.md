# RESULTS — runlist whole-layer decode (Qwen3-0.6B), 2026-09-09

Goal `mtunui03-ekarlt` (launch-count reduction): wire the FastFlowLM-validated
whole-layer per-ctx ELF executor (`RuntimeLayerEngine`) into the native engine
decode path (`npu_engine_universal.cpp`), replacing the launch-bound 112-launch
split loop.

## Wiring (task-2)

- `engine/npu/src/npu_runlist_bridge.{h,cpp}` — bridge TU isolating npu-infer's
  `ModelConfig` (common.h) from the engine's `model_config.h` (name clash).
- `engine/npu/src/npu_engine_universal.cpp` — `NPU_RUNLIST=1` gate before the
  split-path machinery, gated on dense Qwen3-0.6B (`NC==28 && H==1024 &&
  NV==151936 && !has_moe`). Zaya dispatch runs BEFORE the gate (untouched);
  the split path is the untouched fallback.
- `engine/npu/CMakeLists.txt` + `engine/npu/build_npu.sh` — link
  `npu-infer/src/model.c` (C) + `npu-infer/src/runtime_layer.cpp` (C++) +
  bridge into every `npu_engine_*` variant; link against the runlist-capable
  XRT 2.26.0 (scoped `/usr/local/xrt-runlist/lib`, RPATH set).
- `npu-infer/src/runtime_layer.cpp` `forward()` — **#2150 single-launch**: batch
  all 28 layer runs + lm_head into ONE `xrt::runlist` submit/token (serial
  path retained for the per-layer debug-dump envs). `NPU_RUNLIST_STATS=1`
  prints "29 runs batched -> 1 submit (ctx=N)" per token.

## Measurement (this box, Strix Halo NPU)

Prompt "The capital of France is" → token ids [785 6722 315 9625 374], 12
decode tokens.

| Path | launches/token | decode | first token | continuation |
|---|---|---|---|---|
| split (baseline, `NPU_RUNLIST` unset) | 112 (4 GEMMs × 28 layers) | 455 ms/tok (2 tok/s) | 7119 | — |
| runlist whole-layer (`NPU_RUNLIST=1`) | **1 submit** (29 runs batched: 28 layers + lm_head) | **11.4 ms/tok (88 tok/s)** | 12095 | **Paris. The capital of Italy is Florence…** |

Command:
```
NPU_RUNLIST=1 NPU_LAYER_ELF_DIR=$PWD/npu-infer/captures/txn-elfs \
  LD_LIBRARY_PATH=/usr/local/xrt-runlist/lib \
  ./engine/npu/build/npu_engine_qwen3_0_6b \
  ~/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx 12 ids.txt
```

## Correctness (task-3)

- The runlist path is **byte-identical** to the FastFlowLM runtime captures
  (task-1: `test_runtime_layer` fwd1+fwd2 act+logits vs `round41-regate`
  references, 0 diffs; argmax 397/88 match). corr = 1.0.
- Greedy token parity: "The capital of France is" → "Paris. The capital of
  Italy is Florence…" — matches the FLM runtime's decode.
- Note: the split path's first token (7119) differs from the runlist path
  (12095 → "Paris"). The runlist path is the FLM-runtime-identical (correct)
  reference; the split path is the int8 approximation (the 2-tok/s baseline the
  original goal was trying to replace).

## Honest framing

- The runlist path now does **1 `xrt::runlist` submit/token** (29 runs batched
  into one atomic submit), satisfying the "112 → 1 submit/token" criterion
  with instrumentation (`NPU_RUNLIST_STATS=1`). 88 tok/s.

## Extension to 1.7B + 4B (task-5) — DONE, byte-identical

Packing generalized to model dims (G = K/128, gate/up CH = H/16, lm_head
G = H/128, norms from metadata). Per-model per-ctx ELFs + lm_head ELFs
captured/generated. All verified byte-identical vs the runtime captures:

| model | decode | greedy argmax (tok 1000) | logits vs runtime capture |
|---|---|---|---|
| 0.6B | 94 tok/s | 397 | byte-identical |
| 1.7B | 45 tok/s | 25 | byte-identical (0 diffs) |
| 4B   | 21 tok/s | 738 | byte-identical (0 diffs) |

All three decode "The capital of France is" → "Paris. The capital of …"
(greedy token parity). The 1.7B layer weight BO (30 MB) and 4B layer weight
BO (61 MB) are byte-identical to the runtime captures (0 diffs).
