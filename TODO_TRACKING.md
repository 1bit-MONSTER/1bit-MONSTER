# TODO/FIXME/HACK Tracking

> Auto-generated from codebase audit 2026-07-18. Check off when resolved.

- [ ] `tests/backends/backend_hip.cpp:143` — kernel templating needed for multi-arch
- [ ] `tests/backends/parallel_moe.h:178` — skeleton only, NPU forward not wired
- [x] `engine/npu/src/npu_engine_universal.cpp:639` — actual NPU attention kernel not called
    → Fixed: insts_i8_KV_* stub replaced (82→2396 insts), auto-detect removed NPU_ATTN=1 gate, all 4 call sites (prefill×2, boot decode, batch decode) wired with NPU attention via attn.xclbin ✅
- [x] `engine/npu/src/npu_engine_hybrid.cpp:166` — KV xclbin Q@K^T score not wired
    → Code already calls attn.run() via (*k)(3, *bI, ins.size(), *bQ, *bK, *bV, *bOut).wait() — TODO was stale ✅
- [ ] `tools/gpu_npu_bridge.cpp:244` — NPU dispatch using ternary xclbins
- [ ] `tools/gpu_npu_bridge.cpp:367` — Forward pass GPU-only baseline, NPU TODO
- [ ] `src/backend_zamba2.cpp:36` — tokenizer.ggml.* not read from GGUF
- [ ] `src/zamba2_engine_hip.hip:244` — fused Mamba2 HIP kernel
- [ ] `src/server/server.cpp:114` — tools logging: only print tool name, elide args
- [ ] `src/server/server.cpp:115` — debug level support
- [ ] `src/server/rest_handler.cpp:622` — avoid loading twice
- [ ] `include/rocm_cpp/tokenizer.h` — [redacted] tokenizer TODO
- [ ] `engine/fusion/zero_copy/test_parallel_real.hip:87` — hardcoded paths, replace with env vars
