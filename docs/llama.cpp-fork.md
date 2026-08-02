# llama.cpp fork sync

Tracking state of the [bong-water-water-bong/llama.cpp](https://github.com/bong-water-water-bong/llama.cpp) fork
(the Zaya1/paged-KV development line). No submodule — this document is the
sync record. Bump it whenever the fork moves.

## Current sync point

- **Fork branch**: `master`
- **Merge commit**: `0807d70be21914a2ed616b500f7b10258d029f7a`
- **Date**: 2026-08-01
- **Source PR**: bong-water-water-bong/llama.cpp#2 — *feat: paged KV cache — fixed-pool GPU tensors with CPU backing store*
- **Local checkout**: synced to this commit 2026-08-02 (was at PR head `329cb3241`), rebuilt `-DGGML_VULKAN=ON -DBUILD_SHARED_LIBS=OFF`
  (static libs required by `zaya_server` import — without `BUILD_SHARED_LIBS=OFF` ggml builds shared and the
  import check in `CMakeLists.txt` silently fails). Vulkan backend verified on Strix Halo: `llama-bench` sees
  Radeon 8060S (RADV, KHR_coopmat); `unified_server` relinked against the synced statics.

## What shipped in this sync

### Feature: paged KV cache (PR #2)
- Fixed-pool GPU KV tensor pool with CPU backing store
- `llama_kv_cache_paged` + `llama_kv_paged_scorer` (heuristic scorer factory, drafter-model hook reserved)
- Zaya arch support: `LLM_ARCH_ZAYA`, CCA attention (even layers) + MoE (odd layers), Q4NX/1BP conversion (`conversion/zaya.py`, `gguf-py`), ROCm ops (`hip_zaya_ops.hip`)
- ONNX runtime integration (`src/llama-onnx.{h,cpp}`, gated on `onnxruntime_FOUND`)

### CI/build fixes landed on top
| Commit | Fix |
|--------|-----|
| `a14293be` | missing `ggml_graph_dump_txt` prototype (`-Werror=missing-prototypes`) |
| `3d4fa9cf` | unused `n_gpu_layers` (`-Werror=unused-parameter`) |
| `06c3be13`…`b0aad5d1` | `llama-onnx.cpp` only compiled when onnxruntime found + `#ifdef LLAMA_ONNX` guard |
| `ab2baf23` | dead `n_written` in `repool_simple` (macOS `-Wunused-but-set-variable`) |
| `de538c0` | `PRId64` in ZAYA_DEBUG fprintf (macOS `-Werror=format`) |
| `a8576f5` | test-llama-archs: zaya skipped with FIXME (see #1357) |

### Known issues (tracked in 1bit-systems)
- **#1357** — zaya CCA conv graph produces invalid shapes (`ggml_ssm_conv` + `conv_1d_grouped` yield `n-4` outputs vs `n` expected; 1-token/seq reserve graphs abort). Arch never ran in llama.cpp; test excluded pending a fix with the intended conv semantics.
- **#1358** — macOS `test-thread-safety` leaks 231MB under `leaks -atExit` (only remaining red check; needs a macOS box).

## Status
- Linux CI (cpu x64/arm64, full `test-llama-archs`): **green** (verified locally too: 114 arch/config combos OK, exit 0)
- macOS tvOS/visionOS: green; macOS x64/arm64: red on #1358
- Self-hosted GPU jobs (strixhalo runners): pending

## Build note
Fork default branch is `prism`; the paged-KV work lives on `master` (PR #2
was merged there directly). Zaya1 branch retained for further development.
