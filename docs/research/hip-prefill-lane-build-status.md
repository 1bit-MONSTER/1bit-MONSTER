# HIP prefill lane (D2) — build status (issue #1942)

> **2026-09-02 (strixhalo).** The hybrid prefill/decode policy (issue #1942)
> recommends path **D2**: give the HIP prefill lane a llama.cpp context by
> building the **vendored llama.cpp with a HIP backend** (`GGML_HIP=ON`,
> TheRock toolchain), so it can prefill and emit a `llama_state` blob that the
> HRX decode lane consumes. This file records the concrete build state found
> and the exact recipe, so the next session can build on it rather than
> re-derive it.

## What was verified

1. **The vendored llama.cpp supports GGML_HIP** — `third_party/llama.cpp/ggml/CMakeLists.txt`
   exposes `option(GGML_HIP ...)` (default OFF). The existing repo build dir
   (`third_party/llama.cpp/build`) is configured `GGML_VULKAN=ON, GGML_HIP=OFF`
   (the GGUF fallback path), so D2 needs a **separate** HIP build dir.

2. **Blocker found + resolved (HIP compiler):** CMake **rejects `hipcc`** as
   `CMAKE_HIP_COMPILER` (`CMakeDetermineHIPCompiler.cmake:73`: "This is not
   supported. Use Clang directly"). With `/home/bcloud/therock100/bin/amdclang++`
   as the HIP compiler, `GGML_HIP=ON` configures cleanly:
   - HIP compiler identification **Clang 23.0.0** (TheRock), `hip::amdhip64`
     resolved as SHARED_LIBRARY, **"HIP and hipBLAS found"**, "Including HIP
     backend", `ggml version 0.20.0` (`4df29be4f`).
   - Project builds with `-march=native` on the CPU backend; ccache active.

## Configure recipe (verified 2026-09-02)

```bash
cmake -S third_party/llama.cpp -B /tmp/llama-hip \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_HIP=ON \
  -DCMAKE_C_COMPILER=/home/bcloud/therock100/bin/amdclang \
  -DCMAKE_CXX_COMPILER=/home/bcloud/therock100/bin/amdclang++ \
  -DCMAKE_HIP_COMPILER=/home/bcloud/therock100/bin/amdclang++ \
  -DLLAMA_CURL=OFF
cmake --build /tmp/llama-hip --target llama ggml-hip ggml-cpu -j 16
```

## Build result

**PASS (2026-09-02, strixhalo).** `cmake --build /tmp/llama-hip --target
llama ggml-hip ggml-cpu -j16` completed **exit 0, 100%** (see
`/tmp/llama-hip-build.log`), producing:

- `libllama.so.0.1.0` — links `libggml-base`/`libggml-cpu`/`libggml-hip`
- `libggml-hip.so.0.20.0` — **HIP-accelerated**: `ldd` shows
  `libhipblas.so.3`, `libamdhip64.so.7`, `libdrm_amdgpu.so.1`
- `libggml-cpu.so.0.20.0`, `libggml-base.so.0.20.0`

This resolves the D2 blocker ("the engine's vendored llama.cpp has **no HIP
backend**") — with the TheRock HIP compiler (`amdclang++`, Clang 23.0.0) and
the system Debian-multiarch HIP runtime, `GGML_HIP=ON` configures and the HIP
lane **builds and links the HIP runtime**. The prefill lane is now an
environment, not a question.

## Next steps after the build

1. Confirm the HIP lane emits a `llama_state` blob (the D2 state-format
   round-trip gate is already PASSED per issue #1942 and
   `hybrid-prefill-decode.md` — the vendored engine llama.cpp and the HRX
   bundle's `libllama` round-trip the blob byte-identically).
2. Wire the hybrid policy (HIP prefill → llama_state blob → HRX warm decode).
3. Run the correctness gate (hybrid continuation == pure-HIP greedy) + the
   benchmark gate.
