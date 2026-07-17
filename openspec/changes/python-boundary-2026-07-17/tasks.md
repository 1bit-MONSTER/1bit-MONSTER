# Tasks — Python→Native/Rust boundary

Ordered by value/risk. Each is independent; do not batch.

## Ready now (low risk)

- [x] **T1/T3. Native tokenizer wired (DONE, 2026-07-17).** Re-scoped after inspecting
      canonical: `src/server` already tokenizes natively via the FLM engine (no Python).
      The real gap was `daemon/npu-cppd.py`, which was wired to TWO BROKEN native
      tokenizers: `fused-engine --tokenize-only` (emitted nothing) and
      `engine/npu/tokenizer/detokenize` (mangled byte-level spaces:
      `9707 11 1879 0` -> `Hello,\u0120world\\\`). The C tokenizer
      `engine/npu/tokenizer/tokenize` also mis-encodes (`world!` -> wrong ids).
      Fix: point the daemon at `engine/fusion/tokenize` (pure C++17, verified bit-exact
      encode+decode, clean roundtrip), added `engine/fusion/Makefile` build target,
      gitignored the binary. No Python tokenizer lib anywhere in the serving path now.
- [ ] **T2. Ship the Rust router.** Fold `unified-router.py` routing policy into
      `rust/onebit` (or land `reference/unified-router-rs/` as a bin). Retire the `.py`.
- [ ] **T3. Remove `tokenizers` / tokenize-subprocess from `daemon/npu-cppd.py`** —
      point tokenization at the native `tokenize` CLI/lib (kills the `pip install
      tokenizers` + torch2aie-venv tokenize dependency).

## Blocked / needs fix first (medium risk)

- [x] **T4. Fix `free(): invalid size` in `npu-infer/src/npu_engine_stdio.cpp`.** DONE.
      Root cause (ASan): NOT a data heap bug — a null-vtable SEGV in `I8Ctx::~I8Ctx`
      at exit. XRT global singletons are torn down before the local
      xclbin/hw_context/kernel dtors run (static-destruction-order fiasco). Fix:
      `std::_Exit(0)` after the stdin loop — skip the cross-boundary teardown, let the
      OS reclaim. Verified clean (exit 0, no ASan/glibc error) on token/continue/reset.
      NEW finding surfaced: engine repeats token 9707 (sampling/output-quality bug —
      separate from the crash; see T7/new T10).
- [ ] **T5. Pin known-good xclbin set + add `r.wait()` timeout** in the NPU engine so a
      faulting kernel fails loudly instead of hanging the device. (findings §3, bug 2)
- [ ] **T6. Replace the torch decode loop** (`tools/npu_runner.py`) in
      `daemon/npu-cppd.py`. RETARGETED (2026-07-17): do NOT use the buggy
      `npu_engine_stdio` (T10). Instead proxy to `flm serve` — FastFlowLM is the
      validated native NPU engine and VERIFIED coherent on hardware today
      (`The capital of France is **Paris**.`). It already speaks OpenAI HTTP
      (`flm serve <tag> --port N`). Simplest path: retire the Python daemon entirely,
      run `flm serve` behind the Rust unified-router (which already routes to
      `qwen3-0.6b-FLM`). Eliminates torch + the Python HTTP wrapper + the Python
      decode loop in one move — a fully Python-free serving path.

## Follow-ups

- [ ] **T7. Move lm_head off CPU** in the NPU engine (151,936-vocab dot product is
      ~2.2 s/token on CPU today).
- [ ] **T10. Fix repeating-token output** in `npu_engine_stdio.cpp`. ROOT-CAUSED
      (findings §3a): the int8 NPU GEMM only does a single K=1024 tile, so O-proj
      (K=2048) and D-proj (K=3072) return EXACTLY ZERO — attention & MLP contribute
      nothing, residual stays ≈ input embedding, tied lm_head predicts the input token
      back (emits last prompt token forever). NOT a sampler/RoPE bug. Fix = K-tiling +
      accumulation in `I8Ctx::go()`, OR xclbins that handle K>1024, OR retarget the FLM
      engine. Also fix hardcoded `5.0/127` activation scale (D-proj input ~8/elem
      saturates). This is NPU-kernel work, not a CPU one-liner.
- [ ] **T8. Reconcile the "Zero Python" README claim** — it is not yet literally true
      while `daemon/npu-cppd.py` + tokenization Python remain. Update the claim OR
      finish T3/T6 first. Do not edit the claim silently.
- [ ] **T9. De-orphan the good engines.** `npu_engine_stdio.cpp` and `tokenize.cpp`
      are not referenced by any CMake/Makefile/build.sh. Add build targets so the next
      engine attempt starts from the last working one instead of a fresh `engine_final_*`.

## A. Native decode target established (2026-07-17)

FLM (FastFlowLM, `/usr/bin/flm` + `fastflowlm-build`) is the native NPU decode engine:
- Verified coherent: `echo 'The capital of France is' | flm run qwen3:0.6b` -> "**Paris**".
- Native C++/NPU, no Python, no torch in its decode path.
- Exposes `flm run` (interactive) and `flm serve` (OpenAI `/v1/chat/completions`).
- The Rust router's SMALL_MODEL (`qwen3-0.6b-FLM`) already targets it.

Consequence: `npu_engine_stdio` (the orphan we fixed in T4 / diagnosed in T10) is
EXPERIMENTAL ONLY — an independent-engine effort, not required for a working serving
path. B (K-tiling the O/D kernels) is now OPTIONAL/strategic: pursue it only to own the
full stack independent of FastFlowLM; it is not on the critical path to Python-free serving.

## Decisions taken in this change

- Defer the decode-engine swap (T6) until T4+T5 land — do not replace working Python
  with crashing C++.
- Router → Rust (consistent with existing `rust/onebit`; matches the "if you can't kill
  Python, make it Rust" rule).
- Offline tooling (LoRA train, hf_to_q4nx, kernel-gen, export_tokenizer) stays Python —
  not in the serving path, so it does not violate the "no Python in the hotpatch" rule.
