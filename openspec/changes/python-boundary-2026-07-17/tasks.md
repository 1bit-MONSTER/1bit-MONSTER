# Tasks — Python→Native/Rust boundary

Ordered by value/risk. Each is independent; do not batch.

## Ready now (low risk)

- [ ] **T1. Wire native tokenizer into `src/server/rest_handler.cpp`.**
      Link `engine/fusion/tokenize.cpp` C ABI (`tokenizer_load/encode/free`), delete
      `simple_tokenize()` (line ~518). Verified bit-exact (findings §2). Add a build
      target for `tokenize.cpp` in CMakeLists.
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
      `daemon/npu-cppd.py` with the native `npu_engine_stdio` binary — ONLY after T4+T5.
      Note protocol mismatch: daemon speaks `{tokens,max_new_tokens}`; engine speaks
      per-token `{token}`/`{continue}`. Prompt-feed + sample loop must move to C++.

## Follow-ups

- [ ] **T7. Move lm_head off CPU** in the NPU engine (151,936-vocab dot product is
      ~2.2 s/token on CPU today).
- [ ] **T10. Fix repeating-token output** in `npu_engine_stdio.cpp` (predicts 9707
      every step in testing). Check RoPE position advance (`sp`), KV write, and the
      rand() multinomial sampler. Surfaced during T4 verification.
- [ ] **T8. Reconcile the "Zero Python" README claim** — it is not yet literally true
      while `daemon/npu-cppd.py` + tokenization Python remain. Update the claim OR
      finish T3/T6 first. Do not edit the claim silently.
- [ ] **T9. De-orphan the good engines.** `npu_engine_stdio.cpp` and `tokenize.cpp`
      are not referenced by any CMake/Makefile/build.sh. Add build targets so the next
      engine attempt starts from the last working one instead of a fresh `engine_final_*`.

## Decisions taken in this change

- Defer the decode-engine swap (T6) until T4+T5 land — do not replace working Python
  with crashing C++.
- Router → Rust (consistent with existing `rust/onebit`; matches the "if you can't kill
  Python, make it Rust" rule).
- Offline tooling (LoRA train, hf_to_q4nx, kernel-gen, export_tokenizer) stays Python —
  not in the serving path, so it does not violate the "no Python in the hotpatch" rule.
