# Hybrid prefill/decode policy — HIP prefill → HRX decode (issue #1942)

**Status:** design + D2 implementation in progress (2026-09-03)
**Owner:** bong-water-water-bong
**Context:** docs/research/hrx-engine-goal.md §Strategic position item 4.
**Live state:** docs/research/hip-prefill-lane-build-status.md (#2054) + issue #1942 thread (09-02/09-03 rounds).

## 0. Status delta vs the 09-01 body below

- **D2 build blocker RESOLVED (09-02)**: the vendored llama.cpp builds with GGML_HIP on TheRock — CMake rejects `hipcc` as CMAKE_HIP_COMPILER, but amdclang++ (Clang 23, /home/bcloud/therock100/bin/amdclang++) configures cleanly (HIP + hipBLAS found, gfx1151). Recipe + artifacts in docs/research/hip-prefill-lane-build-status.md (#2054). §2's "no HIP backend" is stale — see the build-status doc.
- **State-format gate PASSED (09-02)**: vendored llama.cpp and the HRX bundle's libllama round-trip the full-state blob byte-identically (round-25j sessions, feat/hrx, FINDINGS.md). D2's binary question is answered — same-family state format.
- **Lane runtime NOT yet trusted (09-02/09-03)**: the 09-02 direct-`llama_decode` harness showed an input-dependent, nondeterministic SIGSEGV (2-token prompts, CPU and HIP alike; gdb-serialized passes; NT=1/4 sometimes pass). 2026-09-03 re-probe (agent round): ~300 llama-bench pp2/tg2 evals clean on BOTH the clang build-hip binary and a fresh gcc build; no upstream ggml threadpool fix in the 08-16→09-03 window. Trigger is suspected in the harness's context config (n_ubatch/n_ctx/KV cache type/flash-attn) or the interposer-capture environment, not the stock llama_decode path. Recommended gate before trusting the lane: N×200 pp2/tg2 evals clean (see #1942).
- **Open items**: reproduce with the exact round-13 harness; decide the HIP-lane vs HRX2-fork-prefill question (the ws12 round-27 GQA-batched prefill work is the parallel path to the same goal — if HRX2 closes its pp32 gap, the hybrid may not need the HIP lane at all).

## 1. Thesis and measured numbers

On gfx1151 (Strix Halo iGPU), Qwen3-30B-A3B Q4_K_M:

| phase | HIP | in-process HRX | subprocess HRX |
|---|---|---|---|
| large prefill (pp) | **1227–1313 tok/s** | — | — |
| warm decode (tg) | ~70 tok/s | **~80–87 tok/s** | ~38 tok/s |

The hybrid: prefill the prompt on HIP (fast), hand the warm KV to HRX, and
decode the continuation on HRX (fastest fused decode). Today (`G1b`) the
router simply skips HRX for large prompts — pure HIP the whole way.

Hybrid total = HIP prefill(t) + KV handoff(t) + HRX decode(N).
For a 2k-token prompt + 1k-token continuation: pure HIP ≈ 1.5 + 14.3 = 15.8 s;
hybrid ≈ 1.5 + handoff + 11.8 s. The handoff must be ≤ ~2.5 s for the hybrid
to win at N=1k — at 85 tok/s decode the break-even handoff budget grows ~12 ms
per continuation token. A raw KV copy of a 2k-token cache is ~0.8 GB
(48 layers × 8 kv-heads × 128 dim × 2 × 2 bytes/token ≈ 393 KB/token) —
~80 ms at DRAM speed. **The transfer itself is cheap; the format question is
the entire project.**

## 2. The core problem: the two lanes are different inference engines

| | HIP lane | HRX lane |
|---|---|---|
| implementation | engine's own kernels (`backend_hip_1bp.cpp`) | llama.cpp bundle `libllama.so` (hrx-b59/b66, llama.h 0.0.10320) |
| weights | 1BP (engine native) | GGUF |
| KV cache | engine's own per-layer tensors | llama.cpp `kv_self` (ggml tensors, rope applied at compute) |
| state export API | none | `llama_state_get_data/set_data` (llama.cpp) |

The engine's vendored llama.cpp previously had no HIP backend; the D2 lane build
now configures GGML_HIP=ON with the TheRock amdclang++ toolchain (§0, build
status doc #2054) — but the lane's runtime is not yet trusted (prefill crash
under investigation, §0), so
the HIP prefill lane cannot today produce a llama.cpp-compatible state blob.
The engine's own 1BP KV layout is unrelated to llama.cpp's `kv_self`. There is
no shared KV format between the lanes.

## 3. Handoff designs (coupling, low → high)

### D1 — re-prefix: tokens-only handoff (correctness MVP)

HIP prefill → pass the prompt token ids to the HRX lane → HRX re-prefills
internally (full or last-K tokens) → HRX decodes.

- **Coupling:** none (token ids only). Works today — both lanes take tokens.
- **Cost:** the HRX prefill time is ADDED. With HRX prefill ≈ 100–200 tok/s
  (unmeasured; decode is its strong point), re-prefixing a 2k prompt costs
  10–20 s — **a loss against pure HIP for every continuation length** on the
  30B. D1 is a correctness gate / fallback, not the shipped policy.
- **Use:** the acceptance-criteria harness (correct continuation) and the
  fallback when the state handoff fails.

### D2 — llama_state blob: both lanes = llama.cpp (recommended direction)

Give the HIP prefill lane a llama.cpp context: build the engine's vendored
llama.cpp with a HIP backend (`GGML_HIP=ON` — ROCm/TheRock 10.1 on this box),
prefill there, `llama_state_get_data` → transfer blob → `llama_state_set_data`
into the HRX bundle's context.

- **Coupling:** medium. Turns the cross-engine KV remap into a same-family
  version check:
  - `LLAMA_STATE_VERSION` / `LLAMA_SESSION_VERSION` (bundle: 9/2 — verify the
    vendored fork matches) and the state blob layout must be compatible.
  - Both contexts must be created with identical model metadata + KV params
    (`n_ctx`, `n_ubatch`, cache type) so the blob's per-layer KV sizing lines
    up. The blob is versioned but not self-describing for context params —
    the transfer must size both sides identically.
  - Model format: the 30B-A3B must load in BOTH (GGUF in both — the engine's
    llama.cpp reads GGUF; the bundle reads GGUF). This lane serves GGUF; the
    engine's 1BP-only HIP kernels are NOT involved.
- **Cost:** ~80 ms transfer + context-creation overhead. Viable.
- **Risk:** the vendored fork (0.1.x-era) vs bundle (0.0.10320-era) state
  format may have drifted — the exact `LLAMA_STATE_VERSION` and blob layout
  must be verified with a byte-level round-trip test BEFORE committing to D2
  (test plan §5.1).
- **Implementation surface:** (1) enable GGML_HIP in the vendored llama.cpp
  (TheRock 10.1, gfx1151 — the HRX bundle already proves the toolchain);
  (2) a `llama_state` shim in the HRX lane (dlsym the bundle's
  `llama_state_get_size/get_data/set_data` — they exist in llama.h 0.0.10320,
  23 `llama_state_*` symbols match the vendored header); (3) a handoff path in
  the router: prefill-context owns the KV, decode-context imports it.

### D3 — raw KV remap: engine 1BP KV → llama.cpp kv_self (not recommended)

Export the engine's own KV tensors and write them into the HRX bundle's
`kv_self` via the (not exported) ggml tensor handles.

- **Coupling:** maximal — the engine must replicate llama.cpp's KV layout,
  rope application convention, cache-type handling, and sliding-window /
  GQA indexing exactly, for every model it serves. One upstream KV refactor
  silently corrupts the handoff.
- **Reject** unless D2 is impossible (state-format drift that can't be
  patched) — D3 is a second inference engine maintaining llama.cpp's internals.

## 4. Policy

```
route(model, prompt):
    if prompt_tokens <  THRESHOLD_PREFILL:        # small prompt — HRX alone
        hrx.prefill_and_decode(prompt)
    else:                                          # large prompt — hybrid
        hip_llama.prefill(prompt)                  # D2 lane
        kv = hip_llama.state_get_data()
        hrx.state_set_data(kv)
        out = hrx.decode(continuation)
    # correctness gate: hybrid continuation must match pure-HIP greedy
    # (token-identical up to the first diverging top-1, or better: the
    # hybrid is only routed when it matches)
```

Threshold tuning per model from the bench table in §1 (large prefill is where
HIP wins; short prompts amortize the handoff cost poorly).

## 5. Acceptance

### 5.1 State-format round-trip (D2 gate, do FIRST)
1. Build the vendored llama.cpp with GGML_HIP (gfx1151, TheRock 10.1).
2. Load the same GGUF in both the vendored build and the HRX bundle.
3. Decode K tokens on each; export state from A (`llama_state_get_data`),
   import into B (`llama_state_set_data`), continue B and A greedy.
4. **Gate:** continuation from the imported state is token-identical to the
   native continuation (same greedy top-1s, same rng state after ≥10 tokens).
   If the blob is rejected (size/version mismatch) → log the exact drift and
   fall back to D1; do NOT start D3.

### 5.2 End-to-end (issue #1942 acceptance)
- One request: prompt prefill on HIP, continuation decode on HRX, correct
  continuation (no context loss) — D1 harness proves correctness; D2 is the
  shipped fast path.
- Benchmark: total time beats either backend alone on the same model
  (30B-A3B Q4_K_M, ≥2k-token prompt, ≥500-token continuation).
- Documented state-format compatibility: the round-trip result + the exact
  `LLAMA_STATE_VERSION`/`LLAMA_SESSION_VERSION` pair.

## 6. Open questions

- HRX bundle prefill speed (needed to confirm D1 is only a fallback).
- Vendored llama.cpp vs bundle `LLAMA_STATE_VERSION` compatibility — the
  binary question that D2 lives or dies on. Check first.
- KV cache type (`KV_CACHE_TYPE_F16/AUTO`) agreement across builds.
- Whether the vendored fork's GGML_HIP can co-exist with the HRX bundle in
  one process (both dlopen HIP/ROCm — the in-process HRX path already loads
  the bundle's libllama; adding the engine's own HIP-linked llama.cpp may
  conflict on ROCm symbols; the subprocess llama-server path avoids this).
