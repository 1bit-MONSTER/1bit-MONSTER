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

## §5.1 gate — RE-VERIFIED 2026-09-07 (rt_session harness)

A = vendored third_party/llama.cpp @4df29be4f GGML_HIP build (session 9) · B = hrx-v2-src
fork @0f52c297a (session 9) · Qwen3-0.6B-Q4_K_M · 5-token prompt + 8 gens → blob
1,491,865 B (llama_state_save_file/load_file, ctx 512/512/8).

- A-imported == A-native: **IDENTICAL** (13/13).
- B-imported == B-native: **IDENTICAL** (13/13) — handoff is faithful (kv+rng fully
  transferred; fork resumed from the vendored blob continues exactly as it natively would).
- A-native vs B-native (no state): diverges after ~4-5 tokens = cross-build ggml-cpu
  numeric drift (fork vs vendored kernels), reproducible under forced F16 and F32 KV
  (rules out cache-type disagreement; §6 open question answered). Not a handoff defect.
- Consequence for D2 hybrid correctness: per §4, continuation equality vs pure-HIP is
  bounded by this drift ("identical up to the first diverging top-1"); the state format
  round-trips losslessly. D1 harness remains the no-context-loss proof.

## §5.2 D2 handoff — correctness PROVEN on the 30B (2026-09-07)

Qwen3-Coder-30B-A3B-Instruct-Q4_K_M (the HRX-bundle model), 2,962-token prompt:
- A (vendored 4df29be4f GGML_HIP, ngl99) prefills on HIP (~620 tok/s incl. save),
  292,011,596 B session blob via llama_state_save_file.
- B (hrx2 fork @0f52c297a, session 9) imports the blob (llama_state_load_file) and
  decodes. **A-run (HIP decode of own kv) vs B-run (fork decode of imported kv):
  48/48 continuation tokens IDENTICAL** — the handoff is lossless; no context loss.
- B-imported vs B-native (fork re-prefill from scratch) diverges from token 0 =
  A-HIP-prefill vs B-CPU-prefill kernel drift (anticipated by §4: equality is bounded
  by the first diverging top-1; decode-on-identical-kv agrees 100%).
- Remaining for full §5.2 acceptance: decode leg on the real HRX device (bundle),
  ≥500-token continuation, and the hybrid-vs-single-backend timing table.

## §5.2 benchmark — measured 2026-09-07 (clean run, quiet box)

Qwen3-Coder-30B-A3B-Instruct-Q4_K_M · 2,962-token prompt · 500-token continuation,
standalone harnesses: rt_A (vendored 4df29be4f GGML_HIP ngl99), rt_HRX (gfx1151
llama-build bundle: libllama + ggml-hrx 0.9.11, HRX0, all layers offloaded).

| config | wall | notes |
|---|---|---|
| **hybrid**: HIP prefill + HRX0 decode | **404.5 s** | prefill 4.50 s (~660 tok/s) + HRX decode 400.0 s (1.25 tok/s) |
| **HIP-only** | **12.2 s** | prefill + 500 decode (~41+ tok/s end-to-end incl. load) |
| **HRX-only** | **596.3 s** | HRX prefill ~196 s (~15 tok/s) + decode ~400 s |

Token comparisons: hybrid-vs-hip-only and hybrid-vs-hrx-only diverge at token 0
(cross-backend prefill/decode kernel numerics — HIP == fork-CPU decode agreed
48/48 on identical kv earlier; HRX0-device decode drifts from token 0 vs both).
Continuations are fluent/coherent on all three paths (no context loss).

**Verdict:** D2 handoff is functionally proven (lossless state transfer, decode on
HIP/fork-CPU/HRX0 all work from the shared blob), but the §5.2 perf criterion
(hybrid total beats either backend alone) is **NOT met**: HRX0 decode measured
1.25 tok/s in this harness vs the 80-87 tok/s documented for the engines


## §5.2 benchmark — measured 2026-09-07 (clean run, quiet box)

Qwen3-Coder-30B-A3B-Instruct-Q4_K_M · 2,962-token prompt · 500-token continuation,
standalone harnesses: rt_A (vendored 4df29be4f GGML_HIP ngl99), rt_HRX (gfx1151
llama-build bundle: libllama + ggml-hrx 0.9.11, HRX0, all layers offloaded).

| config | wall | notes |
|---|---|---|
| **hybrid**: HIP prefill + HRX0 decode | **404.5 s** | prefill 4.50 s (~660 tok/s) + HRX decode 400.0 s (1.25 tok/s) |
| **HIP-only** | **12.2 s** | prefill + 500 decode end-to-end incl. model load |
| **HRX-only** | **596.3 s** | HRX prefill ~196 s (~15 tok/s) + decode ~400 s |

Token comparisons: hybrid-vs-hip-only and hybrid-vs-hrx-only diverge at token 0
(cross-backend prefill/decode kernel numerics — HIP == fork-CPU decode agreed
48/48 on identical kv earlier; HRX0-device decode drifts from token 0 vs both).
Continuations are fluent/coherent on all three paths (no context loss).

**Verdict:** D2 handoff is functionally proven (lossless state transfer; decode on
HIP, fork-CPU and HRX0 all work from the shared blob), but the §5.2 perf criterion
(hybrid total beats either backend alone) is **NOT met** with this bundle: HRX0
decode measured 1.25 tok/s in the standalone harness vs the 80-87 tok/s documented
for the engine's in-process path (hrx_inprocess) and HIP decode ~70 tok/s. Open
question: is the decode-rate gap harness/bundle-config specific (engine path uses
dlopen+DEEPBIND + engine ctx params) or a bundle regression? Resolve via the
engine-path integration (wire llama_state import into HrxBackend, bench through
1bit unified) before the hybrid policy ships. The §0 HRX2 decision stands.


## §5.2 benchmark — measured 2026-09-07 (clean run, quiet box)

Qwen3-Coder-30B-A3B-Instruct-Q4_K_M · 2,962-token prompt · 500-token continuation,
standalone harnesses: rt_A (vendored 4df29be4f GGML_HIP ngl99), rt_HRX (gfx1151
llama-build bundle: libllama + ggml-hrx 0.9.11, HRX0, all layers offloaded).

| config | wall | notes |
|---|---|---|
| **hybrid**: HIP prefill + HRX0 decode | **404.5 s** | prefill 4.50 s (~660 tok/s) + HRX decode 400.0 s (1.25 tok/s) |
| **HIP-only** | **12.2 s** | prefill + 500 decode end-to-end incl. model load |
| **HRX-only** | **596.3 s** | HRX prefill ~196 s (~15 tok/s) + decode ~400 s |

Token comparisons: hybrid-vs-hip-only and hybrid-vs-hrx-only diverge at token 0
(cross-backend prefill/decode kernel numerics — HIP == fork-CPU decode agreed
48/48 on identical kv earlier; HRX0-device decode drifts from token 0 vs both).
Continuations are fluent/coherent on all three paths (no context loss).

**Verdict:** D2 handoff is functionally proven (lossless state transfer; decode on
HIP, fork-CPU and HRX0 all work from the shared blob), but the §5.2 perf criterion
(hybrid total beats either backend alone) is **NOT met** with this bundle: HRX0
decode measured 1.25 tok/s in the standalone harness vs the 80-87 tok/s documented
for the engine's in-process path (hrx_inprocess) and HIP decode ~70 tok/s. Open
question: is the decode-rate gap harness/bundle-config specific (engine path uses
dlopen+DEEPBIND + engine ctx params) or a bundle regression? Resolve via the
engine-path integration (wire llama_state import into HrxBackend, bench through
1bit unified) before the hybrid policy ships. The §0 HRX2 decision stands.

## 2026-09-07 follow-up — HRX0 import-decode blocker filed (#2145)

b66 release bundle and the amd-hrx-graph fork both FAIL llama_decode at token 2 on
HRX0 after llama_state_load_file of the 292 MB HIP-prefill blob (graph compute -1);
the local llama-build (GET_ROWS-capable) decodes from the imported state (500 tokens,
coherent) but at 1.25 tok/s. Handoff losslessness stands (fork-CPU + HIP 48/48).
Filed as 1bit-MONSTER/1bit-MONSTER#2145 — blocks the D2 shipped fast path; D1
tokens-only re-prefix remains the correctness fallback; §0 HRX2 decision untouched.

## #2145 triage refinement (2026-09-07) — not import-specific

amd-hrx-graph fork fails NATIVE 30B-Coder decode on HRX0 with no state import
(graph compute -1 in the chunked prefill itself); its envelope is qwen3 <= 4B +
zaya. b66 stock = engine non-default for this model (GET_ROWS limit); the
engine serves the 30B through the GET_ROWS-capable llama-build bundle, which
decodes imported state correctly at 1.25 tok/s. The 80-87 tok/s historical
rate remains unreproduced on current bundles (flash-attn irrelevant); 30B
qwen3moe decode on the HRX2 stack is fleet roadmap. D2 shipped fast path stays
gated on HRX decode throughput; shim landing point merged (PR #2146).


## §5.2 benchmark — measured 2026-09-07 (clean run, quiet box)

Qwen3-Coder-30B-A3B-Instruct-Q4_K_M · 2,962-token prompt · 500-token continuation,
standalone harnesses: rt_A (vendored 4df29be4f GGML_HIP ngl99), rt_HRX (gfx1151
llama-build bundle: libllama + ggml-hrx 0.9.11, HRX0, all layers offloaded).

| config | wall | notes |
|---|---|---|
| **hybrid**: HIP prefill + HRX0 decode | **404.5 s** | prefill 4.50 s (~660 tok/s) + HRX decode 400.0 s (1.25 tok/s) |
| **HIP-only** | **12.2 s** | prefill + 500 decode end-to-end incl. model load |
| **HRX-only** | **596.3 s** | HRX prefill ~196 s (~15 tok/s) + decode ~400 s |

Token comparisons: hybrid-vs-hip-only and hybrid-vs-hrx-only diverge at token 0
(cross-backend prefill/decode kernel numerics — HIP == fork-CPU decode agreed
48/48 on identical kv earlier; HRX0-device decode drifts from token 0 vs both).
Continuations are fluent/coherent on all three paths (no context loss).

**Verdict:** D2 handoff is functionally proven (lossless state transfer; decode on
HIP, fork-CPU and HRX0 all work from the shared blob), but the §5.2 perf criterion
(hybrid total beats either backend alone) is **NOT met** with this bundle: HRX0
decode measured 1.25 tok/s in the standalone harness vs the 80-87 tok/s documented
for the engine's in-process path (hrx_inprocess) and HIP decode ~70 tok/s. Open
question: is the decode-rate gap harness/bundle-config specific (engine path uses
dlopen+DEEPBIND + engine ctx params) or a bundle regression? Resolve via the
engine-path integration (wire llama_state import into HrxBackend, bench through
1bit unified) before the hybrid policy ships. The §0 HRX2 decision stands.

## 2026-09-07 follow-up — HRX0 import-decode blocker filed (#2145)

b66 release bundle and the amd-hrx-graph fork both FAIL llama_decode at token 2 on
HRX0 after llama_state_load_file of the 292 MB HIP-prefill blob (graph compute -1);
the local llama-build (GET_ROWS-capable) decodes from the imported state (500 tokens,
coherent) but at 1.25 tok/s. Handoff losslessness stands (fork-CPU + HIP 48/48).
Filed as 1bit-MONSTER/1bit-MONSTER#2145 — blocks the D2 shipped fast path; D1
tokens-only re-prefix remains the correctness fallback; §0 HRX2 decision untouched.

## 2026-09-07 rate-gap RESOLVED — llama-bench: 40.9 tok/s tg256 at KV 2.9-3.2k (llama-build/HRX0, 30B Coder)

The 1.25 tok/s harness figure was an artifact of the auto-pos decode loop at
large n_pos (flat 400-730 ms/decode; thread- and config-invariant; not seen by
llama-bench on the same libllama). Bundle long-ctx decode capability = ~41
tok/s. Hybrid re-estimate with real rate: HIP prefill 4.5 s + 500-token HRX
decode ~12-13 s ≈ 17 s vs HIP-only 12.2 s — within 1.4x, not beating. Decisive
remaining question: does the ENGINE in-process decode (explicit-pos, own
counter, n_batch 2048) hit the auto-pos stall at long ctx? Measure in-engine.


## §5.2 benchmark — measured 2026-09-07 (clean run, quiet box)

Qwen3-Coder-30B-A3B-Instruct-Q4_K_M · 2,962-token prompt · 500-token continuation,
standalone harnesses: rt_A (vendored 4df29be4f GGML_HIP ngl99), rt_HRX (gfx1151
llama-build bundle: libllama + ggml-hrx 0.9.11, HRX0, all layers offloaded).

| config | wall | notes |
|---|---|---|
| **hybrid**: HIP prefill + HRX0 decode | **404.5 s** | prefill 4.50 s (~660 tok/s) + HRX decode 400.0 s (1.25 tok/s) |
| **HIP-only** | **12.2 s** | prefill + 500 decode end-to-end incl. model load |
| **HRX-only** | **596.3 s** | HRX prefill ~196 s (~15 tok/s) + decode ~400 s |

Token comparisons: hybrid-vs-hip-only and hybrid-vs-hrx-only diverge at token 0
(cross-backend prefill/decode kernel numerics — HIP == fork-CPU decode agreed
48/48 on identical kv earlier; HRX0-device decode drifts from token 0 vs both).
Continuations are fluent/coherent on all three paths (no context loss).

**Verdict:** D2 handoff is functionally proven (lossless state transfer; decode on
HIP, fork-CPU and HRX0 all work from the shared blob), but the §5.2 perf criterion
(hybrid total beats either backend alone) is **NOT met** with this bundle: HRX0
decode measured 1.25 tok/s in the standalone harness vs the 80-87 tok/s documented
for the engine's in-process path (hrx_inprocess) and HIP decode ~70 tok/s. Open
question: is the decode-rate gap harness/bundle-config specific (engine path uses
dlopen+DEEPBIND + engine ctx params) or a bundle regression? Resolve via the
engine-path integration (wire llama_state import into HrxBackend, bench through
1bit unified) before the hybrid policy ships. The §0 HRX2 decision stands.

## 2026-09-07 follow-up — HRX0 import-decode blocker filed (#2145)

b66 release bundle and the amd-hrx-graph fork both FAIL llama_decode at token 2 on
HRX0 after llama_state_load_file of the 292 MB HIP-prefill blob (graph compute -1);
the local llama-build (GET_ROWS-capable) decodes from the imported state (500 tokens,
coherent) but at 1.25 tok/s. Handoff losslessness stands (fork-CPU + HIP 48/48).
Filed as 1bit-MONSTER/1bit-MONSTER#2145 — blocks the D2 shipped fast path; D1
tokens-only re-prefix remains the correctness fallback; §0 HRX2 decision untouched.

## 2026-09-07 rate-gap RESOLVED — llama-bench: 40.9 tok/s tg256 at KV 2.9-3.2k (llama-build/HRX0, 30B Coder)

The 1.25 tok/s harness figure was an artifact of the auto-pos decode loop at
large n_pos (flat 400-730 ms/decode; thread- and config-invariant; not seen by
llama-bench on the same libllama). Bundle long-ctx decode capability = ~41
tok/s. Hybrid re-estimate with real rate: HIP prefill 4.5 s + 500-token HRX
decode ~12-13 s ≈ 17 s vs HIP-only 12.2 s — within 1.4x, not beating. Decisive
remaining question: does the ENGINE in-process decode (explicit-pos, own
counter, n_batch 2048) hit the auto-pos stall at long ctx? Measure in-engine.

## 2026-09-07 handoff — CPU-busy, llama_decode-internal

The slow engine-shaped decode at long ctx is CPU-busy (user+sys ~= wall, sys-heavy
per-decode churn), thread- and config-invariant, with llama-bench fast on the
identical libs+params. Root cause is inside llama_decode per-call behavior on the
HRX build (ctx-state-dependent), needs llama_decode/kv-cache instrumentation —
HRX-fork owner territory (llama-build built from a wiped /tmp tree; rebuild from
~/hrx-ws/hrx-v2-src enables source work). Repro artifacts on strixhalo:
/tmp/m2/rt_HRX + rt_session.cpp + /tmp/m2/rt_30b_f16.bin. Full record: #2145.


## §5.2 benchmark — measured 2026-09-07 (clean run, quiet box)

Qwen3-Coder-30B-A3B-Instruct-Q4_K_M · 2,962-token prompt · 500-token continuation,
standalone harnesses: rt_A (vendored 4df29be4f GGML_HIP ngl99), rt_HRX (gfx1151
llama-build bundle: libllama + ggml-hrx 0.9.11, HRX0, all layers offloaded).

| config | wall | notes |
|---|---|---|
| **hybrid**: HIP prefill + HRX0 decode | **404.5 s** | prefill 4.50 s (~660 tok/s) + HRX decode 400.0 s (1.25 tok/s) |
| **HIP-only** | **12.2 s** | prefill + 500 decode end-to-end incl. model load |
| **HRX-only** | **596.3 s** | HRX prefill ~196 s (~15 tok/s) + decode ~400 s |

Token comparisons: hybrid-vs-hip-only and hybrid-vs-hrx-only diverge at token 0
(cross-backend prefill/decode kernel numerics — HIP == fork-CPU decode agreed
48/48 on identical kv earlier; HRX0-device decode drifts from token 0 vs both).
Continuations are fluent/coherent on all three paths (no context loss).

**Verdict:** D2 handoff is functionally proven (lossless state transfer; decode on
HIP, fork-CPU and HRX0 all work from the shared blob), but the §5.2 perf criterion
(hybrid total beats either backend alone) is **NOT met** with this bundle: HRX0
decode measured 1.25 tok/s in the standalone harness vs the 80-87 tok/s documented
for the engine's in-process path (hrx_inprocess) and HIP decode ~70 tok/s. Open
question: is the decode-rate gap harness/bundle-config specific (engine path uses
dlopen+DEEPBIND + engine ctx params) or a bundle regression? Resolve via the
engine-path integration (wire llama_state import into HrxBackend, bench through
1bit unified) before the hybrid policy ships. The §0 HRX2 decision stands.

## 2026-09-07 follow-up — HRX0 import-decode blocker filed (#2145)

b66 release bundle and the amd-hrx-graph fork both FAIL llama_decode at token 2 on
HRX0 after llama_state_load_file of the 292 MB HIP-prefill blob (graph compute -1);
the local llama-build (GET_ROWS-capable) decodes from the imported state (500 tokens,
coherent) but at 1.25 tok/s. Handoff losslessness stands (fork-CPU + HIP 48/48).
Filed as 1bit-MONSTER/1bit-MONSTER#2145 — blocks the D2 shipped fast path; D1
tokens-only re-prefix remains the correctness fallback; §0 HRX2 decision untouched.

## 2026-09-07 rate-gap RESOLVED — llama-bench: 40.9 tok/s tg256 at KV 2.9-3.2k (llama-build/HRX0, 30B Coder)

The 1.25 tok/s harness figure was an artifact of the auto-pos decode loop at
large n_pos (flat 400-730 ms/decode; thread- and config-invariant; not seen by
llama-bench on the same libllama). Bundle long-ctx decode capability = ~41
tok/s. Hybrid re-estimate with real rate: HIP prefill 4.5 s + 500-token HRX
decode ~12-13 s ≈ 17 s vs HIP-only 12.2 s — within 1.4x, not beating. Decisive
remaining question: does the ENGINE in-process decode (explicit-pos, own
counter, n_batch 2048) hit the auto-pos stall at long ctx? Measure in-engine.

## 2026-09-07 handoff — CPU-busy, llama_decode-internal

The slow engine-shaped decode at long ctx is CPU-busy (user+sys ~= wall, sys-heavy
per-decode churn), thread- and config-invariant, with llama-bench fast on the
identical libs+params. Root cause is inside llama_decode per-call behavior on the
HRX build (ctx-state-dependent), needs llama_decode/kv-cache instrumentation —
HRX-fork owner territory (llama-build built from a wiped /tmp tree; rebuild from
~/hrx-ws/hrx-v2-src enables source work). Repro artifacts on strixhalo:
/tmp/m2/rt_HRX + rt_session.cpp + /tmp/m2/rt_30b_f16.bin. Full record: #2145.

## 2026-09-07 ROOT CAUSE: stale-binary artifact — modern fork has NO stall

amd-hrx-graph fork (current, HRX0): 0.6B 2,962-token prefill + 16 decodes @
pos 2962+ in 2 s total. The 400-730 ms/decode stall was specific to the old
llama-build lib (wiped-tree build, older commit). HRX long-ctx decode is fine
on the current stack (validated envelope). D2 hybrid decode-leg blocker =
30B support on the HRX2 fork (roadmap) + engine bundle modernization; no
llama_decode bug to fix. #2145 rate sub-question CLOSED.


## §5.2 benchmark — measured 2026-09-07 (clean run, quiet box)

Qwen3-Coder-30B-A3B-Instruct-Q4_K_M · 2,962-token prompt · 500-token continuation,
standalone harnesses: rt_A (vendored 4df29be4f GGML_HIP ngl99), rt_HRX (gfx1151
llama-build bundle: libllama + ggml-hrx 0.9.11, HRX0, all layers offloaded).

| config | wall | notes |
|---|---|---|
| **hybrid**: HIP prefill + HRX0 decode | **404.5 s** | prefill 4.50 s (~660 tok/s) + HRX decode 400.0 s (1.25 tok/s) |
| **HIP-only** | **12.2 s** | prefill + 500 decode end-to-end incl. model load |
| **HRX-only** | **596.3 s** | HRX prefill ~196 s (~15 tok/s) + decode ~400 s |

Token comparisons: hybrid-vs-hip-only and hybrid-vs-hrx-only diverge at token 0
(cross-backend prefill/decode kernel numerics — HIP == fork-CPU decode agreed
48/48 on identical kv earlier; HRX0-device decode drifts from token 0 vs both).
Continuations are fluent/coherent on all three paths (no context loss).

**Verdict:** D2 handoff is functionally proven (lossless state transfer; decode on
HIP, fork-CPU and HRX0 all work from the shared blob), but the §5.2 perf criterion
(hybrid total beats either backend alone) is **NOT met** with this bundle: HRX0
decode measured 1.25 tok/s in the standalone harness vs the 80-87 tok/s documented
for the engine's in-process path (hrx_inprocess) and HIP decode ~70 tok/s. Open
question: is the decode-rate gap harness/bundle-config specific (engine path uses
dlopen+DEEPBIND + engine ctx params) or a bundle regression? Resolve via the
engine-path integration (wire llama_state import into HrxBackend, bench through
1bit unified) before the hybrid policy ships. The §0 HRX2 decision stands.

## 2026-09-07 follow-up — HRX0 import-decode blocker filed (#2145)

b66 release bundle and the amd-hrx-graph fork both FAIL llama_decode at token 2 on
HRX0 after llama_state_load_file of the 292 MB HIP-prefill blob (graph compute -1);
the local llama-build (GET_ROWS-capable) decodes from the imported state (500 tokens,
coherent) but at 1.25 tok/s. Handoff losslessness stands (fork-CPU + HIP 48/48).
Filed as 1bit-MONSTER/1bit-MONSTER#2145 — blocks the D2 shipped fast path; D1
tokens-only re-prefix remains the correctness fallback; §0 HRX2 decision untouched.

## 2026-09-07 rate-gap RESOLVED — llama-bench: 40.9 tok/s tg256 at KV 2.9-3.2k (llama-build/HRX0, 30B Coder)

The 1.25 tok/s harness figure was an artifact of the auto-pos decode loop at
large n_pos (flat 400-730 ms/decode; thread- and config-invariant; not seen by
llama-bench on the same libllama). Bundle long-ctx decode capability = ~41
tok/s. Hybrid re-estimate with real rate: HIP prefill 4.5 s + 500-token HRX
decode ~12-13 s ≈ 17 s vs HIP-only 12.2 s — within 1.4x, not beating. Decisive
remaining question: does the ENGINE in-process decode (explicit-pos, own
counter, n_batch 2048) hit the auto-pos stall at long ctx? Measure in-engine.

## 2026-09-07 handoff — CPU-busy, llama_decode-internal

The slow engine-shaped decode at long ctx is CPU-busy (user+sys ~= wall, sys-heavy
per-decode churn), thread- and config-invariant, with llama-bench fast on the
identical libs+params. Root cause is inside llama_decode per-call behavior on the
HRX build (ctx-state-dependent), needs llama_decode/kv-cache instrumentation —
HRX-fork owner territory (llama-build built from a wiped /tmp tree; rebuild from
~/hrx-ws/hrx-v2-src enables source work). Repro artifacts on strixhalo:
/tmp/m2/rt_HRX + rt_session.cpp + /tmp/m2/rt_30b_f16.bin. Full record: #2145.

## 2026-09-07 ROOT CAUSE: stale-binary artifact — modern fork has NO stall

amd-hrx-graph fork (current, HRX0): 0.6B 2,962-token prefill + 16 decodes @
pos 2962+ in 2 s total. The 400-730 ms/decode stall was specific to the old
llama-build lib (wiped-tree build, older commit). HRX long-ctx decode is fine
on the current stack (validated envelope). D2 hybrid decode-leg blocker =
30B support on the HRX2 fork (roadmap) + engine bundle modernization; no
llama_decode bug to fix. #2145 rate sub-question CLOSED.

## Engine modernization scope (2026-09-07, recorded not started)

hrx_inprocess.h ABI mirrors are pinned to the b59-era bundle llama.h (0.0.10320:
4-arg llama_batch_get_one, static_asserted sizes). The modern HRX2 fork's llama.h
differs (2-arg batch_get_one, kv_unified/flash_attn_type era, GDN fused flags).
Wiring a modern bundle into the engine = repin the ABI mirrors + ctx-param usage
(offload_kqv/op_offload defaults changed between eras) + re-run the P2 smoke on a
lane-stable bundle. Premature while the fork fix branch is mid-flight and a stable
hrx-system release is upstream-gated (#1945); dependency: lane ships a stable
modern bundle -> repin -> retest 30B decode leg. Tracked via #2145/#2147.


## §5.2 benchmark — measured 2026-09-07 (clean run, quiet box)

Qwen3-Coder-30B-A3B-Instruct-Q4_K_M · 2,962-token prompt · 500-token continuation,
standalone harnesses: rt_A (vendored 4df29be4f GGML_HIP ngl99), rt_HRX (gfx1151
llama-build bundle: libllama + ggml-hrx 0.9.11, HRX0, all layers offloaded).

| config | wall | notes |
|---|---|---|
| **hybrid**: HIP prefill + HRX0 decode | **404.5 s** | prefill 4.50 s (~660 tok/s) + HRX decode 400.0 s (1.25 tok/s) |
| **HIP-only** | **12.2 s** | prefill + 500 decode end-to-end incl. model load |
| **HRX-only** | **596.3 s** | HRX prefill ~196 s (~15 tok/s) + decode ~400 s |

Token comparisons: hybrid-vs-hip-only and hybrid-vs-hrx-only diverge at token 0
(cross-backend prefill/decode kernel numerics — HIP == fork-CPU decode agreed
48/48 on identical kv earlier; HRX0-device decode drifts from token 0 vs both).
Continuations are fluent/coherent on all three paths (no context loss).

**Verdict:** D2 handoff is functionally proven (lossless state transfer; decode on
HIP, fork-CPU and HRX0 all work from the shared blob), but the §5.2 perf criterion
(hybrid total beats either backend alone) is **NOT met** with this bundle: HRX0
decode measured 1.25 tok/s in the standalone harness vs the 80-87 tok/s documented
for the engine's in-process path (hrx_inprocess) and HIP decode ~70 tok/s. Open
question: is the decode-rate gap harness/bundle-config specific (engine path uses
dlopen+DEEPBIND + engine ctx params) or a bundle regression? Resolve via the
engine-path integration (wire llama_state import into HrxBackend, bench through
1bit unified) before the hybrid policy ships. The §0 HRX2 decision stands.

## 2026-09-07 follow-up — HRX0 import-decode blocker filed (#2145)

b66 release bundle and the amd-hrx-graph fork both FAIL llama_decode at token 2 on
HRX0 after llama_state_load_file of the 292 MB HIP-prefill blob (graph compute -1);
the local llama-build (GET_ROWS-capable) decodes from the imported state (500 tokens,
coherent) but at 1.25 tok/s. Handoff losslessness stands (fork-CPU + HIP 48/48).
Filed as 1bit-MONSTER/1bit-MONSTER#2145 — blocks the D2 shipped fast path; D1
tokens-only re-prefix remains the correctness fallback; §0 HRX2 decision untouched.

## 2026-09-07 rate-gap RESOLVED — llama-bench: 40.9 tok/s tg256 at KV 2.9-3.2k (llama-build/HRX0, 30B Coder)

The 1.25 tok/s harness figure was an artifact of the auto-pos decode loop at
large n_pos (flat 400-730 ms/decode; thread- and config-invariant; not seen by
llama-bench on the same libllama). Bundle long-ctx decode capability = ~41
tok/s. Hybrid re-estimate with real rate: HIP prefill 4.5 s + 500-token HRX
decode ~12-13 s ≈ 17 s vs HIP-only 12.2 s — within 1.4x, not beating. Decisive
remaining question: does the ENGINE in-process decode (explicit-pos, own
counter, n_batch 2048) hit the auto-pos stall at long ctx? Measure in-engine.

## 2026-09-07 handoff — CPU-busy, llama_decode-internal

The slow engine-shaped decode at long ctx is CPU-busy (user+sys ~= wall, sys-heavy
per-decode churn), thread- and config-invariant, with llama-bench fast on the
identical libs+params. Root cause is inside llama_decode per-call behavior on the
HRX build (ctx-state-dependent), needs llama_decode/kv-cache instrumentation —
HRX-fork owner territory (llama-build built from a wiped /tmp tree; rebuild from
~/hrx-ws/hrx-v2-src enables source work). Repro artifacts on strixhalo:
/tmp/m2/rt_HRX + rt_session.cpp + /tmp/m2/rt_30b_f16.bin. Full record: #2145.

## 2026-09-07 ROOT CAUSE: stale-binary artifact — modern fork has NO stall

amd-hrx-graph fork (current, HRX0): 0.6B 2,962-token prefill + 16 decodes @
pos 2962+ in 2 s total. The 400-730 ms/decode stall was specific to the old
llama-build lib (wiped-tree build, older commit). HRX long-ctx decode is fine
on the current stack (validated envelope). D2 hybrid decode-leg blocker =
30B support on the HRX2 fork (roadmap) + engine bundle modernization; no
llama_decode bug to fix. #2145 rate sub-question CLOSED.

## Engine modernization scope (2026-09-07, recorded not started)

hrx_inprocess.h ABI mirrors are pinned to the b59-era bundle llama.h (0.0.10320:
4-arg llama_batch_get_one, static_asserted sizes). The modern HRX2 fork's llama.h
differs (2-arg batch_get_one, kv_unified/flash_attn_type era, GDN fused flags).
Wiring a modern bundle into the engine = repin the ABI mirrors + ctx-param usage
(offload_kqv/op_offload defaults changed between eras) + re-run the P2 smoke on a
lane-stable bundle. Premature while the fork fix branch is mid-flight and a stable
hrx-system release is upstream-gated (#1945); dependency: lane ships a stable
modern bundle -> repin -> retest 30B decode leg. Tracked via #2145/#2147.

## 2026-09-07 (evening) — 30B qwen3moe path UNBLOCKED; D2 functional leg PROVEN on the modern stack

Combined build (their local fix/hrx-ngl-init-order incl. 8000c0925 + 540e9815e +
my ADD/CLAMP/DIV standalone-claim exclusion, commit b2975bb1d on
fix/qwen35-prefill-coverage):
- 30B Coder prefill (5/8/16 tok) + decode: FULL PASS, no workarounds (was graph
  compute -1 at three successive layers: SUM_ROWS -> ADD [2048,5] -> ADD [2048,1]).
- D2 decode leg: HIP-prefill blob (292 MB, 2,962 tok) imported -> 200-token
  coherent decode at pos 2962+ (stream 2581 6932... = HRX-family), exit 0.
- Measured decode ~8-10 tok/s (residual ADDs split to CPU per-layer at 48
  layers — qwen3moe decode ADDs lack the fused attention-output coverage that
  dense qwen3 decode has).
- §5.2 perf verdict: hybrid ≈ 4.5 s prefill + ~55 s/500-token HRX decode ≈
  60 s vs HIP-only 12.2 s — criterion NOT met on current stacks (HRX decode of
  the 30B at ~8-10 tok/s CPU-ADD-bound now; even with ADD fusion (~40 tok/s)
  it would not beat HIP ~70). Product-level conclusion stands: on the 30B
  workload HIP-only wins; the hybrid policy's value would be on workloads where
  HRX decode > HIP decode (small-ctx warm decode) — the §0 decision's domain.

## 2026-09-08 (post-reboot) — conditional-claim A/B on the 30B path (fresh numbers)

Lane head 15ff48549 (conditional standalone ADD claim + alias-skip) verified on the
30B side after the 23:06 ADT cold reboot (q35-hrx-fix build survived in ~/wt):

- D2 imported-blob decode leg: HIP blob regenerated (292,011,596 B, 2,962-tok prompt,
  vendored 4df29be4f GGML_HIP) -> llama_state_load_file on HRX0 -> 120/200/500-token
  decode, exit 0, deterministic (identical streams across runs) — no context loss.
- 30B decode rate ~6 tok/s (CPU-residual-ADD bound, 48 layers) — qwen3moe decode ADDs
  still lack fused attention-output coverage under the conditional claim (by design:
  VIEW-wrapped/MUL_MAT_ID-src ADDs stay CPU). Perf verdict unchanged: hybrid ~60-85 s
  vs HIP-only 12.2 s on the 30B ≥2k workload — criterion NOT met on current stacks.
- Dense qwen3-0.6B decode restored (~98 tok/s via harness; lane llama-bench tg128
  227.96 ± 1.70) — the conditional claim is strictly better than the blanket exclusion
  (b2975bb1d) for dense models and does not regress the 30B qwen3moe path.
- Recorded on #2147 (comment 5578201386). Engine bundle modernization remains gated on
  a stable modern hrx-system release (#1945); qwen3moe-30B decode ADD fusion = fork roadmap.

## 2026-09-08 — HRX decode-fusion root cause: the model is NOT yet built to the engine decode plan

Goal mtsjhonv (unified two-engine stack), task 1 prerequisite. The modern fork
(amd-hrx-graph / q35-hrx-fix @ 15ff48549) decodes Qwen3-30B-A3B Q4_K_M at
~11.6-12 tok/s (llama-bench tg256, BOTH short KV<2048 and long KV 2.9-3.2k),
vs the old llama-build bundle (ggml-hrx 0.9.11) at 39.9-40.1 tok/s on the same
box/model — a 3.4x regression. Three layered causes, all "model not built to
the engines own decode design:

1. DECODE RESIDUAL ADDs split to CPU (dominant). The conditional-claim policy
   (device_supports_op, ggml-hrx.cpp ~line 1017) deliberately routes qwen3moe
   VIEW-wrapped / MUL_MAT_ID-src ADDs to CPU (#2147 legacy: claiming them
   orphaned the graph). Consequence: llama.cpp's ggml_backend_sched fragments
   each decode token into ~98 executor graph_compute calls (~2/layer x 48) at
   ~140-350us fixed cost each — the old bundle kept whole layers in 1-2 fused
   calls (it had MUL_MAT_ADD / ADD_RMS_NORM_MUL kernels: 38 fused-op symbols
   in libggml-hrx.so.0.9.11).

2. Long-context decode-split FA unwired. The loom corpus defines the
   long-context path (flash_attention_decode_split_produce_partials + _reduce_f32
   exports, both 32768-capable, reduce via an execution barrier) but the C++
   dispatch (dispatch-flash-attention.cpp) only emits the SHORT fused kernel
   (decode_split..._next_q8). is_supported_decode_key_value_token_count caps KV
   at 2048; the reduce_fused loom template only has bands [64,256] and
   [257,2048] — no >2048 band. Raising the C++ cap alone (tested) makes
   KV>2048 decode FAIL at loom JIT link (1 reachable template applications

## 2026-09-08 — HRX decode-fusion root cause: the model is NOT yet built to the engine decode plan

Goal mtsjhonv (unified two-engine stack), task 1 prerequisite. The modern fork
(amd-hrx-graph / q35-hrx-fix @ 15ff48549) decodes Qwen3-30B-A3B Q4_K_M at
~11.6-12 tok/s (llama-bench tg256, BOTH short KV<2048 and long KV 2.9-3.2k),
vs the old llama-build bundle (ggml-hrx 0.9.11) at 39.9-40.1 tok/s on the same
box/model — a 3.4x regression. Three layered causes, all "model not built to
the engine's own decode design":

1. DECODE RESIDUAL ADDs split to CPU (dominant). The conditional-claim policy
   (device_supports_op, ggml-hrx.cpp ~line 1017) deliberately routes qwen3moe
   VIEW-wrapped / MUL_MAT_ID-src ADDs to CPU (#2147 legacy: claiming them
   orphaned the graph). Consequence: llama.cpp's ggml_backend_sched fragments
   each decode token into ~98 executor graph_compute calls (~2/layer x 48) at
   ~140-350us fixed cost each — the old bundle kept whole layers in 1-2 fused
   calls (it had MUL_MAT_ADD / ADD_RMS_NORM_MUL kernels: 38 fused-op symbols
   in libggml-hrx.so.0.9.11).

2. Long-context decode-split FA unwired. The loom corpus defines the
   long-context path (flash_attention_decode_split_produce_partials + _reduce_f32
   exports, both 32768-capable, reduce via an execution barrier) but the C++
   dispatch (dispatch-flash-attention.cpp) only emits the SHORT fused kernel
   (decode_split..._next_q8). is_supported_decode_key_value_token_count caps KV
   at 2048; the reduce_fused loom template only has bands [64,256] and
   [257,2048] — no >2048 band. Raising the C++ cap alone (tested) makes
   KV>2048 decode FAIL at loom JIT link ("1 reachable template applications
   remain unresolved") — the missing band/export wiring is required, not just a
   number bump.

3. The engine's OWN intended decode plan (benchmarks/loom/qwen3_30b_a3b_q4_k_m.tg8.json:
   14 fused dispatches incl. flash_attention_decode_split_next_q8,
   dense_linear_q4k_q8_1_x4_next_q8, routed_down_q4k/q6k_next_q8 — ZERO
   standalone ADDs) does not match what llama.cpp actually schedules, because
   the fused matchers are gated to patterns the live qwen3moe decode graph
   does not produce (e.g. attention_output_next_q8 requires input_size==4096;
   the decode attn-out mm is 2048-wide on this model) and the residual-ADD
   claim gap (cause 1) prevents chain formation.

Fix direction (per owner: complete new build, each model built to the engine
design): wire the long-context produce+reduce decode path in dispatch +
scheduler (two-dispatch sequence with execution barrier; add the >2048
reduce_fused loom band or emit the corpus exports), and give qwen3moe decode
residual ADDs fused-chain coverage (absorb into next_q8 chains like the old
bundle's MUL_MAT_ADD) so llama.cpp stops fragmenting per layer. Target: modern
fork tg256 >= 40 tok/s at KV 2.9-3.2k with the fused decode plan engaged.
Current state: baseline re-measured (old 40.0, modern 11.96); cap-raise
attempt reverted (tree restored, working at 11.64).
