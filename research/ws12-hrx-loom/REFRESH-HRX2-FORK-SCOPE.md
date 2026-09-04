# Round 28 — scope: refresh the HRX2 fork onto AMD `hrx-graph-develop-v2` + loomc alignment

**Date:** 2026-09-03 (scoping — read-only; no fork edits made)
**Live fork (current):** `~/hrx-ws/hrx-v2-src` = `bong-water-water-bong/llama.cpp` branch
`hrx-v2` @ `8df3330` (round-25t era; local build `5e4f14a`, Sep 2). Provenance confirmed via
git ls-remote — nothing newer has ever been pushed on `hrx-v2`.
**Target base:** `AMD-Ecosystem/llama.cpp` branch `hrx-graph-develop-v2` head `6319038`
(2026-09-03, "#98 ggml-hrx: adapt Loom compile configuration"). Cloned at
`~/hrx-ws/amd-hrx-graph`.
**Runtime deps:** loomc/HRX installed at `~/hrx-ws/install-new` (loomc 0.1.0, custom 1bit
config providing `hrx::hrx`, `loom::binding::c::loomc`, `loom::binding::c::target::amdgpu`);
source checkout of `ROCm/hrx.git` at `~/hrx-ws/hrx-rocm` (HEAD `0bc22fb`, loom source present).

## Why (the stale symptoms, all observed this session on the 8df3330 build)
1. **Decode NaN bug**: fused-decode kernels return all-NaN logits on hidden-2048 models
   (qwen25-3b, zaya); workaround `GGML_HRX2_DISABLE_F=1` (qwen) / `DISABLE_FUSION=1` (zaya).
   qwen3-06b (hidden 1024) is clean — shape-dependent kernel bug in the 1bit decode corpus.
2. **Multi-seq SEGV** (npl≥2): wild CPU `dup_bytes` on the zaya recurrent conv-state cache copy
   (`zaya.cpp:219-229` + `llama-memory-recurrent`); single-seq-only recurrent path.
3. **No llama-server** in the fork (no `examples/server`, no impl lib) → no continuous batching.
4. Provenance: the 1bit layer (zaya/type42/decode kernels) was never pushed past `8df3330`;
   AMD's branch moved on without it and without 1bit's newer-loomc API usage.

## Non-goals (this round)
- Do NOT fix the decode-NaN/segfault kernels in place on the stale fork (short-term churn;
  the refresh should absorb or supersede them).
- Do NOT port the full zaya/Q4NX serving stack yet — only make the refreshed base load +
  decode correctly and re-baseline multi-seq.

## Workstreams

### W1 — loomc/HRX runtime alignment (prereq for any compile)
AMD `ggml-hrx` at `6319038` expects loomc with `loomc_compile_options_t.config_flags`
(its own ExternalProject fetches hrx+loom); 1bit's installed loomc 0.1.0 predates it. Steps:
1. Pin the loom/HRX commit AMD's branch builds against (from the branch's `ggml-hrx-deps`
   ExternalProject URL/ref, `ggml/src/ggml-hrx/CMakeLists.txt`).
2. Build loomc/HRX from `~/hrx-ws/hrx-rocm` at that commit → new install prefix
   (`install-rsync` style, non-destructive — do NOT overwrite `install-new` used by the stale
   fork).
3. C0 gate: `cmake --build ... --target ggml-hrx` compiles clean at `6319038` (config_flags
   resolves); `hrx-backend-test`/`test-hrx-loom-jit` pass (or LLAMA_BUILD_TESTS scoped on).

### W2 — isolate the true 1bit delta (rename-aware)
The 640-file diff between `hrx-v2-src` and `amd-hrx-graph` is mostly UPSTREAM drift
(1bit's fork base = PrismML-Eng line; AMD base = ggml-org line), not 1bit work. Steps:
1. Diff `hrx-v2-src` vs its own base (bong fork default head) → 1bit-only change list
   (expect: zaya.cpp/llama-arch/type42 in ggml.c+quants, the decode kernel corpus +
   routes under `ggml-hrx2/catalog`, recurrent-memory integration, converter scripts).
2. Diff `amd-hrx-graph@6319038` vs ggml-org master → AMD-only list (ggml-hrx dir, corpus).
3. Compute the port set = (1bit-only) minus (already-in-AMD). Rename map `ggml-hrx2`→`ggml-hrx`
   where AMD owns the dir; keep 1bit corpus/routes as a delta layer on AMD's corpus.
4. C1 gate: produce `port-manifest.txt` (file-by-file: add/merge/conflict) reviewed before any
   cherry-pick. **No blind merge** — this lane's failure mode is silent numerical wrongness.

### W3 — port + re-verify (decode/multi-seq baseline)
1. Apply port set onto `amd-hrx-graph@6319038` (branch `round-28-hrx2-refresh`), keeping
   AMD's `#95` BF16 support and `#93` merge-mode corpus build.
2. C2: qwen25-3b-q4nx + qwen3-06b decode **clean with NO DISABLE_* flags** (logits probe
   `/tmp/dprobe` pattern: NaN=0, sane argmax).
3. C3: zaya-q4nx.gguf loads + decodes clean; logits sane; compare vs stale baseline
   (pp ~216 t/s, tg ~16.8 t/s @ DISABLE_FUSION).
4. C4: multi-seq `llama-batched-bench -npl 1,2,4,8` on qwen (non-recurrent) — no segfault,
   record the agg-t/s curve vs stale's 43→138.
5. C5 (zaya multi-seq, stretch): zaya npl≥2 no SEGV — re-examine the recurrent conv-state
   copy against AMD's recurrent framework state (may be fixed upstream or still single-seq).
6. C6: re-check decode kernel corpus coverage — do the r16x8t/fused decode routes still
   exist post-port (they are 1bit-only; confirm they compile into the new corpus).

## Risks
- **API drift compounding**: loomc (W1) + ggml backend (W2) + llama layer (W3) all move; each
  mismatch costs a compile-fix cycle. Budget 3-5 such cycles.
- **Silent wrongness**: refreshed decode may "work" with wrong logits (the lane's documented
  failure mode). Every gate above checks NUMERICS (dprobe NaN/argmax), not just exit codes.
- **Loom Rust build time**: rebuilding loomc at a newer commit can take 20-60+ min and may
  pull new IREE/ROCm bits; the box is in active use — build in a dedicated prefix, run
  non-destructively.
- **Box is production**: stale fork must remain runnable until C2/C3 pass (keep `hrx-v2-src`
  untouched; new work lives in `amd-hrx-graph`).

## Acceptance criteria
- W1 + W2 land; port manifest reviewed; no in-place "fixes" to the stale fork.
- C0-C4 green (compile, clean decode w/o flags, zaya decode, multi-seq no segfault).
- Perf at or above stale baseline (pp ≥216 t/s, tg ≥16.8 t/s single-seq zaya).
- Stretch: C5 (zaya multi-seq) and/or C6 (corpus parity) tracked as follow-ups if red.

## Logging
Updates in `~/okf/log.md` (round-28 section); scope file next to ROUND27 docs.
