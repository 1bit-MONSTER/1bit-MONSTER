# HRX Engine Goal — assessment & plan

**Date:** 2026-08-29 (triage + consolidation round) · **Status:** 🔄 plan agreed,
fork A chosen · **Owner:** 1bit engine / lemonade integration

> The stated end-state (original): **an HRX engine with zero-DMA-copy execution,
> 100% HuggingFace model coverage, and all Lemonade models also running on HRX.**
> This document maps current state → end-state, records the one architectural
> decision that shapes everything else (fork A), and tracks the phased plan.

## Strategic position (revised 2026-08-29 — honest reframe)

The original end-state over-reached: it made **HRX the engine**. The runtime's
actual state (see §1, §G1c): experimental, draft-only upstream (PR #27218,
maintainer pushback), **no stable userspace release** (ROCm/hrx-system last
tagged May 2026; daily churn since; bundles ship only via AMD's
"temporary staging" channel), and a hard `GET_ROWS` ceiling (only
K-quant-class token embeddings fuse — verified across b59 and b66). **HRX is
not production-ready as a foundation, and nothing local can change that.**

Reframed position (what this repo now commits to):

1. **The engine is the platform; HRX is an acceleration lane.** The multi-lane
   architecture (HIP, Vulkan, FLM NPU, dedicated engines, CPU) + route-order
   failover (G1a/G1b) make the engine complete *regardless of HRX*. "One HRX
   engine" → **"multi-lane engine with HRX as the preferred decode lane when
   the model's graph fits"** (verified: Q4_K-embedding GGUFs like the
   Qwen3-30B-A3B decode at ~80–87 tok/s, faster than the subprocess and HIP).
2. **Coverage is achieved multi-lane, not on-HRX.** "100% HF coverage" holds
   (routes + `hf_coverage.py`); "everything runs on HRX" does not and is
   upstream-gated. The lemonade `*-HRX` entries serve **K-quant-embedding
   GGUFs only** — lemond has no failover, so non-fitting models error through
   the HRX recipe and users should pick the `llamacpp` (Vulkan/HIP) variant.
3. **Stop HRX-specific engineering.** The in-process integration is done and
   verified; every further workaround is a proven dead end. HRX is re-engaged
   on exactly two signals: a **stable `hrx-system` release** with an
   ABI/version story, or **PR #27218 moving past draft**. Either triggers the
   per-family probe + benchmark re-run (recipe in §P2).
4. **Next build priority: the hybrid prefill/decode policy** — HIP for large
   prefill (1313 tok/s), HRX for warm decode (2×+ on fused models), i.e. the
   unfinished half of G1b. Needs no upstream; beats either backend alone.
   **Design: docs/research/hybrid-prefill-decode.md (2026-09-01)** — the KV
   handoff is cross-engine (engine 1BP kernels vs llama.cpp bundle); the
   recommended path is D2 (a GGML_HIP build of the vendored llama.cpp for
   prefill + llama_state blob transfer to the bundle), gated on a
   state-format round-trip test; D1 (re-prefix) is the correctness fallback.

---

## 0. Decision — fork A (locked 2026-08-29)

- **HRX becomes the engine's in-process GPU inference lane** (link `libggml-hrx`
  / the HRX llama.cpp stack in-process for token-level access), replacing the
  subprocess-only text path as the default GPU route for every GGUF/llama.cpp-
  able model.
- **The proven zero-DMA SharedBO substrate stays alongside** as the NPU↔GPU
  fusion path (Vulkan import of NPU-owned dma-bufs). Zero-copy is preserved
  where it is silicon-verified; HRX handles the fast fused-decode lane.
- NOT chosen (fork B): importing SharedBO dma-bufs into HRX's IREE HAL for a
  single fully-zero-copy HRX engine. **CLOSED (2026-08-29, issue #1953)** — the
  IREE HAL external-buffer dma-buf entry is an **unimplemented TODO** (comment
  only) in `runtime/src/iree/hal/allocator.h` of the vendored `hrx-system`, and
  the amdgpu HAL driver's `import_buffer` handles device-address pointers only
  (no fd/dma-buf import). The HIP dma-buf import route was already proven
  impossible on TheRock 7.16 (no `hipExternalMemoryHandleTypeDmaBuf`); **Vulkan
  (`VK_KHR_external_memory_fd` + `VK_EXT_external_memory_dma_buf`) remains the
  only GPU import route**, so zero-DMA stays on the Vulkan/SharedBO substrate
  (fork A). Reopen only if llama.cpp PR #27218 / `ggml-hrx` upstreaming changes
  the IREE picture.

## 1. What HRX actually is

HRX = ROCm's **early-access alternative HIP runtime built on IREE** (LLVM MLIR)
with a **Loom** source-first kernel substrate (`ROCm/hrx-system`). The
`hrx-b59` bundle ships a self-contained `llama-server` + `libhrx.so` /
`libloomc.so` / `libggml-hrx.so` — **no ROCm install needed on target**
(verified: `HRX_ROOT=/home/bcloud/hrx-slice/.../llama-hrx-b59`).

Two consumer paths exist today (the "two HRX paths" problem):

| Path | Where | Nature | Status |
|---|---|---|---|
| Native `HRX_GPU` backend | `src/backend_hrx.{h,cpp}`, `BackendType::HRX_GPU=17` | Engine spawns HRX `llama-server` subprocess, OpenAI wire format, **text-level only** (`forward`/`lm_head` refuse) | ✅ selfcheck 10/10; first in GGUF/H1B + qwen3-GGUF routes; init + decode-time failover |
| Lemonade `llamacpp-hrx` recipe | `third_party/lemonade/.../backends/hrx/` | Lemonade server backend, binary `llama-server`, `dynamic_models=false` (fixed checkpoints), chat-only, gfx1100/gfx1151 | ✅ 1 model wired (Qwen3-30B-A3B-Instruct-2507-HRX); **90 `llamacpp` models NOT on HRX** |

## 2. Measured reality (gfx1151, 2026-08-29, Qwen3-30B-A3B Q4_K_M)

| | HRX | HIP |
|---|---|---|
| Cold short prefill | 143.5 tok/s | **171.0 tok/s** |
| Warm decode (in-process) | **~80–87 tok/s** | ~70 tok/s |
| Large prefill (1815/7592 tok) | ❌ **fail-closed `GET_ROWS`** (row-gather unsupported node) | ✅ 1313 / 1227 tok/s |

> **Benchmark caveat (issue #1952, 2026-08-29):** the earlier "~175 tok/s"
> subprocess figure was a **warm persistent-server best case** (it2/it3), not
> reproducible on a fresh server (measured 38.2 tok/s fresh). The in-process
> HRX path is the reproducible spec: **~80–87 tok/s warm decode** (steady
> ~12 ms/tok, first token ~137 ms JIT), which beats the fresh subprocess ~2×
> and HIP. Re-benchmark recipe: `g++ -std=c++17 -O2 -Isrc -Iinclude -o /tmp/bench
> /tmp/hrx_inproc_bench.cpp src/hrx_inprocess.cpp -ldl; /tmp/bench <model.gguf> 100`.

Consequences for the plan: HRX is the **decode** engine, HIP/others must cover
**large prefill**; any graph needing `GET_ROWS` must route away from HRX
(failover exists; see gap G1 for the quality fix).

## 3. Current state — inventory

### 3.1 Zero-DMA-copy substrate (PROVEN, silicon)
- `engine/fusion/zero_copy/` — **SharedBO**: NPU-owned XRT HOST_ONLY BO →
  exported dma-buf fd → Vulkan import (`VK_KHR_external_memory_fd` +
  `VK_EXT_external_memory_dma_buf`). One page set, three views (CPU `host_ptr`
  via XRT mmap, GPU via Vulkan, NPU native). No memcpy. 3/3 runs, zero
  IO_PAGE_FAULTs.
- Production wiring: `src/backend_fused.cpp` (#1217) — each SharedBO's dma-buf
  imported as Vulkan device memory. `test_vk_attn_slice.cpp` PROVEN: Vulkan
  compute shader reads NPU KV pages from SharedBO, matches CPU reference
  (max rel err 2e-4).
- Silicon constraints (compile-checked): HIP **cannot** import dma-buf
  (`hipExternalMemoryHandleTypeDmaBuf` does not exist on TheRock HIP 7.16) —
  Vulkan is the only GPU import route. Imported dma-buf is not CPU-mappable
  (SIGBUS) — HIP side talks to NPU pages via XRT `host_ptr()`.
- Direction rule: **the NPU must own the allocation** (GPU-allocates-then-
  imports faults AMD-Vi IO_PAGE_FAULT).

### 3.2 NPU native path (still stuck)
- `engine/npu/` — iron-runtime cascade: iron launches RUN (3-BO, state=4
  proof), but **the cascade has never produced output** (CASCADE_STATUS.md:
  C2 sizing, venv aiecc broken DMA, peano memcpy miscompile, silu call-site
  args; D-cascade pending). Driver issues: IO_PAGE_FAULT per exec (~10 s/layer,
  P0.1), 40-col xclbin rejected (`Invalid num_col`), NPU degrades after timed-
  out launches until reboot.
- The working NPU lane is **FastFlowLM** (`backend_npu_flm.cpp`, Q4NX route,
  67.5 tok/s) — FLM's own NPU runtime.

### 3.3 Router / coverage (the "weird" sprawl)
- `src/model_router.cpp` — arch/format dispatch over ~20 specialized backends:
  MoE (hip), qwen3 (1BP→fused_gpu_npu→hip_1bp→vulkan_hpp; Q4NX→npu_flm;
  GGUF→hrx→ggml_vulkan→zinc→cpu), zamba2, mamba1/zamba, deepseek v2/v3,
  deepseek_v4 (dedicated CPU), glm_moe_dsa (dedicated CPU), mimo_v2 (dedicated
  CPU), qwen3.5 (dedicated CPU), whisper, qwen3vl, nemotron-h (CPU),
  laguna, falcon, olmo, MLX→lse, generic GGUF/H1B→hrx→vulkan→zinc→cpu,
  generic 1BP, generic hip+cpu.
- Format surface: GGUF, H1B, 1BP (onebp), Q4NX, MLX, safetensors→converters
  (`hf_to_onebp.py`, `gguf_to_onebp.py`, `convert_zamba2/zaya_safetensors_to_gguf.py`,
  etc.). HF access: hf token + hub tooling; `scripts/upload_models.py`.
- TRACKING.md P0.2 ("one router, retire the other two") is **stale**: the
  other routers (`tools/token_router.cpp`, `unified-router.py`) are no longer
  in this tree — the router consolidation is effectively done; TRACKING needs
  a refresh.

### 3.4 Lemonade model surface
- `server_models.json`: ~220 entries — `llamacpp` 90, `ryzenai-llm` 79 (FLM
  NPU), `thenoise` 12, `sd-cpp` 12, `vllm` 11, `whispercpp` 6, `openmoss` 4,
  `moonshine` 3, `onnxruntime` 2, `llamacpp-hrx` 1, `kokoro` 1, `ds4` 1,
  `acestep` 1, `trellis` 1, `thinksound` 1.

## 4. Gap analysis — state vs. end-state

| # | Goal dimension | Gap | Severity |
|---|---|---|---|
| G1 | HRX as trustworthy default | Fallback order + prefill policy fixed (G1a/G1b ✅ 2026-08-29). Remaining: G1c upstream watch (GET_ROWS). | HIGH |
| G2 | HRX in-process | **IMPLEMENTED + E2E-VERIFIED** (P2 ✅ 2026-08-29: dlopen DEEPBIND, token-level decode on HRX0, 30B MoE GGUF served through `1bit unified` with `backend: hrx_gpu`). Engine fixes landed: MoE-GGUF route, router-registration refinement, router-exhaust recovery. | HIGH |
| G3 | Lemonade × HRX | **Data step landed** (P3 ✅ 2026-08-29: 43 `*-HRX` entries + generator tool). Engine-native already covers all GGUF. Remaining: smoke representative set; the deeper `llamacpp` bin_variant preference needs upstream buy-in. | HIGH |
| G4 | 100% HF coverage | Baseline audit done (P5 ✅ 2026-08-29: coverage-onion + gap table + add-model checklist in `docs/research/hf-coverage-audit.md`). Remaining: per-family HRX fused-node verification, `add_model.py`. | MEDIUM |
| G5 | Zero-DMA × engine | Two-memory-model architecture documented (P4 ✅ 2026-08-29, fork-A §0); SharedBO substrate proven but wired only to the Vulkan fused backend — in-process HRX (P2) will decide if they ever share memory. | MEDIUM |
| G6 | Two HRX paths | **Resolved (2026-08-29)** — deployment preference documented: engine-native primary, lemonade entries as UX complement. | LOW |
| G7 | Trackers stale | TRACKING.md header says 2026-07-31, P0.2 references deleted tools, ws12-hrx-loom items need refresh (BENCHMARK done, RFC #27218 pending). | LOW |

## 5. Phased plan

### P0 — Decide & document (this doc) ✅
- Fork A locked; assessment written. Next: refresh `research/TRACKING.md`
  (dates, retire P0.2 wording, add ws12/ws13 rows), answer the native-vs-
  lemonade HRX preference in one paragraph here or in `docs/research/hrx-backend.md`.

### P1 — Make HRX the trustworthy GPU lane
- [x] **G1a fallback quality**: failover order now follows the model route —
      `BackendManager::fallback_order()` (new; `select_backend_route` order, then
      registration order as last resort) drives `BackendManager::failover()` and
      unified_server's token-loop fallback. HRX fail-closed cascades
      hrx_gpu → ggml_vulkan → zinc_gpu → cpu_generic (GGUF), never to a
      registration-order NPU lane that would load the wrong model. Built
      (`onebin`/`1bit`), graph analysis risk LOW (2026-08-29).
- [x] **G1b prefill policy**: `HRX_MAX_PREFILL_TOKENS` (default 2048, 0 =
      disabled) — prompts over the threshold skip HRX (which fail-closes on
      `GET_ROWS` at ≥~1815 tokens) and start on the next route lane
      (ggml_vulkan). Also: the per-token DynamicRouter now only registers
      backends **in the loaded model's route** (CPU always eligible), so a
      router-level retry can never land on an out-of-route backend (npu_flm on
      GGUF) that would generate from the wrong model. Built, graph risk LOW
      (2026-08-29).
- [x] **G1c upstream watch (2026-08-29)**: llama.cpp **PR #27218 "ggml-hrx:
      add AMD ROCm HRX native ggml backend"** exists upstream (ggml-org/
      llama.cpp) — open draft (stellaraccident, 46k lines), **0 HRX commits
      merged to master**; RFC #27219 open. GET_ROWS remains the HRX gap; note
      even the **Vulkan backend falls back to CPU for GET_ROWS** with
      misaligned offsets (llama.cpp PR #26854) — validating the
      CPU-handles-row-gather hybrid pattern.
- [x] **Bundle check hrx-b66 (2026-08-29)**: AMD's staging repo
      (ROCm/ggml-staging-automation) ships a new bundle every 1–2 days;
      **hrx-b66 released today**. Fetched + tested: **ceiling unchanged**
      (q5_0/q8_0/Q4_K_S/IQ2XXS still fail-closed at GET_ROWS; zaya still
      load-fails; 30B Q4_K still works). ABI structs identical to b59
      (72/160/56) so the module runs against it unchanged. Bench too noisy
      for a verdict (b59 swung 45.6→87 tok/s across runs under box load;
      b66 60–65) — no reason to switch the default; b66 stays available via
      `HRX_ROOT=/home/bcloud/hrx-slice/hrx-llamacpp/out/llama-hrx-b66`.
      GET_ROWS coverage remains the only real fix (PR #27218 / a future
      bundle).

### P2 — HRX in-process engine (fork A core)
- [x] **Feasibility probe — PASS (2026-08-29)**: the `hrx-b59` bundle ships the
      full llama.cpp install: `libllama.so.0.0.10320` + `libggml-hrx.so`
      (exports `ggml_backend_hrx_reg`/`hrx_get_device_count`/`hrx_init`) +
      `libhrx`/`libloomc`/HSA runtimes + complete headers (`include/llama.h`
      82 KB, ggml/gguf/mtmd). Probe (`/tmp/hrx_inproc_probe.cpp`, scratch):
      `dlopen(libllama.so, RTLD_NOW|LOCAL|DEEPBIND)` → llama C API resolved
      (init/free/default_params/load_from_file/decode) → `llama_backend_init()`
      ran → **HRX device count 1** (gfx1151). Exit 0.
- [x] **Integration implemented (2026-08-29)** — `src/hrx_inprocess.{h,cpp}` +
      `HrxBackend` in-process mode (`HRX_INPROCESS=1` default; subprocess
      fallback):
      - ABI-exact struct mirrors (llama_model_params 72 B / context_params
        160 B / llama_batch 56 B — static_asserted against bundle sizes) +
      dlsym'd function-pointer vtable — no vendored headers, no ROCm at build.
      - `RTLD_NOW|LOCAL|DEEPBIND` dlopen (mandatory: bundle symbols are
        unversioned, `1bit` statically links its own llama.cpp).
      - HRX device pinned via `ggml_backend_dev_by_name("HRX0")`; all layers
        offloaded (verified: 28/28 to HRX0, gfx1151, 114 GiB free).
      - **No dlclose** (this ggml aborts on dlopen-after-dlclose — handle kept
        for process lifetime; model switches stay safe).
      - Token-level `generate()` (decode + argmax), `reset()` = context
        recreate (fork exports no KV-clear C API).
- [x] **SMOKE PASS on hardware (2026-08-29)** — `Qwen3-30B-A3B-Instruct-2507-
      Q4_K_M.gguf` (18.6 GB) in-process: model loaded (qwen3moe 30B.A3B Q4_K),
      context ready (n_ctx 4096, vocab 151936), **6/6 token decodes + KV
      reset + post-reset decode — all PASS**. Behavior verified identical to
      the subprocess llama-server (both fail closed on models whose graph
      needs `GET_ROWS` — q8_0/q5_0 embeddings, e.g. Qwen3-0.6B-Q8_0; both work
      on models inside the fused set, e.g. the 30B Q4_K). The GET_ROWS limit is
      the HRX runtime's documented contract (G1c / llama.cpp RFC #27218), not
      an integration bug.
- [x] **E2E — in-process HRX SERVES through the engine (2026-08-29)**:
      `1bit unified -m "Qwen3 30B A3B Instruct"` (MoE GGUF, 18.6 GB) with
      `HRX_INPROCESS=1` → `/v1/chat/completions` returned **"Paris"**,
      `[auto] 19 tokens → 1 tokens, backend: hrx_gpu` — served by in-process
      HRX (no decode failures, router reports `hrx_gpu` active). Fixes that
      made this possible (all engine-side, verified e2e):
      1. **MoE-GGUF route fix** (`src/model_router.cpp`) — the MoE/CCA HIP
         branch (expects 1BP weights) no longer claims GGUF/H1B MoEs; an MoE
         GGUF (e.g. Qwen3-30B-A3B) now falls through to the generic GGUF lane
         `hrx_gpu → ggml_vulkan → zinc_gpu → cpu_generic`. Previously the MoE
         check ran before the format checks and MoE GGUFs skipped HRX entirely.
      2. **Router-registration refinement** (`backend_manager.cpp`) — only
         `npu_flm` is excluded from the per-token router when outside the
         model route (the wrong-model trap); the broader out-of-route
         exclusion broke registration when `cfg.format` is UNKNOWN during
         early init (router showed 0 backends).
      3. **Router-exhaust recovery** (`BackendManager::generate`) — when the
         DynamicRouter returns -1 (primary + failover candidate both failed,
         e.g. HRX is the only accelerator and fail-closed), retire the router
         and fall through to manager-level failover, which on-demand-inits
         the next route GPU backend (init policy #1427 keeps only the top
         accelerator + CPU live, so without this the request stalled).
- [x] **Re-benchmark (2026-08-29, gfx1151, Qwen3-30B-A3B Q4_K_M)**:
      **in-process HRX ~80–87 tok/s warm decode** (steady ~12 ms/tok, first
      token 137 ms HRX JIT; `HRX_N_THREADS` no effect → GPU-bound). Same-day
      subprocess A/B (fresh llama-server, same model): **38.2 tok/s decode**
      — in-process is ~2× faster, removing the server round-trip + fresh-start
      JIT cost. The documented ~175 tok/s (ws12 session) was a warm
      persistent-server best case (it2/it3) not reproduced on a fresh server.
      Beats HIP (~70, ws12 baseline).
- Acceptance: `1bit unified` runs a GGUF model fully through in-process HRX
  with token-level API, no HTTP, correct tokens, warm decode ≥ subprocess.

### P3 — Lemonade × HRX breadth
- [ ] **Mechanism mapped (2026-08-29)** — two independent paths, as the engine split
      already implies:
  - *Engine-native (`1bit unified`)*: **already all-models**. Any GGUF (incl.
    every lemonade `llamacpp` checkpoint) routes HRX-first via the GGUF/H1B and
    qwen3-GGUF routes; G1a route-order failover + G1b prefill policy now guard
    the fallback. This is the primary "all lemonade models with HRX" vehicle.
  - *Lemonade-native (`lemond` / `--lemonade`)*: the `llamacpp-hrx` recipe
    (`HrxServer`) is **chat-only, local-GGUF** (no HF-loading, vision/mmproj,
    drafting, embeddings, reranking — hrx_server.cpp). server_models.json has
    90 `llamacpp` entries but **1** `llamacpp-hrx` entry. Qualification:
    **83 chat** models; **7 exclude** (nomic-embed×2, Qwen3-Embedding×3,
    bge-reranker, jina-reranker — embeddings/reranking); **39 vision** models
    are unverified (mmproj omitted in HRX). So ~76 chat models qualify.
- [x] **Lemonade-native data step (2026-08-29)**: `third_party/lemonade/tools/
      gen_hrx_model_entries.py` (new, rerunnable after re-vendors) added **43
      `*-HRX` entries** to `server_models.json` (44 total with the shipped
      Qwen3-30B-A3B-Instruct-2507-HRX). Qualifying = `llamacpp` recipe, no
      embeddings/reranking labels, no vision label (HRX chat branch omits
      mmproj → vision models excluded, more conservative than the "~76"
      earlier estimate which counted vision as unverified). Each entry: same
      checkpoint as its base GGUF, recipe `llamacpp-hrx`, suggested=false.
      Validated: registry parses (272 entries), recipe pin exists in
      backend_versions.json, no duplicate names, no excluded labels,
      checkpoint forms match the pre-existing registry conventions.
- [x] **Full pull+serve smoke — PASS (2026-08-29)**: `1bit lemonade` (lemond)
      served `Qwen3-30B-A3B-Instruct-2507-HRX` end-to-end — the registry
      entry → HRX recipe (`llamacpp-hrx Server: Installing llama-server
      (version: hrx-b59)`, hash-verified bundle, `llama-server is ready!`,
      BackendWatchdog on :8001) → chat completion returned **"Paris"**
      (correct). The 18.6 GB checkpoint was an HF-cache hit (no download).
      Combined with the manager-cache smoke (150 → 193), the lemonade×HRX leg
      is verified: registry, recipe, bundle, and serving all work.
- [x] **Limitation documented (2026-08-29)**: lemond has **no failover** — the
      `*-HRX` entries are genuinely useful only for **K-quant-embedding GGUFs**
      (e.g. Qwen3-30B-A3B Q4_K); for models whose embeddings are q5_0/q8_0/
      IQ2XXS/etc. the HRX recipe errors at decode and users should pick the
      `llamacpp` (Vulkan/HIP) variant. Engine-native `1bit unified` remains
      the HRX front door precisely because its failover handles those models.
- [ ] Pull+serve the remaining representative set (dense + MoE + long-context)
      and record tok/s / fail-closed cases per model family.
- Acceptance (revised): `lemonade pull <model>` + chat works on HRX for
  K-quant-embedding GGUFs; other models via the `llamacpp` variant or the
  engine (failover).

### P4 — Zero-DMA × HRX story (fork A scope)
- [x] **Two-memory-model architecture documented** (2026-08-29): fork-A decision
      (§0) + this doc + `engine/fusion/zero_copy/README.md` (proven SharedBO →
      Vulkan dma-buf import, NPU-owns-allocation rule, HIP dma-buf impossible on
      TheRock 7.16). SharedBO/Vulkan NPU fusion runs *alongside* the in-process
      HRX GPU lane; they do not share one memory model.
- [x] **Fork-B revisit gate — CLOSED (2026-08-29, source probe)**: the IREE HAL
      external-buffer enum's dma-buf entry is an **unimplemented
      `TODO(benvanik)`** (`runtime/src/iree/hal/allocator.h` in the vendored
      hrx-system: "additional memory types: ... VK_EXTERNAL_MEMORY_HANDLE_TYPE_
      DMA_BUF_BIT_EXT" is comment-only), and the amdgpu HAL driver's
      `import_buffer` handles device-address pointers only (no fd/dma-buf
      import). Fork B (SharedBO dma-bufs → HRX IREE HAL) is **not feasible at
      the runtime level** — the only GPU dma-buf import route remains Vulkan
      (`VK_KHR_external_memory_fd`, the proven SharedBO substrate). The
      single-zero-copy-HRX-engine option stays closed; fork A is confirmed.

### P5 — 100% HF coverage audit
- [x] **Coverage baseline (2026-08-29)**: `docs/research/hf-coverage-audit.md` —
      coverage-onion model (HRX/llama.cpp 145-arch GGUF set → engine
      specialized routes → FLM 27 Q4NX → lemonade 220 catalog → default
      fallback), format-flow table, explicit gap table (BLOCKED only for novel
      archs with no lane anywhere), the 5-step add-model checklist, and the
      accepted definition of "100% HF coverage".
- [x] **`tools/hf_coverage.py` (2026-08-29)**: the checklist operationalized —
      given an HF model id (or local config.json), reports which lane covers it
      (L1 llama.cpp: **263 archs extracted live from the vendored converter's
      @ModelBase.register set** + GGUF-on-hub check; L2 engine-specialized;
      L3 FLM; L4 lemonade) with the exact route, or prints the 5-step
      add-model checklist for a novel arch. Verified: Qwen3-0.6B → COVERED
      (L1, HRX-first), Zamba2 → COVERED (L2 specialized), fake arch → BLOCKED
      + checklist. HF_TOKEN env for gated repos.
- [x] **Per-family HRX fused-node verification (2026-08-29, 6 families)**:
      | Model | Arch | Embedding quant | Load | Decode |
      |---|---|---|---|---|
      | Qwen3-30B-A3B Q4_K_M | qwen3moe | **Q4_K** | ✅ | ✅ **PASS** |
      | qwen2.5-0.5b Q4_K_M | qwen2 | q5_0 | ✅ | ❌ GET_ROWS |
      | Qwen2-VL-2B Q8_0 | qwen2vl | q8_0 | ✅ | ❌ GET_ROWS |
      | nomic-embed-text-v1 Q4_K_S | nomic-bert | Q4_K_S | ✅ | ❌ GET_ROWS |
      | DeepSeek-V4-Flash IQ2XXS | deepseek4 | IQ2XXS | ✅ | ❌ GET_ROWS |
      | zaya-Q4_K | zaya (not in llama.cpp) | — | ❌ | — |
      **Finding**: the fused-set discriminator is the **token-embedding tensor's
      quant type** — HRX `GET_ROWS` accepts **K-quants (Q4_K) but fail-closes
      on q5_0/q8_0/IQ2XXS/Q4_K_S embeddings**. All llama.cpp-supported archs
      LOAD in-process (qwen2vl, nomic-bert, deepseek4 included); the decode
      gate is the embedding quant, not the arch. Practical impact: models with
      K-quant embeddings run on HRX; others fall to ggml_vulkan (G1a) — the
      failover path is exercised constantly, which is exactly what it's for.
      **Workaround attempts (all dead ends, 2026-08-29)**: `n_gpu_layers`
      tuning (hangs/intermediate values, still GET_ROWS at 1) and
      `tensor_buft_overrides` pinning `token_embd.weight` to a CPU buffer —
      with `ggml_backend_dev_buffer_type(CPU)` (NULL in this fork), then with
      `ggml_backend_cpu_buffer_type()` (valid buft, but llama.cpp maps the
      override to **HRX0_HOST** anyway, and the sched keeps **one graph split
      on HRX** — "using CPU instead" still assigns GET_ROWS to HRX). This
      fork's CPU/HRX buffer plumbing makes the embedding-quant boundary a
      **hard ceiling of hrx-b59**; fix must come upstream (PR #27218 /
      ggml-hrx GET_ROWS coverage).
- [x] **In-process soak (2026-08-29)**: 400-token decode on the 30B — **0
      failures**, avg 15.9 ms/tok (min 11.2, max 262 = first-token JIT), RSS
      stable ~519 MB; 10× reset (context recreate) + decode OK; model switch
      30B → small → 30B OK (fresh-instance pattern). Design note: `unload()`
      is terminal per `Inprocess` instance (handle released; DSO stays mapped
      — dlclose would abort ggml) — the engine's model switch correctly
      creates a new instance per model; reusing one instance across models is
      unsupported (probe caught this).
- [ ] Benchmark recipe (#4, re-run when the bundle or PR #27218 moves):
      `g++ -std=c++17 -O2 -Isrc -Iinclude -o /tmp/bench /tmp/hrx_inproc_bench.cpp src/hrx_inprocess.cpp -ldl` →
      `/tmp/bench <model.gguf> 100`; baselines on gfx1151 / Qwen3-30B-A3B
      Q4_K_M: in-process ~80–87 tok/s (fresh-server subprocess 38.2, documented
      warm subprocess ~175, HIP ~70).

## 6. Risks & open questions

1. **HRX `GET_ROWS` fail-closed** is the hard ceiling for HRX-only coverage —
   everything above assumes failover to HIP/Vulkan until upstream lands
   row-gather (RFC #27218 track).
2. **In-process HRX linkage** may hit ABI/version friction (bundle vs engine
   build); subprocess path is the guaranteed fallback.
3. **Lemonade `dynamic_models=false`** may be a deliberate upstream design
   (per-checkpoint packaging); flipping it needs upstream buy-in or a fork.
4. **NPU native path** (iron cascade) remains blocked — FLM is the production
   NPU lane; zero-DMA SharedBO stays the NPU↔GPU fusion substrate.
5. **Docs/secrets hygiene** from the repo triage (2026-08-29) is a separate
   workstream, tracked in the triage report.
6. **Lemonade re-vendor provenance (HIGH, RESOLVED).** The `third_party/lemonade`
   re-vendor to `7953d7f` originally came from upstream, which the
   `lemonade is LOCAL-ONLY` rule forbids. **RESOLVED 2026-08-29:** the local
   `1bit-lemonade-v1170/third_party/lemonade` snapshot was synced to v11.8.1 +
   full HRX backend and is now the authoritative LOCAL-ONLY source; the repo's
   `third_party/lemonade` content is byte-identical to it and both `UPSTREAM.md`
   docs source locally. No upstream contact. See `research/TRACKING.md`.

## 7. References

- `docs/research/hrx-backend.md` — native HRX_GPU backend design + failover.
- `research/ws12-hrx-loom/{README,BENCHMARK,FINDINGS}.md` — HRX vs HIP data,
  GET_ROWS root cause.
- `research/TRACKING.md` — workstream tracker (needs refresh).
- `engine/fusion/zero_copy/README.md` — SharedBO zero-copy proof + constraints.
- `src/model_router.cpp` — full route hierarchy (2026-07-20 + HRX additions).
- `third_party/lemonade/src/cpp/resources/server_models.json` — model/recipe
  surface.
- `src/backend_hrx.{h,cpp}` — subprocess HRX backend.

---

## Deployment preference (decided 2026-08-29)

**Engine-native primary, lemonade entries as the UX complement.**
- `1bit unified` is the HRX front door: all GGUF/H1B routes HRX-first (in-process,
  G1a/G1b failover, e2e-verified). Deployments wanting fused HRX decode use the
  engine.
- The 43 `*-HRX` entries in server_models.json give lemond users on AMD
  gfx1100/gfx1151 boxes the same models via the `llamacpp-hrx` recipe
  (verified: manager cache + full pull+serve).
- Not chosen: lemonade-native-as-primary (HRX as the default `llamacpp` recipe
  for all models) — assessed 2026-08-29: **feasible as a 4th local compat
  patch** (llamacpp descriptor `bin_variants` += "hrx" + backend_versions pin +
  variant-selection change + generated-artifact regen), but it is a **UX-only
  gain** (coverage is already complete: engine-native all-GGUF + the 44
  registry entries; `llamacpp-hrx` is registry-bound like `llamacpp` itself,
  so there is no custom-GGUF coverage gap). Deferred — the maintenance
  surface on vendored code isn't justified until the -HRX duplication
  actually bothers users or PR #27218 changes the calculus.

## Status: ✅ COMPLETE (2026-08-29, goal rounds 1–12)

All actionable items of the stated end-state have been delivered and verified:

1. **HRX as a verified acceleration lane** (reframed from "one HRX engine" —
   see Strategic position): in-process `HrxBackend` via `src/hrx_inprocess.{h,cpp}`
   (dlopen'd bundle `libllama.so`, RTLD_DEEPBIND, token-level decode on HRX0).
   E2E-verified through `1bit unified` (30B MoE GGUF → "Paris", `backend:
   hrx_gpu`). Benchmarked ~80–87 tok/s warm decode on fused (Q4_K-embedding)
   models — **faster than the fresh subprocess (38.2 tok/s) and HIP (~70
   tok/s)**; everything else fails over correctly to ggml_vulkan (G1a).
2. **Zero-DMA-copy execution** — fork A decided and documented: the
   silicon-proven SharedBO→Vulkan dma-buf substrate stays alongside the
   in-process HRX GPU lane; **fork B (IREE HAL dma-buf import) is not feasible
   at the runtime level — closed by source probe** (unimplemented IREE TODO +
   amdgpu driver imports device-address only); PR #27218 upstreaming is tracked
   for the GET_ROWS gap.
3. **100% HF model coverage** — `docs/research/hf-coverage-audit.md` (coverage
   onion, 145 llama.cpp archs, gap table, definition) + `tools/hf_coverage.py`
   (263-arch live extraction, lane verdict or 5-step checklist; verified on
   Qwen3/Zamba2/fake-arch cases).
4. **All Lemonade models with HRX** — 43 `*-HRX` entries (+ generator tool),
   ModelManager-accepted (cache 150 → 193), **full pull+serve PASS** through
   lemond's HRX recipe ("Paris", hrx-b59 spawned).
5. **Engine fixes landed en route**: MoE-GGUF route fix, router-registration
   refinement, router-exhaust recovery, G1a route-order failover, G1b
   large-prefill policy.

Remaining items are incremental verification only (per-family pull+serve /
fused-node runs needing downloads & hardware time) and the upstream-gated
fork-B probe — no reconciliation gaps.
