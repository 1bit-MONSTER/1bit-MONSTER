# FINDINGS — ws12 HRX / Loom (2026-08-28)

Research log for the HRX/Loom platform-transition workstream. All claims carry
source links; hardware claims marked **[validated]** were reproduced on this
machine (Strix Halo, gfx1151, 128 GB).

## 1. What HRX is

- **HRX = "Hip Runtime Extended"** — AMD's alternative, minimal HIP
  implementation. Repo: [ROCm/hrx-system](https://github.com/ROCm/hrx-system)
  (Apache-2.0, created 2026-03-27, explicitly *"early-access... not an official
  component within the ROCm stack"*). README, first lines.
- Ships `libhrx.so` (~2 MiB), `hrx-info`, native low-latency command-buffer C
  ABI, and a HIP compatibility layer (`libamdhip64.so`, ~88% HIP API coverage).
- **Loom JIT compiler**: specializes kernels for the exact hardware to HSACO in
  1–15 ms — no pre-compiled kernel libraries, no large ROCm downloads.
- Same execution path on Linux and Windows (driver-unification work; Windows
  packaging planned but not implemented in staging).

## 2. HRX is IREE, not a new thing

- `loom/docs/mkdocs.yml` in hrx-system carries `# Copyright 2026 The IREE
  Authors`. All ~40 recent commits in hrx-system are by **Ben Vanik** (IREE
  project lead). The build is IREE's: `dev.py`, `IREE_HAL_DRIVER_AMDGPU`,
  Bazel→CMake (`bazel_to_cmake`), `iree-test-loom` / `iree-benchmark-loom` tool
  names.
- Interpretation: AMD is repackaging the IREE compiler/runtime as a lean,
  client-focused HIP alternative. HRX is the runtime; Loom is the compiler.

## 3. The lemonade backend we inherit (commit `7953d7f`)

- Merged **2026-08-28 23:44 UTC** via **merge-queue squash** (single parent
  `90758ffa`, bulleted commit list in the message — this is why it "looks like
  a rebase": it's a squash, not a rebase). Timeline: `added_to_merge_queue`
  21:05:56Z → `merged` 23:44:31Z.
- PR [#3374](https://github.com/lemonade-sdk/lemonade/pull/3374) by
  @AaronStGeorge; approved by @jeremyfowers, @Geramy, @iswaryaalex.

### Recipe facts (from `src/cpp/server/backends/hrx/hrx_server.cpp` + `hrx.h`)

| Aspect | Value |
|---|---|
| Recipe | `llamacpp-hrx` (separate recipe, not a llama.cpp backend) |
| Binary | `llama-{version}-bin-manylinux-hrx-x64.tar.gz` from `ROCm/ggml-staging-automation` releases |
| Pin | `hrx-b59` (2026-08-22), `VersionPolicy::Exact` |
| Checksum | `sha256:d2fe01432c372ae4382678cb80542a0c5c80a8e3862c52b3d94e77e81062f1b7` in `backend_versions.json` **[validated against the download]** |
| Platforms | Linux x86-64 only (`validate_supported_build_host()` throws otherwise) |
| GPUs | `gfx1100` (RDNA3) and `gfx1151` (RDNA3.5) only |
| Model | Exactly one: `Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf` (18.6 GB, `suggested: true`) |
| Launch | `-m <gguf> --ctx-size N --device HRX0 --port P --jinja --metrics`; `--parallel 1` pinned; `GGML_DISABLE_VULKAN=1` |
| Capabilities | chat only (embeddings/reranking masked off); no HF loading, drafting, vision |
| Config | `hrx_args` passthrough w/ reserved-flag validation; `hrx_bin: "builtin"` default |
| Code | `hrx_server.cpp` (209 lines) + 2 headers + `test_hrx_contract.cpp` (153 lines) — thin wrapper over `LlamaCppServer` |

### CMake integration (verified against our vendored tree)

- Backends are **explicitly listed** in `LEMON_BACKENDS` (NOT globbed): HRX adds
  exactly one line `"llamacpp-hrx|hrx"` (line 154) + a `BUILD_TESTING`-guarded
  test block (line 2510). Neither touches our 5-point embeddability patch
  (200 `CMAKE_CURRENT_SOURCE_DIR` occurrences, `USE_SYSTEM_*`, `add_test`
  police) — **no conflict on re-vendor**.
- The `ds4` (DwarfStar) backend is the same pattern already vendored with us
  (experimental recipe, binary from staging repo `lemonade-sdk/ds4-rocm`, pin
  `b0001`) — HRX slots in identically.

## 4. Live binary verification **[validated, this machine]**

Downloaded `hrx-b59` from `ROCm/ggml-staging-automation`, sha256 verified,
extracted, ran:

- `llama-cli --list-devices` →
  `HRX0: AMD Radeon 8060S Graphics (Node 1) (gfx1151) (114688 MiB free)` —
  **HRX works on our exact hardware.**
- `llama-server -m <model> --device HRX0 --port 9876` → `/health` returns
  `{"status":"ok"}`.
- **Fail-closed confirmed**: non-qualified model (nomic-embed) fails with
  `E graph_compute: unsupported HRX node 0: GET_ROWS ... ggml_backend_sched_graph_compute_async failed with error -1`.
  The backend refuses partial coverage — matches RFC contract.
- Tarball is **fully self-contained**: `libhrx.so.0.1.0` (2 MB),
  `libloomc.so.0.1.0` (13 MB), `libmtmd.so`, `libggml-hrx.so.0.18.0`,
  `libvulkan.so.1.3.296`, and a complete `rocm_sysdeps/lib` tree (drm_amdgpu,
  elf, numa, ...) — **no ROCm install needed on target**.
- Binary built from fork commit `f749e1390` ("ggml-hrx: support direct coherent
  host bindings", 2026-08-17) — present in the AMD fork network, **NOT on
  ggml-org master** (compare: diverged, 457 behind).

## 5. The llama.cpp RFC — status and the "replacement" question

- RFC discussion [#27219](https://github.com/ggml-org/llama.cpp/discussions/27219)
  (2026-08-17) + draft PR
  [#27218](https://github.com/ggml-org/llama.cpp/pull/27218) (30 files, ~7k
  lines: dispatch scheduler, command-program, transient-allocator, JIT bundle).
  **Still draft, not merged.** Author: @stellaraccident (Stella Laurenzo) with
  AMD co-authors (Suderman, Vanik, Vasishta). AI-assisted (Codex) disclosed.
- Claims: 30–50% prefill uplift, parity→+15% decode; ~32 MiB added; ~1-min
  build on Strix Halo; static Windows build path (26.1 MiB llama.exe) planned.
- Maintainer skepticism ([@0cc4m](https://github.com/ggml-org/llama.cpp/pull/27218)):
  "doesn't currently qualify as a GGML backend... fused Qwen3-specific hardcoded
  compute." AMD's answer: generic backend exists on private branches; they lead
  with the performant end state. Two possible landing paths offered.
- **"Replace llama.cpp" evidence**: the docs' own
  [GGML/llama.cpp oracle page](https://rocm.github.io/hrx-system/loom/workflows/oracles/ggml-llama-cpp/)
  is a porting playbook — GGUF/GGML are "physical contracts", llama.cpp is the
  oracle to port *from*; Loom README roadmap: kernels → model execution →
  serving → LoRA/training → "whole host programs". Replacement is the documented
  trajectory; llama.cpp compat is the entry ramp, not the destination.

## 6. Loom docs crawl (rocm.github.io/hrx-system/loom/)

- Full MkDocs site, "One program, specialized all the way down". Dialect op
  counts: scalar 101, vector 151, kernel 54, index 24, check 22, command 8,
  template 6, + ireevm/amdgpu/spirv/x86/wasm target records.
- `loomc` C API: stable embedding surface (AOT/JIT, caller-owned artifact
  caches, autotune) with no GPU runtime dependency; target packages add
  AMDGPU/SPIR-V emission.
- Authoring corpus (`loom/src/loom/test/corpus/authoring/`): hand-written model
  kernels — `ffn_gate_up_swiglu_q6q8.loom` (q6_K × q8_1 gate/up SwiGLU fusion
  with `template.def` provider boundaries), `mlp_down_projection_residual_bf16.loom`,
  `indexed_row_gather_f32.loom`, plus a `hip/` cookbook mapping HIP/CUDA idioms
  to Loom spellings. Tooling: `loom-check`, `loom-compile`, `loom-link`,
  `iree-test-loom`, `iree-benchmark-loom`, `loom-compile-report`
  (show/suggest/diff, evidence tiers silicon-calibrated vs experimental).
- Agent-driven kernel development workflow is explicitly designed for
  AI-agent kernel search with audit trails ("The agent is not an oracle").

## 7. AMD fork activity (AMD-Ecosystem/llama.cpp)

- 9 HRX branches: `users/stella/hrx-rfc-v1`, `hrx-v2` (179 commits ahead, 1997
  behind upstream — long-running private effort), `hrx-integration`,
  `hrx-graph-develop-v2` (updated 2026-08-28 22:57 — "generalize kernel support
  and add llama support"), `hrx-kernel-lib-v1`, `hrx-diffbase`, etc.
- `hrx-v2` commit subjects show the trajectory: "hrx2: remove temporary HIP
  bridge kernels", "hrx2: enable rms norm mul fusion by default", q4/q5/q6/q8
  prompt-route enablements.
- Staging releases b2 (2026-06-23) → b59 (2026-08-22): ~13 releases in 2
  months, nightly cadence. Binary + ROCm sysdeps bundled; no ROCm install
  needed.

## 8. Risks / open questions

1. **Provenance**: binary comes from AMD *staging*, pin `hrx-b59` is exact and
   checksummed, but if AMD rotates tags or the RFC path changes, the pin
   breaks. Same exposure as our `ds4` pin — known tradeoff.
2. **Experimental flag**: `experimental: true`, `selectable_backend: false`,
   `dynamic_models: false` — not surfaced by default in the UI; reachable via
   the model's recipe.
3. **Windows excluded** in lemonade (`validate_supported_build_host` throws);
   AMD's own Windows HRX story is still in the "static llama.exe" plan.
4. **llama.cpp maintainer resistance** could stall upstreaming; AMD is clearly
   all-in regardless (9 branches, 179 commits, daily staging, hrx-demos on
   PyPI running Ideogram-4 without llama.cpp).
5. **Our fork decision**: `third_party/llama.cpp` (bong-water-water-bong fork,
   10454 commits, no HRX work) sits on upstream master — when ggml-hrx lands
   upstream, HIP/Vulkan become the legacy AMD path. Decide whether to track HRX.

## 9. Timeline

| Date | Event |
|---|---|
| 2026-03-27 | ROCm/hrx-system created |
| 2026-06-23 | First staging release `hrx-b2` |
| 2026-08-13 | RFC discussion #27219 + draft PR #27218 |
| 2026-08-22 | `hrx-b59` staging release (current lemonade pin) |
| 2026-08-26 | 1bit-MONSTER re-vendors lemonade v11.8.0 (`e1b31683`) — PR #1889 |
| 2026-08-28 | Lemonade merges hrx backend `7953d7f` (PR #3374) |
| 2026-08-28 | hrx-graph-develop-v2 branch: "generalize kernel support + llama support" |
| next | Our re-vendor e1b31683 → 7953d7f picks up HRX automatically |

## D2 state-format round-trip — GATE PASSED (2026-09-02)

hybrid-prefill-decode.md §5.1 executed. The "binary question" is answered:
the vendored llama.cpp (third_party/llama.cpp 0.1.0, 4df29be4f, LLAMA_SESSION_VERSION 9)
state blob round-trips into the hrx-v2 fork (cdb8110 round-25i, LLAMA_SESSION_VERSION 9).

Harnesses: harnesses/state_exp.cpp (A: vendored, exports via llama_state_get_data)
and harnesses/state_imp.cpp (B: fork, imports via llama_state_set_data).

- Qwen3-0.6B-Q4_K_M: 5-token prompt + 8 gen → 1,491,797 B blob → import OK →
  continuation token-identical (A == B, byte-for-byte).
- Qwen3-Coder-30B-A3B-Instruct-Q4_K_M (the §1 target): 49-token prompt + 8 gen →
  5,605,192 B blob → import OK → continuation token-identical:
  " crashing waves—\nGuiding ships home<|im_end".

Notes:
- State size is cell-count-dependent (empty ctx reports ~17 B); do NOT size-check
  against a fresh ctx — import directly (llama-cli --session path).
- Both contexts must match on n_ctx (512 here) and model; rope/cache-type defaults
  line up between the two builds (both 2026-era, session v9).
- A-side ran CPU (n_gpu_layers=0) — the format gate is backend-independent.
  Next: GGML_HIP=ON build of the vendored fork for the actual HIP prefill lane.

## Hybrid direction reset — "HRX is supposed to eliminate that need" (2026-09-02)

After the D2 gate, the GGML_HIP build of the vendored llama.cpp (the D2 prefill
lane) was stopped: the strategic steer is that HRX2 prefill catching up should
eliminate the need for an iGPU HIP prefill lane + state transfer, not that the
engine adds a second llama.cpp HIP stack. Engine context: `backend_hip.cpp`
(Zaya) and `backend_hip_1bp.cpp` are standalone HIP engines with custom KV —
no llama.cpp state path; the only llama.cpp-context GPU backend is ggml-vulkan
(spec-decode demo). hrx_inprocess.cpp (G2) dlsym's the bundle with a vtable
that has NO llama_state_* entries yet (the D2 shim surface if ever needed).

Measured on the fork (cdb8110, llama-bench -t 16 -ngl 99, sequential):
- 30B-A3B Q4_K pp32: 83–105 t/s (contention variance). GGML_HRX2_TRACE_JSONL
  shows ALL 576 attention MUL_MATs dispatch to HRX2 via
  mul_mat_f16_f32_batched_attention_wg256 (rows 256/128, cols 32, k 128/256).
  NO MUL_MAT_ID dispatch: MoE expert mms run CPU (Q4_K claim reverted, 25f/g).
  The prefill gap vs HIP (1227–1313) is the MoE mms being CPU-side.
- q4nx-converted GGUF (qwen25-3b-q4nx, qwen3coder-30b-q4nx): llama-bench
  labels "all F32 (guessed)" (cosmetic — no ftype case for GGML_TYPE_Q4NX=42),
  pp32 77 t/s (3B) / 42.9 (30B). The q4nx kernels compile+cache
  (provider_compile: mul_mat_q4nx_fused_tbl_tiled, mul_mat_q4nx_fused_f32_r16)
  but the dispatch trace shows only the attention batched route — the Q4NX
  FFN/MoE mms never dispatch to HRX2. Suspected blocker: q4nx weight tensors
  land on CPU buft (same "cannot be used with preferred buffer type
  HRX20_host" path seen for Q4_K), so MUL_MAT_Q4NX never reaches the NPU.
  (Round-25d's 121.5 t/s dense prefill was measured via the ws12 dump
  harnesses, not llama-bench on the converted GGUF.)

Next lever (fork-side, "eliminate the need"): get Q4NX weights into HRX2
buffers so the compiled fused_tbl_tiled kernel dispatches at prefill — verify
with the ws12 dump32_prefill harness (round-25h methodology), not llama-bench.

## D2 gate + round 25j verification (2026-09-02, harnesses/state_exp|state_imp)

- **D2 state-format gate (hybrid-prefill-decode.md §5.1) PASSED**: the vendored
  llama.cpp (third_party/llama.cpp 0.1.0, LLAMA_SESSION_VERSION 9) state blob
  round-trips into the hrx-v2 fork (cdb8110, session 9). Qwen3-0.6B (5-token
  prompt + 8 gen → 1,491,797 B blob) and Qwen3-Coder-30B-A3B (49-token prompt
  + 8 gen → 5,605,192 B blob): import OK, continuation byte-identical. Note:
  state size is cell-count-dependent — an empty ctx reports ~17 B; import
  directly (llama-cli --session path), don't size-check a fresh ctx.
  Strategic reset after the gate: HRX prefill catching up should eliminate the
  need for the iGPU HIP prefill lane + transfer machinery (GGML_HIP build of
  the vendored llama.cpp stopped). Engine HIP lanes (Zaya/1BP) have custom KV
  and no llama.cpp state path; hrx_inprocess.cpp vtable has no llama_state_*.
- **Round 25j verified (fork 4518abd)**: 30B-A3B Q4_K pp32 **133.16 ± 3.33 t/s**
  (cdb8110 was 83–105; the attention-mms-on-NPU + prefill coherency-tax fix
  lands ~+30-60%). 0.6B pp32 502.8 (contention). Generation correct
  ("The capital of France is Paris.").
- **Q4NX routing corrected**: on the q4nx-converted GGUFs the earlier
  "FFN mms CPU-side" reading was wrong — the q4nx kernels DO dispatch.
  Load log: weights land on HRX20_w (device-local, 1839 MiB on the 3B);
  "cannot be used with preferred buffer type HRX20_host" refers only to the
  host-visible buft preference. mul_mat_q4nx_fused_tbl_tiled JIT-compiles and
  runs for the prefill shapes (r2048/r11008 × c32 × k2048/k11008), plus the
  c1 lm_head tail via mul_mat_q4nx_fused_f32_r16. "all F32 (guessed)" is
  cosmetic (no ftype case for GGML_TYPE_Q4NX=42); the 253 q4nx tensors load
  as Q4NX and hit the NPU. ngl=0 fails to load (ggml-cpu claims no Q4NX ops)
  — the NPU is the only execution path for Q4NX models, as intended.

## 30B Q4NX conversion is BROKEN (experts scrambled) — fork MoE path exonerated (2026-09-02)

Chasing the 30B q4nx prefill (46.9 t/s, corr 0.55 to Q4_K reference): the
q4nx-converted MoE models are unusable — the EXPERT (3-D) tensors are
scrambled. Verified with the f32-twin dequant (make-q4nx-f32-twin.py, the
proven Q4NX dequant) vs the Q4_K source dequant (make-q4nx-model.py):

- DENSE attn_q (blk.0.attn_q): corr 0.99197 — conversion correct.
- MoE blk.0.ffn_gate_exps expert 0: corr **0.0025** vs q4_K expert 0; vs
  q4_K experts 0..7 all |corr| < 0.006. NOT an expert-order permutation.
- Sorted-value corr 0.99927 (same value multiset, same ranges) — the data
  is present but the tile/position mapping is wrong (global scramble across
  the 3-D expert tensor). Suspect: q4nx_to_gguf.py / upstream .q4nx expert
  tile layout (axis order or tile-grid mapping for ne=[8192, tpe, n_expert]).

Consequences:
- qwen3coder-30b-q4nx.gguf (and by extension qwen3next-80b-q4nx) cannot be
  used for HRX2 prefill/decode validation — the wrong logits (top5 disjoint
  vs Q4_K CPU, corr 0.55) are the FILE, not the fork.
- The fork's MUL_MAT_ID_Q4NX path (fused_tbl_tiled compiled for the 30B
  shapes r768-cN-k2048-nt32; kill-switches change nothing) is unproven until
  a correctly-converted MoE Q4NX model exists. The 3B dense Q4NX path IS
  proven correct (top1 matches CPU).
- Fix target: the converter's 3-D expert tiling. Validation gate: expert
  dequant corr > 0.99 vs source (like the dense 0.99197) before any MoE
  Q4NX bench.
