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

## 30B Q4NX MoE prefill — NO fork bug; degenerate-input artifact (FINAL) (2026-09-02)

The "wrong prefill" was a harness artifact, not a fork bug. The dump32_prefill
harness feeds 32 IDENTICAL tokens (all id=2); with all-identical hidden states
the final-token logits are quantization-noise-sensitive and corr collapses for
ANY backend pair. Proven: Qwen3-Coder-30B Q4_K, HRX2(ngl=99) vs CPU(ngl=0),
SAME weights, both correct backends, degenerate input -> corr 0.33. The q4nx
0.55/0.19 readings were the same effect, not a grouped-kernel defect.

Correct behavior (real inputs):
- 17-token real prompt: q4nx HRX2 vs Q4_K CPU top1 576 == 576, corr 0.9584
  (quant-rounding ceiling; the f32-twin target for corr 1.0).
- 32 varied tokens (1000..1031): top1 321 == 321, corr 0.9771.
- Decode (real prompts): "Gravity is the natural force..." correct, matching
  Q4_K quality.
- Routing verified correct throughout: grouped c32 dispatches (1128 events,
  tile_base = e*tpe, per-expert src1 detection, dst/src1 tables exact),
  ids identical across tokens (CORRECT for identical inputs), kernel dequant
  indexing identical to the proven r16 decode kernel.

Surgery trail (all reverted; fork clean at 4518abd):
- Suspected src1 binding under-declare (k*ntokens vs k*src1_cols_count for
  per-expert src1 in the grouped fused/tbl bindings) — real but cosmetic
  (bindings not enforced; decode with the same under-declare works). Worth
  fixing upstream if the runtime ever enforces binding ranges.
- Suspected table/async race (b_tbl shared scratch across groups) — a
  flush/wait before dispatch changed the result (0.55->0.19) proving the
  async ordering affects output, but BOTH were degenerate artifacts; with
  real inputs the result is correct regardless.
- Debug instrumentation (env-gated HRX2_DEBUG_IDS) removed.


## 30B q4nx prefill is SYNC-BOUND, not compute-bound (2026-09-02)

The 45 t/s (pp32) vs Q4_K's 133 t/s gap is NOT kernel throughput — it's the
per-MUL_MAT_ID ids-sync serializing the NPU pipeline.

GGML_HRX2_TRACE_JSONL on the 30B q4nx pp32 (clean 4518abd):
- dispatch execution: ~59 ms per eval (32 tokens -> ~540 t/s ceiling).
- stream synchronize: 1186 calls, 3812 ms total, avg 3.2 ms each.
- cold eval: ~611 ms (JIT compiles included); steady-state graph-cached
  replay measures 0 ms (cache artifact, not a real number).

Mechanism: dispatch_mul_mat_id_q4nx copies the ids tensor to the host-visible
q4nx_ids scratch, then calls hrx_stream_synchronize() before the host groups
(144 ops/eval x 48 layers x gate/up/down). Every sync drains ALL pending
dispatches -> the async pipeline collapses per MoE mm. The Q4_K MoE runs on
CPU (round 25f/g) and never hits this sync -> 133 t/s.

Lever (est. 45 -> ~300-500 t/s on the 30B q4nx): eliminate the per-op full
sync. Options: (a) two-phase MUL_MAT_ID eval — copy all ids up-front, ONE
sync per graph, then dispatch all groups; (b) read the ids directly from the
producer (CPU argsort -> host tensor) without the device copy+sync round-trip
when src2 is host-backed; (c) batch the syncs per ubatch. The ids are tiny
(256 i32/op); the sync cost is the drain, not the copy.

Also noted: the kernel itself is correct and fast (59 ms of real compute for
all 48 layers of MoE+attention+pointwise on the NPU). The grouped
fused_tbl_tiled path at c32 is NOT the problem (previous finding corrected).

## Round 25k — MUL_MAT_ID ids direct-read (fork 5e4f14a)

The remaining prefill bottleneck after the sync-bound finding: the ROUTER.
Per layer: scores MUL_MAT (NPU) -> softmax (NPU) -> argsort (CPU for the
[128, 32] prefill shape — HRX2 only claims n128_r1) -> the CPU reads the NPU
scores -> stream synchronize. 313 of the 322 remaining syncs follow MUL ops
(avg 14 ms each — each drains the accumulated submit batch).

Fix landed (round 25k): the MUL_MAT_ID ids tensor is CPU-written into the
host-coherent GTT (round-25i zero-copy); the dispatch copied it to a host
scratch + full stream synchronize before grouping (144x/eval). Now reads
src2->data directly when host-buft-backed (ids_direct) — zero copy, zero
sync. Fallback copy+sync path retained for device-produced ids.

Verified: 30B q4nx pp32 logits unchanged (top1 576, corr 0.9584 vs Q4_K
CPU); sync calls 1186 -> 322 on the pp32 trace. Speed: 45 -> 48 t/s
(marginal — the router argsort syncs still dominate).

Remaining lever: the router. Options: (a) HRX2 argsort for [128, nrows>1]
(removes the CPU round-trip; the ids still need a host read for grouping —
but host-coherent device-written ids could be read without a full drain if
the sync is scoped to the argsort's completion); (b) flush the submit batch
before the router sync so each drain is short; (c) on-device grouping (big).
The dispatch compute is only ~59 ms/eval — the sync structure is the whole
gap to the ~500 t/s ceiling.

## 30B q4nx prefill is COMPUTE-bound (correction to the sync-bound claim) (2026-09-02)

HRX2_OPTIME_SYNC (per-op execution-inclusive) shows MUL_MAT_Q4NX at ~6.9-7.1 ms
per op — the syncs were waiting on REAL NPU execution, not draining idle
batches. The "59 ms compute" reading was dispatch-SUBMIT time, not execution.

Fused-path check (30B q4nx pp32, llama-bench):
- default: 48.2 t/s (fused_tbl_tiled engaged — the fast path IS used)
- HRX2_NO_FUSED_TBL=1: 43.3 (fallback dequant+b_w+mm ~10% slower)
- HRX2_NO_FUSED_PAIR=1: 41.0
- GGML_HRX2_DISABLE_SUBMIT_BATCHING: 44.1 (no win — the syncs aren't
  batch-drain dominated)

Root cause of the ~7 ms/mm: the fused_tbl_tiled c32 kernel dequants ONE
scattered byte per (row, k) — lane reads packed[lane*2048 + (i%256)*8 +
byte_idx] at stride 8 (the round-25e "~43 GB/s effective" bandwidth wall).
The r16 DECODE kernel fixed this (16 rows/workgroup, 8 CONTIGUOUS packed
bytes per k-step) — decode is bandwidth-fine; the c32 PREFILL kernel never
got the contiguous-read restructure.

Lever: restructure mul_mat_q4nx_fused_tbl_tiled's workgroup to read 8
contiguous packed bytes per k-step (16 rows x 8 cols per workgroup, 16
k-block lanes/row like r16), keeping the table-scatter. Estimated 1.5-2x on
the mms (round-25e estimate) -> 30B q4nx pp32 48 -> ~80-100 t/s. The
48-layer c32 MoE path is the wall; the kernel is otherwise correct.

## Round 25l — r16-tbl kernel experiment: correct but NOT faster (reverted) (2026-09-02)

Built hrx2_mul_mat_q4nx_fused_tbl_tiled_r16 (16 rows x 8 table-cols per
workgroup, 256 lanes = 16 rows x 16 k-block lanes, r16-style coalesced
packed reads via workgroup scratch + barrier reduction) + route
(mul_mat_q4nx_fused_tbl_tiled_r16) + dispatch preference when rows%16==0.
Compiled clean through loom-link (SSA name collisions fixed, reduction
generated programmatically).

Result (30B q4nx pp32, llama-bench): 43.5 t/s vs base kernel 48.2 — the
r16-tbl restructure is CORRECT (corr 0.9584 unchanged) but SLOWER. The
coalesced packed reads did NOT unlock the expected 1.5-2x (round-25e
estimate). The ~7 ms/mm is NOT the scattered 1-byte packed-read pattern.

Reverted (fork clean at 5e4f14a / round 25k): dispatch preference + kernel +
route removed; baseline 48.3 t/s + corr 0.9584 restored.

Open question: what IS the ~7 ms/mm? Candidates: (a) the per-group dequant
weight traffic (192 tiles x 983 KB/expert read per group dispatch at the
NPU's ~43-49 GB/s effective = ~20 ms if BW-bound — the fused path should
avoid the b_w round trip but still reads 983 KB/expert inline); (b) the
scale/zp scattered reads; (c) the 1128 per-group dispatches' launch
overhead; (d) the router syncs (the actual remaining wall, ~313 x 14 ms).
The compute-vs-sync split needs a cleaner measurement (the HRX2_OPTIME_SYNC
7 ms/mm includes the drain).
