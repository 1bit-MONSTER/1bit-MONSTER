# ws12-hrx-loom — HRX / Loom platform transition watch

**Status:** ✅ vendored + validated (2026-08-29) — lemonade re-vendored to
`7953d7f6` (HRX commit) and the `llamacpp-hrx` backend now registers and runs
a real model in `onebin` on gfx1151. See "Validation" below.
**Papers:** none (platform-intelligence workstream — sources are the
lemonade/llama.cpp/ROCm repos and https://rocm.github.io/hrx-system/loom/).
**Owner:** bong-water-water-bong

## Goal

Track and de-risk AMD's HRX ("Hip Runtime Extended") + Loom compiler stack as it
migrates into lemonade, and decide how 1bit-MONSTER's vendored lemonade core and
`third_party/llama.cpp` fork should ride it. HRX is AMD's client-native
replacement for the HIP/Vulkan execution path in llama.cpp — and the docs' own
roadmap ("whole host programs", serving/LoRA/training) points at eventual
llama.cpp replacement. We inherit the leading edge of that transition on the
next lemonade re-vendor, so we must know what we're pulling in, what it costs,
and what breaks if the upstream RFC stalls.

## Why it matters (thesis)

1. Lemonade merged the `hrx` recipe **before** the llama.cpp RFC
   ([PR #27218](https://github.com/ggml-org/llama.cpp/pull/27218), still draft)
   landed. Our re-vendor past `7953d7f` pulls it in automatically — no CMake
   conflict with our embeddability patch (verified), backends are explicitly
   listed (`LEMON_BACKENDS` gains `"llamacpp-hrx|hrx"`).
2. HRX is IREE-derived (Ben Vanik; `# Copyright 2026 The IREE Authors`), a real
   compiler with hand-written model kernels (q6/q8 SwiGLU fusion in
   `loom/src/loom/test/corpus/authoring/`).
3. The `hrx-b59` binary is checksum-pinned in `backend_versions.json`, works on
   our gfx1151 (Strix Halo) today — live-verified on this machine — but is
   built from fork commit `f749e1390` (NOT on upstream master).
4. The official llama.cpp-oracle doc is a **porting playbook**: GGUF/GGML are
   "physical contracts", llama.cpp is the oracle to port *from*. That's the
   replacement endgame, documented.

## Tasks

### P0 (do now — next re-vendor)
- [x] Re-vendor lemonade `e1b31683` → `7953d7f` (9 commits: FLM 1.0.3, bench
      checkpoints, log rotation, MTP fix, FHS cache recovery #3393, sdcpp CI,
      gpu-hang test, hrx backend) — re-apply embeddability patch per
      `third_party/lemonade/UPSTREAM.md`. **DONE 2026-08-29 (HRX backend
      landed; see compliance note in TRACKING.md re: upstream provenance).**
- [x] Verify `hrx` backend registers in `onebin` (gfx1151 build, EMBED_LEMONADE=ON)
      and that `build/1bit unified --lemonade` exposes
      `Qwen3-30B-A3B-Instruct-2507-HRX` (18.6 GB, chat-only, `suggested: true`).
      **DONE 2026-08-29.**
- [x] Smoke-test the recipe end-to-end on gfx1151: `llama-server --device HRX0`
      boots, `/health` ok, single chat completion on the pinned model (or record
      the fail-closed error for a non-qualified model — that is expected
      behavior, not a bug). **DONE 2026-08-29 ("Paris", hrx-b59 spawned).**

### P1 (next)
- [x] Track llama.cpp RFC #27218 / discussion #27219 status; record when
      ggml-hrx moves from AMD staging (`ROCm/ggml-staging-automation`) to
      upstream releases — that changes the binary provenance story.
      **PR #27218 "ggml-hrx: add AMD ROCm HRX native ggml backend" exists
      upstream; still draft (0 HRX commits on master). GET_ROWS remains the gap.
      Re-benchmark when it lands.**
- [x] **GET_ROWS on gfx1151 — FIXED locally (2026-08-30).** Root cause: the
      shipped bundle's ggml-hrx kernel catalog is compiled only for
      `GGML_HRX_AMDGPU_TARGETS=gfx1100` (AMD CI default); on gfx1151 the
      get_rows lookup fails -> fail-closed. Fix: build the same stack
      (llama.cpp hrx-v2 + pinned loom e8275fb) with
      `GGML_HRX_AMDGPU_TARGETS=gfx1151` — reproducible via
      `scripts/build-hrx-gfx1151.sh`, served via `scripts/run-hrx-gfx1151.sh`.
      Built artifacts persist at `~/hrx-gfx1151/` (llama-build + hrx-runtime;
      survives reboot). Verified: a 2249-token prompt that
      hard-fails on hrx-b66 completes with 0 GET_ROWS errors. Perf trade-off:
      ~26 tok/s warm decode vs b66's ~67 (untuned gfx1151 kernels), but large
      prompts now work at all. **The `hrx-v2` *ggml-hrx2* backend is NOT
      buildable against any public loom** (its loom-jit includes
      `loomc/target/amdgpu.h`, which exists in no released loom) — v1 is the
      path.
- [ ] Audit `hrx-v2`/`hrx-integration` branches (AMD-Ecosystem/llama.cpp fork,
      179 commits ahead) for "remove HIP bridge kernels" work; decide if our
      `third_party/llama.cpp` fork should track HRX or stay on HIP/Vulkan.
- [x] Benchmark HRX vs our HIP baseline on gfx1151 once the qualified model is
      runnable (RFC claims 30–50% prefill uplift, parity→+15% decode — verify
      with honesty tags). **DONE 2026-08-29 — RFC claim NOT reproduced: HIP wins
      large prefill (1227–1313 tok/s); HRX fails closed on GET_ROWS; HRX wins
      warm decode (~175 vs ~70). See `BENCHMARK.md`.**
- [x] **HRX is the live lane (2026-08-30).** The engine's `hrx_gpu` backend
      (first in the GGUF route) now serves the custom gfx1151 build — verified
      `1bit unified --model Qwen3-0.6B ...` → `Backend: hrx_gpu` → "Paris".
      Built artifacts persist at `~/hrx-gfx1151/` (RUNPATH-fixed, no env vars,
      survives reboot): `scripts/build-hrx-gfx1151.sh` + `scripts/run-hrx-gfx1151.sh`.
- [x] **Native Q4NX port: first Loom kernel (2026-08-30).**
      `research/ws12-hrx-loom/loom-kernels/q4nx_dequant_f32.loom` — a Loom
      kernel that dequantizes a real Q4NX tile (signed two's-complement int4 +
      BF16 row-major scales, the exact `dequant_q4nx.cpp` math) with
      `scalar.extui/sitofp/bitcast`, `scf.if` selection, and 2D views.
      **COMPILES to `.loombc` via the pinned loom-link.** This proves 1BP's
      format can ride the Loom/HXR compiler surface — the strategic port is
      real, not speculative.
- [ ] **Runtime wiring of the Q4NX kernel is BLOCKED on AMD's unreleased
      loom**: the ggml-hrx2 backend's loom-jit includes `loomc/target/amdgpu.h`
      (types `loomc_amdgpu_emit_options_t`, `loomc_amdgpu_profile_options_t`)
      which exist in NO public loom (not e8275fb, not hrx-system main). Track
      when AMD ships it; the kernel + route format are ready to plug in.

### P2 (if the bet pays off)
- [ ] Evaluate Loom (`loomc` C API, `iree-test-loom`, `iree-benchmark-loom`) as
      an authoring surface for our own NPU/GPU kernels — the same stack AMD is
      using for HRX could host 1bit-specific fusions.
- [ ] If HRX goes upstream-default in llama.cpp, plan the fork migration path
      (kernel vocab, gfx targets, Windows story).

## Validation

- Re-vendor builds clean on gfx1151, `onebin` links, `hrx` backend listed in
  `--help`/registry — harness: `cmake --build build --target onebin`. ✅
  2026-08-29: `onebin` built with the vendored HRX backend
  (`lemonade-server-core` compiles `backends/hrx/`).
- `Qwen3-30B-A3B-Instruct-2507-HRX` chat completion returns tokens (validated)
  or fails closed with `unsupported HRX node N: <op>` (correct per contract).
  ✅ 2026-08-29: served via `1bit unified --lemonade` — "Paris" at 130.8 tok/s
  prompt / 35.2 tok/s gen on `HRX0` (gfx1151).
- Checksum of downloaded `hrx-b59` asset == `sha256:d2fe01...` from
  `backend_versions.json` (validated on 2026-08-28). ✅

## Notes

- Live test 2026-08-28: `hrx-b59` tarball is fully self-contained (libhrx 2 MB,
  libloomc 13 MB, libggml-hrx, libvulkan, rocm_sysdeps tree) — no ROCm install
  needed on target. `llama-cli --list-devices` shows `HRX0: AMD Radeon 8060S
  (gfx1151), 114688 MiB`. Non-qualified model fails closed:
  `E graph_compute: unsupported HRX node 0: GET_ROWS`.
- Full evidence dump: `FINDINGS.md`.

## UPDATE 2026-08-30 (round 2): amdgpu.h blocker DISSOLVED

The missing `loomc/target/amdgpu.h` landed in **ROCm/hrx main @ 4890cb5d7**
(after our pinned e8275fb). The new loom exports
`loomc_target_environment_create_amdgpu` from `libloomc.so` and installs the
header. ggml-hrx2 now configures against it (shim: `loom::binding::c::loomc`
→ `loomc::loomc`).

**Our `q4nx_dequant_f32.loom` compiles with the new loom** — the Q4NX port is
fully unblocked at the kernel level.

Remaining: the fork's ~30 legacy mul_mat kernels use the OLD loom dialect
(`#dense` views, `func.template`, `{...}` attrs on barrier/alloca) which the
new loom rejects. I migrated the mechanics (34× `#dense` strip, 2×
func.template→template.decl/def, 15× alloca, 12× barrier) but the new loom's
stricter verifier flags barrier-in-lane-region in mul_mat_q8_0. This is AMD's
legacy-kernel porting debt — a bounded but multi-hour migration, OR wait for
AMD to refresh the fork's kernels. Our kernel (no templates, no barriers) is
the proof and it compiles.
