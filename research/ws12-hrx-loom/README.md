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
- [ ] Re-vendor lemonade `e1b31683` → `7953d7f` (9 commits: FLM 1.0.3, bench
      checkpoints, log rotation, MTP fix, FHS cache recovery #3393, sdcpp CI,
      gpu-hang test, hrx backend) — re-apply embeddability patch per
      `third_party/lemonade/UPSTREAM.md`.
- [ ] Verify `hrx` backend registers in `onebin` (gfx1151 build, EMBED_LEMONADE=ON)
      and that `build/1bit unified --lemonade` exposes
      `Qwen3-30B-A3B-Instruct-2507-HRX` (18.6 GB, chat-only, `suggested: true`).
- [ ] Smoke-test the recipe end-to-end on gfx1151: `llama-server --device HRX0`
      boots, `/health` ok, single chat completion on the pinned model (or record
      the fail-closed error for a non-qualified model — that is expected
      behavior, not a bug).

### P1 (next)
- [ ] Track llama.cpp RFC #27218 / discussion #27219 status; record when
      ggml-hrx moves from AMD staging (`ROCm/ggml-staging-automation`) to
      upstream releases — that changes the binary provenance story.
- [ ] Audit `hrx-v2`/`hrx-integration` branches (AMD-Ecosystem/llama.cpp fork,
      179 commits ahead) for "remove HIP bridge kernels" work; decide if our
      `third_party/llama.cpp` fork should track HRX or stay on HIP/Vulkan.
- [ ] Benchmark HRX vs our HIP baseline on gfx1151 once the qualified model is
      runnable (RFC claims 30–50% prefill uplift, parity→+15% decode — verify
      with honesty tags).

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
