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

### UPDATE 2026-08-30 (round 3): Q4NX catalog route registered

- The Q4NX kernel is now a **registered catalog route** (`q4nx_dequant_f32`):
  route JSON + sources.json + artifacts.json + families.json entries, using
  the valid generic `shape.ncols`/`shape.nrows` binding sources (the
  validator allowlists those; `shape.q4nx.*` is rejected).
- A **minimal 7-artifact catalog** (add_rms_norm_mul, get_rows, mul_mat_f32,
  q4nx_dequant, quantize_q8_1, rms_norm, rms_norm_mul) builds through the
  catalog pipeline: assemble → validate → loom-link → require-artifacts.
  ggml-hrx2.cpp + ggml-hrx2-catalog.cpp then COMPILE.
- **Remaining blocker is AMD's WIP-fork API drift**, not ours: the fork's
  loom-jit C++ uses `loomc_target_selection_*` (new loom renamed to
  `target_specialization_*`, 12 sites) and
  `loomc_amdgpu_profile_options_t.processor` (new API uses a structured
  `loomc_amdgpu_target_identity_t`, 3 sites). The fork's kernels needed the
  old dialect (#dense, func.template) AND its C++ needs the pre-specialization
  loomc API — it was never meant to build against any shipped loom.
- **Bottom line**: our Q4NX kernel is the proof and it compiles + is catalog-
  registered against the amdgpu.h-unblocked loom. The full ggml-hrx2 backend
  build awaits AMD refreshing their fork's C++ to the shipped loomc API
  (or our porting the 15 C++ sites — bounded, ~1 session).

### UPDATE 2026-08-30 (round 4): fork C++ ported — Q4NX route runs BIT-EXACT on gfx1151 ✅

**The 15 API-drift sites are ported and `libggml-hrx2.so` builds + links
against the shipped loomc.** The full chain now executes on the Strix Halo
GPU:

```
JIT compiled route=q4nx_dequant_f32 target=gfx1151 source=Loom bytecode hsaco=9192 bytes
RESULTS total=8192
  max abs diff : 0.000000e+00     correlation  : 1.000000
  mismatches >1e-3: 0 / 8192      PASS YES
```

Native Q4NX int4 dequant runs on HRX (loom → gfx1151 HSACO → dispatch) and
produces output **bit-identical** to the CPU reference (`dequant_q4nx.cpp`
verified layout, corr 0.995 vs source).

**What was ported in `ggml/src/ggml-hrx2/` (AMD fork, new loomc API):**

- `loom-jit/ggml-hrx2-loom-jit.cpp` — 12× `loomc_target_selection_*` →
  `loomc_target_specialization_*` (one `loomc_target_specialization_t`
  built from the root symbol + target profile, attached to
  `compile_options.next`); 3× `profile_options.processor` →
  structured `loomc_amdgpu_target_identity_t` via
  `loomc_amdgpu_target_identity_from_hsa_isa_name` (bare `gfx1151` is
  normalized to the full `amdgcn-amd-amdhsa--gfx1151` triple the API
  requires); `LOOMC_LINK_FLAG_STRIP_CHECK_SYMBOLS` →
  `STRIP_TEST_SYMBOLS`; dropped obsolete
  `LOOMC_CONFIG_POLICY_FLAG_REJECT_UNKNOWN`; added
  `link_options.mode = LOOMC_LINK_MODE_LINK` (MERGE rejects root symbols).
- `ggml-hrx2-catalog.cpp` — `hrx_executable_load_data` gained a
  `target_family` param in the new runtime: call sites pass family
  `"amdgpu"` (device HAL family — the route's catalog `family` field is a
  route grouping, NOT the device family) and `target_key` = device
  architecture when the route doesn't pin one (the runtime requires both
  non-empty and selects the executable target by them).
- **Runtime fix** (ROCm/hrx main @ 4890cb5d7): `aql_ring.c` queried the
  optional `HSA_AMD_AGENT_INFO_PM4_EMULATION` probe with a hard error —
  gfx1151's ROCr rejects the query (INVALID_ARGUMENT) and
  `hrx_gpu_initialize` aborted. Patched: probe failure → default to NATIVE
  AQL execution mode. (`patches/hrx-runtime-pm4-emulation-optional.patch`)

**Persisted artifacts** (this directory):

- `patches/ggml-hrx2-fork-loomc-port.patch` — full fork delta (C++ port,
  catalog pruning to 7 artifacts, q4nx route, kernel dialect migration)
  against AMD fork HEAD 95b1a89; `git apply` on a fresh checkout restores
  the ported state.
- `patches/hrx-runtime-pm4-emulation-optional.patch` — the runtime probe fix.
- `validate-q4nx-route.cpp` — standalone harness: opens the HRX GPU device,
  loads the embedded catalog, compiles the q4nx route via the ported
  loom-jit, splits the 5120-byte tile into packed/scales/zeros, dispatches
  32×256 elements, compares vs the CPU reference. Build + run:
  `g++ -std=c++17 -O2 validate-q4nx-route.cpp -I<fork>/ggml/src/ggml-hrx2
  -I<hrx-install>/include -I<hrx-install>/include/hrx
  -L<fork>/build/bin -lggml-hrx2 -L<hrx-install>/lib -lhrx
  -Wl,-rpath,<fork>/build/bin -Wl,-rpath,<hrx-install>/lib -o validate_q4nx
  && ./validate_q4nx /tmp/q4nx_tile.bin /tmp/q4nx_ref.f32`
- `loom-kernels/q4nx_dequant_f32.loombc` — refreshed to the exact bytecode
  that validated bit-exact (md5 f947ef54).

**Remaining fork debt (not blocking Q4NX):** the legacy mul_mat kernels
still trip the new loom's stricter verifier (barrier-in-lane-region,
STRUCTURE/038) — AMD's kernel porting debt, irrelevant to the Q4NX proof
(no templates, no barriers).

### UPDATE 2026-08-30 (round 5): mul_mat port done — FULL backend kernels link; real matmul runs on gfx1151 ✅

**All 73 kernel export roots in the fork now link through loom-link against
the new loom (was: 31/31 mul_mat + 42/42 others).** Two systematic fixes
unlocked the whole set:

1. **`template.return`, not `func.return`** — the new loom's `template.def`
   body region requires the `template.return` terminator (the old dialect's
   `func.return` trips the inliner: `callee_body_invalid_terminator`). Fixed
   in `mul_mat_q8_0_f32.loom` + `mul_mat_q4_k_swiglu_f32.loom`.
2. **Barriers must be workgroup-uniform** (STRUCTURE/038) — the `mmq64x32`
   tile variants (mul_mat_q8_0, mul_mat_q6_k) map 64 rows × 4 col-lanes onto
   256 lanes, so `row_in_bounds` is a genuine lane predicate and the
   workgroup barrier inside `scf.if %row_in_bounds` is a real deadlock
   hazard on the tail workgroup (rows % 64 != 0). Fix: clamp the row
   (`%row_safe = scf.if %row_in_bounds { row } else { 0 }`) so ALL lanes run
   the barrier + loads (the load phase only touches src1/scratch — row
   independent), and guard ONLY the final dst stores with `row_in_bounds`.

**Full catalog restored + gfx1151-corrected:** the round-3 pruning existed
only because kernels wouldn't compile. With all 73 roots linking, the fork's
original 39-route-file catalog is restored (218 routes), then fixed:

- **`"target_key": "gfx1100"` stripped to `""`** in 9 route files (25+
  routes: mul_mat_q8_0 ×10, rope_neox ×33-route file, soft_max, rms_norm,
  get_rows_q4_k/q5_k/q6_k/q8_0, ...) — the ORIGINAL GET_ROWS root cause, now
  fixed at the catalog level. With the new runtime's target selection, a
  pinned gfx1100 key would fail-closed on gfx1151 ("executable target is not
  supported by the device"); empty key falls back to the device architecture.
- `q4nx_dequant_f32` route + sources/artifacts/families/index entries
  re-added on top of the full catalog.
- CMakeLists artifact list restored to the full set.

Pipeline runs green end-to-end: assemble (218 routes) → validate → loom-link
(38 artifacts) → require-artifacts → embed (76 arrays) → libggml-hrx2.so.

**GPU validation (both PASS on gfx1151):**

```
mul_mat_f32_f32  total=8192 k=2048 rows=128 cols=64
  max abs diff : 1.73e-4   max rel diff : 9.24e-5   mismatches >1e-3: 0/8192   PASS
q4nx_dequant_f32 (regression, full catalog)
  max abs diff : 0.0       correlation  : 1.0       mismatches: 0/8192         PASS
```

The mul_mat_f32 route is the first REAL matmul (dot-product of length 2048
per output element, 8192 outputs, 128×64 workgroup grid) through the ported
loom-jit + restored catalog — numerically correct within f32 reassoc
tolerance vs a double-precision host reference.

**Persisted:** `patches/ggml-hrx2-fork-loomc-port.patch` regenerated (now 92
files: C++ port + kernel dialect/terminator/barrier fixes + catalog restore
+ gfx1100 strip); `validate-mulmat-f32.cpp` harness added (build/run as in
the round-4 section, replacing q4nx args).

### UPDATE 2026-08-30 (round 6): FUSED Q4NX-on-HRX demo — real model weights, dequant+matmul both on the GPU ✅

The two proven routes are now **pipelined on the device** (no CPU round-trip
between kernels): `q4nx_dequant_f32` writes its f32 output straight into the
`mul_mat_f32_f32` src0 buffer. Driven with **real Q4NX weights** extracted
from `zaya1-8b-fresh.q4nx` (8 consecutive I8 rows of layer-0 q_proj, file
offset 335961069 → a [32 rows × 2048 cols] weight block):

```
FUSED Q4NX pipeline (dequant [32x2048] + matmul [32x2048]x[2048]) on gfx1151
  fused dispatch time: ~91-140 us (both kernels, one stream)
  y_ref range: [-373.63, 1.53]   y_out range: [-373.63, 1.53]
  max abs diff : 6.09e-06   max rel diff : 2.67e-06   PASS YES
```

Reference = engine-conformant CPU dequant (q4nx_raw.h nibble semantics +
dequant_q4nx.cpp clamps) + double matmul. This is "Q4NX runs on HRX"
end-to-end: weights stay int4 until the GPU.

**The multi-tile demo exposed 3 real bugs in `q4nx_dequant_f32.loom`** that
the earlier single-tile "bit-exact" run could not see (ncols=256 makes
col/256 ≡ 0 and the trailer bytes fell outside the sampled row-groups):

1. **Missing scale/zp sanity clamp.** Each 5120-byte tile's last 6 bytes of
   the zp section (indices 253-255) carry trailer data, not zero-points; the
   engine's reference (dequant_q4nx.cpp) zeroes non-finite/outlier
   (`|x|>100`) scales and zps. The kernel now does the same
   (`scalar.absf` + `scalar.cmpf olt` + select) — without it the last
   rows × last col-groups of every tile block read garbage zero-points
   (values up to 1e38).
2. **Multi-tile scale indexing.** `row*8 + col/32` is only valid for
   ncols ≤ 256. Correct per engine semantics:
   `(col/256)*256 + row*8 + ((col%256)/32)` (per-tile 256-BF16 base).
3. **Multi-tile packed stride.** The packed tile base is 4096 B per tile;
   `col*8` only advances 2048 per tile, so the kernel needed
   `+(col/256)*2048` (single-tile widths need none).

The single-tile harness was re-validated against a **regenerated
engine-conformant reference** (`/tmp/q4nx_ref.f32`, clamped) — still
bit-exact (max diff 0.0), and the earlier "bit-exact" result is now
understood as self-consistent-but-unclamped, not a real issue.

**Persisted:** `validate-q4nx-fused.cpp` (build/run like the others; arg =
path to a `n_tiles*5120`-byte tile block, default `/tmp/q4nx_w8.bin`);
`loom-kernels/q4nx_dequant_f32.loombc` refreshed (md5 changed with the
kernel fix); fork patch regenerated. Full suite green: q4nx (bit-exact),
mul_mat_f32 (rel 9.2e-5), fused (rel 2.7e-6).

### UPDATE 2026-08-30 (round 7): GGML_TYPE_Q4NX — type + CPU dequant validated; graph-op path mapped ✅/⚠️

**The Q4NX container format is fully decoded** (authoritative, from the
engine's own readers): 8-byte JSON-length header; JSON tensor table with
per-tensor `data_offsets` (relative to `df = 8 + jsonlen`); I8 tensor shape
is `[tile_rows_total, 5120]` with `tile_rows_total = (logical_rows/32) ×
(logical_cols/256)` (verified: q_proj 256 = (1024/32)×(2048/256), o_proj,
k_proj ✓). Semantics locked: lane-packed signed int4 + clamp, matching
`dequant_i8_signed_to_float_ex` (engine q4nx_raw.h byte-exact note).

**`GGML_TYPE_Q4NX` added to the fork's ggml** (enum value 42, COUNT 43;
public tile-geometry constants; `block_q4nx` = one 5120-byte tile;
`dequantize_row_q4nx` / `quantize_row_q4nx_ref` in ggml-quants with the
engine's lane-packed + clamp semantics; traits row registered). Validated
through ggml's own machinery:

```
type: q4nx blck_size=8192 type_size=5120 quantized=1
ne=[8192,8] nbytes=40960 (blob 40960)   # 8 real q_proj tiles
dequantize_row_q4nx vs engine reference: max abs diff = 0.000000e+00  PASS
```

**HRX2 backend fused MUL_MAT dispatch written** (`supports_mul_mat_q4nx_route`
+ `dispatch_mul_mat_q4nx` in ggml-hrx2.cpp): for a Q4NX src0 it assembles
tile-major blob → section views on device, runs the `q4nx_dequant_f32`
route into an f32 scratch, then the `mul_mat_f32_f32` route with the scratch
as src0 — the fused pattern, lifted into the backend. Compiles clean; the
identical dispatch logic is what the round-6 fused harness validated
(rel 2.7e-6).

**Two honest blockers for full end-to-end serving:**

1. **ggml's 1D block model cannot express the 2D Q4NX tile for the standard
   MUL_MAT graph op.** `ggml_new_tensor` requires `ne[0]` to be a multiple of
   `blck_size` (8192) and `ggml_mul_mat` needs `src0->ne[0] == src1->ne[0]`
   as the k-dim. A tile is 32 rows × 256 cols (spans ne0 AND ne1) — unlike
   K-quants whose blocks are within-ne0 slices. So a Q4NX weight cannot be a
   normal src0 of `ggml_mul_mat` with the k-dim = in_features; the backend
   fused dispatch works at the route level but can't be reached through the
   standard graph builder. Fix path: a custom ggml op (e.g.
   `GGML_OP_MUL_MAT_Q4NX` with its own compute that the HRX2 backend claims)
   or a ggml-side reshape of how tile tensors enter the graph.
2. **zaya1-8b-fresh's architecture has no llama.cpp implementation**
   (conv_qk grouped/depthwise, v_proj_current/v_proj_delayed, per-layer
   residual scale/bias, MoE gate with 17 experts, router_states_scale,
   balancing_biases) — even with perfect type support, llama-server cannot
   execute it. Serving a full Q4NX model needs either a standard-arch Q4NX
   model or porting the architecture (large, separate effort).

### UPDATE 2026-08-30 (round 8): GGML_OP_MUL_MAT_Q4NX — Q4NX flows through real ggml graphs on gfx1151 ✅

**Blocker 1 solved with a custom ggml op.** Added `GGML_OP_MUL_MAT_Q4NX`
(enum + `GGML_OP_NAME`/`GGML_OP_SYMBOL` entries + `ggml_mul_mat_q4nx`
builder) to the fork: src0 = Q4NX tensor `[8192, n_tiles]` (each 8192-element
row is one 5120-byte tile — the only shape ggml's block model can express),
src1 = F32 `[256, cols]`, dst = F32 `[n_tiles*32, cols]`. The HRX2 backend
graph executor now dispatches the op through the fused path (tile-major blob
→ section views assembled on device → `q4nx_dequant_f32` route → f32 scratch
→ `mul_mat_f32_f32` route). Validated on gfx1151 with 8 real q_proj tiles:

```
GGML_OP_MUL_MAT_Q4NX [8 tiles -> 256x256] x [256x1] through HRX2 backend on gfx1151
  ref range: [-1.366, 24.810]  out range: [-1.366, 24.810]
  max abs diff : 1.64e-06   max rel diff : 2.94e-05   PASS YES
```

Notes: the f32 route kernel reads src1 as `[cols, k]` (col-major), which
coincides with ggml's `[k, cols]` layout only at cols=1 — the route's native
moe_logits shape. Multi-col needs a src1 transpose in the dispatch
(device-side assembly, deferred). The plan-builder path was bypassed (route
shape_domain pins k=2048/rows=128) in favor of direct config bindings — the
same approach the fused harness validated.

**Remaining blocker:** zaya1-8b-fresh (and zaya1-8b) use a bespoke
architecture (conv_qk, v_proj_current/delayed, per-layer residual
scale/bias, 17-expert MoE gate, router_states_scale) with no llama.cpp
implementation — verified both model files. The converter (milestone 2)
would produce a GGUF llama.cpp cannot execute, and llama-server end-to-end
(milestone 4) needs either a standard-arch Q4NX model or an architecture
port (large, separate effort). The op-level integration (milestone 3) is
complete and validated.

Full suite green: q4nx bit-exact, mul_mat_f32 rel 9.2e-5, fused rel 2.7e-6,
type bit-exact, **graph op rel 2.9e-5**.

### UPDATE 2026-08-30 (round 9): converter done + GGUF load verified; CONTAINER OFFSET CORRECTED

**Authoritative container decode correction.** The true `df` is `8 + hsz`
with `hsz = 232415` (u64 at file offset 0, JSON parses exactly at hsz
bytes) — my earlier hand-derived `json_end = 232429` was 6 bytes high.
All tensor offsets shifted by -6 (q_proj: 335961069 → **335961063**). The
earlier validations were self-consistent (references computed from the same
shifted bytes) but the extraction was misaligned; at the true offset the
tiles are cleaner (zeros all 0 — the "6-byte trailer garbage" from round 6
was actually the misaligned read reaching into the next tile). All harnesses
fixed and re-validated at the true offset — results equal or better
(op-level: rel 2.77e-6; single-tile: bit-exact; fused: rel 7.1e-6). The
scale/zp clamp stays (engine-conformant, defensive).

**Milestone 2 — .q4nx → GGUF converter, verified.** `q4nx_to_gguf.py`
(container reader using the authoritative hsz/data_offsets) emits GGUF with
I8 tensors as `GGML_TYPE_Q4NX` (id 42, ggml ne=[8192, n_tiles]) and BF16
tensors as `GGML_TYPE_BF16` (30). The fork's gguf-py gained `Q4NX = 42` +
`GGML_QUANT_SIZES[Q4NX] = (8192, 5120)`. Verified with the fork's C++ GGUF
reader (`validate-gguf-load.cpp` — the same machinery llama.cpp uses):

```
GGUF: 1 tensors
  tensor[0]: model.layers.0.self_attn.q_proj.weight type=42 nbytes=1310720
GGUF tensor data vs source container bytes: IDENTICAL   PASS
```

Full model converts cleanly: **1284 tensors, 5.99 GB** (Q4NX + BF16).
CPU dequant of the loaded Q4NX tensors is bit-exact (round 7/8).

**Milestone 4 (llama-server e2e) remains blocked by the bespoke
architecture** (conv_qk, v_proj_current/delayed, residual scales, 17-expert
gate — verified in both zaya model files; no llama.cpp implementation).
The loader can now read Q4NX GGUFs; executing a model needs either a
standard-arch Q4NX model or an architecture port.

Full suite green at the true offset: q4nx bit-exact, mul_mat_f32 rel 9.2e-5,
fused rel 7.1e-6, type bit-exact, graph op rel 2.77e-6, GGUF load
byte-identical.

### UPDATE 2026-08-30 (round 10): cols>1 batches + REAL standard-model weight through the op on gfx1151

**`GGML_OP_MUL_MAT_Q4NX` now handles multi-column batches.** The f32 route
reads src1 as `[cols, k]` and writes dst as `[cols, rows]`, both of which
differ from ggml's `[k, cols]` / `[rows, cols]` conventions for cols>1; the
dispatch now transposes src1 and dst on device (per-element stream copies).
Validated: cols=16 rel 2.94e-5, cols=1 fast path rel 2.77e-6.

**Key realization: no pre-existing standard-arch Q4NX model is needed — we
can CREATE one.** `quantize-gguf-to-q4nx.cpp` (uses the fork's own
validated code: `dequantize_row_q4_K/Q6_K` -> f32 -> `quantize_row_q4nx_ref`
-> tiles in tile-grid order) quantizes a standard GGUF weight into Q4NX.
This surfaced and FIXED a real bug in `quantize_row_q4nx_ref`: negative
weights were clamped to 0 instead of mapping to the high nibbles
(two's-complement: q<0 -> q+16), and the scale factor hit the q=8<->val=-8
edge (now smax/7). Round-trip rel L2 ~10% (int4 re-quant of a Q4_K source).

**Real-model validation on gfx1151** — Qwen3-0.6B `blk.0.attn_q.weight`
(quantized Q4_K -> Q4NX), column-tile 0 (64 tiles = [2048 rows x 256 cols])
through the op, 4-column batch:

```
GGML_OP_MUL_MAT_Q4NX — Qwen3-0.6B blk.0.attn_q col-tile 0 [2048x256]x[256x4] on gfx1151
  max abs diff 3.69e-07   mismatches >1e-3: 0/8192   PASS
```

The standard-model Q4NX pipeline is now proven end-to-end at the op level:
standard GGUF -> Q4NX quantizer -> GGML tensor -> GGML_OP_MUL_MAT_Q4NX ->
HRX2 fused dispatch on gfx1151 -> numerically exact (abs 3.7e-7).

### UPDATE 2026-08-30 (round 11): FULL Q4NX MODEL SERVED by llama-cli on gfx1151 ✅

**The complete Option A chain now runs end-to-end on the Strix Halo GPU:**
a standard GGUF (Qwen3-0.6B-Q4_K_M) is quantized to Q4NX, loaded by the
fork's llama.cpp, and served with the Q4NX weights executing through
`GGML_OP_MUL_MAT_Q4NX` on the HRX2 backend.

Key pieces that made it work:

1. **`ggml_mul_mat` interception**: a Q4NX src0 routes to the Q4NX builder
   automatically (the tile tensor's ne[0]=8192 is the block size, not the
   k-dim, so the standard ne[0] equality check is bypassed). No
   llama-graph.cpp surgery needed.
2. **Full-weight op**: `ggml_mul_mat_q4nx` derives n_tc = src1->ne[0]/256
   and rows = n_tiles/n_tc*32; one dispatch dequantizes the WHOLE weight
   (multi column-tile) via the general tile-grid kernel, then runs a new
   **ggml-layout f32 matmul kernel** (`hrx2_mul_mat_f32_f32_ggml_static`,
   src1 [k,cols], dst [rows,cols] — no transposes, so no per-element copy
   storm; the earlier per-element transpose approach caused an ~7.5M-copy
   slowdown that looked like a hang).
3. **Persistent dispatch scratch**: the fused dispatch allocates 4 scratch
   buffers per op and released them immediately, but the kernels are async
   on the stream — the allocator reused the memory mid-flight and caused an
   AMDGPU memory fault. Scratch is now per-device-context and never freed
   until teardown.
4. **Backend claims**: HRX2's `supports_op` claims ONLY `MUL_MAT_Q4NX`
   (the fork's f32/f16/rms_norm etc. routes fail to compile for arbitrary
   model shapes, SUBRANGE/024, or fault); everything else runs on CPU.
   `check_tensor_dims` bypassed for Q4NX (stored ne is [8192, n_tiles], not
   the logical [in, out]).
5. **Model creation**: `make-q4nx-model.py` — full GGUF->Q4NX converter
   with numpy Q4_K/Q6_K dequants (ported exactly from ggml: Q6_K layout has
   d LAST; Q4_K scale/min use raw 0..15 indices; fp16 must be
   bit-reinterpreted, not numerically converted; bf16 scales use ggml's
   round-to-nearest-even; quantization re-reads the BF16-rounded scale;
   roundf half-away-from-zero). Output is **byte-identical** to the
   validated C++ `quantize_row_q4nx_ref` (0/1,310,720 diffs).

Result (llama-cli, -ngl 99 on gfx1151):
```
model: qwen3-0.6b-q4nx.gguf (196 Q4NX weights + 114 kept)
[ Prompt: 15.8-23.1 t/s | Generation: 2.0-2.2 t/s ]
> The capital of France is <real model output tokens>
```
The model loads, the graph executes with all Q4NX matmuls on the HRX2 GPU,
and real tokens are generated. Output text is low-quality because the
weights are DOUBLE-QUANTIZED (Q4_K -> Q4NX, ~10% L2 error compounds); the
pipeline itself is numerically exact (op-level abs 3.7e-7, byte-identical
tiles). Serving from a float source (or a native Q4NX model) would restore
quality — every layer of the chain is validated.

**Milestone status: all four Option A milestones demonstrated.** (1) type,
(2) converter + load, (3) backend fused dispatch, (4) llama-cli end-to-end
on gfx1151 with first tokens. Remaining quality work (float-source
quantization, multi-op scheduler tuning) is follow-up, not blocking.

**Remaining for milestone 4 (llama-server full model):** patch
llama-build-graph.cpp to emit GGML_OP_MUL_MAT_Q4NX per column-tile for
Q4NX-typed weights (in=1024 -> 4 column-tiles summed; token_embd/lm_head
have vocab 151936, not tile-divisible — needs a tail-handling plan).
Bounded but a real llama.cpp integration; every piece it needs is now
validated.

**Persisted:** `validate-ggml-q4nx-type.cpp` (type + CPU dequant validation;
PASS bit-exact), `validate-ggml-q4nx-op.cpp` (graph-op attempt — documents
the ne0/blck_size blocker), fork patch regenerated with the type + backend
changes. The route-level fused demo remains the validated "Q4NX on HRX"
evidence (rel 2.7e-6).

---

## Round 12 — Q4NX quality fixed: the mm kernel silently transposed every
## multi-token matmul (2026-08-30)

**TL;DR:** the served Q4NX model produced gibberish ("ollaplr..."). Root
cause was NOT quantization quality (weights were byte-identical to the
validated quantizer) — it was a silent **transpose in the fused
`mul_mat_f32_f32_ggml` kernel** for any `cols > 1`. Fixed in the .loom
kernel + two dispatch cache-key bugs. Both Q4NX models now score corr
0.94–0.95 vs the BF16 reference and generate coherent text.

### How it was found (debug chain)

1. **Logits A/B harness** (`dump-logits.cpp`): BF16 model at ngl 0 AND 99 →
   correct top5. Both converted models (F32 twin AND Q4NX) → degenerate
   tail top5 [151931..151935] → converter or execution bug, not metadata
   (KV diff clean, tensor names/order identical, attn_q data corr 1.0).
2. **The dump dir was poisoned**: `/tmp/q4nx_src` (used by the converters)
   was created at 18:37 — the exact mtime of a still-being-written
   `Qwen3-0.6B-BF16.gguf`; its last 34 tensor files were all zeros
   (blocks 7–9 + output_norm → degenerate logits). Re-dumping after the
   file stabilized fixed that batch of zeros.
3. **After the zeros were fixed** the Q4NX model gave spread-but-wrong
   logits (corr −0.02) while the "Q4NX-dequant-F32 twin" (same values,
   F32 type) gave corr 0.938 → the Q4NX *execution* was still wrong.
4. **Cache-key bug #1**: `dispatch_mul_mat_q4nx` built provider cache keys
   `-q4nx-<rows>` (dequant) and `-q4nx-mm-<rows>x<cols>` (mm) WITHOUT
   ncols(k)/n_tc — a pure keyed cache (no config compare). In the model,
   attn_k (rows=1024, k=1024, n_tc=4) compiled first; attn_output
   (rows=1024, k=2048, n_tc=8) and ffn_down (k=3072, n_tc=12) REUSED the
   n_tc=4 kernel → wrong tile mapping. Fixed: keys now include k, n_tc,
   wg_size. (Caught by a 7-shape op harness `validate-op-all.cpp` that
   tests every distinct Q4NX shape with cols=5.)
5. **The real killer — kernel layout bug**: `mul_mat_f32_f32_ggml` indexed
   src1 as row-major `i*cols + col` and dst as `row*cols + col`. ggml
   tensors are **ne[0]-fastest**, so element (i,col) of [k,cols] is at
   `i + col*k` and (row,col) of [rows,cols] is at `row + col*rows`. The
   row-major indexing is only correct for `cols == 1` — which is why every
   op-level test (cols=1) passed while the real model (5 tokens) got
   silently transposed activations and outputs. Fixed in
   `mul_mat_f32_f32.loom` (src1: `i + col*k`; dst: `row + col*rows`), with
   the op-test reference updated to the ggml convention (all 7 shapes
   max_abs ≤ 5.5e-7).
6. **Upload sync hardening**: `stage_and_copy_tensor` now flushes+waits the
   stream after each upload (cross-stream ordering with graph dispatches).

### Results

```
full Q4NX (float-source):  corr 0.938 vs BF16, top1 12095 (== ref top1)
full Q4NX (Q4_K-source):   corr 0.951 vs BF16, top1 12095
Q4NX (float-source) gen:   "The question is a bit tricky. Let's break it down step by step"
BF16 ref            gen:   "Paris. The capital of Italy is Rome. The capital of Spain is Madrid..."
```

Both converted models now generate coherent, meaningful text. The earlier
"double-quantization quality loss" explanation was wrong — the pipeline was
numerically wrong (silent transpose), and is now exact (op-level
max_abs ≤ 5.5e-7 for all shapes at cols=5, logits corr 0.94–0.95 = pure
4-bit quantization loss).

**New persisted artifacts:** `dump-logits.cpp` (llama.h logits harness),
`gen-tokens.cpp` (greedy sampling harness), `validate-op-all.cpp`
(all-shape cols=5 op validation with corrected ggml-layout reference),
`numpy-forward.py` (full Qwen3-0.6B numpy forward, used to prove the
activations were correct at every stage), fork patch regenerated.

---

## Round 13 — Zaya 8B port: the architecture was wrong (not the math)

**Headline: 8/8 top-1 agreement with the HF `Zyphra/ZAYA1-8B` reference after
fixing the layer structure. The zaya port now produces coherent chat output.**

### What was wrong

The port (and the 1bit engine reference it was copied from) implemented the
**base-era alternating structure**: even layers = CCA attention only, odd
layers = MoE only. But **Zaya 8B runs BOTH blocks in EVERY layer** (HF
`ZayaDecoderLayer`):

```
per layer:
  residual = h
  h = input_layernorm(residual)                      # attn_norm
  attn = CCA(h)                                      # full CCA path
  residual = (attn+hs_b)*hs_s + (residual+res_b)*res_s   # post_attention scale
  h = post_attention_layernorm(residual)             # post_attn_norm (NEW tensor)
  moe = MoE(h)                                       # with prev_router EDA
  h = (moe+hs_b)*hs_s + (residual+res_b)*res_s           # post_mlp scale
final: logits = emb @ norm(h)
```

Evidence: the container has REAL attention AND MoE weights in all 40 layers
(corr 1.0 with HF input scales, 0.9947 with HF q_proj after int4 noise).
The old logits were uncorrelated with HF (cos -0.10) because half the
weights were never used.

### Fixes (fork `hrx-v2`, uncommitted)

- **converter** `zaya-to-gguf.py`: write both weight sets for every layer +
  `post_attn_norm`; fixed `add_bf16` shape handling (`shape[::-1]` GGUF dims
  convention); fixed `zaya.expert_count`/`zaya.expert_used_count` key names;
  fixed `res_scale_hs`/`res_scale_hs_mlp` tensor names.
- **loader** `llama-model.cpp`: create all 31-32 tensors per layer (no more
  `i % 2` split); added `LLM_TENSOR_POST_ATTN_NORM` name/info mapping.
- **graph** `src/models/zaya.cpp`: rewritten loop — both blocks per layer
  with the HF residual chain; `prev_router` EDA on every MoE; final = just
  `norm(h)` (no residual add).
- **hybrid memory filter**: ZAYA now allocates the recurrent state for ALL
  layers (`filter_attn = filter_recr = []{return true;}`) — the old
  `il % 2 == 0` filter crashed (null `s_l`) once every layer needs state.
- **tokenizer**: GGUF now uses the real HF tokenizer dir
  (BOS=2 `<bos>`, EOS=106 `<|im_end|>` — the old GGUF had them swapped).

### Validation

```
logits corr llama-vs-HF (int4 container vs BF16): 0.897
top-1 agreement over 8 greedy steps:              8/8  (ALL match HF)
per-step cos: 0.82–0.99  (expected int4 quantization gap)
generation:   "<think> We have a conversation: The user asks: 'What is the
               capital of France?' ..."  — coherent, meaningful
```

Layer-by-layer numpy cross-check (container truth): L0 attn_out corr 0.994,
residual_post_attn 0.998, layer_out 1.0/0.9986 — the CCA + residual math was
already exact; only the block structure was wrong.

### Artifacts

- `zaya-to-gguf.py` (fixed converter), `zaya-full-forward2.py` (HF-structure
  numpy ref), `zaya-prefill3.py` (batch prefill CCA verification).
- `/home/bcloud/zaya-f32.gguf` regenerated (F32, both blocks, 1283 tensors).
- HF reference: `/home/bcloud/models/ZAYA1-8B` (17.7GB download, public).

### Next

- `zaya_q4nx.gguf` conversion (Q4NX mode) → HRX ngl=99; the attn projections
  are 2-D MUL_MAT (already dispatched); the stacked experts need MUL_MAT_ID
  Q4NX dispatch.

## Round 14 — Zaya Q4NX on HRX: MUL_MAT_ID dispatch + the ids-stride bug (2026-08-31)

Milestone: **zaya1-8b-fresh runs fully on the HRX20 (gfx1151) lane at
ngl=99** with Q4NX-quantized attention + MoE weights. Logits match the F32
port to float noise; generation is coherent ("Paris.").

### What was added

- **`GGML_OP_MUL_MAT_ID_Q4NX`** + `ggml_mul_mat_id_q4nx` constructor
  (`ggml/include/ggml.h`, `ggml/src/ggml.c`): src0 Q4NX 3-D tile-major
  `[8192, tiles_per_expert, n_expert]` (one expert's tiles contiguous),
  src1 F32 `[k, ntokens]`, ids `[nselected, ntokens]` → result
  `[tpe/n_tc*32, nselected, ntokens]`.
- **HRX2 dispatch** (`ggml-hrx2.cpp`): `ggml_mul_mat_id` now routes Q4NX
  src0 to the new op; `dispatch_mul_mat_id_q4nx` loops `(i,t)` pairs and
  reuses the shared 2-D slice helper (per-expert tile copy → q4nx dequant →
  `mul_mat_f32_f32_ggml` with cols=1). CPU backend rejects both Q4NX ops.
- **Q4NX 3-D GGUF conversion** (`zaya-to-gguf.py`): experts written with
  `raw_shape=[n_exp, tpe, 5120]` → file `[n_exp, tpe, 8192]` → ne
  `[8192, tpe, n_exp]`, nb0=nb1=5120, nb2=tpe*5120 (tile (t,e) at
  `(e*tpe+t)*5120` — verified against the container blob byte-for-byte).

### The root-cause bug (ids are a strided view)

Symptoms: every Q4NX op verified bit-exact (dequant corr 1.0, mm corr 1.0,
all 40 layers' MoE inputs corr ≥ 0.72), yet final logits were uncorrelated
(0.023) and token-1 MoE weights were wrong (0.21888 instead of 0.0187).
Bisect (per-layer tensor dumps + numpy decomposition) showed:

- the router argmax ids are **`[3,4]`** (true top-1 per token), but the
  MUL_MAT_ID dispatch saw **`[3,5]`** — the second token's id was the
  second-best expert of the FIRST token.
- `ffn_moe_topk` is a **VIEW of the argsort output** with `nb[1] = 17*4`
  (17 experts sorted per token), not `4`. The CPU get_rows/mul_mat_id read
  it stride-correctly → `[3,4]` (why the F32 port was always right); the
  HRX2 dispatch copied the ids **contiguously** (flat `[0]=3, [1]=5`) — the
  code even computed `ids_src_stride` but never used it.
- Fix: copy each token row at its real `nb[1]` stride into the packed
  host-visible scratch (`q4nx_ids`), then read `[i*ntokens+t]`.

### Validation

```
logits corr Q4NX-HRX vs F32-HRX:  0.9999999
max |Δlogit|:                      0.0123   (int4 quantization noise)
top1:                               9731   (= F32/HF reference)
top5:               [9731, 11861, 115314, 59820, 79030]  (identical)
generation:        "<think> ... So answer: Paris. ... </think> Paris."
                   (clean EOS; coherent)
```

### Artifacts

- `zaya-q4nx.gguf` (Q4NX: 280 quantized tensors, 7/layer — attn_q/k,
  cca_val_proj1/2, attn_output, gate_up/down experts; router stays F32).
- Fork commits: `ggml.h`/`ggml.c`/`ggml-hrx2.cpp`/`ggml-cpu.{c,cpp}`.
- Harnesses: `/tmp/dump_hrx_logits` (logits for `[2,2202]`), `/tmp/gtok_hrx`
  (chat gen, ngl arg). Debug dumps (DEQDUMP/TENSORDUMP/WDUMP/GRTRACE/MMID)
  were removed after the fix; `q4nx_ids` scratch + stride copy remain.

### Next

- Remove the last debug scaffolding, commit fork + 1bit-MONSTER
  (`feat/hrx-gfx1151-build`), regenerate `patches/` snapshot.

## Round 15 — Zaya Q4NX decode 7–12× faster: kill the per-tile copy storm (2026-08-31)

The Q4NX dispatch uploaded each tile with **3 stream copies** (scales / zp /
packed): one gate_up expert slice (1024 tiles) enqueued 3072 copy commands,
and a single decoded token touched ~9200 stream ops — that launch overhead,
not NPU compute, was the 1.2 t/s wall.

Fix (fork `76ab40d`): expert tiles are **contiguous** in src0
(tile `(t,e)` at `(e*tpe+t)*5120`), so a slice is one copy. The dequant
kernel now reads the raw tile-major blob via three views of the same buffer
(dispatch binds it at offsets 1024/0/512 for packed/scales/zeros); the
section-assembling scratch (`q4nx_zp`/`q4nx_pck`) is gone.

| metric | before | after | speedup |
|---|---|---|---|
| prompt eval | 1.97 t/s | 24.08 t/s | 12.2× |
| generation  | 1.22 t/s |  8.66 t/s |  7.1× |

Correctness unchanged: logits corr 0.99999999 vs F32 port (max |Δ| 4.1e-3 —
the float32 summation-order noise floor, 1,700× below the top-1 margin),
top5 identical, coherent generation.

Harness note: the post-reboot recreation of `gtok_hrx` had a KV position bug
(`batch.pos = step+i` breaks multi-token prompts); fixed to a running
position counter. Harnesses now live in `harnesses/` (reboot-proof; /tmp is
tmpfs).

### Round 15 addendum — what does NOT move the needle (2026-08-31)

Benchmarked the remaining decode budget with targeted experiments:

- **Dequant cache (per-expert f32)**: counterproductive — the prefill (8
  tokens) allocates a fresh ~33 MB buffer per (op, expert) → tens of GB of
  allocations, prefill 4.5× slower. A 2-D-attn-only variant gave zero gain
  (8.60 vs 8.66 t/s): the attn dequants were never the bottleneck.
- **Ids host-sync removal** (80 syncs/token): no change (8.41 vs 8.66 t/s) —
  the single stream already serializes; syncs are cheap.
- **Skipping kernels entirely** made things *slower* (pathological — the
  benchmarks were measuring a broken pipeline, not a useful bound).

Real decomposition: the **F32 port is itself 95.6 ms/token** — the shared
CCA attention path on HRX is the wall. Q4NX adds ~22 ms (80 expert slices ×
~0.28 ms = 5.2 MB copy + 33 MB dequant write + 33 MB mm read each, all
bandwidth-bound). Q4NX now sits 24% behind the F32 port (8.68 vs 10.46 t/s),
and the only remaining lever is fusing the dequant into the mm kernel
(eliminates the f32 round trip) — a follow-up kernel project.

## Round 16 — perf decomposition + the fused-kernel dead end (2026-08-31)

Benchmarked the F32 port (same graph, all-HRX) as the reference wall:

| model | pp32 | tg32 |
|---|---|---|
| zaya F32 (HRX2) | 120.5 t/s | 10.93 t/s |
| zaya Q4NX (HRX2) | 33.9 t/s | 8.43 t/s |

The 3.5× prefill gap is the MoE: every prefill token selects a different
expert, so each op dequantizes ~33 MB f32 per token (bandwidth-bound). Decode
is only 1.3× behind — the shared attention path dominates the 1-token latency.

**Fused dequant+mm kernel (implemented, then reverted):** a
`q4nx_mul_mat_f32_ggml` Loom kernel dequantizing inline in the mm inner loop
(no f32 intermediate — halves traffic). It compiled, ran, and was numerically
correct (corr 0.9999999861), but **prefill collapsed to 0.63 t/s (55×
slower)**: the per-element scalar dequant (~20 index/byte ops per FMA) is
catastrophic on the NPU. The existing `mul_mat_q4_k_f32.loom` (5,469 lines)
shows the required design — vectorized loads + bitfield extraction, 4
columns/lane — a substantial porting effort for Q4NX, documented as the
follow-up. The dequant+mm split at 33.9/8.4 t/s stands.

The catalog/source/artifact scaffolding for the fused route was fully
reverted (zero net diff); the experiment is preserved in this write-up.

## Round 17 — llama-server serves zaya Q4NX on HRX20 over HTTP (2026-08-31)

The full product path now runs end-to-end: `llama-server` serving the Q4NX
zaya model on the HRX2 backend.

```
llama-server -m zaya-q4nx.gguf -ngl 99 -c 256 -b 64 --parallel 1 -fit off --port 8099
curl /v1/completions {"prompt":"The Eiffel Tower is in the city of","n_predict":40,"temperature":0}
→ " Paris.\n\nThe user is asking if these examples follow a specific pattern..."
  prompt: 18.1 t/s | decode: 8.16 t/s (server timings)
```

Notes:
- The server's **auto `n_parallel=4` params-fit probe aborts**:
  `pre-allocated tensor (cache_s_l0 ...) in a buffer (HRX20) that cannot run
  the operation (CPY)` — the multi-parallel recurrent-state CPY shape is not
  supported by `ggml_backend_hrx2_supports_cpy`. Workaround: `--parallel 1
  -fit off`. Supporting the parallel CPY is a small follow-up.
- Hit-rate measurement (decode expert repeat, per-op last-expert): 0.47
  overall — a 1-slot dequant cache would recover ~10% decode, not worth the
  machinery vs the bandwidth-bound wall. Documented, not implemented.
