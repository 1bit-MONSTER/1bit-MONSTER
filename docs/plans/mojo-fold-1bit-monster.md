# Plan: The Mojo Fold — 1bit.systems → 1bit.monster

**Status:** Draft · **Date:** 2026-08-13 · **Scope:** fold the 1bit engine into Mojo/Modular's platform, cut everything the platform makes redundant.

## Thesis

The engine (C++23 compute kernels, one binary, NPU+GPU+CPU) stays. Everything around it — servers, converters, tooling, control planes, bindings — becomes Mojo 1.0, distributed through Modular's platform (pixi/conda, agent skills, MAX ecosystem). Mojo solves the problems we hand-rolled: glue-language sprawl, FFI bindings, packaging, cross-platform tooling.

## Verified platform facts (2026-08-13, research + ingested docs)

| Surface | Reality |
|---|---|
| Mojo ↔ C++ | `extern "C"` only. No C++ ABI. `std.ffi` (`OwnedDLHandle`, `external_call`) dlopens shared libs. Pattern already proven in `npu-infer/ffi_bridge.cpp`. |
| Mojo linking | No static linking / no single-ELF. `mojo build` exes dynamically link `libKGENCompilerRTShared.so`. Ship exe + runtime .so (container pattern). |
| Mojo packages | `mojo precompile` → `.mojoc` (not distributable). Real distribution: pixi + rattler-build conda packages → `modular-community` channel on prefix.dev. |
| max serve | **Cannot host external engines.** Serving is hardwired to `PIPELINE_REGISTRY`. 1bit keeps its own OpenAI-compatible server. |
| MAX fold-ins that DO work | Custom architectures (MAX Graph models — not us), CustomOps (export 1bit kernels as Mojo `CustomOp`s — later), `max benchmark` can measure external servers (validation), agent skills (open standard, no registry gate). |
| Mojo 1.0 stdlib gaps | No `std.net`, no regex, no `std.json` (confirmed in ingested stdlib source). Hand-rolled libc patterns stay (already the UPDATE 34 decision). |
| Toolchain | `mojo` + `pixi` installed on dev machine. No `pixi.toml` in repo yet. Pin: `mojo==1.0.0`. |

## Phase 0 — Land the burn on `main`

**Fact:** the burn (19k lines cut, through-line, UPDATE 34) exists only on `feat/jarvis-v2-rewrite` (pushed). `origin/main` is pre-burn — SaaS (`tools/jarvis/auth.cpp`…), `zaya_audio/`, agent stack, JARVIS v1 side-servers all still present.

- [ ] Fast-forward `main` to `feat/jarvis-v2-rewrite` (`c0327534`), push. Clean 2-commit ff.
- [ ] Delete `backup/local-main-stale-2026-07-19` after confirming the jarvis branch is the true line.
- [ ] All subsequent work happens on `main`.

## Phase 1 — The seam: one C ABI for the whole engine

Mojo can only call `extern "C"`. `npu-infer/ffi_bridge.cpp` already proves the pattern (opaque handle + flat C functions for the raw NPU engine). Extend it to the full engine:

- [ ] New `include/onebit_c.h` + `src/onebit_c.cpp`: opaque `OneBitHandle`, flat C surface — `onebit_create/destroy/init(model_path)/generate(tokens)/config/health/server_lifecycle`. Thin wrappers over `backend_manager` (already static libs: `libbackend_manager.a`, `libgguf_reader.a`, `libonebp_model.a`, `liblora_runtime.a`).
- [ ] New CMake target `onebit_engine` → `libonebit.so` (the only shared lib Mojo/MAX/Rust-era-tools ever touch).
- [ ] `npu-infer/ffi_bridge.cpp` folds into the same header family (single ABI story).
- [ ] Smoke test: a 30-line Mojo program dlopens `libonebit.so`, loads a model, generates tokens — via `OwnedDLHandle` (runtime) and `external_call` (compile-time) variants.

## Phase 2 — The Mojo envelope (rewrite, then cut)

Every non-kernel component gets a Mojo twin; the C++/Python original is deleted when the twin passes parity. Pattern already set by the Adrenalin rewrite (M0–M2: toolchain, JSON+sysfs, HTTP+GETs).

Priority order (existing momentum first):

1. **Adrenalin control plane** (M0–M2, from UPDATE 34) — first production Mojo target.
2. **Python tooling → Mojo** (24 files in `tools/`, 5 in `scripts/`): `gguf_to_onebp.py`, `hf_to_onebp.py`, `qwen3_to_onebp.py`, `hf_tokenizer_to_htok.py`, `convert_*_to_gguf.py`, `safetensors_to_onnx_int8.py`… These are the "converters" of the through-line. Mojo native exes, same CLI contract. Delete the `.py` on parity.
3. **`tools/train/`** (SFT/RL training scripts) — check against MAX training story; likely cut entirely (Modular owns training).
4. **`npu-infer/rust/`** (187-line Rust FFI binding) → Mojo binding over the same C FFI. **Delete Rust.**
5. **Onebit CLI** (`tools/onebit.cpp`/`onebin.cpp` dispatch: chat/up/serve/config/auth/pull) — the control-plane CLI becomes a Mojo exe; C++ keeps only compute subcommands (`zaya`/`unified`/`vision` server cores, `jarvis` pipeline).
6. **`engine/npu` Python** (13 src + 8 generators files) — audit; generators/validators → Mojo where they're tooling, keep Python only if it's a test harness with no Mojo equivalent yet (`mojo test` exists — migrate).
7. **Repackaging** (`pixi.toml`, `mojoproject.toml`-era none) — pin `mojo==1.0.0`, channels `https://conda.modular.com/max`, `conda-forge`.

## Phase 3 — Platform fold (1bit in the Modular ecosystem)

- [ ] **Publish Mojo-built tooling as conda packages** on `modular-community` (prefix.dev) via `github.com/modular/modular-community` PRs (rattler-build `recipe.yaml`). First package: the converter set (`1bit-tools`).
- [ ] **Agent skills repo** `github.com/1bit-systems/skills` (agentskills.io SKILL.md format, `modular/skills` pattern): engine runbook, 1bit-format converter skills, NPU bring-up. Zero gate to publish.
- [ ] **Contribute a `mojo-cpp-interop` skill upstream** — `modular/skills` has none; we own the hard-won knowledge (ffi_bridge pattern, dlopen + OwnedDLHandle). Good-faith platform citizenship, fits the fold.
- [ ] **Validation via `max benchmark`** — external-server mode measures our server alongside vllm/sglang/trtllm; wire into CI.
- [ ] **Optional later**: export 1bit NPU kernels as MAX Graph `CustomOp`s (Mojo) so MAX pipelines can use them; only if a concrete consumer appears (YAGNI until then).

## Phase 4 — 1bit.monster

- [ ] Rebrand pass: site/ content, README lockup, docs header — "1bit.monster: the Mojo-native engine".
- [ ] Packaging story rewrite: engine ships as before (deb/snap/tarball/docker/ollama/AUR) + Mojo exes ship exe-with-`libKGENCompilerRTShared.so` (container pattern) + conda packages.
- [ ] Docs: journey UPDATE 35 covers the fold.

## Redundancy cut list (redundant to MOJO → delete)

| Item | Verdict | When |
|---|---|---|
| `npu-infer/rust/` FFI bindings | **CUT** → Mojo binding | P2.4 |
| `experimental/bit1_mlx` (dead Rapid-MLX extraction) | **CUT** | P2 (now) |
| `zaya_audio/` voice-cloning stack | **CUT** (burned on branch, still on main) | P0 |
| SaaS: `tools/jarvis/auth.cpp` `billing.cpp` `usage.cpp` `beacon.cpp`, `workers/`, `site/dashboard/` | **CUT** (burned on branch) | P0 |
| Agent stack (personas/prompts/skills/awareness scripts) | **CUT** (burned on branch) | P0 |
| `tools/unified-router.py` (P0.2 of research/PLAN.md: "pick one router, retire the other two") | **CUT** → C++ router | P2 |
| Python converters + tooling (24+5 files) | **CUT** → Mojo twins | P2.2 |
| `tools/train/` | **CUT** if Modular's training story covers it | P2.3 |
| `hackathon/` (demo mp4 + scripts) | **CUT** (or archive/archive) | P2 (now) |
| `engine/npu` Python generators/validators | Convert → Mojo | P2.6 |
| Embedded `lemonade-server-core` | **KEEP** — max serve can't host us; it's serving compute, not glue | — |
| `integrations/comfyui`, `vllm-toolbox` | **KEEP** (external-system integrations, not Mojo-redundant) | — |
| Flutter mobile app | **KEEP** (Mojo has no mobile UI story) | — |
| `ggml_vulkan` / llama.cpp fork, kernels, `ck-prefill`, `spec-decode` | **KEEP** (compute) | — |

## Walls (stated honestly)

1. **No C++ ABI in Mojo** → the C shim (P1) is mandatory and permanent. Not a blocker; it's the seam.
2. **Mojo exes aren't single-ELF** → the "one binary" through-line narrows to the engine. Mojo exes carry `libKGENCompilerRTShared.so`. Still zero interpreter at runtime.
3. **max serve can't host 1bit** → our OpenAI-compatible server stays C++/Mojo-owned. No platform serving for us until Modular opens a backend interface.
4. **stdlib gaps** (net/regex/json) → hand-rolled libc patterns (UPDATE 34 already committed to this; `std.ffi` `external_call` covers it).

## Immediate next actions (this session)

1. P0: ff-merge burn onto `main`, push, delete stale backup branch.
2. P1: `onebit_c.h` + `onebit_c.cpp` + `libonebit.so` CMake target + Mojo dlopen smoke test.
3. P2.1: Adrenalin M0 (toolchain) — `pixi.toml` with `mojo==1.0.0`.
