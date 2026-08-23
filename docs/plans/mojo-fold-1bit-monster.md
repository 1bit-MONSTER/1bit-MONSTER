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
| Toolchain | `mojo` + `pixi` installed on dev machine. `pixi.toml` + `pixi.lock` in repo since the htok twin (P2.2). Pin: `mojo==1.0.0`. |

## Phase 0 — Land the burn on `main`

**Fact:** the burn (19k lines cut, through-line, UPDATE 34) exists only on `feat/jarvis-v2-rewrite` (pushed). `origin/main` is pre-burn — SaaS (`tools/jarvis/auth.cpp`…), `zaya_audio/`, agent stack, JARVIS v1 side-servers all still present.

- [ ] Fast-forward `main` to `feat/jarvis-v2-rewrite` (`c0327534`), push. Clean 2-commit ff.
- [ ] Delete `backup/local-main-stale-2026-07-19` after confirming the jarvis branch is the true line.
- [ ] All subsequent work happens on `main`.

## Phase 1 — The seam: one C ABI for the whole engine

Mojo can only call `extern "C"`. `npu-infer/ffi_bridge.cpp` already proves the pattern (opaque handle + flat C functions for the raw NPU engine). Extend it to the full engine:

- [x] New `include/onebit_c.h` + `src/onebit_c.cpp`: opaque `OneBitHandle`, flat C surface — `onebit_create/destroy/init(model_path)/generate(tokens)/config/health/server_lifecycle`. Thin wrappers over `backend_manager` (already static libs: `libbackend_manager.a`, `libgguf_reader.a`, `libonebp_model.a`, `liblora_runtime.a`). — landed `a73e6844`
- [x] New CMake target `onebit_engine` → `libonebit.so` (the only shared lib Mojo/MAX/Rust-era-tools ever touch). — landed `a73e6844`
- [ ] `npu-infer/ffi_bridge.cpp` folds into the same header family (single ABI story).
- [x] Smoke test: a 30-line Mojo program dlopens `libonebit.so`, loads a model, generates tokens — via `OwnedDLHandle` (runtime) and `external_call` (compile-time) variants. — landed `a73e6844` (OwnedDLHandle variant; `external_call` variant still TODO)

## Phase 2 — The Mojo envelope (rewrite, then cut)

Every non-kernel component gets a Mojo twin; the C++/Python original is deleted when the twin passes parity. Pattern already set by the Adrenalin rewrite (M0–M2: toolchain, JSON+sysfs, HTTP+GETs).

Priority order (existing momentum first):

1. **Adrenalin control plane** (M0–M2, from UPDATE 34) — first production Mojo target. — DONE in `AMD-gui` repo (`app/adrenalin.mojo`, single binary, wire-compatible, deployed to strixhalo)
2. **Python tooling → Mojo** (24 files in `tools/`, 5 in `scripts/`): `gguf_to_onebp.py`, `hf_to_onebp.py`, `qwen3_to_onebp.py`, `hf_tokenizer_to_htok.py`, `convert_*_to_gguf.py`, `safetensors_to_onnx_int8.py`… These are the "converters" of the through-line. Mojo native exes, same CLI contract. Delete the `.py` on parity.
   - [x] `tokenizer_json_to_htok` → `tools/tokenizer_json_to_htok.mojo` (byte-identical on qwen2.5-0.5b / qwen3.5-4b / bonsai-1.7b + synthetic escape/gap-id edge cases; engine loader round-trip verified; `convert_model.py` rewired). Supersedes BOTH `tokenizer_json_to_htok.py` and `hf_tokenizer_to_htok.py` — the latter's `<>|` specials heuristic lives on in the GGUF route (`gguf_htok.cpp` → `build_htok_from_gguf`).
   - [x] `qwen3_to_onebp` → `tools/qwen3_to_onebp.mojo` (byte-identical on a synthetic full-path checkpoint incl. padding/flat-group/round-tie cases; real Qwen2.5-0.5B end-to-end: 218 tensors → 343 MB Q4NX in 3 s, engine loader dequant sanity passed; shared `tools/jsonx.mojo` JSON scanner + raw-binary file reader extracted from the htok twin). Supersedes `qwen3_to_onebp.py`; fixes two of its bugs — the `model-*` shard glob (single-shard repos) and the missing `model.` prefix strip (real Qwen3 repos never mapped).
   - [x] `gguf_to_onebp.py` — **CUT, not ported**: the C++ twin `tools/gguf_to_onebp.cpp` (826 lines, v3 header with GGUF metadata, bos/eos/rope/max_seq correct) is the production converter (`batch_convert.sh`, `reconvert_1bp_catalog.sh`) and its tensor data is byte-identical to the .py's. The .py wrote a stale v1 header with wrong bos/eos. `build_1bp.sh` + `build_all_1bp.sh` rewired to the twin; .py deleted. A Mojo port would duplicate 826 tested lines for a compute-adjacent converter the fold already de-Pythoned.
   - [x] `safetensors_to_onnx_int8` → `tools/safetensors_to_onnx_int8.mojo` (INT8 QDQ ONNX for bitnet_decode via a hand-rolled protobuf writer — the onnx package is a Python dep the fold removes). Byte-identical .onnx + config.json on a synthetic qwen2-style checkpoint, and byte-identical external-data output (.onnx + .data) on a big-path fixture (threshold patched for the test; real 2 GB cap restored); onnx.checker validates both. The `.py` couldn't even run without the onnx package installed (PEP 668) — the twin removes the dependency. `convert_model.py` safetensors route rewired.
   - [x] `hf_to_onebp` → `tools/hf_to_onebp.mojo` (Moonshot/Kimi + general transformers; Q4NX/TQ2/MXFP4/F16/F32 quant modes). All-but-offsets byte-identical to the .py on a synthetic moonlight checkpoint across all six quant labels + kimi_k3/kimi_vl/kimi headers; engine gates: Q4NX + TQ2 outputs load + dequantize in NpuOnebpModel (per-row scales — the engine's dequant_tile reads scales[r*groups+g]). `.py` broken bits not replicated: its data offsets start at 256 but the loader treats them as data-relative (files read 256 bytes past every tensor) — twin writes 0-based offsets. Also the `.py` needs numpy with bf16 (its astype crashes on BF16 checkpoints) — the twin handles BF16. `download_moonshot.sh` rewired. Shared jsonx grows the safetensors parser (STensor/parse_safetensors/elem_f32) + config helpers; the qwen3 twin was refactored onto it AND its Q4NX tile was corrected to per-row stats (the earlier "group-wide" parity was a fixture artifact — the .py's min(axis=2) is per-row; re-verified byte-identical).
3. **`tools/train/`** (SFT/RL training scripts) — check against MAX training story; likely cut entirely (Modular owns training).
4. **`npu-infer/rust/`** (187-line Rust FFI binding) → Mojo binding over the same C FFI. **Delete Rust.**
5. **Onebit CLI** (`tools/onebit.cpp`/`onebin.cpp` dispatch: chat/up/serve/config/auth/pull) — the control-plane CLI becomes a Mojo exe; C++ keeps only compute subcommands (`zaya`/`unified`/`vision` server cores, `jarvis` pipeline).
6. **`engine/npu` Python** (13 src + 8 generators files) — audit; generators/validators → Mojo where they're tooling, keep Python only if it's a test harness with no Mojo equivalent yet (`mojo test` exists — migrate).
7. **Repackaging** (`pixi.toml`, `mojoproject.toml`-era none) — pin `mojo==1.0.0`, channels `https://conda.modular.com/max`, `conda-forge`. — [x] `pixi.toml` + `pixi.lock` landed with the htok twin; env verified.

## Phase 3 — Platform fold (1bit in the Modular ecosystem)

- [ ] **Publish Mojo-built tooling as conda packages** on `modular-community` (prefix.dev) via `github.com/modular/modular-community` PRs (rattler-build `recipe.yaml`). First package: the converter set (`1bit-tools`).
- [ ] **Agent skills repo** `github.com/1bit-MONSTER/skills` (agentskills.io SKILL.md format, `modular/skills` pattern): engine runbook, 1bit-format converter skills, NPU bring-up. Zero gate to publish.
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
3. P2.1: Adrenalin M0 (toolchain) — `pixi.toml` with `mojo==1.0.0`. — DONE (pixi.toml + pixi.lock landed; Adrenalin M0–M2 done in `AMD-gui`). Next: P2.2 converters — `gguf_to_onebp.py` / `hf_to_onebp.py` / `qwen3_to_onebp.py` are the remaining big ones.
