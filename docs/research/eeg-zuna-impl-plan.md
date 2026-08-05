# ZUNA EEG Autoencoder → 1bit Engine — Implementation Plan

Status: **scout + plan complete, verified against real code** (2026-08, branch `feat/eeg-zuna-research`)
Backing research: `docs/research/eeg-zuna-research.md`

This is the output of the scout(engine) → scout(zuna reference) → planner orchestration,
with every file/line claim checked against the repo by the orchestrator.

---

## Goal

Add a `1bit zuna` subcommand that loads ZUNA1.1 (380M bf16) weights, runs the EEG
autoencoder (encoder → 50-step Euler rectified-flow decode → reconstructed EEG), and
serves it over HTTP — validated for numerical parity against the reference `zuna`
Python package.

## Verified repo anchors (orchestrator-checked, not assumed)

- Arch enum: `include/rocm_cpp/bitnet_model.h` `rcpp_arch_t` (0–22) + `rcpp_arch_from_string()`
- Format arch: `include/onebp_format.h` `OnebpArch` (SD/LTX diffusion block already at 30–35) + `ONEBP_F16 = 4` quant
- 1BP raw loader: `include/onebp_loader.h` `OnebpModel` (mmap, `tensor_data()`), `src/onebp_model.cpp`
- F16 + Q4NX read pattern: `src/vision_encoder.cpp:737-838` (the template to copy)
- Existing diffusion arch: `src/diffusion_bridge.cpp` — wraps stable-diffusion.cpp struct C API (non-AR example)
- 2D RoPE: `src/prim_kernels.hip:239-267`; 3D/4D RoPE CPU ref: `src/vision_encoder.cpp:965-989` `mage_vit_rope_one`
- Subcommand dispatch: `tools/onebin.cpp:40-110` (`vision_server` template)
- Backend interface: `src/backend.h` `Backend` (token/AR-oriented — ZUNA will NOT use it; it's a standalone forward)
- Server stack: cpp-httplib + nlohmann_json already linked (`CMakeLists.txt:239-251`)

## De-risked unknowns (probed before building)

1. **Reference availability:** `zuna==1.1.6` installs + imports on Python 3.11 (`/tmp/zuvenv2`).
   torch 2.13 CPU available for golden-trace generation.
2. **1BP F16 pass-through:** `vision_encoder.cpp` already reads F16 tensors from 1BP
   (`f16[i]<<16` → f32). No new format work; quant isn't needed (380M is small).
3. **Lazy weight conversion:** a Python script dumps safetensors → raw `.bin` (no C++
   safetensors parser). Matches plan.

## Build phases (CPU-correctness first)

- **Phase 1 — registration:** `RCPP_ARCH_ZUNA=23`, `ONEBP_ZUNA=60`, `model_discovery` case, `1bit zuna` dispatch.
- **Phase 2 — weights + loader:** `src/zuna.cpp` structs (`ZunaEncoderWeights`, `ZunaDecoderWeights`), `load_1bp`.
- **Phase 3 — CPU forward:** `zuna_encode`, `zuna_rope4d_apply`, `zuna_sample` (50 Euler steps), `zuna_main` CLI.
- **Phase 4 — converter:** `tools/export_zuna_1bp.py` (safetensors → raw F16 1BP).
- **Phase 5 — reference parity harness:** `tools/gen_zuna_traces.py` golden enc_out + reconstruction; `tests/check_zuna_parity.py` compares 1bit output.
- **Phase 6 (deferred):** HTTP server, GPU 4D RoPE kernel, preprocessing pipeline (client-side tokenization in v1).

## Design decisions

- **F16, no quant.** Diffusion latents are quantization-sensitive; model too small to benefit.
- **Standalone forward, NOT the AR `Backend` interface.** ZUNA is a diffusion autoencoder;
  shoehorning it into the token decode loop is wrong. It gets its own `zuna_main` + loader.
- **Client-side tokenization in v1.** The server takes `{tokens, tok_idx}`; raw-EEG preprocessing
  (resample/notch/reference/z-score, 10-20 montage) is a later, separable phase.
- **CLI parity first.** `1bit zuna --model zuna.1bp --in raw.bin --out recon.bin` for testing;
  HTTP server deferred. Fewer moving parts to validate correctness.

## Testing / parity

1. `gen_zuna_traces.py`: toy EEG (8ch × 256 samples, 10 Hz sine on ch0) → reference encoder latent + 50-step reconstruction.
2. Export weights to `zuna.1bp` (F16). Run `1bit zuna` on the same tokens/tok_idx.
3. `check_zuna_parity.py`: encoder RMS < 1e-4; reconstruction RMS < 1e-3 (bf16 vs fp32).
4. If the diffusion loop matches with same RNG seed, tighten to < 1e-4.
