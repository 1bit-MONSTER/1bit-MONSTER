# Local roster — complete inventory (2026-09-09)

Machine: Ryzen AI MAX+ 395 (Strix Halo), 32t / 125 GB
NPU: NPU Strix Halo (XDNA 2, 17f0 rev 11, fw 1.1.2.65) · GPU: Radeon 8060S (gfx1151)

## Lanes
1. native-npu — engine/npu/build/npu_engine (reverse-engineered XDNA 2, XRT); reads `.q4nx`
2. npu-flm    — /opt/fastflowlm/bin/flm; reads FLM `*-NPU2/model.q4nx` + xclbins
3. hrx        — HRX bundle b59/b66 (`HRX0`) + hrx-v2 fork (`HRX20`, Q4NX ops); reads `.gguf`
4. vulkan     — ggml-vulkan/RADV (`Vulkan0`); reads standard `.gguf`

## Roster (all files in ~/models, by format)

### Standard GGUF (→ Vulkan; HRX where embedding quant allows)
| Model | File |
|---|---|
| Qwen3-0.6B | Qwen3-0.6B-Q4_K_M.gguf (+ -fused.gguf) |
| Qwen3-1.7B | Qwen3-1.7B-Q4_K_M.gguf |
| Qwen3-4B | Qwen3-4B-Q4_K_M.gguf |
| Qwen2.5-7B-Instruct | qwen2.5-7b-split/qwen2.5-7b-instruct-q4_k_m-*-of-00002.gguf (split) |
| Qwen3-Coder-30B-A3B (MoE) | Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf |
| Qwen3.6-35B-A3B (MoE) | Qwen3.6-35B-A3B-Q8_0.gguf |
| Zaya 8B | zaya-Q4_K.gguf, zaya-Q4_K-fixed.gguf, zaya1-8b-ft-merged[1-7].gguf |

### Q4NX-converted GGUF (→ HRX2 fork only; custom Q4NX tile ops)
glm47-flash, minicpm4-8b, minicpm5-1b, qwen25-05b, qwen25-3b, qwen25-7b,
qwen3-06b, qwen3coder-30b, qwen3next-80b — all `q4nx-converted/<name>-q4nx.gguf`

### Q4NX (→ native NPU; FLM for the FLM-tagged subset)
| Model | File(s) |
|---|---|
| Qwen3-0.6B/1.7B/4B | ~/.config/flm/models/Qwen3-{0.6B,1.7B,4B}-NPU2/model.q4nx |
| Qwen3.6-35B-A3B (MoE) | ~/.config/flm/models/Qwen3.6-35B-A3B-NPU2/model.q4nx |
| Llama-3.2-1B | ~/.config/flm/models/Llama-3.2-1B-NPU2/model.q4nx |
| Zaya 8B | zaya1-8b-fresh.q4nx, zaya1-8b.q4nx |

### Out of scope (other platform)
- vek385-eval/ZAYA1PREVIEW-74B-A4B-Q4_K_M.gguf + ZAYA1-74B-preview.1bp — VEK385
  (Versal AI Edge Gen 2) evaluation artifacts, not Strix Halo targets.
- models/ZAYA1-8B/*.safetensors — source weights (conversion input, not runnable).

## Roster refresh — 2026-09-13 (native-NPU lane)

Supersedes the Q4NX table above for engine coverage. `~/.config/flm/models/` now
holds **18 NPU2 bundles** (13 before 2026-09-13; `flm list` marks the same 18
tags ✅); five were pulled on
2026-09-13 for engine variants that had no weights at all — Gemma3-1B (1.14 GB),
Gemma3-4B (3.51 GB), Qwen3-VL-4B-Instruct (3.07 GB), Qwen3.5-4B (2.56 GB),
Llama-3.1-8B (5.35 GB), 15.6 GB total, each verified by `flm` on arrival.

### Runnable pairs — 14 engine variants (`engine/npu/build/`) ↔ bundle

| Engine variant | Bundle |
|---|---|
| `npu_engine_qwen3_0_6b` | Qwen3-0.6B-NPU2 |
| `npu_engine_qwen3_1_7b` | Qwen3-1.7B-NPU2 |
| `npu_engine_qwen3_4b` | Qwen3-4B-NPU2 |
| `npu_engine_qwen3_5_4b` | Qwen3.5-4B-NPU2 |
| `npu_engine_qwen3_8b` | Qwen3-8B-NPU2 |
| `npu_engine_qwen3_vl_4b` | Qwen3-VL-4B-Instruct-NPU2 |
| `npu_engine_qwen3_6_moe_35b` | Qwen3.6-35B-A3B-NPU2 |
| `npu_engine_llama` | Llama-3.2-1B / 3.2-3B / 3.1-8B-NPU2 |
| `npu_engine_gemma3_1b` | Gemma3-1B-NPU2 |
| `npu_engine_gemma3_4b` | Gemma3-4B-NPU2 |
| `npu_engine_gemma4_e2b` | Gemma4-E2B-IT-NPU2 |
| `npu_engine_gemma4_e4b` | Gemma4-E4B-IT-NPU2 |
| `npu_engine_nanbeige4_1_3b` | Nanbeige4.1-3B-NPU2 |
| `npu_engine_phi4_mini_4b` | Phi4-mini-Instruct-NPU2 |

### Engine variants with no bundle — they build, but cannot run here

`qwen3_14b`, `smollm2_135m`, `zr1`, `deepseek_v4_flash`. No bundle for any of
them exists in FLM's catalog: `/opt/fastflowlm/share/flm/model_list.json` carries
no `zr1`, `smollm` or `qwen3:14b` entry at all, and its only `deepseek` entry is
`deepseek-r1:8b` — an 8B distill, not V4-Flash. The `flm104` install lists the
same 38 tags. These four cannot be measured on this box until upstream ships a
bundle; their presence in `build_npu.sh`'s `MODELS` is not evidence they run.

### Bundles with no engine variant

`LFM2-1.2B-NPU2`, `LFM2-2.6B-NPU2`. The engine has no LFM2 forward pass: the only
LFM2 knowledge in `npu_engine_universal.cpp` is the FLM-delegated
`NPU_FLM_PREFILL` family switch (`NV == 65536 → family = "lfm2"`), so a native run
is family implementation work, not a missing download.

### Dims note (checked 2026-09-13)

A bundle's `config.json` does not have to match the compiled `npu_dims.h` block.
The engine takes H/NC/NH/NKV/HD/IM/NV from the **q4nx manifest** at runtime
(`parse_q4nx_header`), with `config.json` used only when that leaves `cfg`
invalid; `Q_I8R`, `XCLBIN_SUFFIX` and `GU_FUSED` from the dims header are not
referenced in `npu_engine_universal.cpp` at all. A block-vs-config divergence is
therefore not by itself a defect — do not "fix" the header on the strength of
that comparison alone.
