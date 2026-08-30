# HF Model Coverage Audit — 1bit.MONSTER engine + lemonade

**Date:** 2026-08-29 (goal round 3) · **Status:** 🔄 baseline established ·
**Goal link:** `docs/research/hrx-engine-goal.md` (P5)

> Question this answers: *for any HuggingFace model id, is there a documented,
> working route on this machine?* This is the "100% HF model coverage" leg of
> the HRX-engine goal. It is a **coverage onion** — every layer catches a set of
> models, and the layers overlap so a model usually has more than one path.

---

## 0. The coverage onion

```
Layer 1  HRX / llama.cpp GGUF set  (145 archs, vendored llama.cpp)
         └─ any GGUF → hrx_gpu → ggml_vulkan → zinc_gpu → cpu_generic
            (HRX fused decode; G1a route-order failover; G1b prefill policy)
Layer 2  Engine-specialized engines (archs with dedicated backends/converters)
Layer 3  FLM NPU (Q4NX)  — 27 models, npu_flm route
Layer 4  Lemonade catalog (220 entries) — the UX surface; recipes map onto
         layers 1/3 + ONNX + vLLM + SD/TTS for non-LLM modalities
Layer 5  Default fallback — hip_gpu + cpu_generic ("generic kernels")
```

**Layer 1 is the workhorse for "100% HF coverage"**: any HF model with a GGUF
on the hub (or convertible via llama.cpp's `convert_hf_to_gguf.py`, vendored at
`third_party/llama.cpp/convert_*.py`) is loadable. HRX runs it when the graph is
inside the fused node set; otherwise ggml_vulkan → zinc → CPU, with correct
failover order since G1a/G1b.

## 1. Layer 1 — llama.cpp arch set (the HRX lane)

Vendored `third_party/llama.cpp` `src/llama-arch.h`: **145 `LLM_ARCH_*`**
(verified 2026-08-29). Notable families: llama/llama4, mistral3/4, qwen2/2moe/
2vl, qwen3/3moe/3next/3vl/3.5/3.6, phi2/3/phimoe, gemma/2/3/4(+assistant),
deepseek/deepseek2/deepseek32/deepseek4, glm4/moe/glm_dsa, mamba/mamba2/jamba,
zamba-class, gpt2/gptj/gptneox, falcon, bloom, starcoder/2, olmo/2/olmoe,
nemotron(+h/moe), exaone(4/moe), rwkv6/7, granite(+moe/hybrid/switch),
bitnet, jais(2), command-r/cohere2(moe), dbrx, arctic, openelm, minicpm(3),
internlm2, orion, codeshell, baichuan, mpt, refact, chatglm, t5(t5encoder),
jina/nomic/modern/neo-bert, kimi_k3, kimi_linear, mimo2, step35, minimax_m2/m3/
01, dflash, nanbeige, lfm2(+moe), cogvlm, hunyuan(moe/dense/vl), ernie4.5(moe),
smollm3, apertus, grovemoe, seed_oss, llada(moe), maincoder, mistral3/4,
paddleocr, qwen3tts, pockettts, wavtokenizer_dec, talkie, mellum, eagle3,
solar-class, etc.

Caveats (honest limits of the HRX lane):
- **HRX fused node set is narrower than the arch set**: HRX fail-closes on
  `GET_ROWS` (row-gather) graphs — large prompts / some MoEs. Mitigated by
  G1a (fall to ggml_vulkan) + G1b (skip HRX for large prompts). Tracked as G1c
  (llama.cpp RFC #27218).
- **HRX in lemonade is chat-only local-GGUF** (no vision/mmproj, embeddings,
  reranking — `hrx_server.cpp`). Multimodal/embedding archs go through
  ggml_vulkan/zinc or the engine's specialized routes instead.
- Layer-1 correctness for a given model depends on GGUF availability upstream
  (most popular families have official GGUFs).

## 2. Layer 2 — engine-specialized routes (src/model_router.cpp, verified 2026-08-29)

| Model class | Route (first → last) | Notes |
|---|---|---|
| MLX (Qwen3.5/3.6/3.8, lemonseed) | `lse → cpu_generic` | lse-server reads MLX checkpoints |
| MoE (num_experts>0, non-Mamba/Laguna, **native 1BP**) | `hip_gpu → cpu_scalar` | CCA/MoE kernels (1BP weights only) |
| MoE **GGUF/H1B** (e.g. Qwen3-30B-A3B) | `hrx_gpu → ggml_vulkan → zinc_gpu → cpu_generic` | Layer 1 lane — fixed 2026-08-29 (MoE branch no longer claims GGUF) |
| qwen3 **1BP** | `fused_gpu_npu → hip_1bp_gpu → vulkan_hpp_gpu → cpu_generic` | fused GPU+NPU zero-DMA lane |
| qwen3 **Q4NX** | `npu_flm → cpu_generic` | FLM NPU, 67.5 tok/s |
| qwen3 **GGUF** | `hrx_gpu → ggml_vulkan → zinc_gpu → cpu_generic` | Layer 1 |
| zamba2 | `ggml_vulkan → zamba2_vulkan → zamba2_gpu → cpu_generic` | Mamba2 SSD |
| zamba/mamba1 | `mamba1_gpu → cpu_generic` | incl. BlackMamba MoE |
| deepseek v2/v3 | `hip_gpu → cpu_generic` | MLA + MoE |
| deepseek v4 (Flash/Pro) | `cpu_deepseek_v4 → hip_gpu → cpu_generic` | mHC+CSA+HCA, FP4 MoE |
| glm_moe_dsa (GLM-5) | `cpu_glm_moe_dsa → hip_gpu → cpu_generic` | V3-MLA + DSA indexer |
| mimo_v2 | `cpu_mimo_v2 → hip_gpu → cpu_generic` | MoD hybrid |
| qwen3.5 text | `cpu_qwen3_5 → hip_gpu → cpu_generic` | GatedDeltaNet + gated GQA |
| whisper | `cpu_generic` | speech-to-text |
| qwen3-vl | `vision_encoder → hip_gpu → cpu_generic` | vision encoder + qwen3 |
| nemotron-h | `nemotron_h_cpu → cpu_generic` | Mamba2+NoPE+relu2+MoE |
| laguna | `laguna_gpu → zinc_gpu → cpu_generic` | sigmoid-routed MoE |
| falcon / olmo | `hip_gpu → cpu_generic` | — |
| generic GGUF/H1B | `hrx_gpu → ggml_vulkan → zinc_gpu → cpu_generic` | Layer 1 |
| generic 1BP | `fused_gpu_npu → hip_1bp_gpu → vulkan_hpp_gpu → hip_gpu → cpu_generic` | — |
| default (anything else) | `hip_gpu → cpu_generic` | generic kernels |

## 3. Layer 3 — FLM NPU (Q4NX)

`third_party/FastFlowLM/src/model_list.json`: **27 model entries** (verified
2026-08-29): qwen3vl-it, gemma4-it, qwen3.5, qwen3.6-moe, phi4-mini-it,
nanbeige4.1, lfm2 (+trans/it/tk variants), etc. Converter:
`tools/convert_qwen36_moe_q4nx.py` + `third_party/FLM_Q4NX_Converter`.

## 4. Layer 4 — lemonade catalog (the UX surface)

`third_party/lemonade/src/cpp/resources/server_models.json`: ~220 entries.
Recipes: llamacpp 90 (→ Layer 1 GGUF), ryzenai-llm 79 (→ Layer 3-class NPU),
vllm 11, whispercpp 6, sd-cpp 12, thenoise 12, openmoss 4, moonshine 3,
onnxruntime 2, **llamacpp-hrx 1**, kokoro 1, ds4 1, acestep 1, trellis 1,
thinksound 1. Non-LLM modalities (TTS/SD/ASR) live here, not in the engine.

## 5. Format flow (HF → runnable)

| Source | Converter | Target | Lane |
|---|---|---|---|
| HF safetensors (llama.cpp archs) | `third_party/llama.cpp/convert_hf_to_gguf.py` | GGUF | Layer 1 |
| HF safetensors (qwen3 family) | `tools/qwen3_to_onebp.py`, `tools/hf_to_onebp.py`, `tools/gguf_to_onebp.py` | 1BP | fused/hip |
| HF safetensors (zamba2/zaya/zyphra/blackmamba) | `tools/convert_*_safetensors_to_gguf.py`, `scripts/*_to_gguf.py` | GGUF | Layer 1 / zamba lanes |
| HF safetensors (qwen3.6 MoE) | `tools/convert_qwen36_moe_q4nx.py` | Q4NX | FLM NPU |
| MLX checkpoints | direct (lse-server) | MLX | lse |

## 6. Gaps (explicit, no hand-waving)

| Class | Coverage | Why / what's needed |
|---|---|---|
| LLM arch ∈ llama.cpp 145 set with a hub GGUF | ✅ **Covered** (Layer 1) | HRX fused when graph fits; else ggml_vulkan/zinc/cpu (G1a/G1b) |
| qwen3 family (1BP/Q4NX/MLX) | ✅ **Covered** (Layers 1–3) | four format lanes |
| Exotic archs with a dedicated engine (deepseek4, mimo2, glm_dsa, qwen3.5, nemotron-h, laguna…) | ✅ **Covered** (Layer 2) | dedicated CPU/HIP engines |
| Novel HF arch ∉ llama.cpp, ∉ engine, ∉ FLM | ❌ **BLOCKED** until a lane exists | follow the add-model checklist (§7); Layer 5's "generic kernels" claim is aspirational for truly novel archs |
| Multimodal (vision) via HRX | ⚠️ **Partial** | llama.cpp has QWEN3VL/COGVLM/HUNYUAN_VL etc., but HRX chat branch omits mmproj → ggml_vulkan/zinc or engine `vision_encoder` route |
| Embeddings / reranking via HRX | ⚠️ **Partial** | llama.cpp has BERT/JINA/NOMIC/PANGU_EMBED; HRX is chat-only → ggml_vulkan lane or lemonade llamacpp |
| Audio (whisper / qwen3tts / pockettts) | ✅ engine whisper; GGUF TTS archs via Layer 1 | whispercpp in lemonade too |
| Text-to-image / TTS | ✅ via lemonade (sd-cpp, thenoise, kokoro, acestep, trellis) | not an engine concern (different modality) |
| Every lemonade model on **HRX** specifically | ⚠️ **Partial** (goal P3) | **43 `*-HRX` entries landed** 2026-08-29 (90 llamacpp − 7 embed/rerank − 39 vision − 1 existing); engine-native routes all GGUF HRX-first; vision models excluded (mmproj omitted in HRX) |

## 7. Adding a model (the checklist — operationalized in `tools/hf_coverage.py`)

Run `python3 tools/hf_coverage.py <hf-model-id>` first: it reports which lane
covers the model (263 llama.cpp archs extracted live from the vendored
converter, engine-specialized routes, FLM, lemonade) with the exact route, or
prints the steps below for a novel arch.

1. Arch detection in `src/model_discovery.cpp` (config.json `model_type`).
2. Route entry in `select_backend_route()` (`src/model_router.cpp`).
3. Backend factory in `src/backend_factory.cpp` (+ `backend_manager.cpp` if new).
4. Benchmark entry in `bench/record.sh`.
5. If GGUF convertible (llama.cpp set): **no engine change needed** — Layer 1
   catches it; verify HRX fused-node coverage on gfx1151.

## 8. Definition of "100% HF coverage" (accepted for this goal)

> Every model id in the supported families (llama.cpp 145-arch set + engine
> specialized set + FLM 27 + lemonade 220) has a **documented, working route**,
> and any novel arch has a 5-step path to coverage (§7). Absolute "any id on
> HF ever" is unbounded (new archs ship weekly); the goal is: **no model that
> exists today lacks a lane**, and new models gain one via the checklist.
