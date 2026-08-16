# Model Families

1bit auto-detects **32 architecture tokens / 120+ HF arch strings** from GGUF/1BP/safetensors headers — no config files, no per-model glue code. Point the engine at a model and run. Coverage is measured against a full HF census: **317,310 / 317,310 arch-bearing text-gen checkpoints (100.00%) map to an engine token** — see the [models SSOT](../wiki/models.md).

Every family below has its **own page** with a full breakdown: parameter sizes, 1BP file size, supported backends, and real measured performance. Numbers trace back to the [performance SSOT](../wiki/performance.md); model-support detail traces back to [wiki/models.md](../wiki/models.md).

**Backend legend:** 🟢 supported & validated · 🟡 functional, perf pending · 🔴 not yet · 🔬 experimental
**Pipeline legend:** 🧠 LLM · 👁️ vision · 🗣️ voice/TTS · 🧬 EEG · 🏁 end-to-end validated

---

## Featured: Zyphra

The flagship ecosystem — a complete stack from EEG → LLM (dense, MoE, Mamba) → TTS → voice cloning, all running on the same binary.

**→ [Zyphra family page](zyphra.md)**

---

## All families

| Family | Type | Sizes | Page |
|--------|------|-------|------|
| **Zyphra** (Zaya1, ZR1, BlackMamba, Zamba/Zamba2) | MoE · SSM · dense | 1.2B–74B | [zyphra.md](zyphra.md) |
| **Qwen** (Qwen2/2.5/3/3.5/3.6, Qwen-VL) | Dense · MoE · VL | 0.5B–72B | [qwen.md](qwen.md) |
| **Llama** (3.1 / 3.2, TinyLlama) | Dense | 1B–8B | [llama.md](llama.md) |
| **Mistral** (Mistral, Ministral, Pixtral) | Dense · VL | 7B–12B | [mistral.md](mistral.md) |
| **Gemma** (Gemma 3/4, MedGemma, Embedding-Gemma) | Dense · embedding | 300M–4B | [gemma.md](gemma.md) |
| **Phi** (Phi-3 / 3.5 / 4-mini) | Dense | 3.8B | [phi.md](phi.md) |
| **Falcon** (Falcon3) | Dense (parallel attn) | 1B–40B | [falcon.md](falcon.md) |
| **OLMo** (OLMo-2) | Dense (LayerNorm) | 7B–13B | [olmo.md](olmo.md) |
| **Granite** (Granite-3.2) | Dense | 2B–8B | [granite.md](granite.md) |
| **SmolLM** (SmolLM2) | Dense | 135M–1.7B | [smollm.md](smollm.md) |
| **DeepSeek** (V2/V3/R1 + distills) | MoE (MLA) | 8B–671B | [deepseek.md](deepseek.md) |
| **GPT-OSS** | MoE | 20B | [gpt-oss.md](gpt-oss.md) |
| **Laguna** (Poolside) | Dense · MoE | 3B–7B | [laguna.md](laguna.md) |
| **Moonshot Kimi** (Moonlight, Kimi-VL) | Gated MLA MoE | 16B (3B active) | [kimi.md](kimi.md) |
| **BitNet / Bonsai** (Deepgrove) | Ternary-native (TQ2) | 1.7B–27B | [bitnet-bonsai.md](bitnet-bonsai.md) |
| **Whisper** | Speech-to-text | V3 Turbo | [whisper.md](whisper.md) |

> The `Mage-ViT` / `Mage-VL` vision path and Whisper speech path together feed the [JARVIS pipeline](../jarvis.md).

## 1BP format note

Dense (non-ternary-trained) models use **Q4NX** 4-bit; only ternary-native checkpoints (BitNet/Bonsai) use **TQ2** 2-bit. Converting a dense model to TQ2 is quality-destructive — see the [1BP format policy](../wiki/models.md#1bp-format-policy-2026-07-31-verdict-ppl-measured).
