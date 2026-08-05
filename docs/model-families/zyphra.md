# Zyphra — Complete End-to-End Pipeline

Zyphra's portfolio spans the entire AI stack: **EEG → LLM (dense, MoE, Mamba) → TTS → voice cloning**. 1bit supports all of it — 11 models in 1BP format, plus EEG and TTS pipelines documented for ecosystem completeness. This is the family the engine was tuned against, and the one that powers the [JARVIS pipeline](../jarvis.md).

**Legend:** 🧠 LLM · 👁️ vision · 🗣️ voice · 🧬 EEG · 🏁 end-to-end validated

## Models

| Model | Params | 1BP Size | Backend(s) | Pipeline | Perf |
|-------|:------:|:--------:|------------|:--------:|:----:|
| **ZAYA1-8B** | 8.8B | 6.6 GB¹ | ZINC / HIP / NPU | 🧠🗣️ | 64 tok/s HIP |
| **ZAYA1-74B-preview** | 74B | 739 MB² | ZINC / HIP | 🧠🗣️ | 17.9 tok/s HIP (prelim) |
| **ZAYA1-VL-8B** | 8.8B | — | ZINC (vision) | 👁️🧠🗣️ | — |
| **ZR1-1.5B** | 1.5B | 781 MB | ZINC / NPU | 🧠🗣️ | 26 tok/s ZINC |
| **BlackMamba-1.5B** | 1.5B | 970 MB | Mamba1 HIP | 🧠🗣️ | **79.4 tok/s** 🏁 |
| **BlackMamba-2.8B** | 2.8B | 1.8 GB | Mamba1 HIP | 🧠🗣️ | 46.0 tok/s 🏁 |
| **Zamba2-1.2B-v2** | 1.2B | 1.1 GB | ZINC ✅ / NPU | 🧠 | 30 tok/s ZINC |
| **Zamba2-2.7B-v2** | 2.7B | 2.4 GB | ZINC ✅ / NPU | 🧠 | ~30 tok/s ZINC |
| **Zamba2-7B-v2** | 7B | 6.6 GB | ZINC ✅ / NPU | 🧠 | — |
| **Zamba-7B-v1** | 7B | 4.3 GB | Mamba1 HIP | 🧠 | — |

> ¹ ZAYA1-8B 1BP is ~6.6 GB full-weight — the 149 MB entry on HF is MoE-expert-stripped; use the complete file. ² 1BP is extremely small (TQ2 ternary quantization on a 74B MoE).

## Architectures in this family

- **Zaya1 — MoE + CCA.** Zyphra MoE architecture with Cross-Channel Attention + MoE FFN. Flagship 1BP ternary model. Tile8 GEMV (28-layer, Zaya1-8B shaped) measured at 77 tok/s on ROCm HIP. Native TQ2 ternary on NPU via `--native-tq2` (`npu_ternaryd.cpp`, `gemm_generate_sequence_tq2()`) — 4× less DDR traffic than the INT8 bridge. CPU AVX-512 portable path ~2.5 tok/s (8B-shaped).
- **ZR1 — dense reasoning.** Reasoning-tuned dense transformer (Qwen2 architecture). End-to-end validated ~26 tok/s on Vulkan ZINC; 1BP conversion complete. Builds via the Qwen3-0.6B xclbin stanzas (same tile template).
- **BlackMamba — Mamba1 + top-1 MoE.** No attention mechanism — alternating SSM scan and MoE FFN per layer. The engine's fastest end-to-end family (79.4 tok/s at 1.5B). SSM scan via `kernel/ssm_selective_scan.cc`.
- **Zamba2 — Mamba2 hybrid.** Mamba2 SSM layers with sparse attention every 6 layers. ~30 tok/s at 2.7B on Vulkan ZINC; Mamba2 decode block benchmarked at 1270 tok/s on ROCm HIP. NPU AIE2 selective-scan kernel (per-head d_state=64 vectorized, 32 heads/tile).
- **Zamba — Mamba1 hybrid.** Original Zamba-7B-v1: Mamba1 SSM + shared attention layers. GGUF through ROCm HIP.

## Full pipeline depth

- 🧠 **LLM** — Zaya (MoE+CCA), ZR1 (dense reasoning), BlackMamba (Mamba1+MoE), Zamba (Mamba1/2 hybrid)
- 👁️ **Vision** — ZAYA1-VL-8B (built-in vision encoder)
- 🗣️ **Voice** — voice cloning pipeline (RVQ-VAE codec + QLoRA adapter + ONNX decoder + streaming + persona), see [JARVIS pipeline](../jarvis.md)
- 🧬 **EEG** — ZUNA1.1 / ZUNA (diffusion autoencoder, not 1BP) → [Zyphra/ZUNA1.1](https://huggingface.co/Zyphra/ZUNA1.1)
- 🗣️ **TTS** — Zonos-v0.1-hybrid / ZONOS2 (neural audio codec + MoE, not 1BP) → [Zyphra/Zonos-v0.1-hybrid](https://huggingface.co/Zyphra/Zonos-v0.1-hybrid)

## 1BP catalog

| Model | 1BP Size | Verified |
|-------|:--------:|:--------:|
| ZAYA1-8B, ZAYA1-74B-preview | 469 MB / 4.2 GB | ✅ loads |
| ZR1-1.5B | 373 MB | hosted |
| Zamba2-1.2B / 2.7B / 7B v2 | 375 MB – 1.8 GB | hosted |
| BlackMamba-1.5B / 2.8B | 1.0 / 1.9 GB | ✅ loads |

**See also:** [full model support detail](../wiki/models.md) · [benchmarks SSOT](../wiki/performance.md) · [all families](README.md)
