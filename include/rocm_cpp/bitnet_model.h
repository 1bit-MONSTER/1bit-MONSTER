#pragma once
#ifndef ROCM_CPP_BITNET_MODEL_H
#define ROCM_CPP_BITNET_MODEL_H

#ifndef ROCM_CPP_NO_SHERRY
#include "rocm_cpp/ck_gemm.h"
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define H1B_FLAG_HADAMARD_ROTATED 0x1u
#define H1B_FLAG_SHERRY_FP16      0x2u
#define H1B_FLAG_BONSAI_Q1        0x4u
#define H1B_FLAG_BONSAI_TQ2       0x8u
#define H1B_FLAG_BLOCK_SCALED     0x10u

typedef enum {
    RCPP_WEIGHT_FORMAT_HALO_V2    = 0,
    RCPP_WEIGHT_FORMAT_SHERRY_I8  = 1,
    RCPP_WEIGHT_FORMAT_TQ1        = 2,
    RCPP_WEIGHT_FORMAT_SHERRY_FP16 = 3,
    RCPP_WEIGHT_FORMAT_BONSAI_Q1  = 4,
    RCPP_WEIGHT_FORMAT_BONSAI_TQ2 = 5,
    RCPP_WEIGHT_FORMAT_WMMA_I8    = 6,
    RCPP_WEIGHT_FORMAT_BLOCK_SCALED_TERNARY = 7,
    RCPP_WEIGHT_FORMAT_Q1_0_BINARY  = 8,   // 1-bit binary (Q1_0, 128-block, fp16 scale + sign bits)
    RCPP_WEIGHT_FORMAT_TQ2_0_LLAMA  = 9,   // llama.cpp TQ2_0 native (2.0625 bpw, 256-block)
    RCPP_WEIGHT_FORMAT_TQ1_0_LLAMA  = 10,  // llama.cpp TQ1_0 native (1.6875 bpw, 256-block)
} rcpp_weight_format_t;

typedef enum {
    RCPP_ARCH_BITNET  = 0,
    RCPP_ARCH_QWEN3   = 1,
    RCPP_ARCH_LLAMA   = 2,
    RCPP_ARCH_MISTRAL = 3,
    RCPP_ARCH_QWEN2   = 4,
    RCPP_ARCH_GEMMA   = 5,
    RCPP_ARCH_PHI     = 6,
    RCPP_ARCH_ZAMBA2  = 7,
    RCPP_ARCH_ZAMBA   = 8,   // Zamba-7B-v1 (Mamba1 + shared attn)
    RCPP_ARCH_MAMBA   = 9,   // BlackMamba (Mamba1 + MoE)
    RCPP_ARCH_LAGUNA  = 10,
    RCPP_ARCH_FALCON  = 11,  // Falcon (tiiuae) — parallel attn+ffn, MQA
    RCPP_ARCH_OLMO    = 12,  // OLMo (AI2) — LayerNorm, no RoPE
    RCPP_ARCH_ZAYA    = 13,  // Zaya MoE (Zyphra — MoE FFN with CCA attention)
    RCPP_ARCH_QWEN2VL = 14,  // Qwen2-VL (vision-language)
    RCPP_ARCH_WHISPER  = 15,  // OpenAI Whisper (speech-to-text)
    RCPP_ARCH_DEEPSEEK = 16,  // DeepSeek V2/V3/R1 — MoE with Multi-Head Latent Attention
    RCPP_ARCH_QWEN3VL  = 17,  // Qwen3-VL (vision-language, Qwen3 text decoder)
    RCPP_ARCH_KIMI_K3  = 18,  // Moonshot Kimi K3 — 2.8T MoE with KDA + Gated MLA + LatentMoE
    RCPP_ARCH_MOONLIGHT = 19, // Moonshot Moonlight-16B-A3B — Gated MLA MoE
    RCPP_ARCH_KIMI_VL  = 20,  // Moonshot Kimi-VL — Moonlight + MoonViT vision encoder
    RCPP_ARCH_QWEN35   = 21,  // Qwen3.5 Gate-Delta Net — fused QKV, SSM path, GDN attention
    RCPP_ARCH_DEEPSEEK_V4 = 22, // DeepSeek V4 Flash/Pro — mHC residual, CSA+HCA hybrid attn, FP4 MoE
    RCPP_ARCH_GPT2 = 23,    // GPT-2 — learned pos embeddings, LN weight+bias, no RoPE, no-gate gelu FFN
    RCPP_ARCH_GPTNEOX = 24, // GPT-NeoX/Pythia — parallel attn+FFN, LN weight+bias, fused qkv, no-gate gelu FFN
    RCPP_ARCH_OPT = 25,     // OPT — learned positions, LN weight+bias, biases everywhere, no-gate RELU FFN
    RCPP_ARCH_GPTNEO = 26,  // GPT-Neo — gpt2-style names, LN+bias, learned wte/wpe, no-gate gelu_new FFN, windowed attn (>256t)
    RCPP_ARCH_CODEGEN = 27, // CodeGen — fused qkv, partial rotary (rotary_dim), LN+bias, no-gate gelu_new FFN
    RCPP_ARCH_GPTJ = 28,    // GPT-J — separate qkv, adjacent partial rotary (rotary_dim), LN+bias, gelu_new
    RCPP_ARCH_GPTOSS = 29,  // GPT-OSS — MXFP4 packed MoE (FP4 blocks+scales, interleaved gate/up), YARN rope, attention sinks, head_dim 64
    RCPP_ARCH_STEP1 = 30,   // Step1 (StepLaw / stepfun Step-Audio) — dense llama-layout, sqrt-ALiBi (no RoPE), num_attention_groups
    RCPP_ARCH_BLOOM = 31,    // Bloom — fused qkv, LayerNorm w/bias, sequential + post_attn_norm, gelu_new, LINEAR ALiBi, embed LN, tied lm_head
    RCPP_ARCH_LFM2 = 32,    // Liquid LFM2/LFM2.5 — conv+attention hybrid: depthwise causal conv1d blocks + full-attention blocks, per-head QK-norm, tied lm_head
    RCPP_ARCH_NANOCHAT = 33, // NanoChat — gpt2-skeleton: unweighted RMSNorm, relu^2 MLP, adjacent-pair RoPE, logit softcap
    RCPP_ARCH_NEMOTRONH = 34, // Nemotron-H — Mamba-2 + NoPE GQA + relu2 MLP + sigmoid MoE hybrid
    RCPP_ARCH_MINIMAXM2 = 35, // MiniMax-M2 — GQA + single flattened q/k RMSNorm + partial rope + sigmoid MoE
    RCPP_ARCH_COHERE2 = 36,  // Cohere2 — parallel attn+FFN, mean-centered LayerNorm, adjacent-pair rope, SWA
    RCPP_ARCH_FALCONH1 = 37, // Falcon-H1 — Mamba-2 SSM + GQA attention + MuP multipliers
    RCPP_ARCH_RWKV = 38,    // RWKV-4/5/6 — linear-attention WKV recurrence + channel mixing
    RCPP_ARCH_GRANITEMOEHYBRID = 39, // GraniteMoeHybrid — Mamba-2 + NoPE GQA + top-k MoE + shared MLP
    RCPP_ARCH_LFM2MOE = 40,  // LFM2-MoE — ShortConv conv1d + GQA + dense-then-MoE
    RCPP_ARCH_HYV3 = 41,    // HY-V3 — GQA + q/k RMSNorm + dense/MoE
    RCPP_ARCH_AFMOE = 42,   // AfMoE — dual-norm GQA + sigmoid-gated sliding attn + shared-expert MoE
    RCPP_ARCH_ERNIE45MOE = 43, // Ernie4.5-MoE — GQA + softmax-router MoE + shared experts
    RCPP_ARCH_MELLUM = 44,  // Mellum — GQA + q/k RMSNorm + per-layer-type rope + dense/MoE
    RCPP_ARCH_PHIMOE = 45,  // PhiMoE — GQA + LayerNorm + sparsemixer MoE
    RCPP_ARCH_MINIMAX = 46, // MiniMax — lightning linear attn + GQA + MoE
    RCPP_ARCH_COHERE2MOE = 47, // Cohere2Moe — parallel GQA + dense/MoE + mean-centered LN
    RCPP_ARCH_EXAONEMOE = 48, // ExaoneMoe — GQA + q/k RMSNorm + group-limited MoE + shared experts
    RCPP_ARCH_FALCONMAMBA = 49, // FalconMamba — Mamba1 SSM + RMSNorm on B/C/dt
    RCPP_ARCH_JETMOE = 50,   // JetMoE — Mixture of Attention + MoE FFN
    RCPP_ARCH_QWEN3NEXT = 51, // Qwen3-Next — GatedDeltaNet linear attention + full attn + MoE
    RCPP_ARCH_PICO = 52,     // PicoDecoderHF — llama-layout with adjacent-pair RoPE (view_as_complex)
    RCPP_ARCH_DYNAMICALIBI = 53, // DynamicAlibiForCausalLM — llama-skeleton + LINEAR ALiBi (static at inference) + fused gate_up swish MLP

    // ── 2026-08-15 census pass-3: new families (registry tokens; engine
    // backends land in the bring-up deck — generic path loads llama-layout
    // members, others abort loudly on tensor mismatch until then) ──
    RCPP_ARCH_LLAMA4 = 54,           // Llama4ForCausalLM — llama-layout MoE, 16E, YARN, shared expert
    RCPP_ARCH_JAIS = 55,             // JAISLMHeadModel — gpt2-ish layout (n_embd keys, swiglu)
    RCPP_ARCH_DYNAMICFORGETTING = 56, // DynamicForgettingForCausalLM
    RCPP_ARCH_DYNAMICSLIDINGWINDOW = 57, // DynamicSlidingWindowForCausalLM
    RCPP_ARCH_KORMO = 58,            // KORMoForCausalLM (Korean, MTP variant)
    RCPP_ARCH_RWKV7 = 59,            // RWKV-7 Goose — data-dependent recurrence (NOT the 4/5/6 engine)
    RCPP_ARCH_CHATGLM = 60,          // ChatGLMModel/ChatGLMForConditionalGeneration (old GLM prefix-LM)
    RCPP_ARCH_SARVAM = 61,           // SarvamMoE/SarvamMLA
    RCPP_ARCH_RAVEN = 62,            // RavenForCausalLM (huginn)
    RCPP_ARCH_TALKIE = 63,           // TalkieForCausalLM
    RCPP_ARCH_LLADA2 = 64,           // LLaDA2MoeModelLM
    RCPP_ARCH_LOOPLM = 65,           // LoopLMForCausalLM
    RCPP_ARCH_STEP3P5 = 66,          // Step3p5ForCausalLM
    RCPP_ARCH_DAISY = 67,            // DaisyForCausalLM
    RCPP_ARCH_MULTISCALE = 68,       // MultiScaleForCausalLM
    RCPP_ARCH_SKIPMIDDLE = 69,       // SkipMiddleForCausalLM
    RCPP_ARCH_MOTIF = 70,            // MotifForCausalLM (poly_norm)
    RCPP_ARCH_QUASAR = 71,           // QuasarForCausalLM
    RCPP_ARCH_HGRN = 72,             // HGRNForCausalLM
    RCPP_ARCH_RETNET = 73,           // RetNetForCausalLM
    RCPP_ARCH_CUBELM = 74,           // CubeLM
    RCPP_ARCH_RECURRENTGEMMA = 75,   // RecurrentGemmaForCausalLM (Griffin)
    RCPP_ARCH_LIGHTNINGTRANSFORMER = 76, // LightningTransformerModel
    RCPP_ARCH_SPIKEWHALE = 77,       // SpikeWhaleLM
    RCPP_ARCH_STL = 78,              // STLDec16
    RCPP_ARCH_XPERTGPT = 79,         // XpertGPT
    RCPP_ARCH_YATGPT = 80,           // YatNMN-GPT
    RCPP_ARCH_CENO = 81,             // Ceno
    RCPP_ARCH_FIMMY = 82,            // Fimmy
    RCPP_ARCH_HYENADNA = 83,         // HyenaDNA (hyena SSM)
    RCPP_ARCH_LLAMAMOE = 84,         // LlamaMoEForCausalLM (mlp.calculator.experts layout)
    RCPP_ARCH_MODERNBERTDECODER = 85, // ModernBERT-decoder
    RCPP_ARCH_ORKHON = 86,           // Orkhon
    RCPP_ARCH_ROFORMER = 87,         // RoFormer
    RCPP_ARCH_STRIPEDHYENA = 88,     // StripedHyena (SSM)
    RCPP_ARCH_ARGONNE = 89,          // Argonne2
    RCPP_ARCH_EMO = 90,              // Emo
    RCPP_ARCH_FORGETTINGTRANSFORMER = 91, // ForgettingTransformer
    RCPP_ARCH_GPTBERT = 92,          // GPT-BERT
    RCPP_ARCH_GPTJXMOE = 93,         // GPT-JX-MoE
    RCPP_ARCH_KEURALMOE = 94,        // KeuralMoE
    RCPP_ARCH_FINANCEDECODER = 95,   // FinanceDecoder (qovaryx)
    RCPP_ARCH_REFORMER = 96,         // ReformerForCausalLM
    RCPP_ARCH_ACIP = 97,             // ACIPModel
    RCPP_ARCH_COGNICAPOE = 98,       // CognicaPoe
    RCPP_ARCH_GRUGMOE = 99,          // GrugMoE
    RCPP_ARCH_LONGCAT = 100,         // LongCatFlash
    RCPP_ARCH_TELECHAT = 101,        // Telechat
    RCPP_ARCH_BTLM = 102,            // BTLM
    RCPP_ARCH_DUCHIFAT = 103,        // Duchifat v2
    RCPP_ARCH_DUO = 104,             // DUO
    RCPP_ARCH_ESHMUN = 105,          // Eshmun
    RCPP_ARCH_GLA = 106,             // GLA (gated linear attention)
    RCPP_ARCH_POLYVERSE = 107,       // Polyverse (VLM)
    RCPP_ARCH_TRANSFOXL = 108,       // Transformer-XL
    RCPP_ARCH_TRANSNORMER = 109,     // TransNormer
    RCPP_ARCH_TWINY = 110,           // Twiny
    RCPP_ARCH_GPTPANGU = 111,        // GPT-Pangu
    RCPP_ARCH_BVV = 112,             // BVV (model_unfrozen)
    RCPP_ARCH_FUYU = 113,            // FuyuForCausalLM (VLM — causal decoder, image tokens inline)
    RCPP_ARCH_MUSE = 114,            // Muse-Glimmer (VLM — causal multimodal decoder)
    // Sentinel for unmapped architecture strings. Unmapped archs used to
    // silently become RCPP_ARCH_BITNET (wrong activation / attention for
    // most families) — now they fail loudly at discovery/load (decision
    // 2026-08-13, bring-up pilot #10).
    RCPP_ARCH_UNKNOWN = 255,
} rcpp_arch_t;

#include <string.h>

static inline rcpp_arch_t rcpp_arch_from_string(const char* s) {
    if (!s || strcmp(s, "bitnet") == 0) return RCPP_ARCH_BITNET;
    if (strcmp(s, "qwen3")   == 0) return RCPP_ARCH_QWEN3;
    if (strcmp(s, "llama")   == 0) return RCPP_ARCH_LLAMA;
    if (strcmp(s, "mistral") == 0) return RCPP_ARCH_MISTRAL;
    if (strcmp(s, "ministral") == 0) return RCPP_ARCH_MISTRAL;  // Ministral (config declares MistralForCausalLM)
    if (strcmp(s, "ministral3") == 0) return RCPP_ARCH_LLAMA;   // Ministral3 — llama-layout + YARN rope + llama-4 attn scale
    if (strcmp(s, "sparsemistralforcausallm") == 0) return RCPP_ARCH_MISTRAL;  // SparseMistral (mistral layout)
    if (strcmp(s, "qwen2")   == 0) return RCPP_ARCH_QWEN2;
    if (strcmp(s, "gemma")   == 0) return RCPP_ARCH_GEMMA;
    if (strcmp(s, "gemma2")  == 0) return RCPP_ARCH_GEMMA;
    if (strcmp(s, "gemma3")  == 0) return RCPP_ARCH_GEMMA;
    if (strcmp(s, "gemma4")  == 0) return RCPP_ARCH_GEMMA;
    if (strcmp(s, "phi")     == 0) return RCPP_ARCH_PHI;
    if (strcmp(s, "mixformersequential") == 0) return RCPP_ARCH_PHI;  // phi-1/1.5 HF class (model_type=phi)
    if (strcmp(s, "zamba2")  == 0) return RCPP_ARCH_ZAMBA2;
    if (strcmp(s, "zamba")   == 0) return RCPP_ARCH_ZAMBA;
    if (strcmp(s, "mamba")   == 0) return RCPP_ARCH_MAMBA;
    if (strcmp(s, "laguna")  == 0) return RCPP_ARCH_LAGUNA;
    if (strcmp(s, "falcon")  == 0) return RCPP_ARCH_FALCON;
    if (strcmp(s, "falcon3") == 0) return RCPP_ARCH_FALCON;
    if (strcmp(s, "rw")      == 0) return RCPP_ARCH_FALCON;  // Falcon 7B/40B HF arch (RWForCausalLM)
    if (strcmp(s, "olmo")    == 0) return RCPP_ARCH_OLMO;
    if (strcmp(s, "olmo2")   == 0) return RCPP_ARCH_OLMO;
    if (strcmp(s, "olmo3")   == 0) return RCPP_ARCH_OLMO;   // OLMo 3 (olmo2 arch: QK-norm, RMSNorm, rope)
    if (strcmp(s, "olmoe")   == 0) return RCPP_ARCH_OLMO;
    if (strcmp(s, "zaya")    == 0) return RCPP_ARCH_ZAYA;
    if (strcmp(s, "qwen2vl") == 0) return RCPP_ARCH_QWEN2VL;
    if (strcmp(s, "qwen3vl") == 0) return RCPP_ARCH_QWEN3VL;
    // DeepSeek LLM (V1, Coder) uses standard attention — map to Qwen2-like
    if (strcmp(s, "deepseek")   == 0) return RCPP_ARCH_QWEN2;
    // DeepSeek V2/V3/R1 use Multi-Head Latent Attention (MLA) — native support
    if (strcmp(s, "deepseek2")  == 0) return RCPP_ARCH_DEEPSEEK;
    if (strcmp(s, "deepseek3")  == 0) return RCPP_ARCH_DEEPSEEK;
    if (strcmp(s, "stablelm")  == 0) return RCPP_ARCH_LLAMA;
    if (strcmp(s, "stablelmepoch") == 0) return RCPP_ARCH_LLAMA;  // StableLM-Epoch (llama-layout, census alias)
    if (strcmp(s, "tinyllama") == 0) return RCPP_ARCH_LLAMA;      // TinyLlama (config declares LlamaForCausalLM)
    if (strcmp(s, "openlm")    == 0) return RCPP_ARCH_LLAMA;      // OpenLM / open-llama (config declares LlamaForCausalLM)
    if (strcmp(s, "mobilellm") == 0) return RCPP_ARCH_LLAMA;      // Meta MobileLLM (llama-layout: rope, RMSNorm, swiglu)
    if (strcmp(s, "customllama") == 0) return RCPP_ARCH_LLAMA;    // cosmetic llama renames (CustomLlamaForCausalLM)
    if (strcmp(s, "yi")        == 0) return RCPP_ARCH_LLAMA;      // Yi (01-ai; config model_type=llama)
    if (strcmp(s, "decilm")    == 0) return RCPP_ARCH_LLAMA;      // DeciLM (llama-layout, GQA)
    if (strcmp(s, "hunyuan")   == 0) return RCPP_ARCH_LLAMA;      // HunYuan dense (llama-layout)
    if (strcmp(s, "nanbeige")  == 0) return RCPP_ARCH_LLAMA;      // Nanbeige (llama-layout)
    if (strcmp(s, "recast8b_llama") == 0) return RCPP_ARCH_LLAMA; // RECAST (llama-layout)
    if (strcmp(s, "hyperllama") == 0) return RCPP_ARCH_LLAMA;
    if (strcmp(s, "sparsellama") == 0) return RCPP_ARCH_LLAMA;
    if (strcmp(s, "constrainedllama") == 0) return RCPP_ARCH_LLAMA;
    if (strcmp(s, "mosaic")    == 0) return RCPP_ARCH_LLAMA;
    if (strcmp(s, "mpt")       == 0) return RCPP_ARCH_LLAMA;
    if (strcmp(s, "pixtral")   == 0) return RCPP_ARCH_MISTRAL;
    if (strcmp(s, "whisper")   == 0) return RCPP_ARCH_WHISPER;
    if (strcmp(s, "granite")  == 0) return RCPP_ARCH_GEMMA;
    if (strcmp(s, "granitemoe") == 0) return RCPP_ARCH_GEMMA;
    if (strcmp(s, "phi3")    == 0) return RCPP_ARCH_PHI;
    if (strcmp(s, "phi3small") == 0) return RCPP_ARCH_PHI;  // Phi-3-small (phi layout)
    if (strcmp(s, "kphi3")   == 0) return RCPP_ARCH_PHI;    // K-Phi3 (phi layout)
    if (strcmp(s, "phi4")    == 0) return RCPP_ARCH_PHI;
    if (strcmp(s, "starcoder") == 0) return RCPP_ARCH_LLAMA;
    if (strcmp(s, "starcoder2") == 0) return RCPP_ARCH_LLAMA;
    if (strcmp(s, "command-r") == 0) return RCPP_ARCH_LLAMA;
    if (strcmp(s, "dbrx")    == 0) return RCPP_ARCH_LLAMA;
    if (strcmp(s, "jamba")   == 0) return RCPP_ARCH_LLAMA;
    // ── 2026-08-13 arch-string coverage batch (LLaMA-layout families) ──
    if (strcmp(s, "baichuan")   == 0) return RCPP_ARCH_LLAMA;  // Baichuan-1/2 (LLaMA-layout)
    if (strcmp(s, "baichuan2")  == 0) return RCPP_ARCH_LLAMA;
    if (strcmp(s, "BaichuanForCausalLM") == 0) return RCPP_ARCH_LLAMA;
    if (strcmp(s, "exaone")     == 0) return RCPP_ARCH_LLAMA;  // LG EXAONE 3 (LLaMA-layout)
    if (strcmp(s, "glm4")       == 0) return RCPP_ARCH_LLAMA;  // GLM-4 (llama + partial-rope 0.5 + qkv bias)
    if (strcmp(s, "ExaoneForCausalLM")   == 0) return RCPP_ARCH_LLAMA;
    if (strcmp(s, "solar")      == 0) return RCPP_ARCH_LLAMA;  // upstage SOLAR (LLaMA-layout)
    if (strcmp(s, "solaropen")  == 0) return RCPP_ARCH_LLAMA;  // SolarOpen (llama-layout)
    if (strcmp(s, "solaropen2") == 0) return RCPP_ARCH_LLAMA;
    if (strcmp(s, "internlm")   == 0) return RCPP_ARCH_LLAMA;  // InternLM-1
    if (strcmp(s, "internlm2")  == 0) return RCPP_ARCH_LLAMA;  // InternLM-2 (LLaMA-layout)
    if (strcmp(s, "xverse")     == 0) return RCPP_ARCH_LLAMA;  // xverse (LLaMA-layout)
    if (strcmp(s, "qwen")       == 0) return RCPP_ARCH_QWEN2;  // Qwen1 (attention-layout ~ Qwen2)
    // ── 2026-08-13 bring-up pilot: LLaMA-layout architectures (GGUF + HF class names) ──
    if (strcmp(s, "openelm")        == 0) return RCPP_ARCH_LLAMA;  // Apple OpenELM (RMSNorm, GQA, RoPE)
    if (strcmp(s, "OpenELMForCausalLM") == 0) return RCPP_ARCH_LLAMA;
    if (strcmp(s, "nemotron")       == 0) return RCPP_ARCH_LLAMA;  // NVIDIA Nemotron (Llama-3.1 layout)
    if (strcmp(s, "NemotronForCausalLM") == 0) return RCPP_ARCH_LLAMA;
    if (strcmp(s, "minicpm")        == 0) return RCPP_ARCH_LLAMA;  // MiniCPM (LLaMA-layout, added bias)
    if (strcmp(s, "MiniCPMForCausalLM")  == 0) return RCPP_ARCH_LLAMA;
    // ── New VLM architectures ──
    if (strcmp(s, "smolvlm")   == 0) return RCPP_ARCH_QWEN2VL;
    if (strcmp(s, "llava")     == 0) return RCPP_ARCH_QWEN2VL;
    if (strcmp(s, "molmo")     == 0) return RCPP_ARCH_OLMO;
    if (strcmp(s, "ovis")      == 0) return RCPP_ARCH_GEMMA;
    if (strcmp(s, "paligemma") == 0) return RCPP_ARCH_GEMMA;
    if (strcmp(s, "florence")  == 0) return RCPP_ARCH_QWEN2VL;
    // ── New MoE reasoning ──
    if (strcmp(s, "phi_moe")   == 0) return RCPP_ARCH_PHI;
    if (strcmp(s, "deepseek_v3") == 0) return RCPP_ARCH_DEEPSEEK;
    // DeepSeek V4 Flash/Pro — mHC + CSA+HCA hybrid attention + FP4 MoE experts
    if (strcmp(s, "deepseek_v4")  == 0) return RCPP_ARCH_DEEPSEEK_V4;
    if (strcmp(s, "deepseek4")    == 0) return RCPP_ARCH_DEEPSEEK_V4;
    if (strcmp(s, "dflash")       == 0) return RCPP_ARCH_DEEPSEEK_V4;
    if (strcmp(s, "deepseek4_dspark") == 0) return RCPP_ARCH_DEEPSEEK_V4;
    if (strcmp(s, "smollm")    == 0) return RCPP_ARCH_LLAMA;
    if (strcmp(s, "smollm2")   == 0) return RCPP_ARCH_LLAMA;
    // ── MONSTER breadth batch 2026-08-14 (from llama.cpp convert_hf_to_gguf
    //    conversion/ registry — HF class names, suffix-stripped by
    //    safetensors_reader) ──
    if (strcmp(s, "smollm3")   == 0) return RCPP_ARCH_LLAMA;   // SmolLM3ForCausalLM (llama-layout)
    if (strcmp(s, "apertus")   == 0) return RCPP_ARCH_LLAMA;   // ApertusForCausalLM (LlamaModel)
    if (strcmp(s, "cohere")    == 0) return RCPP_ARCH_LLAMA;   // CohereForCausalLM (= command-r)
    if (strcmp(s, "gptbigcode")== 0) return RCPP_ARCH_LLAMA;   // GPTBigCodeForCausalLM (StarCoder1)
    if (strcmp(s, "internlm3") == 0) return RCPP_ARCH_LLAMA;   // InternLM3ForCausalLM
    if (strcmp(s, "mixtral")   == 0) return RCPP_ARCH_MISTRAL; // MixtralForCausalLM (mistral layout, MoE)
    if (strcmp(s, "qwen2moe")  == 0) return RCPP_ARCH_QWEN2;   // Qwen2MoeForCausalLM (shared-expert MoE: warned+ignored, pilot #8)
    if (strcmp(s, "qwen3moe")  == 0) return RCPP_ARCH_QWEN3;   // Qwen3MoeForCausalLM (128/8 experts, mixtral-style)
    if (strcmp(s, "deepseekv2")== 0) return RCPP_ARCH_DEEPSEEK;   // DeepseekV2ForCausalLM (MLA)
    if (strcmp(s, "deepseekv3")== 0) return RCPP_ARCH_DEEPSEEK;   // DeepseekV3ForCausalLM (MLA)
    if (strcmp(s, "deepseekv32")== 0) return RCPP_ARCH_DEEPSEEK;  // DeepseekV32ForCausalLM (V3.2, MLA)
    if (strcmp(s, "deepseekv4")== 0) return RCPP_ARCH_DEEPSEEK_V4; // DeepseekV4ForCausalLM
    if (strcmp(s, "gpt2")     == 0) return RCPP_ARCH_GPT2;   // GPT2LMHeadModel (custom tensor map)
    if (strcmp(s, "gpt2lmheadcustom") == 0) return RCPP_ARCH_GPT2;  // GPT2LMHeadCustomModel (gpt2 layout)
    if (strcmp(s, "biogpt")    == 0) return RCPP_ARCH_GPT2;       // BioGPT (gpt2-layout: learned pos emb, gelu)
    if (strcmp(s, "xglm")      == 0) return RCPP_ARCH_GPT2;       // XGLM (gpt2-layout)
    if (strcmp(s, "gpjtgpt2model") == 0) return RCPP_ARCH_GPT2;
    if (strcmp(s, "gpt2almhead") == 0) return RCPP_ARCH_GPT2;
    if (strcmp(s, "poptorchpipelinedgpt2lmhead") == 0) return RCPP_ARCH_GPT2;
    if (strcmp(s, "gptneox")   == 0) return RCPP_ARCH_GPTNEOX; // GPTNeoXForCausalLM (parallel attn+FFN, LN+bias)
    if (strcmp(s, "opt")       == 0) return RCPP_ARCH_OPT;    // OPTForCausalLM (learned pos, relu)
    if (strcmp(s, "gptneo")    == 0) return RCPP_ARCH_GPTNEO; // GPTNeoForCausalLM
    if (strcmp(s, "codegen")   == 0) return RCPP_ARCH_CODEGEN; // CodeGenForCausalLM (fused qkv, partial rotary)
    if (strcmp(s, "gptj")      == 0) return RCPP_ARCH_GPTJ;    // GPTJForCausalLM (adjacent partial rotary)
    if (strcmp(s, "gptjiang")  == 0) return RCPP_ARCH_GPTJ;    // GPTJiang (gptj layout)
    if (strcmp(s, "gptoss")    == 0) return RCPP_ARCH_GPTOSS;  // GptOssForCausalLM (packed FP4 MoE)
    if (strcmp(s, "step1")     == 0) return RCPP_ARCH_STEP1;   // Step1ForCausalLM (sqrt-ALiBi, no RoPE)
    if (strcmp(s, "step1moe")  == 0) return RCPP_ARCH_STEP1;   // Step1MoEForCausalLM (dense weights in practice; MoE cfg ignored until an expert-bearing ckpt is seen)
    if (strcmp(s, "bloom")     == 0) return RCPP_ARCH_BLOOM;   // BloomForCausalLM (fused qkv, linear ALiBi, LayerNorm)
    if (strcmp(s, "lfm2")      == 0) return RCPP_ARCH_LFM2;
    // ── 2026-08-15 census tail sweep (auto-generated, model_type-verified) ──
    // Pass-3 aliases (2026-08-15 evening): real-config verified, configs fetched
    // for each class below (bailing_moe v2/v2.5/v3/linear, bamba, kimik25=DeepseekV3,
    // instella=deepseek_v3, hybridqwen3, gpt2moe=CustomGPT2, sparsetral=mistral).
    if (strcmp(s, "bailingmoe") == 0) return RCPP_ARCH_LLAMA;  // BailingMoeForCausalLM (llama-layout MoE, verified 2026-08-15)
    if (strcmp(s, "bailingmoev2") == 0) return RCPP_ARCH_LLAMA;  // BailingMoeV2ForCausalLM (llama-layout MoE, verified 2026-08-15)
    if (strcmp(s, "bailingmoev2_5") == 0) return RCPP_ARCH_LLAMA;  // BailingMoeV2_5 (llama-layout MoE, verified 2026-08-15)
    if (strcmp(s, "bailingmoev3") == 0) return RCPP_ARCH_LLAMA;  // BailingMoeV3 (llama-layout MoE, verified 2026-08-15)
    if (strcmp(s, "bailingmoelinearv2") == 0) return RCPP_ARCH_LLAMA;  // BailingMoeLinearV2 (llama-layout MoE, verified 2026-08-15)
    if (strcmp(s, "bamba") == 0) return RCPP_ARCH_LLAMA;  // IBM Bamba (llama profile: rms 1e-05 rope 10000 silu)
    if (strcmp(s, "kimik25") == 0) return RCPP_ARCH_DEEPSEEK;  // Kimi-K2.5 (arch declares DeepseekV3ForCausalLM — MLA MoE)
    if (strcmp(s, "instella") == 0) return RCPP_ARCH_DEEPSEEK;  // AMD Instella-MoE (config model_type=deepseek_v3)
    if (strcmp(s, "hybridqwen3") == 0) return RCPP_ARCH_LLAMA;  // HybridQwen3 (dump-verified llama profile)
    if (strcmp(s, "gpt2moe") == 0) return RCPP_ARCH_GPT2;  // GPT2MoE (CustomGPT2 — gpt2-layout + experts)
    if (strcmp(s, "modeling_sparsetral.mistral") == 0) return RCPP_ARCH_MISTRAL;  // SparseTral (sparse mistral, dump-verified)
    if (strcmp(s, "adavocabgemma") == 0) return RCPP_ARCH_GEMMA;  // gemma
    if (strcmp(s, "aicraftar-tharo.g-conditionalgeneration") == 0) return RCPP_ARCH_QWEN2VL;  // qwen2_vl
    if (strcmp(s, "antihal") == 0) return RCPP_ARCH_GEMMA;  // gemma4
    if (strcmp(s, "asvdopt") == 0) return RCPP_ARCH_OPT;  // opt
    if (strcmp(s, "automodel") == 0) return RCPP_ARCH_MISTRAL;  // mistral
    if (strcmp(s, "backpackgpt2") == 0) return RCPP_ARCH_GPT2;  // gpt2
    if (strcmp(s, "bunnyqwen") == 0) return RCPP_ARCH_QWEN2;  // llava-qwen2
    if (strcmp(s, "careaqa") == 0) return RCPP_ARCH_LLAMA;  // llama
    if (strcmp(s, "chexagent") == 0) return RCPP_ARCH_PHI;  // phi
    if (strcmp(s, "codebharat") == 0) return RCPP_ARCH_LLAMA;  // llama
    if (strcmp(s, "cogpt2") == 0) return RCPP_ARCH_GPT2;  // gpt2
    if (strcmp(s, "craneai") == 0) return RCPP_ARCH_GEMMA;  // gemma3
    if (strcmp(s, "custom_mpt") == 0) return RCPP_ARCH_LLAMA;  // mpt
    if (strcmp(s, "custombiogpt") == 0) return RCPP_ARCH_GPT2;  // biogpt
    if (strcmp(s, "custommixtral") == 0) return RCPP_ARCH_MISTRAL;  // mixtral
    if (strcmp(s, "custommodel3") == 0) return RCPP_ARCH_GPTNEOX;  // gpt_neox
    if (strcmp(s, "dashqphi3") == 0) return RCPP_ARCH_PHI;  // phi3
    if (strcmp(s, "deepseekv2mobe") == 0) return RCPP_ARCH_DEEPSEEK;  // deepseek_v2
    if (strcmp(s, "deepseekv2sparsemobe") == 0) return RCPP_ARCH_DEEPSEEK;  // deepseek_v2
    if (strcmp(s, "deepseekv3forcausallmnextn") == 0) return RCPP_ARCH_DEEPSEEK;  // deepseek_v3
    if (strcmp(s, "denseformer") == 0) return RCPP_ARCH_LLAMA;  // llama
    if (strcmp(s, "dflashlaguna") == 0) return RCPP_ARCH_LAGUNA;  // laguna
    if (strcmp(s, "dribble") == 0) return RCPP_ARCH_PHI;  // phi
    if (strcmp(s, "dribblellama") == 0) return RCPP_ARCH_LLAMA;  // llama
    if (strcmp(s, "duolaguna") == 0) return RCPP_ARCH_LAGUNA;  // laguna
    if (strcmp(s, "dusmistral") == 0) return RCPP_ARCH_MISTRAL;  // mistral
    if (strcmp(s, "edullm") == 0) return RCPP_ARCH_MISTRAL;  // mixtral
    if (strcmp(s, "efficientdlm") == 0) return RCPP_ARCH_QWEN3;  // qwen3
    if (strcmp(s, "exaonetd") == 0) return RCPP_ARCH_LLAMA;  // exaone
    if (strcmp(s, "flashgptneox") == 0) return RCPP_ARCH_GPTNEOX;  // gpt_neox
    if (strcmp(s, "flaxgptj") == 0) return RCPP_ARCH_GPTJ;  // gptj
    if (strcmp(s, "forcausallm") == 0) return RCPP_ARCH_GEMMA;  // gemma4
    if (strcmp(s, "fsdpgptoss") == 0) return RCPP_ARCH_GPTOSS;  // gpt_oss
    if (strcmp(s, "gemma4text") == 0) return RCPP_ARCH_GEMMA;  // gemma4
    if (strcmp(s, "gemmagain") == 0) return RCPP_ARCH_GEMMA;  // gemma3
    if (strcmp(s, "gfusionfordiffusionlm") == 0) return RCPP_ARCH_DEEPSEEK;  // deepseek_v3
    if (strcmp(s, "gistgptneo") == 0) return RCPP_ARCH_GPTNEO;  // gpt_neo
    if (strcmp(s, "glamm") == 0) return RCPP_ARCH_QWEN2VL;  // llava
    if (strcmp(s, "glus") == 0) return RCPP_ARCH_QWEN2VL;  // llava
    if (strcmp(s, "gpt2forquestionanswering") == 0) return RCPP_ARCH_GPT2;  // gpt2
    if (strcmp(s, "gpt2forsequenceclassification") == 0) return RCPP_ARCH_GPT2;  // gpt2
    if (strcmp(s, "gpt2lmandvaluehead") == 0) return RCPP_ARCH_GPT2;  // gpt2
    if (strcmp(s, "gptbigcodeforsequenceclassification") == 0) return RCPP_ARCH_LLAMA;  // gpt_bigcode
    if (strcmp(s, "gretriever") == 0) return RCPP_ARCH_LLAMA;  // llama
    if (strcmp(s, "int8opt") == 0) return RCPP_ARCH_OPT;  // opt
    if (strcmp(s, "internlm2forreward") == 0) return RCPP_ARCH_LLAMA;  // internlm2
    if (strcmp(s, "internlmxcomposer2") == 0) return RCPP_ARCH_LLAMA;  // internlm
    if (strcmp(s, "interns2preview") == 0) return RCPP_ARCH_QWEN35;  // qwen3_5_moe
    if (strcmp(s, "kblamphi3") == 0) return RCPP_ARCH_PHI;  // phi3
    if (strcmp(s, "kimik2") == 0) return RCPP_ARCH_QWEN35;  // qwen3_5
    if (strcmp(s, "layerwiseminicpm") == 0) return RCPP_ARCH_LLAMA;  // minicpm
    if (strcmp(s, "leanllama") == 0) return RCPP_ARCH_LLAMA;  // llama
    if (strcmp(s, "leanmixtral") == 0) return RCPP_ARCH_MISTRAL;  // mixtral
    if (strcmp(s, "lexadelta") == 0) return RCPP_ARCH_GPTOSS;  // gpt_oss
    if (strcmp(s, "lfm2bidirectionalformaskedlm") == 0) return RCPP_ARCH_LFM2;  // lfm2
    if (strcmp(s, "lightonocr") == 0) return RCPP_ARCH_MISTRAL;  // mistral3
    if (strcmp(s, "llamaforcausallmeagle3") == 0) return RCPP_ARCH_LLAMA;  // llama
    if (strcmp(s, "llamaforsequenceclassification") == 0) return RCPP_ARCH_LLAMA;  // llama
    if (strcmp(s, "llavaqwen") == 0) return RCPP_ARCH_QWEN2VL;  // llava
    if (strcmp(s, "loragpt2") == 0) return RCPP_ARCH_GPT2;  // gpt2
    if (strcmp(s, "mahler60/prueba") == 0) return RCPP_ARCH_GPTNEOX;  // gpt_neox
    if (strcmp(s, "mamba2") == 0) return RCPP_ARCH_MAMBA;  // mamba2
    if (strcmp(s, "mambamodel") == 0) return RCPP_ARCH_MAMBA;  // mamba
    if (strcmp(s, "memllama") == 0) return RCPP_ARCH_LLAMA;  // llama
    if (strcmp(s, "meteormamba") == 0) return RCPP_ARCH_MAMBA;  // mamba
    if (strcmp(s, "mimoaudio") == 0) return RCPP_ARCH_QWEN2;  // qwen2
    if (strcmp(s, "miniphi3") == 0) return RCPP_ARCH_PHI;  // phi3
    if (strcmp(s, "mixformervlsequential") == 0) return RCPP_ARCH_PHI;  // mixformer-sequential
    if (strcmp(s, "mobillama") == 0) return RCPP_ARCH_LLAMA;  // llama
    if (strcmp(s, "monoformer") == 0) return RCPP_ARCH_LLAMA;  // llama
    if (strcmp(s, "moyi") == 0) return RCPP_ARCH_QWEN2;  // deepseek
    if (strcmp(s, "multiheadgptneo") == 0) return RCPP_ARCH_GPTNEO;  // gpt_neo
    if (strcmp(s, "multimodalstarcoder2") == 0) return RCPP_ARCH_LLAMA;  // starcoder2
    if (strcmp(s, "mutorgemma") == 0) return RCPP_ARCH_GEMMA;  // gemma
    if (strcmp(s, "mybaichuan") == 0) return RCPP_ARCH_LLAMA;  // baichuan
    if (strcmp(s, "myqwen") == 0) return RCPP_ARCH_QWEN2;  // qwen
    if (strcmp(s, "myxverse") == 0) return RCPP_ARCH_LLAMA;  // xverse
    if (strcmp(s, "nemotronhaugmented") == 0) return RCPP_ARCH_NEMOTRONH;  // nemotron_h
    if (strcmp(s, "notagen") == 0) return RCPP_ARCH_GPT2;  // gpt2
    if (strcmp(s, "olmo2forsequenceclassification") == 0) return RCPP_ARCH_OLMO;  // olmo2
    if (strcmp(s, "olmo3sink") == 0) return RCPP_ARCH_OLMO;  // olmo3
    if (strcmp(s, "olmomodel") == 0) return RCPP_ARCH_OLMO;  // olmo
    if (strcmp(s, "opt_prompttuned_for_sentimentanalysis") == 0) return RCPP_ARCH_OPT;  // opt
    if (strcmp(s, "pawqwen3") == 0) return RCPP_ARCH_QWEN3;  // qwen3
    if (strcmp(s, "phi3forsequenceclassification") == 0) return RCPP_ARCH_PHI;  // phi3
    if (strcmp(s, "poptorchpipelinedgpt2") == 0) return RCPP_ARCH_GPT2;  // gpt2
    if (strcmp(s, "poptorchpipelinedwhisper") == 0) return RCPP_ARCH_WHISPER;  // whisper
    if (strcmp(s, "quark") == 0) return RCPP_ARCH_QWEN35;  // qwen3_5_moe
    if (strcmp(s, "qwen2forcausallmpostblocksteeringfixed") == 0) return RCPP_ARCH_QWEN2;  // qwen2
    if (strcmp(s, "qwen2forprocessreward") == 0) return RCPP_ARCH_QWEN2;  // qwen2
    if (strcmp(s, "qwen2forsequenceclassification") == 0) return RCPP_ARCH_QWEN2;  // qwen2
    if (strcmp(s, "qwen2reasoning") == 0) return RCPP_ARCH_QWEN2;  // qwen2
    if (strcmp(s, "qwen2vlaudio") == 0) return RCPP_ARCH_QWEN2VL;  // qwen2_vl
    if (strcmp(s, "qwen2vlextended") == 0) return RCPP_ARCH_QWEN2VL;  // qwen2_vl
    if (strcmp(s, "qwen2vlforconditionalgenerationwithaudio") == 0) return RCPP_ARCH_QWEN2VL;  // qwen2_vl
    if (strcmp(s, "qwen3_5dllm") == 0) return RCPP_ARCH_QWEN35;  // qwen3_5
    if (strcmp(s, "qwen3_5text") == 0) return RCPP_ARCH_QWEN35;  // qwen3_5
    if (strcmp(s, "qwen3forsequenceclassification") == 0) return RCPP_ARCH_QWEN3;  // qwen3
    if (strcmp(s, "qwen3gated") == 0) return RCPP_ARCH_QWEN3;  // qwen3
    if (strcmp(s, "qwen3mobe") == 0) return RCPP_ARCH_QWEN3;  // qwen3_moe
    if (strcmp(s, "qwen3sparsemobe") == 0) return RCPP_ARCH_QWEN3;  // qwen3_moe
    if (strcmp(s, "qwen3vlseg") == 0) return RCPP_ARCH_QWEN3VL;  // qwen3_vl
    if (strcmp(s, "rnj1") == 0) return RCPP_ARCH_GEMMA;  // gemma3
    if (strcmp(s, "ruqwen2") == 0) return RCPP_ARCH_QWEN2;  // qwen2
    if (strcmp(s, "serayuki") == 0) return RCPP_ARCH_LLAMA;  // llama
    if (strcmp(s, "sewy3") == 0) return RCPP_ARCH_GEMMA;  // gemma
    if (strcmp(s, "smollm3model") == 0) return RCPP_ARCH_LLAMA;  // smollm3
    if (strcmp(s, "stablediffcoder") == 0) return RCPP_ARCH_LLAMA;  // llama
    if (strcmp(s, "streamvln") == 0) return RCPP_ARCH_QWEN2VL;  // llava
    if (strcmp(s, "symbolicgpt") == 0) return RCPP_ARCH_GPT2;  // gpt2
    if (strcmp(s, "titansmactransformer") == 0) return RCPP_ARCH_LLAMA;  // llama
    if (strcmp(s, "trimkvphi3") == 0) return RCPP_ARCH_PHI;  // phi3
    if (strcmp(s, "trimkvqwen3") == 0) return RCPP_ARCH_QWEN3;  // qwen3
    if (strcmp(s, "uyu2") == 0) return RCPP_ARCH_GEMMA;  // gemma4
    if (strcmp(s, "vlclipgptneox") == 0) return RCPP_ARCH_GPTNEOX;  // gpt_neox
    if (strcmp(s, "whaleye") == 0) return RCPP_ARCH_DEEPSEEK;  // deepseek_v32
    if (strcmp(s, "xcuros") == 0) return RCPP_ARCH_QWEN2;  // qwen2
    if (strcmp(s, "nanochat") == 0) return RCPP_ARCH_NANOCHAT;  // NanoChatForCausalLM (verified vs modeling_nanochat.py 2026-08-15)
    if (strcmp(s, "pico") == 0) return RCPP_ARCH_PICO;  // PicoDecoderHF (llama-layout + adjacent rope)
    if (strcmp(s, "picodecoder") == 0) return RCPP_ARCH_PICO;  // PicoDecoderHF (llama-layout + adjacent rope)
    if (strcmp(s, "picodecoderhf") == 0) return RCPP_ARCH_PICO;  // PicoDecoderHF (llama-layout + adjacent rope)
    if (strcmp(s, "caca") == 0) return RCPP_ARCH_LLAMA;  // CacaForCausalLM (llama profile, rms+rope+GQA, verified 2026-08-15)
    if (strcmp(s, "gateddeltanet") == 0) return RCPP_ARCH_QWEN3NEXT;  // GatedDeltaNet (same attention as qwen3next backend)
    if (strcmp(s, "dynamicalibi") == 0) return RCPP_ARCH_DYNAMICALIBI;  // DynamicAlibiForCausalLM (static ALiBi at inference, verified 2026-08-15)
    if (strcmp(s, "glm") == 0) return RCPP_ARCH_LLAMA;  // GlmForCausalLM (glm-4-9b config: partial-rope + qkv bias — needs the glm4 quirks)
    // ── end census tail sweep ──
    if (strcmp(s, "roleslm") == 0) return RCPP_ARCH_LLAMA;  // RoleSLM (sathishphdai SLM family — llama layout, blocks.N names, verified vs model.py)
    if (strcmp(s, "slm") == 0) return RCPP_ARCH_LLAMA;  // SLM (sathishphdai SLM family)
    if (strcmp(s, "slmforcausallm") == 0) return RCPP_ARCH_LLAMA;  // SLMForCausalLM (SLM family)
    if (strcmp(s, "slmmodel") == 0) return RCPP_ARCH_LLAMA;  // SLMModel (SLM family)
    if (strcmp(s, "industryslm") == 0) return RCPP_ARCH_LLAMA;  // IndustrySLM (SLM family)
    if (strcmp(s, "hfhealthslm") == 0) return RCPP_ARCH_LLAMA;  // HFHealthSLM (SLM family)
    if (strcmp(s, "sdlcslm") == 0) return RCPP_ARCH_LLAMA;  // SDLC-SLM (SLM family)

    if (strcmp(s, "hrmtext") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama (rms 1e-6 + rope 10000 + silu, sapient HRM-Text)
    if (strcmp(s, "longcatcausallm") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama (num_layers 38, rms 1e-5, rope 1e6, LongCat-2.0)
    if (strcmp(s, "gpt") == 0) return RCPP_ARCH_GPT2;  // GptForCausalLM — dominant layout gpt2 (htmLLM/mgpt2/nanogpt configs)
    if (strcmp(s, "pldrllm") == 0) return RCPP_ARCH_QWEN2;  // PLDR-LLM — qwen2-layout (attention_bias true, silu)

    if (strcmp(s, "rwkv5") == 0) return RCPP_ARCH_RWKV;  // rwkv5 (RWKV backend covers 4/5/6)
    if (strcmp(s, "rwkv6") == 0) return RCPP_ARCH_RWKV;  // rwkv6 (RWKV backend covers 4/5/6)
    if (strcmp(s, "llavaphi") == 0) return RCPP_ARCH_PHI;  // LLaVA-Phi (VLM, phi text decoder)

    if (strcmp(s, "adelicllama") == 0) return RCPP_ARCH_LLAMA;  // loose llama (rms+silu, rope default)
    if (strcmp(s, "aligngpt") == 0) return RCPP_ARCH_LLAMA;  // loose llama (rms+silu, rope default)
    if (strcmp(s, "axk2") == 0) return RCPP_ARCH_LLAMA;  // loose llama (rms+silu, rope default)
    if (strcmp(s, "biomedgpt") == 0) return RCPP_ARCH_LLAMA;  // loose llama (rms+silu, rope default)
    if (strcmp(s, "blockffn") == 0) return RCPP_ARCH_LLAMA;  // loose llama (rms+silu, rope default)
    if (strcmp(s, "dots1") == 0) return RCPP_ARCH_LLAMA;  // loose llama (rms+silu, rope default)
    if (strcmp(s, "dummyllama") == 0) return RCPP_ARCH_LLAMA;  // loose llama (rms+silu, rope default)
    if (strcmp(s, "experiemental") == 0) return RCPP_ARCH_LLAMA;  // loose llama (rms+silu, rope default)
    if (strcmp(s, "fabric") == 0) return RCPP_ARCH_LLAMA;  // loose llama (rms+silu, rope default)
    if (strcmp(s, "fingpt") == 0) return RCPP_ARCH_LLAMA;  // loose llama (rms+silu, rope default)
    if (strcmp(s, "flamingo") == 0) return RCPP_ARCH_LLAMA;  // loose llama (rms+silu, rope default)
    if (strcmp(s, "gptmini") == 0) return RCPP_ARCH_LLAMA;  // loose llama (rms+silu, rope default)
    if (strcmp(s, "h3") == 0) return RCPP_ARCH_LLAMA;  // loose llama (rms+silu, rope default)
    if (strcmp(s, "internlmxcomposer") == 0) return RCPP_ARCH_LLAMA;  // loose llama (rms+silu, rope default)
    if (strcmp(s, "internvl") == 0) return RCPP_ARCH_LLAMA;  // loose llama (rms+silu, rope default)
    if (strcmp(s, "k3dspark") == 0) return RCPP_ARCH_LLAMA;  // loose llama (rms+silu, rope default)
    if (strcmp(s, "kmoshi") == 0) return RCPP_ARCH_LLAMA;  // loose llama (rms+silu, rope default)
    if (strcmp(s, "livemem") == 0) return RCPP_ARCH_LLAMA;  // loose llama (rms+silu, rope default)
    if (strcmp(s, "llama2") == 0) return RCPP_ARCH_LLAMA;  // loose llama (rms+silu, rope default)
    if (strcmp(s, "llavabaichuan2") == 0) return RCPP_ARCH_LLAMA;  // loose llama (rms+silu, rope default)
    if (strcmp(s, "lmdeploy") == 0) return RCPP_ARCH_LLAMA;  // loose llama (rms+silu, rope default)
    if (strcmp(s, "lumma") == 0) return RCPP_ARCH_LLAMA;  // loose llama (rms+silu, rope default)
    if (strcmp(s, "megha") == 0) return RCPP_ARCH_LLAMA;  // loose llama (rms+silu, rope default)
    if (strcmp(s, "midm") == 0) return RCPP_ARCH_LLAMA;  // loose llama (rms+silu, rope default)
    if (strcmp(s, "minibanana") == 0) return RCPP_ARCH_LLAMA;  // loose llama (rms+silu, rope default)
    if (strcmp(s, "minigpt") == 0) return RCPP_ARCH_LLAMA;  // loose llama (rms+silu, rope default)
    if (strcmp(s, "namer") == 0) return RCPP_ARCH_LLAMA;  // loose llama (rms+silu, rope default)
    if (strcmp(s, "ndl") == 0) return RCPP_ARCH_LLAMA;  // loose llama (rms+silu, rope default)
    if (strcmp(s, "nextchat") == 0) return RCPP_ARCH_LLAMA;  // loose llama (rms+silu, rope default)
    if (strcmp(s, "ngme") == 0) return RCPP_ARCH_LLAMA;  // loose llama (rms+silu, rope default)
    if (strcmp(s, "olmohybrid") == 0) return RCPP_ARCH_LLAMA;  // loose llama (rms+silu, rope default)
    if (strcmp(s, "omnillama") == 0) return RCPP_ARCH_LLAMA;  // loose llama (rms+silu, rope default)
    if (strcmp(s, "plm") == 0) return RCPP_ARCH_LLAMA;  // loose llama (rms+silu, rope default)
    if (strcmp(s, "regqwen") == 0) return RCPP_ARCH_LLAMA;  // loose llama (rms+silu, rope default)
    if (strcmp(s, "rllama") == 0) return RCPP_ARCH_LLAMA;  // loose llama (rms+silu, rope default)
    if (strcmp(s, "sky21b") == 0) return RCPP_ARCH_LLAMA;  // loose llama (rms+silu, rope default)
    if (strcmp(s, "sllama") == 0) return RCPP_ARCH_LLAMA;  // loose llama (rms+silu, rope default)
    if (strcmp(s, "stllama") == 0) return RCPP_ARCH_LLAMA;  // loose llama (rms+silu, rope default)
    if (strcmp(s, "taffy") == 0) return RCPP_ARCH_LLAMA;  // loose llama (rms+silu, rope default)
    if (strcmp(s, "tensormind") == 0) return RCPP_ARCH_LLAMA;  // loose llama (rms+silu, rope default)
    if (strcmp(s, "vllama") == 0) return RCPP_ARCH_LLAMA;  // loose llama (rms+silu, rope default)
    if (strcmp(s, "xmodel") == 0) return RCPP_ARCH_LLAMA;  // loose llama (rms+silu, rope default)
    if (strcmp(s, "xmodellm") == 0) return RCPP_ARCH_LLAMA;  // loose llama (rms+silu, rope default)
    if (strcmp(s, "yayiuie") == 0) return RCPP_ARCH_LLAMA;  // loose llama (rms+silu, rope default)
    if (strcmp(s, "youtu") == 0) return RCPP_ARCH_LLAMA;  // loose llama (rms+silu, rope default)
    if (strcmp(s, "a2dqwen") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: dataopsnick/adapt-diff-qwen-0.8b)
    if (strcmp(s, "a2dqwen3_5") == 0) return RCPP_ARCH_QWEN2;  // VLM qwen text decoder (sd17js2/arcLM-0.8B)
    if (strcmp(s, "aetherv211attn") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: FINAL-Bench/Aether-6B-11Attn-base)
    if (strcmp(s, "aetherv27way") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: FINAL-Bench/Aether-7B-5Attn)
    if (strcmp(s, "alexallama") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: sfulay/zephyr-7b-sft-full-amazon)
    if (strcmp(s, "apriel") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: ServiceNow-AI/Apriel-5B-Base)
    if (strcmp(s, "arcanalama") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: syp115/Arcana_star)
    if (strcmp(s, "arcanallama") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: syp115/Arcana)
    if (strcmp(s, "armormforsequenceclassification") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: newmindai/Muhakim)
    if (strcmp(s, "auroragpt2") == 0) return RCPP_ARCH_GPT2;  // dump-id config (gpt2 profile: ThatHungarian/Aurora-10M)
    if (strcmp(s, "beit3llavallama") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: Yirany/Muffin-13B)
    if (strcmp(s, "blastmodel") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: cwoolee/blast-llama-4B)
    if (strcmp(s, "c3qwen") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: liufanfanlff/C3-Context-Cascade-Compression)
    if (strcmp(s, "cbhybridllama") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: cerebras/Llama-3-CBHybridL-8B)
    if (strcmp(s, "chatunivillama") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: Chat-UniVi/Chat-UniVi-ScienceQA)
    if (strcmp(s, "codalanguage") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: Salesforce/CoDA-v0-Base)
    if (strcmp(s, "cogvlmvideo") == 0) return RCPP_ARCH_LLAMA;  // VLM llama text decoder (zai-org/cogvlm2-video-llama3-chat)
    if (strcmp(s, "continue1") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: SVECTOR-CORPORATION/Continue-1-OSS)
    if (strcmp(s, "crystalcoder") == 0) return RCPP_ARCH_GPT2;  // dump-id config (gpt2 profile: IFM/Crystal)
    if (strcmp(s, "ddllama") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: xuan-luo/FlexiDepth-Llama-3-8B-Instruct)
    if (strcmp(s, "deepseekfixed") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: vltnmmdv/deepseek-moe-16b-base)
    if (strcmp(s, "deepstackllama") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: menglc/deepstack-l-vicuna-7b)
    if (strcmp(s, "dharaar") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: codelion/dhara-250m-ar-base)
    if (strcmp(s, "dharaformaskeddiffusion") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: codelion/dhara-70m)
    if (strcmp(s, "editgptmistral") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: luoruipu1/Volcano-7b)
    if (strcmp(s, "energytransformer") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: cccczshao/CALM-M)
    if (strcmp(s, "erk") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: ecloudtech/Erk-14B)
    if (strcmp(s, "erniepixel") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: ernie-research/PixelGPT)
    if (strcmp(s, "evellama") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: BAAI/EVE-7B-HD-v1.0)
    if (strcmp(s, "evomistral") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: SakanaAI/EvoLLM-JP-v1-10B)
    if (strcmp(s, "exaone4forcausallmconv") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: SKIS-AI-Research/EXAConvo-Exp)
    if (strcmp(s, "extendedllama") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: normalcomputing/extended-mind-llama-2-7b)
    if (strcmp(s, "fast_dllm_qwen") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: Stalemartyr/finetune_fast_dLLM_1.5B_v2)
    if (strcmp(s, "fegeollama") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: NaughtyDog97/DFE-GPS-9B)
    if (strcmp(s, "fm9g") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: bowmanhan/jiuge-9G4B)
    if (strcmp(s, "freedomomega") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: UMBRANETWORK/Goblin-Glaude-4.5-Alcoholics)
    if (strcmp(s, "fuse2") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: Akahsizrr/Mini-Whale-1-12B)
    if (strcmp(s, "gemma3n") == 0) return RCPP_ARCH_GEMMA;  // VLM gemma text decoder (h4shy/gemma-3n-E2B-prototype-pytorch)
    if (strcmp(s, "geochatllama") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: MBZUAI/geochat-7B)
    if (strcmp(s, "gexqwen") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: MosRat/Gex_V1)
    if (strcmp(s, "gigachat35") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: ai-sage/GigaChat3.5-432B-A28B)
    if (strcmp(s, "gptjlora") == 0) return RCPP_ARCH_GPTJ;  // dump-id config (gpt2 profile: Enkhai/gpt-j-6b-8bit-lora)
    if (strcmp(s, "gptoptim") == 0) return RCPP_ARCH_GPT2;  // dump-id config (gpt2 profile: distributed/optimized-gpt2-250m-v0.1.2)
    if (strcmp(s, "gptrefact") == 0) return RCPP_ARCH_GPT2;  // dump-id config (gpt2 profile: refactai/Refact-1_6-base)
    if (strcmp(s, "gpts14m") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: AxiomicLabs/GPT-S-1.4M)
    if (strcmp(s, "gpts3") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: AxiomicLabs/GPT-S2-5M)
    if (strcmp(s, "gptsdprelu") == 0) return RCPP_ARCH_GPT2;  // dump-id config (gpt2 profile: adamroberts/tinystories-5090-sdprelu)
    if (strcmp(s, "gptx3") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: AxiomicLabs/GPT-S-5M)
    if (strcmp(s, "graphllama") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: Jiabin99/GraphGPT-7B-mix-all)
    if (strcmp(s, "heterollama") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: Jiabin99/HiGPT)
    if (strcmp(s, "hunyuanimage3forcausalmm") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: Runware/hunyuan-image)
    if (strcmp(s, "hybrid") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: Parveshiiii/Terminator-X)
    if (strcmp(s, "hyv3vl") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: sasa2000/Hy-Embodied-VLM-1.0-Text-Only)
    if (strcmp(s, "illuminator") == 0) return RCPP_ARCH_GPT2;  // dump-id config (gpt2 profile: Anipal/iLLuMinator)
    if (strcmp(s, "infllmv2_llama") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: LCM-Lab/infllm_llama)
    if (strcmp(s, "iquestpltcoder") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: Multilingual-Multimodal-NLP/LoopCoder-V2)
    if (strcmp(s, "jumplanderpython") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: jumplander/JL-Code-Python-97M)
    if (strcmp(s, "kanana2tiny") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: kakaocorp/kanana-2-1.3b-base)
    if (strcmp(s, "kanana2vec") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: kakaocorp/kanana-nano-2.1b-embedding)
    if (strcmp(s, "kangpt2") == 0) return RCPP_ARCH_GPT2;  // dump-id config (gpt2 profile: paolvz/gpt2kanpart12)
    if (strcmp(s, "kormoforcausallmwithmtp") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: kormo-lm/mtp_1view_1B_base_60BT)
    if (strcmp(s, "lam") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: Akhrots/LAM8B)
    if (strcmp(s, "lamedllama") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: GoodBaiBai88/M3D-LaMed-Llama-2-7B)
    if (strcmp(s, "lamedphi3") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: GoodBaiBai88/M3D-LaMed-Phi-3-4B)
    if (strcmp(s, "latentqwen") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: Tioe/LaTER-14B)
    if (strcmp(s, "lckvllama") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: whynlp/tinyllama-lckv-w2-100b)
    if (strcmp(s, "legollama") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: zwli/GroundingGPT)
    if (strcmp(s, "litallama") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: nateraw/lita)
    if (strcmp(s, "lizzy") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: flwrlabs/Lizzy-7B)
    if (strcmp(s, "llaaallama") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: LinkSoul/LLaSM-Baichuan)
    if (strcmp(s, "llamabutler") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: akhauriyash/Llama-3.2-3B-Butler)
    if (strcmp(s, "llamadeepseek") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: erax-ai/EraX-LLaMA3.1-8B-DeepSeekR1-MLA-MoE-Raw)
    if (strcmp(s, "llamahydra") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: Tweeties/tweety-tatar-hydra-base-7b-v24a)
    if (strcmp(s, "llamaladder") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: nanami/ladder-last16L-llama3.1-8binstruct-sft4k-stage2v03-bsize32-rkl8b)
    if (strcmp(s, "llamalongbel") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: AnonymousARR42/LongBEL_8B_MedMentions_st21pv)
    if (strcmp(s, "llamamla") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: BarraHome/llama3_2-1B-deepseek)
    if (strcmp(s, "llamaskipconnection") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: vkkhare/llama-skip)
    if (strcmp(s, "llamasparse") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: seele123/DeepSeek-R1-Distill-Llama-8B-TEAL)
    if (strcmp(s, "llamasyncabel") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: Aremaki/SynCABEL_QUAERO_EMEA)
    if (strcmp(s, "llamavidllava") == 0) return RCPP_ARCH_LLAMA;  // VLM llama text decoder (Nilesh360/llama-vid-7b-full-224-video-fps-1)
    if (strcmp(s, "llasa") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: bezzam/Llasa-1B)
    if (strcmp(s, "llavacrystal") == 0) return RCPP_ARCH_GPT2;  // dump-id config (gpt2 profile: qazimbhat1/my-model-repo3)
    if (strcmp(s, "llavallamaimagebindselect") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: dreamerlin/chatbind-7b-delta)
    if (strcmp(s, "llavaminerva") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: aimagelab/LLaVA-MORE-Minerva)
    if (strcmp(s, "llavaonevision1_5_") == 0) return RCPP_ARCH_QWEN2;  // VLM qwen text decoder (Jinghao-Guo/llavaov1.5-4B-instruct-converted-qwen)
    if (strcmp(s, "llavasearchllama") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: craigwu/seal_vqa_7b)
    if (strcmp(s, "llavastablelm_1_6b") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: FreedomIntelligence/ALLaVA-StableLM2-1_6B)
    if (strcmp(s, "llavastablelmepoch") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: NousResearch/Obsidian-3B-V0.5)
    if (strcmp(s, "llm2slmgpt2") == 0) return RCPP_ARCH_GPT2;  // dump-id config (gpt2 profile: toshi456/LLM-to-SLM-Alpaca)
    if (strcmp(s, "lumina") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: communityai/apt_zp_v1)
    if (strcmp(s, "mambainqwen") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: ucmp137538/miqhybrid_iter3)
    if (strcmp(s, "mgmllama") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: YanweiLi/MGM-8B)
    if (strcmp(s, "mimogdn") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: arianraje/mimo-7b-gdn-hybrid-init)
    if (strcmp(s, "minigeminillama") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: YanweiLi/MGM-34B)
    if (strcmp(s, "minimindomni") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: jingyaogong/minimind-3o)
    if (strcmp(s, "ministraldualrope") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: fin-ai-lab/aux-2015)
    if (strcmp(s, "mistralreconfig3") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: gyudong123/mistral_reconfigured_ver3_DPO)
    if (strcmp(s, "mixsensellama") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: Zero-Vision/Llama-3-MixSense)
    if (strcmp(s, "mllama") == 0) return RCPP_ARCH_LLAMA;  // VLM llama text decoder (RedHatAI/Llama-3.2-90B-Vision-Instruct-FP8-dynamic)
    if (strcmp(s, "modeling_camelidae.llama") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: hywu/Camelidae-8x7B)
    if (strcmp(s, "modeling_llama_butler.llamabutler") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: akhauriyash/Llama-3.2-1B-Butler)
    if (strcmp(s, "modelstarolmhead") == 0) return RCPP_ARCH_GPT2;  // dump-id config (gpt2 profile: C-a-Star-Technology-Official/StarO-AI-2.69-Super)
    if (strcmp(s, "monoid") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: NoesisLab/Spartacus-1B-Instruct)
    if (strcmp(s, "morllama") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: sudeshmu/fine_tune)
    if (strcmp(s, "moss") == 0) return RCPP_ARCH_GPT2;  // dump-id config (gpt2 profile: OpenMOSS-Team/moss-moon-003-base)
    if (strcmp(s, "multimodalllama") == 0) return RCPP_ARCH_LLAMA;  // VLM llama text decoder (AdrianBZG/llama-3-8B-Instruct-VisualQuestionAnswering)
    if (strcmp(s, "mymoss") == 0) return RCPP_ARCH_GPT2;  // dump-id config (gpt2 profile: ssbuild/moss-moon-003-sft-int4)
    if (strcmp(s, "nemotronh_nano_omni_reasoning_v3") == 0) return RCPP_ARCH_NEMOTRONH;  // VLM nemotron-h text decoder (unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning)
    if (strcmp(s, "nerfllmllama") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: andreamaduzzi/LLaNA-7B)
    if (strcmp(s, "neuralnet") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: metadeeai/neural-net)
    if (strcmp(s, "neutrino") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: neuralcrew/neutrino-instruct)
    if (strcmp(s, "olmo1124") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: shanearora/i-am-a-good-big-instruct-model)
    if (strcmp(s, "olmo1124forsequenceclassification") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: allenai/OLMo-2-1124-7B-RM-Preview)
    if (strcmp(s, "olmo2noqknormprenorm") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: allenai/Dense_1b_130B)
    if (strcmp(s, "olmo2retrofit") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: allenai/Olmo-3-7B-RL-Zero-Mix)
    if (strcmp(s, "olmo3siamesedepth") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: ArchSpace-Collection/OLMo3-1B-SiameseNorm-DepthAttention-stage1)
    if (strcmp(s, "omnispeech2sllama") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: Mihaiii/Llama-3.1-8B-Omni-abliterated)
    if (strcmp(s, "openllama") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: aerner/lm-v2)
    if (strcmp(s, "openpanguv2") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: blockblockblock/openPangu-2.0-Flash-exl3-4.0bpw)
    if (strcmp(s, "oryxllama") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: THUdyh/Oryx-34B-Image)
    if (strcmp(s, "oryxqwen") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: THUdyh/Oryx-7B)
    if (strcmp(s, "ospreyllama") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: sunshine-lwt/Osprey-7b)
    if (strcmp(s, "palo") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: MBZUAI/PALO-13B)
    if (strcmp(s, "panguembedded") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: FreedomIntelligence/openPangu-Embedded-7B)
    if (strcmp(s, "parallax") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: YifeiZuo/Parallax-0.6B)
    if (strcmp(s, "parambharatgen") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: bharatgenai/LegalParam)
    if (strcmp(s, "pointllmllama") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: RunsenXu/PointLLM_13B_v1.1)
    if (strcmp(s, "positionxlnet") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: efittschen/xlnet_2o)
    if (strcmp(s, "progressiveyocollama") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: hosseinbv/prog-y-tiny-llama-CDL-19)
    if (strcmp(s, "quasarlong") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: silx-ai/Quasar-Preview)
    if (strcmp(s, "qwerkyllamamambahybrid") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: QwerkyAI/Qwerky-Optimized-Llama3.2-Mamba-0.2-3B-Instruct)
    if (strcmp(s, "qyrouarch") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: Qyrou/Vega-1-65m-exp-base)
    if (strcmp(s, "rbdashllama") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: RBDash-Team/rbdash-v1-13b)
    if (strcmp(s, "recast1b_llama") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: appledora/recast-llama3.2-f8t2)
    if (strcmp(s, "recast7b_llama") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: appledora/recast2-G8W16H4)
    if (strcmp(s, "rixis1") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: rubenroy/NeuraNET-Zero-18B-Preview)
    if (strcmp(s, "rosex1") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: GODELEV/Rose-Medium)
    if (strcmp(s, "rwkvhybrid") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: RWKV-Red-Team/ARWKV-7B-Preview-0.1)
    if (strcmp(s, "sageloopcoder") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: sagea-ai/sage-oss-40b)
    if (strcmp(s, "scrapegoat") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: scrapegoat/Scrapegoat-Tiny-Coder)
    if (strcmp(s, "scratchllama") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: nagohachi/tiny-lm-japanese-500m-base-v1)
    if (strcmp(s, "share4vllama") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: Lin-Chen/ShareGPT4V-7B_Pretrained_vit-large336-l12_vicuna-7b-v1.5)
    if (strcmp(s, "shatest") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: gshasiri/myllama-for-vllm)
    if (strcmp(s, "sheikhf1") == 0) return RCPP_ARCH_GPT2;  // dump-id config (gpt2 profile: Sheikh-F1/Sheikh-F1)
    if (strcmp(s, "shikrallama") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: shikras/shikra-7b-delta-v1)
    if (strcmp(s, "siq_vl") == 0) return RCPP_ARCH_QWEN2;  // VLM qwen text decoder (duoan/siq-vl_siglip2-large-patch16-512_qwen2.5-1.5b-instruct_stage1)
    if (strcmp(s, "sky") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: 0labs-in/Sky-V1_3-5.5B)
    if (strcmp(s, "skycrest") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: 0labs-in/Sky-v2.0-11B)
    if (strcmp(s, "slicedllama") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: atultomar/prunedllama_0_5)
    if (strcmp(s, "sumiformaskgeneration") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: tohoku-nlp/sumi-7b)
    if (strcmp(s, "switchgpt2") == 0) return RCPP_ARCH_GPT2;  // dump-id config (gpt2 profile: crumb/Ducky-MoMoe-prototype-e4-ul2)
    if (strcmp(s, "tridafordlm") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: trillionlabs/Trida-7B-Preview)
    if (strcmp(s, "typhoon2audio2audio") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: typhoon-ai/llama3.1-typhoon2-audio-8b-instruct)
    if (strcmp(s, "upcycledsmollm") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: anothy1/SmolLM2-MoE-214M-A135M)
    if (strcmp(s, "valleyllama") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: luoruipu1/valley-13b-v1-delta)
    if (strcmp(s, "vcoderdsllavallama") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: shi-labs/vcoder_ds_llava-v1.5-7b)
    if (strcmp(s, "vcoderllavallama") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: shi-labs/vcoder_llava-v1.5-7b)
    if (strcmp(s, "videochatgptllama") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: heldJan/llama-2-7b-miniplatypus)
    if (strcmp(s, "vstreamllama") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: IVGSZ/Flash-VStream-7b)
    if (strcmp(s, "windedge") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: North-ML1/Wind-Edge-1.6-Base)
    if (strcmp(s, "wrappedllamav2") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: wolfgangshen/llama3-8b-musicai_maps_j0_multi)
    if (strcmp(s, "xllm") == 0) return RCPP_ARCH_LLAMA;  // dump-id config (llama profile: bgchoi/convx-16x-mistral)
    if (strcmp(s, "dmtdqwen3") == 0) return RCPP_ARCH_QWEN3;  // loose llama (rms+silu, rope default) [qwen3 family]
    if (strcmp(s, "a2dqwen2") == 0) return RCPP_ARCH_QWEN2;  // dump-id config (llama profile: ChaosAIVision/qwen2.5-1.5b-orca-bd3lm-sft-orca) [qwen2 family]
    if (strcmp(s, "a2dqwen3") == 0) return RCPP_ARCH_QWEN3;  // dump-id config (llama profile: dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1) [qwen3 family]
    if (strcmp(s, "bunnyqwen2") == 0) return RCPP_ARCH_QWEN2;  // dump-id config (llama profile: BAAI/Bunny-v1_0-2B-zh) [qwen2 family]
    if (strcmp(s, "devilqwen2") == 0) return RCPP_ARCH_QWEN2;  // dump-id config (llama profile: gaostar/DeViL-7B) [qwen2 family]
    if (strcmp(s, "fegeoqwen2") == 0) return RCPP_ARCH_QWEN2;  // dump-id config (llama profile: NaughtyDog97/DiagramFormalizer) [qwen2 family]
    if (strcmp(s, "hicomqwen2") == 0) return RCPP_ARCH_QWEN2;  // dump-id config (llama profile: lntzm/HICom_7B_qwen25_directg_local43_global32) [qwen2 family]
    if (strcmp(s, "impqwen2") == 0) return RCPP_ARCH_QWEN2;  // dump-id config (llama profile: MILVLG/Imp-v1.5-2B-Qwen1.5) [qwen2 family]
    if (strcmp(s, "infllmv2_qwen3") == 0) return RCPP_ARCH_QWEN3;  // dump-id config (llama profile: LCM-Lab/infllm_qwen3-8b) [qwen3 family]
    if (strcmp(s, "minigeminiqwen2") == 0) return RCPP_ARCH_QWEN2;  // dump-id config (llama profile: MonolithFoundation/Bumblebee) [qwen2 family]
    if (strcmp(s, "mobilintqwen2eagle3") == 0) return RCPP_ARCH_QWEN2;  // dump-id config (llama profile: mobilint/EAGLE3-JPharmatron-7B) [qwen2 family]
    if (strcmp(s, "mobilintqwen3eagle3") == 0) return RCPP_ARCH_QWEN3;  // dump-id config (llama profile: mobilint/EAGLE3-Qwen3-4B) [qwen3 family]
    if (strcmp(s, "oryxqwen2") == 0) return RCPP_ARCH_QWEN2;  // dump-id config (llama profile: THUdyh/Oryx-7B-Image) [qwen2 family]
    if (strcmp(s, "penguinvlqwen3") == 0) return RCPP_ARCH_QWEN3;  // dump-id config (llama profile: tencent/Penguin-VL-8B) [qwen3 family]
    if (strcmp(s, "qwen2_5_xray") == 0) return RCPP_ARCH_QWEN2;  // dump-id config (llama profile: astromindinc/am-xray-7b) [qwen2 family]
    if (strcmp(s, "qwen2adapter") == 0) return RCPP_ARCH_QWEN2;  // dump-id config (llama profile: zeliang0426/Fix-Strict_Darpo-cache-adapter-3k) [qwen2 family]
    if (strcmp(s, "qwen2bl") == 0) return RCPP_ARCH_QWEN2;  // dump-id config (llama profile: kartmannXu/Qwen2.5-3B-bl-0.4) [qwen2 family]
    if (strcmp(s, "qwen2ch") == 0) return RCPP_ARCH_QWEN2;  // dump-id config (llama profile: kartmannXu/Qwen2.5-3B-ch-0.25-tuned) [qwen2 family]
    if (strcmp(s, "qwen2forcausallmwithhrm") == 0) return RCPP_ARCH_QWEN2;  // dump-id config (llama profile: FlippyDora/Qwen2_5_3B_inst_hrm_init) [qwen2 family]
    if (strcmp(s, "qwen2hybrid") == 0) return RCPP_ARCH_QWEN2;  // dump-id config (llama profile: abcsk123/PyraCode-1.5B) [qwen2 family]
    if (strcmp(s, "qwen2layerwisesae") == 0) return RCPP_ARCH_QWEN2;  // dump-id config (llama profile: Vadim21221/qwen2_1_5b-instruct-layerwise-sae_lr_1e_7) [qwen2 family]
    if (strcmp(s, "qwen2mm") == 0) return RCPP_ARCH_QWEN2;  // dump-id config (llama profile: MentaCapture/qmodel) [qwen2 family]
    if (strcmp(s, "qwen2nomicvision") == 0) return RCPP_ARCH_QWEN2;  // dump-id config (llama profile: jonathanjordan21/Qwen2.5-Nomic-Vision) [qwen2 family]
    if (strcmp(s, "qwen2parscale") == 0) return RCPP_ARCH_QWEN2;  // dump-id config (llama profile: ParScale/ParScale-1.8B-P1) [qwen2 family]
    if (strcmp(s, "qwen2steeringvector") == 0) return RCPP_ARCH_QWEN2;  // dump-id config (llama profile: Vadim21221/qwen2_1_5b-instruct-steering-vector) [qwen2 family]
    if (strcmp(s, "qwen2ts") == 0) return RCPP_ARCH_QWEN2;  // dump-id config (llama profile: bytedance-research/ChatTS-14B) [qwen2 family]
    if (strcmp(s, "qwen3asvd") == 0) return RCPP_ARCH_QWEN3;  // dump-id config (llama profile: stealavie/Qwen3-4B-Thinking-2507-ASVD-2) [qwen3 family]
    if (strcmp(s, "qwen3attnres") == 0) return RCPP_ARCH_QWEN3;  // dump-id config (llama profile: Ethangou/attention-residuals-100M-block) [qwen3 family]
    if (strcmp(s, "qwen3audiowrapped") == 0) return RCPP_ARCH_QWEN3;  // dump-id config (llama profile: Blinorot/ALARM-P) [qwen3 family]
    if (strcmp(s, "qwen3kvpop") == 0) return RCPP_ARCH_QWEN3;  // dump-id config (llama profile: sirluk/Qwen3-8B-KVpop-4x) [qwen3 family]
    if (strcmp(s, "qwen3lcqatforcompression") == 0) return RCPP_ARCH_QWEN3;  // dump-id config (llama profile: w-hy21/lcqat_qwen3_1.7B) [qwen3 family]
    if (strcmp(s, "qwen3mhcforcausallmv2") == 0) return RCPP_ARCH_QWEN3;  // dump-id config (llama profile: theCoderWithHat/mhc-qwen3) [qwen3 family]
    if (strcmp(s, "qwen3mtp") == 0) return RCPP_ARCH_QWEN3;  // dump-id config (llama profile: Babu420/ninko-pinko-inference) [qwen3 family]
    if (strcmp(s, "qwen3recovered") == 0) return RCPP_ARCH_QWEN3;  // dump-id config (llama profile: atlasium-efficient/Qwen3-12B-20pct-Compressed-14B-EN-V1) [qwen3 family]
    if (strcmp(s, "qwen3scaleseq") == 0) return RCPP_ARCH_QWEN3;  // dump-id config (llama profile: tencent/Sequential-Hidden-Decoding-8B-n8-Instruct) [qwen3 family]
    if (strcmp(s, "rwkv7qwen2") == 0) return RCPP_ARCH_QWEN2;  // dump-id config (llama profile: recursal/QRWKV7-7B-Instruct) [qwen2 family]
    if (strcmp(s, "slicedqwen2") == 0) return RCPP_ARCH_QWEN2;  // dump-id config (llama profile: atultomar/prunedqwen_0_5) [qwen2 family]
    if (strcmp(s, "slicegptqwen2") == 0) return RCPP_ARCH_QWEN2;  // dump-id config (llama profile: qingfengyuhuoda/vllm-sliced-qwen2.5-14b-v12cuda) [qwen2 family]
    if (strcmp(s, "tpullama") == 0) return RCPP_ARCH_QWEN3;  // dump-id config (llama profile: benjamin/Qwen3-0.6B-Base-flax) [qwen3 family]
    if (strcmp(s, "tpuqwen3") == 0) return RCPP_ARCH_QWEN3;  // dump-id config (llama profile: benjamin/Qwen3-4B-Base-flax) [qwen3 family]
    if (strcmp(s, "videollama2qwen2") == 0) return RCPP_ARCH_QWEN2;  // dump-id config (llama profile: QuangTuan/MultiMood-7B-GRPO-VisualAudioText-Comp) [qwen2 family]
    if (strcmp(s, "videollama3qwen2") == 0) return RCPP_ARCH_QWEN2;  // dump-id config (llama profile: Fiaa/videollama3-s-t-a-g-e-5) [qwen2 family]
    if (strcmp(s, "vllmtfbqwen2") == 0) return RCPP_ARCH_QWEN2;  // dump-id config (llama profile: n1h111sm/TFB-Qwen2.5-3B-Instruct) [qwen2 family]
    if (strcmp(s, "wisentqwen2") == 0) return RCPP_ARCH_QWEN2;  // dump-id config (llama profile: wisent-ai/qwen2.5-coder-7b-wisent-caa) [qwen2 family]

    if (strcmp(s, "abia") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (abia)
    if (strcmp(s, "ahaqwen3") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (ahaqwen3)
    if (strcmp(s, "apollo") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (apollo)
    if (strcmp(s, "aprielh") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (apriel_h)
    if (strcmp(s, "aquiladense") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (aquiladense)
    if (strcmp(s, "armt") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (armt)
    if (strcmp(s, "axiom") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (axiom)
    if (strcmp(s, "axk1") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (axk1)
    if (strcmp(s, "baiwen3") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (baiwen3)
    if (strcmp(s, "boomer") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (boomer)
    if (strcmp(s, "breen") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (breen)
    if (strcmp(s, "brumby") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (brumby)
    if (strcmp(s, "buddygpt") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (buddygpt)
    if (strcmp(s, "bunnyminicpm") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (bunny-minicpm)
    if (strcmp(s, "cambrianphi3") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (cambrian_phi3)
    if (strcmp(s, "causallm") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (transformer)
    if (strcmp(s, "chemq3mtp") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (chemq3_mtp)
    if (strcmp(s, "clinicalllama") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (clinical-llama)
    if (strcmp(s, "clokcem") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (clokcem)
    if (strcmp(s, "continuum") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (continuum)
    if (strcmp(s, "cosmos") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (cosmos)
    if (strcmp(s, "creek") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (creek)
    if (strcmp(s, "crowelogicmini") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (crowe_logic_mini)
    if (strcmp(s, "cs336") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (cs336_transformer)
    if (strcmp(s, "dat") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (dat)
    if (strcmp(s, "deepllama") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (deep_llama)
    if (strcmp(s, "diffllama") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (diffllama)
    if (strcmp(s, "dpmm") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (dpmm)
    if (strcmp(s, "duchifat") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (duchifat)
    if (strcmp(s, "egollm") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (ego_llm)
    if (strcmp(s, "forgelm") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (forgelm)
    if (strcmp(s, "g9v3") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (g9v3)
    if (strcmp(s, "gigachataudio") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (gigachat_audio)
    if (strcmp(s, "gopu") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (gopu)
    if (strcmp(s, "gptosspuzzle") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (gpt_oss_puzzle)
    if (strcmp(s, "gritlm") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (None)
    if (strcmp(s, "grok") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (grok)
    if (strcmp(s, "h") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (h_model)
    if (strcmp(s, "halos") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (halo_s)
    if (strcmp(s, "hanzi") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (hanzi)
    if (strcmp(s, "helionosc") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (helion-osc)
    if (strcmp(s, "henyo") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (custom_henyo_culturax)
    if (strcmp(s, "hnet") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (None)
    if (strcmp(s, "icarus") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (icarus)
    if (strcmp(s, "illada") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (illada)
    if (strcmp(s, "induction") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (induction_lm)
    if (strcmp(s, "interns1") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (interns1)
    if (strcmp(s, "jetnemotron") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (jet_nemotron)
    if (strcmp(s, "jibay2") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (jibay2)
    if (strcmp(s, "jinsoollm") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (jinsoo_llm)
    if (strcmp(s, "jirackternary") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (jirack_ternary)
    if (strcmp(s, "jirackternary1b") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (jirack_ternary)
    if (strcmp(s, "jirackternarypro1b") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (jirack_ternary)
    if (strcmp(s, "kinoe") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (kinoe)
    if (strcmp(s, "led") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (led)
    if (strcmp(s, "ligergsa") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (liger_gsa)
    if (strcmp(s, "lingowhale") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (lingowhale)
    if (strcmp(s, "llamamixlora") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (llama_mixlora)
    if (strcmp(s, "lulu2") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (luluv2)
    if (strcmp(s, "mae") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (MAELM)
    if (strcmp(s, "magic") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (None)
    if (strcmp(s, "maplept") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (maplept)
    if (strcmp(s, "medusa") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (medusa)
    if (strcmp(s, "metadiffusion") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (metadiffusion)
    if (strcmp(s, "microllama") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (micro-llama)
    if (strcmp(s, "mightyllama") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (mighty-llama)
    if (strcmp(s, "mindi") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (mindi)
    if (strcmp(s, "minillama3") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (mini_llama3)
    if (strcmp(s, "minimix") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (None)
    if (strcmp(s, "miniqwen") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (mini_qwen)
    if (strcmp(s, "mistraldenseformer") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (mistral_denseformer)
    if (strcmp(s, "mists") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (mists)
    if (strcmp(s, "mnemosyne") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (mnemosyne)
    if (strcmp(s, "mobilereasoningllm") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (arka_v2_mobile)
    if (strcmp(s, "mobilintexaone4") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (mobilint-exaone4)
    if (strcmp(s, "moho") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (None)
    if (strcmp(s, "moss2") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (llava_moss2)
    if (strcmp(s, "mugen") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (mugen)
    if (strcmp(s, "murzik") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (murzik)
    if (strcmp(s, "myqwen2") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (myqwen_hf)
    if (strcmp(s, "nanollama") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (nano_llama)
    if (strcmp(s, "nanolm") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (nano-lm)
    if (strcmp(s, "nanoqwen") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (None)
    if (strcmp(s, "nda") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (nda)
    if (strcmp(s, "nebulax") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (nebula-x)
    if (strcmp(s, "neollm") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (neo)
    if (strcmp(s, "nextgen") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (next_gen_gpt)
    if (strcmp(s, "ngpt") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (None)
    if (strcmp(s, "omnilmm") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (omnilmm)
    if (strcmp(s, "open") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (openmodel)
    if (strcmp(s, "opengpt") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (None)
    if (strcmp(s, "palm") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (None)
    if (strcmp(s, "palullama") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (palullama)
    if (strcmp(s, "pebblelm") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (pebblellm)
    if (strcmp(s, "pheonix") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (Pheonix)
    if (strcmp(s, "pica") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (pica)
    if (strcmp(s, "pluto") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (astrai_pluto)
    if (strcmp(s, "pothana") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (pothana)
    if (strcmp(s, "privatewhisper") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (private-whisper)
    if (strcmp(s, "qed") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (qed)
    if (strcmp(s, "quietqwen") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (quietqwen)
    if (strcmp(s, "qwen2_5_memory") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (qwen2_5_memory)
    if (strcmp(s, "qwen3reasoning") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (None)
    if (strcmp(s, "race") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (None)
    if (strcmp(s, "rafflesia") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (rafflesia1)
    if (strcmp(s, "rapnss") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (rapnss)
    if (strcmp(s, "re_gpt") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (re_gpt)
    if (strcmp(s, "recallmllama") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (recallm_llama)
    if (strcmp(s, "recallmqwen2") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (recallm_qwen2)
    if (strcmp(s, "repeated") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (None)
    if (strcmp(s, "rio3") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (rio3)
    if (strcmp(s, "rnd1") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (rnd1)
    if (strcmp(s, "rogue") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (rogue_text)
    if (strcmp(s, "sardine") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (sardine)
    if (strcmp(s, "seed") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (seed)
    if (strcmp(s, "shivik") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (shivik)
    if (strcmp(s, "shivikcode") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (shivik_code)
    if (strcmp(s, "shrnk") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (shrnk)
    if (strcmp(s, "simple") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (simple_model)
    if (strcmp(s, "sixpert") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (sixpert)
    if (strcmp(s, "skyai") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (skyai)
    if (strcmp(s, "soka") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (Soka1.0)
    if (strcmp(s, "spec") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (spec-1-mini)
    if (strcmp(s, "ssllm") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (ssllm)
    if (strcmp(s, "swen") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (swen)
    if (strcmp(s, "switchllama") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (switchllama)
    if (strcmp(s, "tachbit") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (tachbit)
    if (strcmp(s, "tcv") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (TCVForCausalLM)
    if (strcmp(s, "teleflm") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (TeleFLM)
    if (strcmp(s, "tharo.g") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (Tharo.G-Eco)
    if (strcmp(s, "tinystate") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (tinystate)
    if (strcmp(s, "transllama") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (None)
    if (strcmp(s, "trouter") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (trouter)
    if (strcmp(s, "turingmm") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (turingMM)
    if (strcmp(s, "unbox") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (unbox)
    if (strcmp(s, "veridian") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (veridian)
    if (strcmp(s, "veronica") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (veronica)
    if (strcmp(s, "vexionlm") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (vexion_lm)
    if (strcmp(s, "veyra2apricot") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (veyra2_apricot)
    if (strcmp(s, "voxtral") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (voxtral)
    if (strcmp(s, "yivl") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (None)
    if (strcmp(s, "yua") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (yua)
    if (strcmp(s, "zagros") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (zagros)
    if (strcmp(s, "zebra") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (zebra)
    if (strcmp(s, "zhiyin") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (zhiyin)
    if (strcmp(s, "amit") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (amit)
    if (strcmp(s, "aries") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (aries)
    if (strcmp(s, "arlow") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (arlow)
    if (strcmp(s, "aurora") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (None)
    if (strcmp(s, "bagel") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (bagel)
    if (strcmp(s, "baseline") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (baseline_decoder)
    if (strcmp(s, "bitnetgpt") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (bitnet_gpt)
    if (strcmp(s, "canary") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (canary)
    if (strcmp(s, "cats") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (cats_model)
    if (strcmp(s, "clever") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (clever)
    if (strcmp(s, "curious") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (curious_text)
    if (strcmp(s, "dart") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (dart)
    if (strcmp(s, "dexv1") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (DexV1)
    if (strcmp(s, "dialogpt") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (None)
    if (strcmp(s, "elysium") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (elysium)
    if (strcmp(s, "ember") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (ember)
    if (strcmp(s, "emg") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (emg)
    if (strcmp(s, "enigma") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (enigma)
    if (strcmp(s, "friday") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (friday)
    if (strcmp(s, "gator") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (gator)
    if (strcmp(s, "gecko") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (gecko)
    if (strcmp(s, "geomotiongpt") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (geomotiongpt)
    if (strcmp(s, "gome") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (gome)
    if (strcmp(s, "gpt2custom") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (None)
    if (strcmp(s, "gpt2withroles") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (gpt2_with_roles)
    if (strcmp(s, "gptx") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (gptx)
    if (strcmp(s, "hanse") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (hanse)
    if (strcmp(s, "hils") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (olmo_hils)
    if (strcmp(s, "humangpt") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (None)
    if (strcmp(s, "humanv") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (humanv)
    if (strcmp(s, "hummingbird") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (hummingbird)
    if (strcmp(s, "ilama") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (ilama)
    if (strcmp(s, "imu1") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (imu_1)
    if (strcmp(s, "lenna") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (None)
    if (strcmp(s, "lime") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (lime)
    if (strcmp(s, "linglong") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (linglong)
    if (strcmp(s, "lizard") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (lizard)
    if (strcmp(s, "manta") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (manta)
    if (strcmp(s, "memory") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (memory_model)
    if (strcmp(s, "molformer") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (molformer)
    if (strcmp(s, "nepaligpt") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (nep_gptv1)
    if (strcmp(s, "ours") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (None)
    if (strcmp(s, "peer") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (peer)
    if (strcmp(s, "pega") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (pega)
    if (strcmp(s, "personamini") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (personamini)
    if (strcmp(s, "phi2") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (phi2)
    if (strcmp(s, "ptp") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (ptp)
    if (strcmp(s, "qgpt2") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (None)
    if (strcmp(s, "raptor") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (raptor)
    if (strcmp(s, "remote") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (remote)
    if (strcmp(s, "reward") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (None)
    if (strcmp(s, "rpt") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (rpt)
    if (strcmp(s, "sakhi") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (sakhi)
    if (strcmp(s, "scan") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (scan)
    if (strcmp(s, "sesame") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (None)
    if (strcmp(s, "shrike") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (shrike_lm)
    if (strcmp(s, "sinhalagpt") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (sinhala_gpt)
    if (strcmp(s, "smoothie") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (smoothie)
    if (strcmp(s, "solo") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (None)
    if (strcmp(s, "spect1") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (spect1)
    if (strcmp(s, "ssai") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (ssai)
    if (strcmp(s, "student") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (student)
    if (strcmp(s, "tensa") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (tensa)
    if (strcmp(s, "theta") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (theta)
    if (strcmp(s, "tinygpt2") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (tinygpt2)
    if (strcmp(s, "trol") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (trol)
    if (strcmp(s, "turkishgpt") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (turkish_gpt)
    if (strcmp(s, "vora") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (vora)
    if (strcmp(s, "yasin") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (yasin)
    if (strcmp(s, "miphaphi") == 0) return RCPP_ARCH_PHI;  // config-verified phi profile (mipha_phi)
    if (strcmp(s, "qwen2vlvae") == 0) return RCPP_ARCH_QWEN2;  // config-verified qwen2 profile (qwen2_vl_vae)

    if (strcmp(s, "asterisk") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (asterisk)
    if (strcmp(s, "baichuanm1") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (baichuan_m1)
    if (strcmp(s, "cambrianllama") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (cambrian_llama)
    if (strcmp(s, "codellama") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (None)
    if (strcmp(s, "cwic") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (cwic)
    if (strcmp(s, "distributedllama") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (None)
    if (strcmp(s, "dockgen") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (dockgen)
    if (strcmp(s, "ernie") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (ernie)
    if (strcmp(s, "fineweb") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (fineweb_decoder)
    if (strcmp(s, "grok1") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (git)
    if (strcmp(s, "keylm75m") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (keylm75m)
    if (strcmp(s, "kirim") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (kirim)
    if (strcmp(s, "kogum") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (kogum)
    if (strcmp(s, "lille") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (lille-130m)
    if (strcmp(s, "llada") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (llada)
    if (strcmp(s, "llama2bias") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (llama2_bias)
    if (strcmp(s, "llavaminillama") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (llava_mini_llama)
    if (strcmp(s, "llavaphi3") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (llava_phi3)
    if (strcmp(s, "lstllama") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (lst)
    if (strcmp(s, "maincoder") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (maincoder)
    if (strcmp(s, "maira2") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (maira2)
    if (strcmp(s, "minicpmsala") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (minicpm_sala)
    if (strcmp(s, "mobilellama") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (mobilevlm)
    if (strcmp(s, "molllama") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (mol_llama)
    if (strcmp(s, "mymodel") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (None)
    if (strcmp(s, "nemotronflash") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (nemotron_flash)
    if (strcmp(s, "neuroblast") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (neuroblast)
    if (strcmp(s, "nova") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (nova)
    if (strcmp(s, "nova1") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (nova1)
    if (strcmp(s, "peftmodel") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (peft)
    if (strcmp(s, "plasmidlm") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (plasmid_lm)
    if (strcmp(s, "smallthinker") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (smallthinker)
    if (strcmp(s, "spatiallmllama") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (spatiallm_llama)
    if (strcmp(s, "susono") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (susono)
    if (strcmp(s, "swarm") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (swarm_agi)
    if (strcmp(s, "telechat3") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (telechat3)
    if (strcmp(s, "tinyllm") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (tinyllm)
    if (strcmp(s, "trillion") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (trillion)
    if (strcmp(s, "tttlinear") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (ttt_linear)
    if (strcmp(s, "tttmlp") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (ttt_mlp)
    if (strcmp(s, "vaetki") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (vaetki)
    if (strcmp(s, "vwllama") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (None)
    if (strcmp(s, "zeus") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (zeusmm)
    if (strcmp(s, "codeshell") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (codeshell)
    if (strcmp(s, "customgpt2") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (None)
    if (strcmp(s, "doge2") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (doge2)
    if (strcmp(s, "gpt2mimo") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (gpt2mimo)
    if (strcmp(s, "kayra") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (kayra)
    if (strcmp(s, "latex") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (latex)
    if (strcmp(s, "lola") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (lola_v1)
    if (strcmp(s, "mola") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (mola_lm)
    if (strcmp(s, "ndm") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (ndm)
    if (strcmp(s, "ysnrfd") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (ysnrfd)
    if (strcmp(s, "tinyllavaphi") == 0) return RCPP_ARCH_PHI;  // config-verified phi profile (tiny_llava_phi)
    if (strcmp(s, "emovaqwen2") == 0) return RCPP_ARCH_QWEN2;  // qwen-family text decoder (emova_qwen2)
    if (strcmp(s, "qwen3ts") == 0) return RCPP_ARCH_QWEN3;  // qwen-family text decoder (qwen3ts)

    if (strcmp(s, "helpingai") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (HelpingAI)
    if (strcmp(s, "maple") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (maple-preview)
    if (strcmp(s, "wedlm") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (Tencent WEDLM)
    if (strcmp(s, "helium") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (L3-8B-helium3)
    if (strcmp(s, "bluelm") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (vivo BlueLM)
    if (strcmp(s, "bunnyllama") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (Bunny VLM, llama text decoder)
    if (strcmp(s, "longllama") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (LongLLaMA)
    if (strcmp(s, "minimind") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (MiniMind)
    if (strcmp(s, "bolmo") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (Bolmo-1B)
    if (strcmp(s, "imp") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (Imp VLM)
    if (strcmp(s, "llavamistral7") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile
    if (strcmp(s, "tpp") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile
    if (strcmp(s, "monet") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (huggingtweets)
    if (strcmp(s, "gpt3dev") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile
    if (strcmp(s, "gpt2l") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (homergpt2l)
    if (strcmp(s, "lordcoder") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile
    if (strcmp(s, "gear") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (dialogpt-small)
    if (strcmp(s, "hawk") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (gpt-morty)
    if (strcmp(s, "smallm") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile
    if (strcmp(s, "aragpt2") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (aragpt2-mega)
    if (strcmp(s, "arctic") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile
    if (strcmp(s, "customgpt") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (custom-gpt2)
    if (strcmp(s, "isaac") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (distilgpt2)
    if (strcmp(s, "otter") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (dialogpt-small)
    if (strcmp(s, "taonet") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile
    if (strcmp(s, "idefics") == 0) return RCPP_ARCH_LLAMA;  // Idefics-80B (VLM, llama-2 text decoder)
    if (strcmp(s, "cogagent") == 0) return RCPP_ARCH_LLAMA;  // CogAgent (VLM, llama-2 text decoder)
    if (strcmp(s, "idefics3") == 0) return RCPP_ARCH_LLAMA;  // Idefics3-8B-Llama3 (VLM, llama-3 text decoder)
    if (strcmp(s, "detikzifycambrian") == 0) return RCPP_ARCH_QWEN2;  // Detikzify-Cambrian (VLM, qwen2 text decoder)
    if (strcmp(s, "lfm2vl") == 0) return RCPP_ARCH_LFM2;  // LFM2-VL (VLM, lfm2 text decoder)
    if (strcmp(s, "cohere2vision") == 0) return RCPP_ARCH_COHERE2;  // Cohere2-Vision (VLM, cohere2 text decoder)
    if (strcmp(s, "llavaqwen1_5") == 0) return RCPP_ARCH_QWEN2;  // LLaVA-Qwen1.5 (VLM, qwen2 text decoder)
    if (strcmp(s, "qwen2_5omnithinker") == 0) return RCPP_ARCH_QWEN2;  // Qwen2.5-OmniThinker (VLM, qwen2.5 text decoder)
    if (strcmp(s, "vmistral") == 0) return RCPP_ARCH_MISTRAL;  // VMistral (VLM, mistral text decoder)
    if (strcmp(s, "deep") == 0) return RCPP_ARCH_DEEPSEEK_V4;  // DeepForCausalLM (deepseek-v4 config)
    if (strcmp(s, "mobilintqwen2") == 0) return RCPP_ARCH_QWEN2;  // mobilint Qwen2.5 (qwen2 layout)
    if (strcmp(s, "qwen2bm") == 0) return RCPP_ARCH_QWEN2;  // QWEN-2B-More (qwen2 layout)

    if (strcmp(s, "openaigpt") == 0) return RCPP_ARCH_GPT2;  // openai-gpt (gpt2 layout, Conv1D)
    if (strcmp(s, "ctrl") == 0) return RCPP_ARCH_GPT2;  // CTRL (gpt2 layout, extra conditioning embed ignored)
    if (strcmp(s, "chessgpt") == 0) return RCPP_ARCH_GPTNEOX;  // ChessGPT (gpt_neox config, verified)
    if (strcmp(s, "opensci") == 0) return RCPP_ARCH_LLAMA;  // OpenSci (med-llama-7b config, llama profile)
    if (strcmp(s, "myllama") == 0) return RCPP_ARCH_LLAMA;  // myllama (LLaMa model_type, silu+rms)
    if (strcmp(s, "plamo") == 0) return RCPP_ARCH_LLAMA;  // PLaMo-13B (llama profile)
    if (strcmp(s, "internvlchat") == 0) return RCPP_ARCH_LLAMA;  // InternVL-Chat (VLM, vicuna/llama text decoder)
    if (strcmp(s, "ncpolmo3") == 0) return RCPP_ARCH_OLMO;  // NCP-Olmo3 (olmo3 family)
    if (strcmp(s, "detikzify") == 0) return RCPP_ARCH_LLAMA;  // Detikzify-CL-7B (VLM, llama text decoder)
    if (strcmp(s, "cogvlm") == 0) return RCPP_ARCH_LLAMA;  // CogVLM (VLM, llama-2 text decoder)
    if (strcmp(s, "aquila") == 0) return RCPP_ARCH_LLAMA;  // Aquila/Aquila2 (llama-derived; qwen3.5-moe VLMs fail loud)
    if (strcmp(s, "dream") == 0) return RCPP_ARCH_GEMMA;  // DreamFast (VLM, gemma-3 text decoder)
    if (strcmp(s, "index") == 0) return RCPP_ARCH_GEMMA;  // Index (VLM, gemma-3 text decoder)
    if (strcmp(s, "mimov2flash") == 0) return RCPP_ARCH_QWEN2;  // MiMo-V2-Flash (qwen2-derived)
    if (strcmp(s, "mimo") == 0) return RCPP_ARCH_QWEN2;  // MiMo-v2.5 (qwen2-derived)
    if (strcmp(s, "qwen2_5omni") == 0) return RCPP_ARCH_QWEN2;  // Qwen2.5-Omni (VLM, qwen2.5 text decoder)
    if (strcmp(s, "llavanext") == 0) return RCPP_ARCH_QWEN2;  // LLaVA-NeXT (qwen text decoder variant)
    if (strcmp(s, "molm") == 0) return RCPP_ARCH_OLMO;  // MoLM/Molmo (VLM, olmo text decoder)
    if (strcmp(s, "moondream") == 0) return RCPP_ARCH_PHI;  // Moondream (VLM, phi-1.5 text decoder)

    if (strcmp(s, "bitllama") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (bit_llama)
    if (strcmp(s, "iquestcoder") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (IQuest-Coder-7B)
    if (strcmp(s, "mplugowl2llama") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (VLM, llama text decoder)
    if (strcmp(s, "zhinao") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (360Zhinao-7B)
    if (strcmp(s, "kimilinear") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (Kimi-Linear, dense-declared)
    if (strcmp(s, "flexolmo") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (FlexOLMo, dense-declared)
    if (strcmp(s, "hymba") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (Hymba, dense-declared)
    if (strcmp(s, "longcatflashngram") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile
    if (strcmp(s, "arcee") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (Arcee trinity-mini)
    if (strcmp(s, "revision") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (Mark1-revision)
    if (strcmp(s, "tinyllava") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (VLM, llama text decoder)
    if (strcmp(s, "ernie4_5_") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (ERNIE-4.5 dense)
    if (strcmp(s, "ernie4_5") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (ERNIE-4.5 dense)
    if (strcmp(s, "mobilintllama") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (mobilint Llama-3.1)
    if (strcmp(s, "minicpm3") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (MiniCPM3)
    if (strcmp(s, "yuan") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (Tencent Yuan)
    if (strcmp(s, "anemone") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (law-guardian-llama)
    if (strcmp(s, "babyllama") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile
    if (strcmp(s, "iquestloopcoder") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile
    if (strcmp(s, "moshi") == 0) return RCPP_ARCH_LLAMA;  // config-verified llama profile (Moshi text tower, dense-declared)
    if (strcmp(s, "transformer") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (codeparrot-small)
    if (strcmp(s, "lisa") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (dialogpt-based)
    if (strcmp(s, "helix") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (dialogpt-small)
    if (strcmp(s, "sdar") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile
    if (strcmp(s, "ouro") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile
    if (strcmp(s, "doge") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (dialogpt)
    if (strcmp(s, "skywork") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (skycode)
    if (strcmp(s, "progen") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (ProGen)
    if (strcmp(s, "gpt2a") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (distilgpt2)
    if (strcmp(s, "quiet") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile
    if (strcmp(s, "nandi") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile
    if (strcmp(s, "pegasus") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (gpt-jonsnow)
    if (strcmp(s, "tinygpt") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (tinygpt2)
    if (strcmp(s, "ttt") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile
    if (strcmp(s, "avey") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile
    if (strcmp(s, "mega") == 0) return RCPP_ARCH_GPT2;  // config-verified gpt2 profile (aragpt2-mega)
    if (strcmp(s, "mobilintqwen3") == 0) return RCPP_ARCH_QWEN3;  // config-verified qwen3 (mobilint Qwen3-0.6B)

    if (strcmp(s, "chess") == 0) return RCPP_ARCH_GPT2;  // ChessForCausalLM (LLM-course chess_transformer — gpt2 layout, verified vs model.py 2026-08-15)
    if (strcmp(s, "chesstransformer") == 0) return RCPP_ARCH_GPT2;  // model_type chess_transformer (gpt2 layout)

    
    if (strcmp(s, "gemma4assistant") == 0) return RCPP_ARCH_GEMMA;  // model_type gemma4_assistant (gemma4 family)
    if (strcmp(s, "phi3v") == 0) return RCPP_ARCH_PHI;  // model_type phi3_v (phi-3 vision, text decoder phi3)
    if (strcmp(s, "phi4mm") == 0) return RCPP_ARCH_PHI;  // model_type phi4mm (phi-4 multimodal, text decoder phi4)
    if (strcmp(s, "moondream1") == 0) return RCPP_ARCH_PHI;  // model_type moondream1 (moondream VLM, phi-1.5 text decoder)
    if (strcmp(s, "llavamistral") == 0) return RCPP_ARCH_MISTRAL;  // model_type llava_mistral (VLM, mistral text decoder)
    if (strcmp(s, "sparsemistral") == 0) return RCPP_ARCH_MISTRAL;  // model_type sparse_mistral (mistral layout)
    if (strcmp(s, "mimov2") == 0) return RCPP_ARCH_QWEN2;  // model_type mimo_v2 (MiMo, qwen2-derived)
    if (strcmp(s, "mplugowl2") == 0) return RCPP_ARCH_LLAMA;  // model_type mplug_owl2 (VLM, llama-2 text decoder)
    if (strcmp(s, "orion") == 0) return RCPP_ARCH_LLAMA;  // model_type orion (Orion-14B, llama layout)
    if (strcmp(s, "qwen35text") == 0) return RCPP_ARCH_QWEN35;  // model_type qwen3_5_text (tag form of qwen3_5)
    if (strcmp(s, "refinedwebmodel") == 0) return RCPP_ARCH_FALCON;  // model_type RefinedWebModel (falcon-rw layout)
    if (strcmp(s, "phimsft") == 0) return RCPP_ARCH_PHI;  // model_type phi-msft (Microsoft Phi)

    
    
    
    
    if (strcmp(s, "nemotronh") == 0) return RCPP_ARCH_NEMOTRONH; // NemotronHForCausalLM
    if (strcmp(s, "nemotron_h") == 0) return RCPP_ARCH_NEMOTRONH; // HF model_type
    if (strcmp(s, "qwen3next") == 0) return RCPP_ARCH_QWEN3NEXT;
    if (strcmp(s, "qwen3_5_moe_text") == 0) return RCPP_ARCH_QWEN3NEXT;  // Qwen3.5-MoE text decoder = GatedDeltaNet  // Qwen3NextForCausalLM
    if (strcmp(s, "qwen3_next") == 0) return RCPP_ARCH_QWEN3NEXT;  // HF model_type
    if (strcmp(s, "minimaxm2") == 0) return RCPP_ARCH_MINIMAXM2;   // MiniMaxM2ForCausalLM
    if (strcmp(s, "minimax_m2") == 0) return RCPP_ARCH_MINIMAXM2;  // HF model_type
    if (strcmp(s, "cohere2") == 0) return RCPP_ARCH_COHERE2;       // Cohere2ForCausalLM
    if (strcmp(s, "cohere2_model") == 0) return RCPP_ARCH_COHERE2;  // HF model_type
    if (strcmp(s, "falconh1") == 0) return RCPP_ARCH_FALCONH1;     // FalconH1ForCausalLM
    if (strcmp(s, "falcon_h1") == 0) return RCPP_ARCH_FALCONH1;    // HF model_type
    if (strcmp(s, "rwkv") == 0) return RCPP_ARCH_RWKV;             // RwkvForCausalLM (RWKV-4)
    if (strcmp(s, "granitemoehybrid") == 0) return RCPP_ARCH_GRANITEMOEHYBRID;  // GraniteMoeHybridForCausalLM
    if (strcmp(s, "lfm2_moe") == 0) return RCPP_ARCH_LFM2MOE;                  // Lfm2MoeForCausalLM
    if (strcmp(s, "lfm2moe") == 0) return RCPP_ARCH_LFM2MOE;                   // class name
    if (strcmp(s, "hy_v3") == 0) return RCPP_ARCH_HYV3;                               // HYV3ForCausalLM
    if (strcmp(s, "hyv3") == 0) return RCPP_ARCH_HYV3;                                // class name
    if (strcmp(s, "afmoe") == 0) return RCPP_ARCH_AFMOE;                              // AfmoeForCausalLM
    if (strcmp(s, "ernie4_5_moe") == 0) return RCPP_ARCH_ERNIE45MOE;                   // Ernie4_5_MoeForCausalLM
    if (strcmp(s, "ernie45moe") == 0) return RCPP_ARCH_ERNIE45MOE;                     // class name
    if (strcmp(s, "mellum") == 0) return RCPP_ARCH_MELLUM;                               // MellumForCausalLM
    if (strcmp(s, "phimoe") == 0) return RCPP_ARCH_PHIMOE;                               // PhimoeForCausalLM
    if (strcmp(s, "minimax") == 0) return RCPP_ARCH_MINIMAX;                             // MiniMaxForCausalLM
    if (strcmp(s, "cohere2_moe") == 0) return RCPP_ARCH_COHERE2MOE;                       // Cohere2MoeForCausalLM
    if (strcmp(s, "cohere2moe") == 0) return RCPP_ARCH_COHERE2MOE;                        // class name
    if (strcmp(s, "exaone_moe") == 0) return RCPP_ARCH_EXAONEMOE;                         // ExaoneMoeForCausalLM
    if (strcmp(s, "exaonemoe") == 0) return RCPP_ARCH_EXAONEMOE;                          // class name
    if (strcmp(s, "falcon_mamba") == 0) return RCPP_ARCH_FALCONMAMBA;                     // FalconMambaForCausalLM
    if (strcmp(s, "falconmamba") == 0) return RCPP_ARCH_FALCONMAMBA;                      // class name
    if (strcmp(s, "jetmoe") == 0) return RCPP_ARCH_JETMOE;                               // JetMoeForCausalLM
    // NOTE: rwkv5/rwkv6 map to the 4/5/6 RWKV backend; rwkv7 (Goose) is
    // data-dependent — its own RCPP_ARCH_RWKV7 token (engine work in deck).
    if (strcmp(s, "rwkv7") == 0) return RCPP_ARCH_RWKV7;                                // RWKV-7 Goose (data-dependent recurrence)
    // ── Moonshot Kimi family ──
    if (strcmp(s, "kimi_k3")   == 0) return RCPP_ARCH_KIMI_K3;
    if (strcmp(s, "kimi")      == 0) return RCPP_ARCH_KIMI_K3;
    if (strcmp(s, "moonlight") == 0) return RCPP_ARCH_MOONLIGHT;
    if (strcmp(s, "kimi_vl")   == 0) return RCPP_ARCH_KIMI_VL;
    if (strcmp(s, "kimi_vl_a3b") == 0) return RCPP_ARCH_KIMI_VL;
    // ── Qwen3.6-MoE (shared-expert MoE, Qwen2-compatible attention) ──
    if (strcmp(s, "qwen35")   == 0) return RCPP_ARCH_QWEN35;
    if (strcmp(s, "qwen35moe") == 0) return RCPP_ARCH_QWEN35;
    // ── VLM conditional-generation classes (MAX-style: text decoder maps to
    //    the base token; vision tower is a separate workstream, NO-MORE-SECRETS
    //    documented in docs/wiki/models.md) ──
    if (strcmp(s, "qwen3_5")     == 0) return RCPP_ARCH_QWEN35;   // Qwen3.5 (GDN dense)
    if (strcmp(s, "qwen3_5moe")  == 0) return RCPP_ARCH_QWEN35;   // Qwen3.5-MoE
    if (strcmp(s, "mistral3")    == 0) return RCPP_ARCH_MISTRAL;  // Mistral3 (text decoder = mistral)
    if (strcmp(s, "qwen2_5_vl")  == 0) return RCPP_ARCH_QWEN2VL;  // Qwen2.5-VL (text decoder = qwen2)
    if (strcmp(s, "qwen3vlmoe")  == 0) return RCPP_ARCH_QWEN3VL;  // Qwen3-VL-MoE
    if (strcmp(s, "gemma4unified") == 0) return RCPP_ARCH_GEMMA;
    if (strcmp(s, "llavallama")   == 0) return RCPP_ARCH_LLAMA;  // LLaVA w/ llama-2 text decoder (MAX-style: decoder maps to base token)
    if (strcmp(s, "cambrianqwen") == 0) return RCPP_ARCH_QWEN2;   // Cambrian-1 (qwen2 text decoder)
    if (strcmp(s, "llavaqwen2")   == 0) return RCPP_ARCH_QWEN2;   // LLaVA w/ qwen2 text decoder
    if (strcmp(s, "hunyuandensev1") == 0) return RCPP_ARCH_LLAMA; // HunYuan dense V1 (llama-layout)
    if (strcmp(s, "seedoss")      == 0) return RCPP_ARCH_LLAMA;   // ByteDance Seed-OSS dense (llama-layout, GQA)
    if (strcmp(s, "glm4")         == 0) return RCPP_ARCH_LLAMA;   // GLM-4 (llama + partial-rope 0.5 + qkv bias)
    if (strcmp(s, "glm4moe")      == 0) return RCPP_ARCH_LLAMA;   // GLM-4-MoE (same attn + deepseek-style gating)
    if (strcmp(s, "glmmoedsa")    == 0) return RCPP_ARCH_LLAMA;   // GLM-4.5 MoE (DSA attention)
    if (strcmp(s, "glm4moelite")  == 0) return RCPP_ARCH_LLAMA;   // GLM-4-MoE-Lite
  // Gemma4-Unified (text decoder = gemma)
    if (strcmp(s, "qwen3_5vl")   == 0) return RCPP_ARCH_QWEN35;   // Qwen3.5-VL (text decoder = qwen3.5)
    // ── HF model_type values (snake_case family tags; the reader falls back
    //    to these when the class name maps UNKNOWN — extraction 2026-08-15) ──
    if (strcmp(s, "gpt_neox")    == 0) return RCPP_ARCH_GPTNEOX;  // GPTNeoXConfig model_type
    if (strcmp(s, "gpt_neo")     == 0) return RCPP_ARCH_GPTNEO;   // GPTNeoConfig model_type
    if (strcmp(s, "gpt_j")       == 0) return RCPP_ARCH_GPTJ;
    if (strcmp(s, "gpt_bigcode") == 0) return RCPP_ARCH_LLAMA;    // StarCoder/GPT-BigCode layout
    if (strcmp(s, "qwen2_vl")    == 0) return RCPP_ARCH_QWEN2VL;  // Qwen2VLConfig model_type
    if (strcmp(s, "qwen3_vl")    == 0) return RCPP_ARCH_QWEN3VL;
    if (strcmp(s, "qwen3_moe")   == 0) return RCPP_ARCH_QWEN3;
    if (strcmp(s, "qwen2_moe")   == 0) return RCPP_ARCH_QWEN2;
    if (strcmp(s, "mistral_moe") == 0) return RCPP_ARCH_MISTRAL;  // mixtral-style
    if (strcmp(s, "granite_moe") == 0) return RCPP_ARCH_GEMMA;    // granite MoE (gemma layout)
    if (strcmp(s, "gemma3_text") == 0) return RCPP_ARCH_GEMMA;    // Gemma3TextConfig
    if (strcmp(s, "gemma4_text") == 0) return RCPP_ARCH_GEMMA;
    if (strcmp(s, "llava")       == 0) return RCPP_ARCH_QWEN2VL;  // LLaVA model_type
    if (strcmp(s, "llava_llama") == 0) return RCPP_ARCH_LLAMA;    // LLaVA-llama text decoder
    if (strcmp(s, "llava_qwen2") == 0) return RCPP_ARCH_QWEN2;
    if (strcmp(s, "deepseek_v2") == 0) return RCPP_ARCH_DEEPSEEK;
    if (strcmp(s, "deepseek_v3") == 0) return RCPP_ARCH_DEEPSEEK;
    if (strcmp(s, "deepseek_v4") == 0) return RCPP_ARCH_DEEPSEEK_V4;
    if (strcmp(s, "stablelm_epoch") == 0) return RCPP_ARCH_LLAMA;
    if (strcmp(s, "openelm")     == 0) return RCPP_ARCH_LLAMA;
    if (strcmp(s, "cohere")      == 0) return RCPP_ARCH_LLAMA;
    if (strcmp(s, "cambrian_qwen") == 0) return RCPP_ARCH_QWEN2;  // Cambrian-1 (qwen2 text)
    if (strcmp(s, "hunyuan_v1_dense") == 0) return RCPP_ARCH_LLAMA;
    if (strcmp(s, "exaone4")     == 0) return RCPP_ARCH_LLAMA;
    if (strcmp(s, "nemotron")    == 0) return RCPP_ARCH_LLAMA;
    if (strcmp(s, "fp8_qwen3")   == 0) return RCPP_ARCH_QWEN3;    // FP8 wrapper, same layout
    if (strcmp(s, "fp8_qwen2")   == 0) return RCPP_ARCH_QWEN2;
    if (strcmp(s, "fp8_llama")   == 0) return RCPP_ARCH_LLAMA;
    if (strcmp(s, "bit_llama")   == 0) return RCPP_ARCH_LLAMA;    // BitNet-style llama

    // ── 2026-08-15 census pass-3: new-family mappings (class + model_type) ──
    // Each line is the class name (stripped) and/or the model_type fallback.
    if (strcmp(s, "llama4") == 0) return RCPP_ARCH_LLAMA4;       // Llama4ForCausalLM / llama4_text
    if (strcmp(s, "llama4_text") == 0) return RCPP_ARCH_LLAMA4;
    if (strcmp(s, "jais") == 0) return RCPP_ARCH_JAIS;           // JAISLMHeadModel
    if (strcmp(s, "dynamicforgetting") == 0) return RCPP_ARCH_DYNAMICFORGETTING;
    if (strcmp(s, "dynamic_forgetting") == 0) return RCPP_ARCH_DYNAMICFORGETTING;
    if (strcmp(s, "dynamicslidingwindow") == 0) return RCPP_ARCH_DYNAMICSLIDINGWINDOW;
    if (strcmp(s, "dynamic_sliding_window") == 0) return RCPP_ARCH_DYNAMICSLIDINGWINDOW;
    if (strcmp(s, "kormo") == 0) return RCPP_ARCH_KORMO;         // KORMoForCausalLM
    if (strcmp(s, "chatglm") == 0) return RCPP_ARCH_CHATGLM;    // ChatGLMModel
    if (strcmp(s, "sarvammoe") == 0) return RCPP_ARCH_SARVAM;    // SarvamMoEForCausalLM
    if (strcmp(s, "sarvam_moe") == 0) return RCPP_ARCH_SARVAM;
    if (strcmp(s, "sarvammla") == 0) return RCPP_ARCH_SARVAM;    // SarvamMLAForCausalLM
    if (strcmp(s, "sarvam_mla") == 0) return RCPP_ARCH_SARVAM;
    if (strcmp(s, "raven") == 0) return RCPP_ARCH_RAVEN;         // RavenForCausalLM
    if (strcmp(s, "huginn_raven") == 0) return RCPP_ARCH_RAVEN;
    if (strcmp(s, "talkie") == 0) return RCPP_ARCH_TALKIE;      // TalkieForCausalLM
    if (strcmp(s, "llada2moemodellm") == 0) return RCPP_ARCH_LLADA2;  // LLaDA2MoeModelLM
    if (strcmp(s, "llada2_moe") == 0) return RCPP_ARCH_LLADA2;
    if (strcmp(s, "looplm") == 0) return RCPP_ARCH_LOOPLM;       // LoopLMForCausalLM
    if (strcmp(s, "loop-lm") == 0) return RCPP_ARCH_LOOPLM;
    if (strcmp(s, "step3p5") == 0) return RCPP_ARCH_STEP3P5;    // Step3p5ForCausalLM
    if (strcmp(s, "daisy") == 0) return RCPP_ARCH_DAISY;        // DaisyForCausalLM
    if (strcmp(s, "multiscale") == 0) return RCPP_ARCH_MULTISCALE;
    if (strcmp(s, "multiscale_transformer") == 0) return RCPP_ARCH_MULTISCALE;
    if (strcmp(s, "skipmiddle") == 0) return RCPP_ARCH_SKIPMIDDLE;
    if (strcmp(s, "motif") == 0) return RCPP_ARCH_MOTIF;        // MotifForCausalLM
    if (strcmp(s, "quasar") == 0) return RCPP_ARCH_QUASAR;      // QuasarForCausalLM
    if (strcmp(s, "hgrn") == 0) return RCPP_ARCH_HGRN;          // HGRNForCausalLM
    if (strcmp(s, "hgrn_bit") == 0) return RCPP_ARCH_HGRN;
    if (strcmp(s, "retnet") == 0) return RCPP_ARCH_RETNET;      // RetNetForCausalLM
    if (strcmp(s, "cubelm") == 0) return RCPP_ARCH_CUBELM;      // CubeLM
    if (strcmp(s, "recurrentgemma") == 0) return RCPP_ARCH_RECURRENTGEMMA;
    if (strcmp(s, "recurrent_gemma") == 0) return RCPP_ARCH_RECURRENTGEMMA;
    if (strcmp(s, "lightningtransformermodel") == 0) return RCPP_ARCH_LIGHTNINGTRANSFORMER;
    if (strcmp(s, "lightning_transformer") == 0) return RCPP_ARCH_LIGHTNINGTRANSFORMER;
    if (strcmp(s, "spikewhalelm") == 0) return RCPP_ARCH_SPIKEWHALE;
    if (strcmp(s, "spike_whale") == 0) return RCPP_ARCH_SPIKEWHALE;
    if (strcmp(s, "stl") == 0) return RCPP_ARCH_STL;            // STLDec16
    if (strcmp(s, "stldec16") == 0) return RCPP_ARCH_STL;
    if (strcmp(s, "xpertgpt") == 0) return RCPP_ARCH_XPERTGPT;
    if (strcmp(s, "yatgpt") == 0) return RCPP_ARCH_YATGPT;
    if (strcmp(s, "yatnmn_gpt") == 0) return RCPP_ARCH_YATGPT;
    if (strcmp(s, "ceno") == 0) return RCPP_ARCH_CENO;
    if (strcmp(s, "fimmy") == 0) return RCPP_ARCH_FIMMY;
    if (strcmp(s, "hyenadna") == 0) return RCPP_ARCH_HYENADNA;
    if (strcmp(s, "llamamoe") == 0) return RCPP_ARCH_LLAMAMOE;
    if (strcmp(s, "llama_moe") == 0) return RCPP_ARCH_LLAMAMOE;
    if (strcmp(s, "modernbertdecoder") == 0) return RCPP_ARCH_MODERNBERTDECODER;
    if (strcmp(s, "modernbert-decoder") == 0) return RCPP_ARCH_MODERNBERTDECODER;
    if (strcmp(s, "modernbert") == 0) return RCPP_ARCH_MODERNBERTDECODER;
    if (strcmp(s, "orkhon") == 0) return RCPP_ARCH_ORKHON;
    if (strcmp(s, "roformer") == 0) return RCPP_ARCH_ROFORMER;
    if (strcmp(s, "stripedhyenamodel") == 0) return RCPP_ARCH_STRIPEDHYENA;
    if (strcmp(s, "stripedhyena") == 0) return RCPP_ARCH_STRIPEDHYENA;
    if (strcmp(s, "argonne") == 0) return RCPP_ARCH_ARGONNE;
    if (strcmp(s, "argonne2") == 0) return RCPP_ARCH_ARGONNE;
    if (strcmp(s, "emo") == 0) return RCPP_ARCH_EMO;
    if (strcmp(s, "forgettingtransformer") == 0) return RCPP_ARCH_FORGETTINGTRANSFORMER;
    if (strcmp(s, "forgetting_transformer") == 0) return RCPP_ARCH_FORGETTINGTRANSFORMER;
    if (strcmp(s, "gptbert") == 0) return RCPP_ARCH_GPTBERT;
    if (strcmp(s, "gpt-bert") == 0) return RCPP_ARCH_GPTBERT;
    if (strcmp(s, "gptjxmoe") == 0) return RCPP_ARCH_GPTJXMOE;
    if (strcmp(s, "keuralmoecausallm") == 0) return RCPP_ARCH_KEURALMOE;
    if (strcmp(s, "keural") == 0) return RCPP_ARCH_KEURALMOE;
    if (strcmp(s, "financedecoder") == 0) return RCPP_ARCH_FINANCEDECODER;
    if (strcmp(s, "qovaryx_finance_decoder") == 0) return RCPP_ARCH_FINANCEDECODER;
    if (strcmp(s, "reformermodelwithlmhead") == 0) return RCPP_ARCH_REFORMER;
    if (strcmp(s, "reformer") == 0) return RCPP_ARCH_REFORMER;
    if (strcmp(s, "acip") == 0) return RCPP_ARCH_ACIP;
    if (strcmp(s, "acip_model") == 0) return RCPP_ARCH_ACIP;
    if (strcmp(s, "cognicapoe") == 0) return RCPP_ARCH_COGNICAPOE;
    if (strcmp(s, "cognica_poe") == 0) return RCPP_ARCH_COGNICAPOE;
    if (strcmp(s, "grugmoe") == 0) return RCPP_ARCH_GRUGMOE;
    if (strcmp(s, "longcatflash") == 0) return RCPP_ARCH_LONGCAT;
    if (strcmp(s, "longcat_flash") == 0) return RCPP_ARCH_LONGCAT;
    if (strcmp(s, "telechat") == 0) return RCPP_ARCH_TELECHAT;
    if (strcmp(s, "btlm") == 0) return RCPP_ARCH_BTLM;
    if (strcmp(s, "duchifatcore") == 0) return RCPP_ARCH_DUCHIFAT;
    if (strcmp(s, "duchifat_v2") == 0) return RCPP_ARCH_DUCHIFAT;
    if (strcmp(s, "duo") == 0) return RCPP_ARCH_DUO;
    if (strcmp(s, "eshmun") == 0) return RCPP_ARCH_ESHMUN;
    if (strcmp(s, "gla") == 0) return RCPP_ARCH_GLA;
    if (strcmp(s, "polyverse") == 0) return RCPP_ARCH_POLYVERSE;
    if (strcmp(s, "transfoxl") == 0) return RCPP_ARCH_TRANSFOXL;
    if (strcmp(s, "transformer_xl") == 0) return RCPP_ARCH_TRANSFOXL;
    if (strcmp(s, "transnormer") == 0) return RCPP_ARCH_TRANSNORMER;
    if (strcmp(s, "twiny") == 0) return RCPP_ARCH_TWINY;
    if (strcmp(s, "gptpangu") == 0) return RCPP_ARCH_GPTPANGU;
    if (strcmp(s, "gpt_pangu") == 0) return RCPP_ARCH_GPTPANGU;
    if (strcmp(s, "bvv") == 0) return RCPP_ARCH_BVV;
    if (strcmp(s, "model_unfrozen") == 0) return RCPP_ARCH_BVV;

    // ── 2026-08-15 census pass-3 batch 2: llama-layout families, VLM text
    // decoders (map to text family), and model_type variants (verified) ──
    if (strcmp(s, "adaptermoellavaqwen3") == 0) return RCPP_ARCH_QWEN3VL;  // llava-qwen3 VLM
    if (strcmp(s, "bananamind2pico") == 0) return RCPP_ARCH_PICO;  // bananamind2-pico (PicoDecoderHF)
    if (strcmp(s, "bunnyphi") == 0) return RCPP_ARCH_PHI;  // bunny-phi VLM
    if (strcmp(s, "bunnyphi3") == 0) return RCPP_ARCH_PHI;  // bunny-phi3 VLM
    if (strcmp(s, "colmaskmoellavaqwen3") == 0) return RCPP_ARCH_QWEN3VL;  // llava-qwen3 VLM
    if (strcmp(s, "deepqwenvl") == 0) return RCPP_ARCH_QWEN2VL;  // deepqwen-vl VLM
    if (strcmp(s, "dyncolmaskmoellavaqwen2") == 0) return RCPP_ARCH_QWEN2VL;  // llava-qwen2 VLM
    if (strcmp(s, "emu3") == 0) return RCPP_ARCH_LLAMA;  // emu3-gen (llama layout, verify ALIAS_LLAMA)
    if (strcmp(s, "gemma4unifiedassistant") == 0) return RCPP_ARCH_GEMMA;  // gemma4 unified assistant
    if (strcmp(s, "gptjx") == 0) return RCPP_ARCH_GPTJ;  // GPT-JX (gpt-j layout, n_embd keys)
    if (strcmp(s, "graniteswitch") == 0) return RCPP_ARCH_GEMMA;  // granite switch (gemma layout)
    if (strcmp(s, "hgrn2") == 0) return RCPP_ARCH_HGRN;  // HGRN2
    if (strcmp(s, "jais2") == 0) return RCPP_ARCH_JAIS;  // Jais-2
    if (strcmp(s, "japanesestablelmalpha") == 0) return RCPP_ARCH_LLAMA;  // stablelm (llama layout)
    if (strcmp(s, "llavagemma") == 0) return RCPP_ARCH_GEMMA;  // llava-gemma VLM
    if (strcmp(s, "llavagpt2") == 0) return RCPP_ARCH_GPT2;  // llava-gpt2 VLM
    if (strcmp(s, "llavamamba") == 0) return RCPP_ARCH_MAMBA;  // llava-mamba VLM
    if (strcmp(s, "llavampt") == 0) return RCPP_ARCH_LLAMA;  // llava-mpt VLM
    if (strcmp(s, "llavaqwen3") == 0) return RCPP_ARCH_QWEN3VL;  // llava-qwen3 VLM
    if (strcmp(s, "maskmoellavaqwen3") == 0) return RCPP_ARCH_QWEN3VL;  // llava-qwen3 VLM
    if (strcmp(s, "minimaxm1") == 0) return RCPP_ARCH_MINIMAX;  // MiniMax-M1 (MoE)
    if (strcmp(s, "minimaxm3sparse") == 0) return RCPP_ARCH_MINIMAX;  // MiniMax-M3 sparse (VLM)
    if (strcmp(s, "mobilintexaone") == 0) return RCPP_ARCH_LLAMA;  // mobilint-exaone (llama layout, config-verified)
    if (strcmp(s, "moellavaqwen3") == 0) return RCPP_ARCH_QWEN3VL;  // llava-qwen3 VLM
    if (strcmp(s, "mosaicgpt") == 0) return RCPP_ARCH_LLAMA;  // mosaic (llama layout)
    if (strcmp(s, "nanogpt") == 0) return RCPP_ARCH_GPT2;  // nanogpt (gpt2 layout)
    if (strcmp(s, "nmmaskmoellavaqwen3") == 0) return RCPP_ARCH_QWEN3VL;  // llava-qwen3 VLM
    if (strcmp(s, "phi4flash") == 0) return RCPP_ARCH_PHI;  // phi4-flash
    if (strcmp(s, "plamo2") == 0) return RCPP_ARCH_LLAMA;  // plamo-2 (llama layout, config-verified)
    if (strcmp(s, "plamo3") == 0) return RCPP_ARCH_LLAMA;  // plamo-3 (llama layout)
    if (strcmp(s, "qwen2chunking") == 0) return RCPP_ARCH_QWEN2;  // qwen2 chunking
    if (strcmp(s, "qwen3omnimoe") == 0) return RCPP_ARCH_QWEN3VL;  // qwen3-omni VLM
    if (strcmp(s, "rwkv6qwen2") == 0) return RCPP_ARCH_QWEN2;  // RWKV6Qwen2 (qwen2-layout hybrid, like rwkv7qwen2)
    if (strcmp(s, "spatiallmqwen") == 0) return RCPP_ARCH_QWEN2VL;  // spatial-lm qwen VLM
    if (strcmp(s, "stablelmalpha") == 0) return RCPP_ARCH_LLAMA;  // stablelm (llama layout)
    if (strcmp(s, "tpugemma3") == 0) return RCPP_ARCH_GEMMA;  // gemma3 on TPU
    // VLM causal decoders (own families)
    if (strcmp(s, "mfuyu") == 0) return RCPP_ARCH_FUYU;         // FuyuForCausalLM
    if (strcmp(s, "fuyu") == 0) return RCPP_ARCH_FUYU;
    if (strcmp(s, "museglimmer") == 0) return RCPP_ARCH_MUSE;  // Muse-Glimmer
    if (strcmp(s, "muse_glimmer") == 0) return RCPP_ARCH_MUSE;

    // ── 2026-08-15 census pass-3 batch 3: verify-pass aliases + family variants ──
    if (strcmp(s, "alibi") == 0) return RCPP_ARCH_GPT2;  // codeparrot ALiBi (gpt2-layout, verify ALIAS_GPT2)
    if (strcmp(s, "gsa") == 0) return RCPP_ARCH_LLAMA;  // illada-8b (verify ALIAS_LLAMA)
    if (strcmp(s, "gptx2") == 0) return RCPP_ARCH_LLAMA;  // GPT-X2.5 (verify ALIAS_LLAMA)
    if (strcmp(s, "mosmamba") == 0) return RCPP_ARCH_MAMBA;  // mos-mamba (mamba hybrid)
    if (strcmp(s, "gemmoe") == 0) return RCPP_ARCH_GEMMA;  // gemma-moe
    if (strcmp(s, "gemma3moe") == 0) return RCPP_ARCH_GEMMA;  // gemma3-moe
    if (strcmp(s, "mixtralmole") == 0) return RCPP_ARCH_MISTRAL;  // mixtral variant
    if (strcmp(s, "hybridgpt2") == 0) return RCPP_ARCH_GPT2;  // hybrid gpt2
    if (strcmp(s, "activationsgptneo") == 0) return RCPP_ARCH_GPTNEOX;  // gpt-neox with activations
    if (strcmp(s, "attnqwen") == 0) return RCPP_ARCH_QWEN3;  // attn-qwen3
    if (strcmp(s, "bitmamba2lm") == 0) return RCPP_ARCH_MAMBA;  // bitmamba (mamba2)
    if (strcmp(s, "replitlm") == 0) return RCPP_ARCH_LLAMA;  // replit code (llama-based)
    if (strcmp(s, "pharia") == 0) return RCPP_ARCH_LLAMA;  // pharia-1-llm (llama-based)
    if (strcmp(s, "step3p7") == 0) return RCPP_ARCH_STEP3P5;  // step3.7 (step family)
    if (strcmp(s, "inflm") == 0) return RCPP_ARCH_LLAMA;  // infllm (llama-based)
    if (strcmp(s, "extendedmpt") == 0) return RCPP_ARCH_LLAMA;  // extended-mpt
    if (strcmp(s, "deltanet") == 0) return RCPP_ARCH_QWEN3NEXT;  // gated-deltanet (qwen3next family)
    if (strcmp(s, "tinygdn") == 0) return RCPP_ARCH_QWEN3NEXT;  // tiny gated-deltanet
    if (strcmp(s, "phi2moe") == 0) return RCPP_ARCH_PHI;  // phi-2-moe
    if (strcmp(s, "latentmoellavaphi") == 0) return RCPP_ARCH_PHI;  // llava-phi VLM
    if (strcmp(s, "nmmaskmoellavaphi") == 0) return RCPP_ARCH_PHI;  // llava-phi VLM
    if (strcmp(s, "qwen3sharedmoe") == 0) return RCPP_ARCH_QWEN3;  // qwen3 shared-moe
    if (strcmp(s, "nanochatgpt") == 0) return RCPP_ARCH_NANOCHAT;  // nanochat-gpt

    // ── 2026-08-15 census pass-3 batch 4: config-verified small families ──
    if (strcmp(s, "pit") == 0) return RCPP_ARCH_GPT2;  // pitchfork (config declares GPT2LMHeadModel)
    if (strcmp(s, "chesstrm") == 0) return RCPP_ARCH_GPT2;  // chess-transformer (gpt2 layout)
    if (strcmp(s, "randygpt") == 0) return RCPP_ARCH_GPT2;  // randygpt (n_embd keys)
    if (strcmp(s, "stickbreaking") == 0) return RCPP_ARCH_GPT2;  // stickbreaking (n_embd keys, gpt2-layout)
    if (strcmp(s, "pinyincode") == 0) return RCPP_ARCH_GPT2;  // pinyin-code (n_embd keys)
    if (strcmp(s, "brujula") == 0) return RCPP_ARCH_GPT2;  // Brujula (n_embd keys, gpt2-layout)
    if (strcmp(s, "phonelm") == 0) return RCPP_ARCH_LLAMA;  // PhoneLM (rms+rope+relu, config-verified)
    if (strcmp(s, "norovoxalphamoe") == 0) return RCPP_ARCH_LLAMA;  // Norovox-Alpha-MoE (rope 1e6, llama-layout MoE)

    // Unmapped architecture — do NOT fall back to BITNET silently.
    return RCPP_ARCH_UNKNOWN;
}

// RoPE weight convention (corrected 2026-08-13, pilot #16/17): the engine's
// half-split pairing (i, i+head_dim/2) is correct for NATURAL weights —
// verified EXACTLY (max diff 0) against transformers for both llama and
// granite at pos > 0. The earlier "pre-rotated GGUF" theory was wrong for the
// engine's pairing: rotated weights + half-split mismatched torch (corr 0.07)
// — the pre-rotation is llama.cpp's internal convention, not applicable to
// the engine's rope. The loader therefore never rotates; the GGUF path
// un-rotates (inverse permutation) to natural at load.
static inline bool rcpp_arch_rotates_rope(rcpp_arch_t arch, const char* architecture) {
    (void)arch; (void)architecture;
    return false;
}

typedef struct {
    void* input_norm_dev;
    void* post_attn_norm_dev;
    void* attn_sub_norm_dev;
    void* ffn_sub_norm_dev;
    void* attn_q_norm_dev;
    void* attn_k_norm_dev;

    // Ternary linear layers — halo-encoded uint8 packed + per-row FP32 scales
    void* q_packed_dev;     float* q_scales_dev;
    void* k_packed_dev;     float* k_scales_dev;
    void* v_packed_dev;     float* v_scales_dev;
    void* o_packed_dev;     float* o_scales_dev;
    void* gate_packed_dev;  float* gate_scales_dev;
    void* up_packed_dev;    float* up_scales_dev;
    void* down_packed_dev;  float* down_scales_dev;

    // WMMA_I8 path: Hadamard-rotated INT8 weights + per-row fp32 scales
    void* q_i8_dev;          float* q_i8_scales_dev;
    void* k_i8_dev;          float* k_i8_scales_dev;
    void* v_i8_dev;          float* v_i8_scales_dev;
    void* o_i8_dev;          float* o_i8_scales_dev;
    void* gate_i8_dev;       float* gate_i8_scales_dev;
    void* up_i8_dev;         float* up_i8_scales_dev;
    void* down_i8_dev;       float* down_i8_scales_dev;

    // Block-Scaled Ternary path: block-scaled ternary packed (5 bytes/block)
    // See include/block_scaled_ternary.h for format
    void* bst_q_packed_dev;     void* bst_q_scales_dev;
    void* bst_k_packed_dev;     void* bst_k_scales_dev;
    void* bst_v_packed_dev;     void* bst_v_scales_dev;
    void* bst_o_packed_dev;     void* bst_o_scales_dev;
    void* bst_gate_packed_dev;  void* bst_gate_scales_dev;
    void* bst_up_packed_dev;    void* bst_up_scales_dev;
    void* bst_down_packed_dev;  void* bst_down_scales_dev;

    // Attention biases (qwen2.5-family; GGUF conversions drop them)
    void* q_bias_dev;
    void* k_bias_dev;
    void* v_bias_dev;
} rcpp_bitnet_layer_t;

typedef struct {
    int hidden_size;
    int intermediate_size;
    int num_layers;
    int num_heads;
    int num_kv_heads;
    int vocab_size;
    int max_seq_len;
    int tie_embeddings;
    float rope_theta;
    float rms_norm_eps;
    int format_version;
    unsigned int flags;
    rcpp_weight_format_t weight_format;
    int is_qwen3;
    rcpp_arch_t arch;
    void* embedding_dev;
    void* embedding_packed_dev;
    void* final_norm_weight_dev;
    void* lm_head_dev;              // untied LM head (NULL = tied to embedding)
    rcpp_bitnet_layer_t* layers;
} rcpp_bitnet_model_t;

rcpp_status_t rcpp_bitnet_load_h1b(const char* path, rcpp_bitnet_model_t* out_model);
rcpp_status_t rcpp_bitnet_load_gguf(const char* path, rcpp_bitnet_model_t* out_model);
rcpp_status_t rcpp_bitnet_load_onnx(const char* path, rcpp_bitnet_model_t* out_model);
void rcpp_bitnet_free(rcpp_bitnet_model_t* model);

#ifdef __cplusplus
}
#endif
#endif
