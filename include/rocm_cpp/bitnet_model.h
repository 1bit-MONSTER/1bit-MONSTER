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
    RCPP_ARCH_LFM2 = 32,    // Liquid LFM2/LFM2.5 — conv+attention hybrid: depthwise causal conv1d blocks + full-attention blocks
    RCPP_ARCH_NEMOTRONH = 33, // Nemotron-H — Mamba-2 + NoPE GQA + relu2 MLP + sigmoid MoE hybrid
    RCPP_ARCH_QWEN3NEXT = 34, // Qwen3-Next — GatedDeltaNet linear attention + full attn + MoE
    RCPP_ARCH_MINIMAXM2 = 35, // MiniMax-M2 — GQA + single flattened q/k RMSNorm + partial rope + sigmoid MoE
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
    // ── end census tail sweep ──
    
    if (strcmp(s, "nemotronh") == 0) return RCPP_ARCH_NEMOTRONH; // NemotronHForCausalLM
    if (strcmp(s, "nemotron_h") == 0) return RCPP_ARCH_NEMOTRONH; // HF model_type
    if (strcmp(s, "qwen3next") == 0) return RCPP_ARCH_QWEN3NEXT;  // Qwen3NextForCausalLM
    if (strcmp(s, "qwen3_next") == 0) return RCPP_ARCH_QWEN3NEXT;  // HF model_type
    if (strcmp(s, "minimaxm2") == 0) return RCPP_ARCH_MINIMAXM2;   // MiniMaxM2ForCausalLM
    if (strcmp(s, "minimax_m2") == 0) return RCPP_ARCH_MINIMAXM2;  // HF model_type
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
