// arch_mapping_selfcheck.cpp — self-check for the one-bin dispatch arch registry.
// Verifies rcpp_arch_from_string: new bring-up mappings + regression on existing ones.
//
// Run:
//   g++ -std=c++17 -Iinclude Testing/arch_mapping_selfcheck.cpp -o /tmp/arch_check && /tmp/arch_check
#include <cstdio>
#include <cstring>
#include "rocm_cpp/bitnet_model.h"

int main() {
    int total = 0, fails = 0;
    auto check = [&](const char* s, rcpp_arch_t expect, const char* label) {
        ++total;
        rcpp_arch_t got = rcpp_arch_from_string(s);
        if (got != expect) {
            std::printf("FAIL %-26s got=%d want=%d\n", label, (int)got, (int)expect);
            ++fails;
        }
    };

    // ── 2026-08-13 bring-up pilot: LLaMA-layout architectures (GGUF + HF class names)
    check("openelm", RCPP_ARCH_LLAMA, "openelm");
    check("OpenELMForCausalLM", RCPP_ARCH_LLAMA, "OpenELMForCausalLM");
    check("nemotron", RCPP_ARCH_LLAMA, "nemotron");
    check("NemotronForCausalLM", RCPP_ARCH_LLAMA, "NemotronForCausalLM");
    check("minicpm", RCPP_ARCH_LLAMA, "minicpm");
    check("MiniCPMForCausalLM", RCPP_ARCH_LLAMA, "MiniCPMForCausalLM");

    // ── Regression: existing mappings
    check("llama", RCPP_ARCH_LLAMA, "llama");
    check("starcoder2", RCPP_ARCH_LLAMA, "starcoder2");
    check("baichuan2", RCPP_ARCH_LLAMA, "baichuan2");
    check("BaichuanForCausalLM", RCPP_ARCH_LLAMA, "BaichuanForCausalLM");
    check("exaone", RCPP_ARCH_LLAMA, "exaone");
    check("ExaoneForCausalLM", RCPP_ARCH_LLAMA, "ExaoneForCausalLM");
    check("solar", RCPP_ARCH_LLAMA, "solar");
    check("internlm2", RCPP_ARCH_LLAMA, "internlm2");
    check("xverse", RCPP_ARCH_LLAMA, "xverse");
    check("qwen", RCPP_ARCH_QWEN2, "qwen (Qwen1)");
    check("qwen3", RCPP_ARCH_QWEN3, "qwen3");
    check("qwen2", RCPP_ARCH_QWEN2, "qwen2");
    check("gemma4", RCPP_ARCH_GEMMA, "gemma4");
    check("granite", RCPP_ARCH_GEMMA, "granite");
    check("phi4", RCPP_ARCH_PHI, "phi4");
    check("deepseek2", RCPP_ARCH_DEEPSEEK, "deepseek2");
    check("deepseek_v4", RCPP_ARCH_DEEPSEEK_V4, "deepseek_v4");
    check("zamba2", RCPP_ARCH_ZAMBA2, "zamba2");
    check("mamba", RCPP_ARCH_MAMBA, "mamba");
    check("falcon3", RCPP_ARCH_FALCON, "falcon3");
    check("rw", RCPP_ARCH_FALCON, "falcon via rw");
    check("tinyllama", RCPP_ARCH_LLAMA, "tinyllama");
    check("openlm", RCPP_ARCH_LLAMA, "openlm");
    check("stablelmepoch", RCPP_ARCH_LLAMA, "stablelmepoch");
    check("mobilellm", RCPP_ARCH_LLAMA, "mobilellm");
    check("customllama", RCPP_ARCH_LLAMA, "customllama");
    check("ministral", RCPP_ARCH_MISTRAL, "ministral");
    check("olmo3", RCPP_ARCH_OLMO, "olmo3");
    check("gpt2lmheadcustom", RCPP_ARCH_GPT2, "gpt2 custom");
    check("yi", RCPP_ARCH_LLAMA, "yi");
    check("glm4", RCPP_ARCH_LLAMA, "glm4");
    check("decilm", RCPP_ARCH_LLAMA, "decilm");
    check("hunyuan", RCPP_ARCH_LLAMA, "hunyuan");
    check("nanbeige", RCPP_ARCH_LLAMA, "nanbeige");
    check("biogpt", RCPP_ARCH_GPT2, "biogpt");
    check("xglm", RCPP_ARCH_GPT2, "xglm");
    check("mixformersequential", RCPP_ARCH_PHI, "phi-1.5");
    check("kimi", RCPP_ARCH_KIMI_K3, "kimi");
    check("qwen35moe", RCPP_ARCH_QWEN35, "qwen35moe");
    check("whisper", RCPP_ARCH_WHISPER, "whisper");

    // ── MONSTER breadth batch 2026-08-14 (llama.cpp conversion/ registry) ──
    check("smollm3", RCPP_ARCH_LLAMA, "smollm3 (SmolLM3ForCausalLM)");
    check("apertus", RCPP_ARCH_LLAMA, "apertus (ApertusForCausalLM)");
    check("cohere", RCPP_ARCH_LLAMA, "cohere (CohereForCausalLM)");
    check("gptbigcode", RCPP_ARCH_LLAMA, "gptbigcode (GPTBigCodeForCausalLM)");
    check("internlm3", RCPP_ARCH_LLAMA, "internlm3 (InternLM3ForCausalLM)");
    check("mixtral", RCPP_ARCH_MISTRAL, "mixtral (MixtralForCausalLM)");
    check("qwen2moe", RCPP_ARCH_QWEN2, "qwen2moe (Qwen2MoeForCausalLM)");
    check("qwen3moe", RCPP_ARCH_QWEN3, "qwen3moe (Qwen3MoeForCausalLM)");
    check("deepseekv2", RCPP_ARCH_DEEPSEEK, "deepseekv2 (DeepseekV2ForCausalLM)");
    check("deepseekv3", RCPP_ARCH_DEEPSEEK, "deepseekv3 (DeepseekV3ForCausalLM)");
    check("deepseekv4", RCPP_ARCH_DEEPSEEK_V4, "deepseekv4 (DeepseekV4ForCausalLM)");
    check("gpt2", RCPP_ARCH_GPT2, "gpt2 (GPT2LMHeadModel)");
    check("gptneox", RCPP_ARCH_GPTNEOX, "gptneox (GPTNeoXForCausalLM)");
    check("opt", RCPP_ARCH_OPT, "opt (OPTForCausalLM)");
    check("gptneo", RCPP_ARCH_GPTNEO, "gptneo (GPTNeoForCausalLM)");
    check("codegen", RCPP_ARCH_CODEGEN, "codegen (CodeGenForCausalLM)");
    check("gptj", RCPP_ARCH_GPTJ, "gptj (GPTJForCausalLM)");
    check("gptoss", RCPP_ARCH_GPTOSS, "gptoss (GptOssForCausalLM)");
    check("step1", RCPP_ARCH_STEP1, "step1 (Step1ForCausalLM)");
    check("step1moe", RCPP_ARCH_STEP1, "step1moe (Step1MoEForCausalLM — dense weights in practice)");
    check("bloom", RCPP_ARCH_BLOOM, "bloom (BloomForCausalLM — fused qkv, linear ALiBi)");

    // ── Decision (pilot #10): unknown archs -> UNKNOWN (loud), not BITNET
    check("totally_unknown_arch", RCPP_ARCH_UNKNOWN, "unknown->UNKNOWN (loud fail)");

    // ── VLM conditional-generation classes (MAX-style text-decoder mapping) ──
    // The reader strips "forconditionalgeneration"/"forvisiontext2text" suffixes
    // (see safetensors_reader.cpp) before calling rcpp_arch_from_string, so these
    // assert the stripped forms the mapping actually receives.
    check("qwen3_5", RCPP_ARCH_QWEN35, "qwen3_5 (Qwen3.5 dense)");
    check("qwen3_5moe", RCPP_ARCH_QWEN35, "qwen3_5moe (Qwen3.5-MoE)");
    check("mistral3", RCPP_ARCH_MISTRAL, "mistral3 (Mistral3-VL text decoder)");
    check("qwen2_5_vl", RCPP_ARCH_QWEN2VL, "qwen2_5_vl (Qwen2.5-VL text decoder)");
    check("qwen3vlmoe", RCPP_ARCH_QWEN3VL, "qwen3vlmoe (Qwen3-VL-MoE)");
    check("gemma4unified", RCPP_ARCH_GEMMA, "gemma4unified (Gemma4-Unified)");
    check("qwen3_5vl", RCPP_ARCH_QWEN35, "qwen3_5vl (Qwen3.5-VL text decoder)");
    // Reader-strip equivalence (suffix -> stripped form -> token)
    check("gemma3", RCPP_ARCH_GEMMA, "gemma3 (from Gemma3ForConditionalGeneration)");
    check("gemma4", RCPP_ARCH_GEMMA, "gemma4 (from Gemma4ForConditionalGeneration)");
    check("qwen2vl", RCPP_ARCH_QWEN2VL, "qwen2vl (from Qwen2VLForConditionalGeneration)");
    check("qwen3vl", RCPP_ARCH_QWEN3VL, "qwen3vl (from Qwen3VLForConditionalGeneration)");
    check("llava", RCPP_ARCH_QWEN2VL, "llava (from LlavaForConditionalGeneration)");
    check("smolvlm", RCPP_ARCH_QWEN2VL, "smolvlm (from SmolVLMForConditionalGeneration)");
    check("paligemma", RCPP_ARCH_GEMMA, "paligemma (from PaliGemmaForConditionalGeneration)");
    // Encoder-decoder stays UNKNOWN (decoder-only engine, out of scope)
    check("t5", RCPP_ARCH_UNKNOWN, "t5 enc-dec stays UNKNOWN");
    check("mt5", RCPP_ARCH_UNKNOWN, "mt5 enc-dec stays UNKNOWN");
    check("bart", RCPP_ARCH_UNKNOWN, "bart enc-dec stays UNKNOWN");

    if (fails) {
        std::printf("ARCH MAPPING: %d/%d FAILED\n", fails, total);
        return 1;
    }
    std::printf("ARCH MAPPING: all %d checks passed\n", total);
    return 0;
}
