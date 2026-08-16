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

    // ── HF model_type values (reader falls back to model_type when the
    //    class name maps UNKNOWN — extraction 2026-08-15) ──
    check("gpt_neox", RCPP_ARCH_GPTNEOX, "model_type gpt_neox");
    check("gpt_neo", RCPP_ARCH_GPTNEO, "model_type gpt_neo");
    check("gpt_bigcode", RCPP_ARCH_LLAMA, "model_type gpt_bigcode");
    check("qwen2_vl", RCPP_ARCH_QWEN2VL, "model_type qwen2_vl");
    check("qwen3_vl", RCPP_ARCH_QWEN3VL, "model_type qwen3_vl");
    check("qwen3_moe", RCPP_ARCH_QWEN3, "model_type qwen3_moe");
    check("qwen2_moe", RCPP_ARCH_QWEN2, "model_type qwen2_moe");
    check("granite_moe", RCPP_ARCH_GEMMA, "model_type granite_moe");
    check("gemma3_text", RCPP_ARCH_GEMMA, "model_type gemma3_text");
    check("llava", RCPP_ARCH_QWEN2VL, "model_type llava");
    check("llava_llama", RCPP_ARCH_LLAMA, "model_type llava_llama");
    check("llava_qwen2", RCPP_ARCH_QWEN2, "model_type llava_qwen2");
    check("deepseek_v2", RCPP_ARCH_DEEPSEEK, "model_type deepseek_v2");
    check("deepseek_v3", RCPP_ARCH_DEEPSEEK, "model_type deepseek_v3");
    check("cambrian_qwen", RCPP_ARCH_QWEN2, "model_type cambrian_qwen");
    check("hunyuan_v1_dense", RCPP_ARCH_LLAMA, "model_type hunyuan_v1_dense");
    check("seedoss", RCPP_ARCH_LLAMA, "model_type seedoss");
    check("glm4", RCPP_ARCH_LLAMA, "model_type glm4");
    check("glm4moe", RCPP_ARCH_LLAMA, "glm4moe (GLM-4-MoE)");
    check("glmmoedsa", RCPP_ARCH_LLAMA, "glmmoedsa (GLM-4.5 MoE)");
    check("glm4moelite", RCPP_ARCH_LLAMA, "glm4moelite (GLM-4-MoE-Lite)");
    check("nemotronh", RCPP_ARCH_NEMOTRONH, "nemotronh (Nemotron-H)");
    check("nemotron_h", RCPP_ARCH_NEMOTRONH, "model_type nemotron_h");
    check("fp8_qwen3", RCPP_ARCH_QWEN3, "model_type fp8_qwen3 (FP8 wrapper)");
    check("fp8_llama", RCPP_ARCH_LLAMA, "model_type fp8_llama");
    check("bit_llama", RCPP_ARCH_LLAMA, "model_type bit_llama");

    // ── census tail sweep checks ──
    check("adavocabgemma", RCPP_ARCH_GEMMA, "adavocabgemma");
    check("aicraftar-tharo.g-conditionalgeneration", RCPP_ARCH_QWEN2VL, "aicraftar-tharo.g-conditionalgeneration");
    check("antihal", RCPP_ARCH_GEMMA, "antihal");
    check("asvdopt", RCPP_ARCH_OPT, "asvdopt");
    check("automodel", RCPP_ARCH_MISTRAL, "automodel");
    check("backpackgpt2", RCPP_ARCH_GPT2, "backpackgpt2");
    check("bunnyqwen", RCPP_ARCH_QWEN2, "bunnyqwen");
    check("careaqa", RCPP_ARCH_LLAMA, "careaqa");
    check("chexagent", RCPP_ARCH_PHI, "chexagent");
    check("codebharat", RCPP_ARCH_LLAMA, "codebharat");
    check("cogpt2", RCPP_ARCH_GPT2, "cogpt2");
    check("custom_mpt", RCPP_ARCH_LLAMA, "custom_mpt");
    check("custombiogpt", RCPP_ARCH_GPT2, "custombiogpt");
    check("custommixtral", RCPP_ARCH_MISTRAL, "custommixtral");
    check("custommodel3", RCPP_ARCH_GPTNEOX, "custommodel3");
    check("dashqphi3", RCPP_ARCH_PHI, "dashqphi3");
    check("deepseekv2mobe", RCPP_ARCH_DEEPSEEK, "deepseekv2mobe");
    check("deepseekv2sparsemobe", RCPP_ARCH_DEEPSEEK, "deepseekv2sparsemobe");
    check("deepseekv3forcausallmnextn", RCPP_ARCH_DEEPSEEK, "deepseekv3forcausallmnextn");
    check("denseformer", RCPP_ARCH_LLAMA, "denseformer");
    check("dflashlaguna", RCPP_ARCH_LAGUNA, "dflashlaguna");
    check("dribble", RCPP_ARCH_PHI, "dribble");
    check("dribblellama", RCPP_ARCH_LLAMA, "dribblellama");
    check("duolaguna", RCPP_ARCH_LAGUNA, "duolaguna");
    check("dusmistral", RCPP_ARCH_MISTRAL, "dusmistral");
    check("edullm", RCPP_ARCH_MISTRAL, "edullm");
    check("efficientdlm", RCPP_ARCH_QWEN3, "efficientdlm");
    check("exaonetd", RCPP_ARCH_LLAMA, "exaonetd");
    check("flashgptneox", RCPP_ARCH_GPTNEOX, "flashgptneox");
    check("flaxgptj", RCPP_ARCH_GPTJ, "flaxgptj");
    check("forcausallm", RCPP_ARCH_GEMMA, "forcausallm");
    check("fsdpgptoss", RCPP_ARCH_GPTOSS, "fsdpgptoss");
    check("gemmagain", RCPP_ARCH_GEMMA, "gemmagain");
    check("gfusionfordiffusionlm", RCPP_ARCH_DEEPSEEK, "gfusionfordiffusionlm");
    check("gistgptneo", RCPP_ARCH_GPTNEO, "gistgptneo");
    check("glamm", RCPP_ARCH_QWEN2VL, "glamm");
    check("glus", RCPP_ARCH_QWEN2VL, "glus");
    check("gpt2forquestionanswering", RCPP_ARCH_GPT2, "gpt2forquestionanswering");
    check("gpt2forsequenceclassification", RCPP_ARCH_GPT2, "gpt2forsequenceclassification");
    check("gpt2lmandvaluehead", RCPP_ARCH_GPT2, "gpt2lmandvaluehead");
    check("gptbigcodeforsequenceclassification", RCPP_ARCH_LLAMA, "gptbigcodeforsequenceclassification");
    check("gretriever", RCPP_ARCH_LLAMA, "gretriever");
    check("int8opt", RCPP_ARCH_OPT, "int8opt");
    check("internlm2forreward", RCPP_ARCH_LLAMA, "internlm2forreward");
    check("internlmxcomposer2", RCPP_ARCH_LLAMA, "internlmxcomposer2");
    check("interns2preview", RCPP_ARCH_QWEN35, "interns2preview");
    check("kblamphi3", RCPP_ARCH_PHI, "kblamphi3");
    check("layerwiseminicpm", RCPP_ARCH_LLAMA, "layerwiseminicpm");
    check("leanllama", RCPP_ARCH_LLAMA, "leanllama");
    check("leanmixtral", RCPP_ARCH_MISTRAL, "leanmixtral");
    check("lexadelta", RCPP_ARCH_GPTOSS, "lexadelta");
    check("lfm2bidirectionalformaskedlm", RCPP_ARCH_LFM2, "lfm2bidirectionalformaskedlm");
    check("lightonocr", RCPP_ARCH_MISTRAL, "lightonocr");
    check("llamaforcausallmeagle3", RCPP_ARCH_LLAMA, "llamaforcausallmeagle3");
    check("llamaforsequenceclassification", RCPP_ARCH_LLAMA, "llamaforsequenceclassification");
    check("llavaqwen", RCPP_ARCH_QWEN2VL, "llavaqwen");
    check("loragpt2", RCPP_ARCH_GPT2, "loragpt2");
    check("mahler60/prueba", RCPP_ARCH_GPTNEOX, "mahler60/prueba");
    check("mamba2", RCPP_ARCH_MAMBA, "mamba2");
    check("mambamodel", RCPP_ARCH_MAMBA, "mambamodel");
    check("memllama", RCPP_ARCH_LLAMA, "memllama");
    check("meteormamba", RCPP_ARCH_MAMBA, "meteormamba");
    check("mimoaudio", RCPP_ARCH_QWEN2, "mimoaudio");
    check("miniphi3", RCPP_ARCH_PHI, "miniphi3");
    check("mixformervlsequential", RCPP_ARCH_PHI, "mixformervlsequential");
    check("mobillama", RCPP_ARCH_LLAMA, "mobillama");
    check("monoformer", RCPP_ARCH_LLAMA, "monoformer");
    check("moyi", RCPP_ARCH_QWEN2, "moyi");
    check("multiheadgptneo", RCPP_ARCH_GPTNEO, "multiheadgptneo");
    check("multimodalstarcoder2", RCPP_ARCH_LLAMA, "multimodalstarcoder2");
    check("mutorgemma", RCPP_ARCH_GEMMA, "mutorgemma");
    check("mybaichuan", RCPP_ARCH_LLAMA, "mybaichuan");
    check("myqwen", RCPP_ARCH_QWEN2, "myqwen");
    check("myxverse", RCPP_ARCH_LLAMA, "myxverse");
    check("notagen", RCPP_ARCH_GPT2, "notagen");
    check("olmo2forsequenceclassification", RCPP_ARCH_OLMO, "olmo2forsequenceclassification");
    check("olmo3sink", RCPP_ARCH_OLMO, "olmo3sink");
    check("olmomodel", RCPP_ARCH_OLMO, "olmomodel");
    check("opt_prompttuned_for_sentimentanalysis", RCPP_ARCH_OPT, "opt_prompttuned_for_sentimentanalysis");
    check("pawqwen3", RCPP_ARCH_QWEN3, "pawqwen3");
    check("phi3forsequenceclassification", RCPP_ARCH_PHI, "phi3forsequenceclassification");
    check("poptorchpipelinedgpt2", RCPP_ARCH_GPT2, "poptorchpipelinedgpt2");
    check("poptorchpipelinedwhisper", RCPP_ARCH_WHISPER, "poptorchpipelinedwhisper");
    check("qwen2forcausallmpostblocksteeringfixed", RCPP_ARCH_QWEN2, "qwen2forcausallmpostblocksteeringfixed");
    check("qwen2forprocessreward", RCPP_ARCH_QWEN2, "qwen2forprocessreward");
    check("qwen2forsequenceclassification", RCPP_ARCH_QWEN2, "qwen2forsequenceclassification");
    check("qwen2reasoning", RCPP_ARCH_QWEN2, "qwen2reasoning");
    check("qwen2vlaudio", RCPP_ARCH_QWEN2VL, "qwen2vlaudio");
    check("qwen2vlextended", RCPP_ARCH_QWEN2VL, "qwen2vlextended");
    check("qwen2vlforconditionalgenerationwithaudio", RCPP_ARCH_QWEN2VL, "qwen2vlforconditionalgenerationwithaudio");
    check("qwen3_5dllm", RCPP_ARCH_QWEN35, "qwen3_5dllm");
    check("qwen3forsequenceclassification", RCPP_ARCH_QWEN3, "qwen3forsequenceclassification");
    check("qwen3gated", RCPP_ARCH_QWEN3, "qwen3gated");
    check("qwen3mobe", RCPP_ARCH_QWEN3, "qwen3mobe");
    check("qwen3sparsemobe", RCPP_ARCH_QWEN3, "qwen3sparsemobe");
    check("qwen3vlseg", RCPP_ARCH_QWEN3VL, "qwen3vlseg");
    check("ruqwen2", RCPP_ARCH_QWEN2, "ruqwen2");
    check("serayuki", RCPP_ARCH_LLAMA, "serayuki");
    check("sewy3", RCPP_ARCH_GEMMA, "sewy3");
    check("smollm3model", RCPP_ARCH_LLAMA, "smollm3model");
    check("stablediffcoder", RCPP_ARCH_LLAMA, "stablediffcoder");
    check("streamvln", RCPP_ARCH_QWEN2VL, "streamvln");
    check("symbolicgpt", RCPP_ARCH_GPT2, "symbolicgpt");
    check("titansmactransformer", RCPP_ARCH_LLAMA, "titansmactransformer");
    check("trimkvphi3", RCPP_ARCH_PHI, "trimkvphi3");
    check("trimkvqwen3", RCPP_ARCH_QWEN3, "trimkvqwen3");
    check("vlclipgptneox", RCPP_ARCH_GPTNEOX, "vlclipgptneox");
    check("whaleye", RCPP_ARCH_DEEPSEEK, "whaleye");
    check("xcuros", RCPP_ARCH_QWEN2, "xcuros");
    if (fails) {
        std::printf("ARCH MAPPING: %d/%d FAILED\n", fails, total);
        return 1;
    }


    // ── 2026-08-15 census pass-3 checks (new families + verified aliases) ──
    check("acip", RCPP_ARCH_ACIP, "acip");
    check("argonne", RCPP_ARCH_ARGONNE, "argonne");
    check("bailingmoe", RCPP_ARCH_LLAMA, "bailingmoe");
    check("bailingmoelinearv2", RCPP_ARCH_LLAMA, "bailingmoelinearv2");
    check("bailingmoev2", RCPP_ARCH_LLAMA, "bailingmoev2");
    check("bailingmoev2_5", RCPP_ARCH_LLAMA, "bailingmoev2_5");
    check("bailingmoev3", RCPP_ARCH_LLAMA, "bailingmoev3");
    check("bamba", RCPP_ARCH_LLAMA, "bamba");
    check("btlm", RCPP_ARCH_BTLM, "btlm");
    check("bvv", RCPP_ARCH_BVV, "bvv");
    check("ceno", RCPP_ARCH_CENO, "ceno");
    check("chatglm", RCPP_ARCH_CHATGLM, "chatglm");
    check("cognicapoe", RCPP_ARCH_COGNICAPOE, "cognicapoe");
    check("cubelm", RCPP_ARCH_CUBELM, "cubelm");
    check("daisy", RCPP_ARCH_DAISY, "daisy");
    check("duchifatcore", RCPP_ARCH_DUCHIFAT, "duchifatcore");
    check("duo", RCPP_ARCH_DUO, "duo");
    check("dynamicforgetting", RCPP_ARCH_DYNAMICFORGETTING, "dynamicforgetting");
    check("dynamicslidingwindow", RCPP_ARCH_DYNAMICSLIDINGWINDOW, "dynamicslidingwindow");
    check("emo", RCPP_ARCH_EMO, "emo");
    check("eshmun", RCPP_ARCH_ESHMUN, "eshmun");
    check("fimmy", RCPP_ARCH_FIMMY, "fimmy");
    check("financedecoder", RCPP_ARCH_FINANCEDECODER, "financedecoder");
    check("forgettingtransformer", RCPP_ARCH_FORGETTINGTRANSFORMER, "forgettingtransformer");
    check("gla", RCPP_ARCH_GLA, "gla");
    check("gpt2moe", RCPP_ARCH_GPT2, "gpt2moe");
    check("gptbert", RCPP_ARCH_GPTBERT, "gptbert");
    check("gptjxmoe", RCPP_ARCH_GPTJXMOE, "gptjxmoe");
    check("gptpangu", RCPP_ARCH_GPTPANGU, "gptpangu");
    check("grugmoe", RCPP_ARCH_GRUGMOE, "grugmoe");
    check("hgrn", RCPP_ARCH_HGRN, "hgrn");
    check("hybridqwen3", RCPP_ARCH_LLAMA, "hybridqwen3");
    check("hyenadna", RCPP_ARCH_HYENADNA, "hyenadna");
    check("instella", RCPP_ARCH_DEEPSEEK, "instella");
    check("jais", RCPP_ARCH_JAIS, "jais");
    check("keuralmoecausallm", RCPP_ARCH_KEURALMOE, "keuralmoecausallm");
    check("kimik25", RCPP_ARCH_DEEPSEEK, "kimik25");
    check("kormo", RCPP_ARCH_KORMO, "kormo");
    check("lightningtransformermodel", RCPP_ARCH_LIGHTNINGTRANSFORMER, "lightningtransformermodel");
    check("llada2moemodellm", RCPP_ARCH_LLADA2, "llada2moemodellm");
    check("llama4", RCPP_ARCH_LLAMA4, "llama4");
    check("llamamoe", RCPP_ARCH_LLAMAMOE, "llamamoe");
    check("longcatflash", RCPP_ARCH_LONGCAT, "longcatflash");
    check("looplm", RCPP_ARCH_LOOPLM, "looplm");
    check("modeling_sparsetral.mistral", RCPP_ARCH_MISTRAL, "modeling_sparsetral.mistral");
    check("modernbertdecoder", RCPP_ARCH_MODERNBERTDECODER, "modernbertdecoder");
    check("motif", RCPP_ARCH_MOTIF, "motif");
    check("multiscale", RCPP_ARCH_MULTISCALE, "multiscale");
    check("orkhon", RCPP_ARCH_ORKHON, "orkhon");
    check("polyverse", RCPP_ARCH_POLYVERSE, "polyverse");
    check("quasar", RCPP_ARCH_QUASAR, "quasar");
    check("raven", RCPP_ARCH_RAVEN, "raven");
    check("recurrentgemma", RCPP_ARCH_RECURRENTGEMMA, "recurrentgemma");
    check("reformermodelwithlmhead", RCPP_ARCH_REFORMER, "reformermodelwithlmhead");
    check("retnet", RCPP_ARCH_RETNET, "retnet");
    check("roformer", RCPP_ARCH_ROFORMER, "roformer");
    check("rwkv7", RCPP_ARCH_RWKV7, "rwkv7");
    check("sarvammla", RCPP_ARCH_SARVAM, "sarvammla");
    check("sarvammoe", RCPP_ARCH_SARVAM, "sarvammoe");
    check("skipmiddle", RCPP_ARCH_SKIPMIDDLE, "skipmiddle");
    check("spikewhalelm", RCPP_ARCH_SPIKEWHALE, "spikewhalelm");
    check("step3p5", RCPP_ARCH_STEP3P5, "step3p5");
    check("stl", RCPP_ARCH_STL, "stl");
    check("stripedhyenamodel", RCPP_ARCH_STRIPEDHYENA, "stripedhyenamodel");
    check("talkie", RCPP_ARCH_TALKIE, "talkie");
    check("telechat", RCPP_ARCH_TELECHAT, "telechat");
    check("transfoxl", RCPP_ARCH_TRANSFOXL, "transfoxl");
    check("transnormer", RCPP_ARCH_TRANSNORMER, "transnormer");
    check("twiny", RCPP_ARCH_TWINY, "twiny");
    check("xpertgpt", RCPP_ARCH_XPERTGPT, "xpertgpt");
    check("yatgpt", RCPP_ARCH_YATGPT, "yatgpt");


    // ── 2026-08-15 census pass-3 batch 2 checks ──
    check("adaptermoellavaqwen3", RCPP_ARCH_QWEN3VL, "adaptermoellavaqwen3");
    check("bananamind2pico", RCPP_ARCH_PICO, "bananamind2pico");
    check("bunnyphi", RCPP_ARCH_PHI, "bunnyphi");
    check("bunnyphi3", RCPP_ARCH_PHI, "bunnyphi3");
    check("colmaskmoellavaqwen3", RCPP_ARCH_QWEN3VL, "colmaskmoellavaqwen3");
    check("deepqwenvl", RCPP_ARCH_QWEN2VL, "deepqwenvl");
    check("dyncolmaskmoellavaqwen2", RCPP_ARCH_QWEN2VL, "dyncolmaskmoellavaqwen2");
    check("emu3", RCPP_ARCH_LLAMA, "emu3");
    check("gemma4unifiedassistant", RCPP_ARCH_GEMMA, "gemma4unifiedassistant");
    check("gptjx", RCPP_ARCH_GPTJ, "gptjx");
    check("graniteswitch", RCPP_ARCH_GEMMA, "graniteswitch");
    check("hgrn2", RCPP_ARCH_HGRN, "hgrn2");
    check("jais2", RCPP_ARCH_JAIS, "jais2");
    check("japanesestablelmalpha", RCPP_ARCH_LLAMA, "japanesestablelmalpha");
    check("llavagemma", RCPP_ARCH_GEMMA, "llavagemma");
    check("llavagpt2", RCPP_ARCH_GPT2, "llavagpt2");
    check("llavamamba", RCPP_ARCH_MAMBA, "llavamamba");
    check("llavampt", RCPP_ARCH_LLAMA, "llavampt");
    check("llavaqwen3", RCPP_ARCH_QWEN3VL, "llavaqwen3");
    check("maskmoellavaqwen3", RCPP_ARCH_QWEN3VL, "maskmoellavaqwen3");
    check("minimaxm1", RCPP_ARCH_MINIMAX, "minimaxm1");
    check("minimaxm3sparse", RCPP_ARCH_MINIMAX, "minimaxm3sparse");
    check("mobilintexaone", RCPP_ARCH_LLAMA, "mobilintexaone");
    check("moellavaqwen3", RCPP_ARCH_QWEN3VL, "moellavaqwen3");
    check("mosaicgpt", RCPP_ARCH_LLAMA, "mosaicgpt");
    check("nanogpt", RCPP_ARCH_GPT2, "nanogpt");
    check("nmmaskmoellavaqwen3", RCPP_ARCH_QWEN3VL, "nmmaskmoellavaqwen3");
    check("phi4flash", RCPP_ARCH_PHI, "phi4flash");
    check("plamo2", RCPP_ARCH_LLAMA, "plamo2");
    check("plamo3", RCPP_ARCH_LLAMA, "plamo3");
    check("qwen2chunking", RCPP_ARCH_QWEN2, "qwen2chunking");
    check("qwen3omnimoe", RCPP_ARCH_QWEN3VL, "qwen3omnimoe");
    check("rwkv6qwen2", RCPP_ARCH_QWEN2, "rwkv6qwen2");
    check("spatiallmqwen", RCPP_ARCH_QWEN2VL, "spatiallmqwen");
    check("stablelmalpha", RCPP_ARCH_LLAMA, "stablelmalpha");
    check("tpugemma3", RCPP_ARCH_GEMMA, "tpugemma3");


    // ── 2026-08-15 census pass-3 batch 3 checks ──
    check("activationsgptneo", RCPP_ARCH_GPTNEOX, "activationsgptneo");
    check("alibi", RCPP_ARCH_GPT2, "alibi");
    check("attnqwen", RCPP_ARCH_QWEN3, "attnqwen");
    check("bitmamba2lm", RCPP_ARCH_MAMBA, "bitmamba2lm");
    check("deltanet", RCPP_ARCH_QWEN3NEXT, "deltanet");
    check("extendedmpt", RCPP_ARCH_LLAMA, "extendedmpt");
    check("gemma3moe", RCPP_ARCH_GEMMA, "gemma3moe");
    check("gemmoe", RCPP_ARCH_GEMMA, "gemmoe");
    check("gptx2", RCPP_ARCH_LLAMA, "gptx2");
    check("gsa", RCPP_ARCH_LLAMA, "gsa");
    check("hybridgpt2", RCPP_ARCH_GPT2, "hybridgpt2");
    check("inflm", RCPP_ARCH_LLAMA, "inflm");
    check("latentmoellavaphi", RCPP_ARCH_PHI, "latentmoellavaphi");
    check("mfuyu", RCPP_ARCH_FUYU, "mfuyu");
    check("mixtralmole", RCPP_ARCH_MISTRAL, "mixtralmole");
    check("mosmamba", RCPP_ARCH_MAMBA, "mosmamba");
    check("museglimmer", RCPP_ARCH_MUSE, "museglimmer");
    check("nanochatgpt", RCPP_ARCH_NANOCHAT, "nanochatgpt");
    check("nmmaskmoellavaphi", RCPP_ARCH_PHI, "nmmaskmoellavaphi");
    check("pharia", RCPP_ARCH_LLAMA, "pharia");
    check("phi2moe", RCPP_ARCH_PHI, "phi2moe");
    check("qwen3sharedmoe", RCPP_ARCH_QWEN3, "qwen3sharedmoe");
    check("replitlm", RCPP_ARCH_LLAMA, "replitlm");
    check("step3p7", RCPP_ARCH_STEP3P5, "step3p7");
    check("tinygdn", RCPP_ARCH_QWEN3NEXT, "tinygdn");


    // ── 2026-08-15 census pass-3 batch 4 checks ──
    check("brujula", RCPP_ARCH_GPT2, "brujula");
    check("chesstrm", RCPP_ARCH_GPT2, "chesstrm");
    check("norovoxalphamoe", RCPP_ARCH_LLAMA, "norovoxalphamoe");
    check("phonelm", RCPP_ARCH_LLAMA, "phonelm");
    check("pinyincode", RCPP_ARCH_GPT2, "pinyincode");
    check("pit", RCPP_ARCH_GPT2, "pit");
    check("randygpt", RCPP_ARCH_GPT2, "randygpt");
    check("stickbreaking", RCPP_ARCH_GPT2, "stickbreaking");


    // ── 2026-08-15 census pass-3 batch 5 checks ──
    check("evo2", RCPP_ARCH_STRIPEDHYENA, "evo2");
    check("progen2forpretraining", RCPP_ARCH_GPT2, "progen2forpretraining");


    // ── 2026-08-15 census pass-3 batch 5b checks ──
    check("eagle3speculator", RCPP_ARCH_LLAMA, "eagle3speculator");
    check("embformer", RCPP_ARCH_LLAMA, "embformer");
    check("starvector", RCPP_ARCH_LLAMA, "starvector");
    std::printf("ARCH MAPPING: all %d checks passed\n", total);
    return 0;
}
