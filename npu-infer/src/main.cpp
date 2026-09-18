#include <vector>
#include <string>
#include "engine.h"
#include "common.h"
#include "qwen3_tokenizer.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>

// usage: npu_infer <model.q4nx> ["prompt text"]
//   Real-prompt mode (round 37 remaining item): the prompt text is wrapped
//   in the Qwen3 chat template, tokenized with the Qwen3 byte-level BPE
//   tokenizer (tokenizer.json next to model.q4nx), run through the engine's
//   prefill+decode, and the sampled tokens are decoded back to readable text.
//
// Env:
//   NPU_PROMPT          prompt text (alternative to argv[2])
//   NPU_RAW_PROMPT=1    no chat template — plain continuation after the text
//   NPU_SYSTEM          system prompt for chat mode (default "You are a helpful assistant.")
//   NPU_TOKENIZER       override tokenizer.json path
//   legacy: NPU_PROMPT_IDS / NPU_PROMPT_TOKEN (raw numeric-token prompts)
//   NPU_MAX_TOKENS / NPU_TEMPERATURE / NPU_TOP_K / NPU_TOP_P / NPU_SEED
static std::string dirname_of(const char* path) {
    std::string p(path);
    auto s = p.rfind('/');
    return (s == std::string::npos) ? std::string(".") : p.substr(0, s);
}

int main(int argc, char** argv) {
    printf("╔═══════════════════════════════════════════════╗\n");
    printf("║  NPU Inference Engine — Qwen3-0.6B           ║\n");
    printf("║  Strix Halo XDNA 2 NPU — Full Pipeline       ║\n");
    printf("╚═══════════════════════════════════════════════╝\n\n");
    
    const char* model_path = getenv("NPU_MODEL_PATH")?getenv("NPU_MODEL_PATH"):"model.q4nx";
    if (argc > 1) model_path = argv[1];
    
    if (argc > 1 && strcmp(argv[1], "--help") == 0) {
        printf("Usage: %s <model.q4nx> [\"prompt text\"]\n"
               "  prompt text via argv[2] or NPU_PROMPT (chat mode); "
               "NPU_RAW_PROMPT=1 for plain continuation.\n", model_path);
        return 0;
    }

    // ---- real-prompt path: tokenize text with the Qwen3 BPE tokenizer ----
    const char* prompt_text = (argc > 2) ? argv[2] : getenv("NPU_PROMPT");
    const char* raw_mode = getenv("NPU_RAW_PROMPT");
    Qwen3Tokenizer tok;
    std::vector<int> prompt;
    bool text_mode = (prompt_text && *prompt_text);
    if (text_mode) {
        const char* tok_path = getenv("NPU_TOKENIZER");
        std::string tp;
        if (tok_path && *tok_path) {
            tp = tok_path;
        } else {
            tp = dirname_of(model_path) + "/tokenizer.json";
        }
        if (!tok.load(tp)) {
            fprintf(stderr, "❌ Cannot load tokenizer %s\n", tp.c_str());
            return 1;
        }
        std::string text;
        if (raw_mode) {
            text = prompt_text;  // plain continuation, no template
            fprintf(stderr, "prompt: raw continuation (%zu chars)\n", text.size());
        } else {
            const char* sys = getenv("NPU_SYSTEM");
            std::string system = (sys && *sys) ? sys : "You are a helpful assistant.";
            text = Qwen3Tokenizer::chat_prompt(system, prompt_text);
            fprintf(stderr, "prompt: chat mode, system=\"%s\", user=\"%.60s%s\"\n",
                    system.c_str(), prompt_text,
                    strlen(prompt_text) > 60 ? "..." : "");
        }
        prompt = tok.encode(text);
        fprintf(stderr, "prompt: %zu tokens\n", prompt.size());

        // NPU_TOK_ONLY=1 — tokenizer self-check: print ids + round-trip
        // decode, skip the engine (used to validate vs python tokenizers).
        if (getenv("NPU_TOK_ONLY")) {
            printf("TOKENS:");
            for (int id : prompt) printf(" %d", id);
            printf("\nROUNDTRIP:%s\n", tok.decode(prompt).c_str());
            return 0;
        }
    } else {
        // ---- legacy numeric-token prompt path ----
        const char* pids = getenv("NPU_PROMPT_IDS");
        const char* pt = getenv("NPU_PROMPT_TOKEN");
        if (pids && *pids) {
            std::string s(pids);
            size_t pos = 0;
            while (pos < s.size()) {
                size_t c = s.find(',', pos);
                prompt.push_back(atoi(s.substr(pos, c - pos).c_str()));
                if (c == std::string::npos) break;
                pos = c + 1;
            }
        } else if (pt) {
            prompt.push_back(atoi(pt));
        } else {
            prompt.push_back(151643);  // BOS
        }
    }
    
    // Init engine
    NpuInferenceEngine engine;
    if (!engine.init(model_path)) {
        fprintf(stderr, "❌ Engine initialization failed\n");
        return 1;
    }
    
    int max_out = getenv("NPU_MAX_TOKENS") ? atoi(getenv("NPU_MAX_TOKENS")) : 16;
    if (max_out < 1) max_out = 16;
    if (max_out > 4096) max_out = 4096;   // MAX_L ceiling (runtime + engine ELF range)
    std::vector<int> output(max_out);
    
    auto t0 = std::chrono::steady_clock::now();
    int num_out = engine.generate(prompt.data(), (int)prompt.size(), output.data(), max_out);
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    
    if (num_out > 0) {
        printf("\n✅ Generated %d tokens (%.0f ms, %.0f ms/tok):\n  ", 
               num_out, ms, ms / num_out);
        for (int i = 0; i < num_out; i++) {
            printf("%d ", output[i]);
        }
        printf("\n");
        if (text_mode && tok.loaded()) {
            // decode until the first EOS token (exclusive), then trim
            std::vector<int> shown;
            for (int i = 0; i < num_out; i++) {
                if (Qwen3Tokenizer::is_eos(output[i])) break;
                shown.push_back(output[i]);
            }
            std::string text = tok.decode(shown);
            // strip a trailing <|im_start|>user turn if the model started one
            auto pos = text.find("<|im_start|>user");
            if (pos != std::string::npos) text = text.substr(0, pos);
            printf("\n📝 Output:\n%s\n", text.c_str());
        }
    } else {
        printf("\n❌ No tokens generated\n");
    }
    
    return 0;
}
