#include <vector>
#include "engine.h"
#include "common.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>

int main(int argc, char** argv) {
    printf("╔═══════════════════════════════════════════════╗\n");
    printf("║  NPU Inference Engine — Qwen3-0.6B           ║\n");
    printf("║  Strix Halo XDNA 2 NPU — Full Pipeline       ║\n");
    printf("╚═══════════════════════════════════════════════╝\n\n");
    
    const char* model_path = getenv("NPU_MODEL_PATH")?getenv("NPU_MODEL_PATH"):"model.q4nx";
    if (argc > 1) model_path = argv[1];
    
    if (argc > 1 && strcmp(argv[1], "--help") == 0) {
        printf("Usage: %s [model_path]\n", model_path);
        return 0;
    }
    
    // Init engine
    NpuInferenceEngine engine;
    if (!engine.init(model_path)) {
        fprintf(stderr, "❌ Engine initialization failed\n");
        return 1;
    }
    
    // Test generate: BOS token → a few output tokens
    // NPU_PROMPT_IDS (comma-separated) or NPU_PROMPT_TOKEN overrides the
    // default BOS prompt. A chat driver encodes a real prompt via the
    // tokenizer and passes the token stream here.
    const char* pids = getenv("NPU_PROMPT_IDS");
    const char* pt = getenv("NPU_PROMPT_TOKEN");
    std::vector<int> prompt;
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
    int max_out = getenv("NPU_MAX_TOKENS") ? atoi(getenv("NPU_MAX_TOKENS")) : 16;
    if (max_out < 1) max_out = 16;
    if (max_out > 64) max_out = 64;
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
    } else {
        printf("\n❌ No tokens generated\n");
    }
    
    return 0;
}
