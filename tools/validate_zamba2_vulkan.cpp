// validate_zamba2_vulkan.cpp — token-for-token validation of the Zamba2
// Vulkan backend (backend_zamba2_vulkan.cpp) against the CPU reference
// (Zamba2Model::forward in zamba2_engine.cpp), per the P1 plan in
// docs/research/zamba2-vulkan.md.
//
// Usage: validate_zamba2_vulkan <model.gguf> [tokens=32] [start_token=1]
// Exit 0 = all tokens match; 1 = mismatch.
#include "backend.h"
#include "zamba2_engine.h"
#include "backend_zamba2_vulkan.cpp"  // factory + load_zamba2_from_gguf (via its loader include)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <chrono>
#include <cmath>

static int argmax_of(const float* v, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (v[i] > v[best]) best = i;
    return best;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf> [tokens=32] [start_token=1]\n", argv[0]);
        return 2;
    }
    const std::string path = argv[1];
    const int tokens = argc > 2 ? atoi(argv[2]) : 32;
    const int start = argc > 3 ? atoi(argv[3]) : 1;

    // ── CPU reference ──
    Zamba2Model cpu;
    if (!load_zamba2_from_gguf(path, cpu)) {
        fprintf(stderr, "FAIL: could not load %s (CPU)\n", path.c_str());
        return 1;
    }
    cpu.init_state();

    // ── Vulkan backend under test ──
    setenv("ZAMBA2_VK", "1", 1);
    Backend* vk = create_zamba2_vulkan_backend();
    ModelConfig cfg;
    cfg.model_path = path;
    cfg.arch = RCPP_ARCH_ZAMBA2;
    if (!vk->init(cfg, path)) {
        fprintf(stderr, "FAIL: zamba2_vulkan init failed\n");
        return 1;
    }
    if (!vk->reset()) { fprintf(stderr, "FAIL: reset\n"); return 1; }

    std::vector<float> cpu_logits(cpu.cfg.vocab_size, 0.f);

    int mismatches = 0;
    int tok_cpu = start, tok_vk = start;

    // CPU reference pass
    auto tc0 = std::chrono::high_resolution_clock::now();
    std::vector<int> cpu_seq;
    for (int t = 0; t < tokens; t++) {
        if (!cpu.forward(tok_cpu, cpu_logits.data())) { fprintf(stderr, "FAIL: cpu forward\n"); return 1; }
        tok_cpu = argmax_of(cpu_logits.data(), cpu.cfg.vocab_size);
        cpu_seq.push_back(tok_cpu);
    }
    double cpu_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - tc0).count();

    // Vulkan pass — same start token, compare against the CPU sequence
    auto tv0 = std::chrono::high_resolution_clock::now();
    for (int t = 0; t < tokens; t++) {
        int next_vk = vk->generate(tok_vk);
        if (next_vk != cpu_seq[t]) {
            mismatches++;
            fprintf(stderr, "  tok %d: cpu=%d vk=%d  <<< MISMATCH\n", t, cpu_seq[t], next_vk);
        }
        tok_vk = next_vk;
    }
    double vk_ms = std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - tv0).count();

    printf("Zamba2 VK validation: %d tokens, %d mismatches, cpu=%.2f ms/tok vk=%.2f ms/tok\n",
           tokens, mismatches, cpu_ms / tokens, vk_ms / tokens);
    vk->destroy();
    delete vk;

    return mismatches == 0 ? 0 : 1;
}
