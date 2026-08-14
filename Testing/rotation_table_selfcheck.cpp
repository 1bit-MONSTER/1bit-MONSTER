// rotation_table_selfcheck.cpp — locks the RoPE rotation table (pilot #11).
// Verifies rcpp_arch_rotates_rope() per family so future arch additions can't
// silently regress the llama.cpp GGUF convention (pre-rotated vs natural).
//
// Run:
//   g++ -std=c++17 -Iinclude Testing/rotation_table_selfcheck.cpp \
//       -o /tmp/rot_check && /tmp/rot_check
#include <cstdio>
#include <cstring>
#include "rocm_cpp/bitnet_model.h"

int main() {
    int total = 0, fails = 0;
    auto check = [&](const char* label, rcpp_arch_t arch, const char* archstr, bool want_rotate) {
        ++total;
        bool got = rcpp_arch_rotates_rope(arch, archstr);
        bool want_neox = !want_rotate;
        if (got != want_rotate) {
            std::printf("FAIL %-28s arch=%d '%s' rotate=%d want=%d\n",
                        label, (int)arch, archstr ? archstr : "", got, want_rotate);
            ++fails;
        }
    };

    // CORRECTED 2026-08-13 (pilot #16/17): NO family rotates. The engine's
    // half-split rope pairing is correct for natural weights (verified EXACT,
    // diff 0, vs transformers for llama + granite at pos > 0). The llama.cpp
    // GGUF pre-rotation is llama.cpp's internal convention and must be
    // un-rotated by the GGUF loader, never applied by the safetensors loader.
    check("llama", RCPP_ARCH_LLAMA, "llama", false);
    check("llama via arch only", RCPP_ARCH_LLAMA, "smollm2", false);
    check("granite", RCPP_ARCH_GEMMA, "granite", false);
    check("granitemoe", RCPP_ARCH_GEMMA, "granitemoe", false);
    check("mistral", RCPP_ARCH_MISTRAL, "mistral", false);
    check("qwen2", RCPP_ARCH_QWEN2, "qwen2", false);
    check("qwen3", RCPP_ARCH_QWEN3, "qwen3", false);
    check("qwen35", RCPP_ARCH_QWEN35, "qwen35", false);
    check("gemma", RCPP_ARCH_GEMMA, "gemma", false);
    check("phi", RCPP_ARCH_PHI, "phi", false);
    check("falcon", RCPP_ARCH_FALCON, "falcon", false);
    check("zamba2", RCPP_ARCH_ZAMBA2, "zamba2", false);
    check("mamba", RCPP_ARCH_MAMBA, "mamba", false);
    check("deepseek_v4", RCPP_ARCH_DEEPSEEK_V4, "deepseek_v4", false);
    check("whisper", RCPP_ARCH_WHISPER, "whisper", false);
    check("kimi", RCPP_ARCH_KIMI_K3, "kimi", false);
    check("unknown", RCPP_ARCH_UNKNOWN, "mystery", false);

    if (fails) { std::printf("ROTATION TABLE: %d/%d FAILED\n", fails, total); return 1; }
    std::printf("ROTATION TABLE: all %d checks passed\n", total);
    return 0;
}
