// metal_smoke.mm — Metal backend smoke gate: synthesize a tiny model's
// .bin weights, init the Metal backend, run generate() for N tokens, and
// assert every token lands in [0, vocab). Catches compile/link failures,
// Metal device/kernel init failures, and gross runtime breakage — the
// backend_metal.mm path has never been exercised in CI.
//
// Build (macOS only):
//   clang++ -std=c++17 -fobjc-arc -framework Metal \
//       -framework MetalPerformanceShaders \
//       -framework MetalPerformanceShadersGraph \
//       -Iinclude Testing/metal_smoke.mm src/backend_metal.mm -o /tmp/metal_smoke
//   /tmp/metal_smoke <weights_dir> <tokens>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>

#include "backend.h"
#include "common.h"

extern "C" Backend* create_metal_backend();

static void write_f32(const std::string& path, const std::vector<float>& v) {
    std::ofstream f(path, std::ios::binary);
    f.write((const char*)v.data(), v.size() * sizeof(float));
}

// MPSNDArray enforces a minimum buffer footprint (~2048B floor that scales
// with dims); synthetic dims below it abort. Round every dim up to clear it.
static int mps_floor(int n) { return n < 2048 ? 2048 : n; }

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <weights_dir> [tokens]\n", argv[0]); return 2; }
    std::string wd = argv[1];
    if (!wd.empty() && wd.back() != '/') wd += '/';
    int n_tokens = argc > 2 ? atoi(argv[2]) : 8;

    // ── Synthesize a tiny llama-ish model (seeded → deterministic) ──
    const int H = 2048, L = 2, NH = 8, NKV = 2, HD = 256, IM = 4096, V = 4096;
    std::mt19937 rng(42);
    auto rnd = [&]() { return std::uniform_real_distribution<float>(-0.1f, 0.1f)(rng); };

    write_f32(wd + "model_embed_tokens_weight.bin", std::vector<float>((size_t)V * H, 0.01f));
    write_f32(wd + "model_norm_weight.bin", std::vector<float>(H, 1.0f));
    write_f32(wd + "model_input_hidden_states_scale.bin", std::vector<float>(H, 1.0f));
    write_f32(wd + "model_input_hidden_states_bias.bin", std::vector<float>(H, 0.0f));
    for (int l = 0; l < L; l++) {
        std::string p = wd + "model_layers_" + std::to_string(l) + "_";
        for (const char* w : {"self_attn_q_proj.weight", "self_attn_k_proj.weight",
                              "self_attn_v_proj.weight", "self_attn_o_proj.weight",
                              "mlp_gate_proj.weight", "mlp_down_proj.weight",
                              "mlp_up_proj.weight", "input_layernorm.weight",
                              "post_attention_layernorm.weight"}) {
            int rows = 0, cols = 0;
            if (strstr(w, "q_proj"))      { rows = H; cols = H; }
            else if (strstr(w, "k_proj")) { rows = H; cols = H; }
            else if (strstr(w, "v_proj")) { rows = H; cols = H; }
            else if (strstr(w, "o_proj")) { rows = H; cols = H; }
            else if (strstr(w, "gate_proj")) { rows = IM; cols = H; }
            else if (strstr(w, "down_proj")) { rows = H; cols = IM; }
            else if (strstr(w, "up_proj")) { rows = IM; cols = H; }
            else { rows = H; cols = 1; }  // layernorm
            std::vector<float> v((size_t)rows * cols);
            for (auto& x : v) x = rnd();
            write_f32(p + w + ".bin", v);
        }
    }

    ModelConfig cfg;
    cfg.hidden = cfg.hidden_size = H;
    cfg.n_layers = cfg.num_layers = L;
    cfg.n_heads = cfg.num_heads = NH;
    cfg.n_kv_heads = cfg.num_kv_heads = NKV;
    cfg.head_dim = HD;
    cfg.n_ff = cfg.intermediate_size = IM;
    cfg.vocab = cfg.vocab_size = V;
    cfg.max_seq_len = 512;

    Backend* b = create_metal_backend();
    if (!b) { fprintf(stderr, "FAIL: create_metal_backend() returned null\n"); return 1; }
    if (!b->init(cfg, wd)) { fprintf(stderr, "FAIL: MetalBackend::init\n"); return 1; }

    int tok = 42;
    for (int i = 0; i < n_tokens; i++) {
        tok = b->generate(tok);
        if (tok < 0 || tok >= V) {
            fprintf(stderr, "FAIL: token %d out of range [0,%d)\n", tok, V);
            return 1;
        }
    }
    delete b;  // ~MetalBackend() calls destroy() — no explicit destroy() (would double-release ARC buffers)
    printf("METAL SMOKE PASS: %d tokens, all in [0,%d)\n", n_tokens, V);
    return 0;
}
