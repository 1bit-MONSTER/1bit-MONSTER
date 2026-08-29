// parity_fused.cpp — real-prompt, logits-level parity harness for the fused
// backend.  De-risks every future shader/kernel rewrite: instead of trusting
// the degenerate "15 13 15 ..." token stream, compare the full logits on real
// text prompts between two modes (GPU FFN vs NPU FFN, HIP attention vs Vulkan
// attention) and gate on both token equality AND a max-logit-diff tolerance.
//
// Usage:
//   parity_fused run  <model.1bp> <prompts.jsonl> <mode> <out.bin>
//       mode: gpu | npu | vk     (sets USE_NPU_FFN / FUSED_VK_ATTN)
//       writes per-step {argmax i32, VOCAB f32 logits} for each prompt token
//       + 4 generated tokens to <out.bin>.
//   parity_fused compare <a.bin> <b.bin> [tol] [label_a] [label_b]
//       prints per-step tokens + max |Δlogit|; exit 0 iff all tokens match
//       and max Δlogit <= tol (default 0.05).
//
// Example (HIP-attn vs VK-attn, GPU FFN both — the shader-rewrite gate):
//   build/parity_fused run  models/Qwen3-0.6B.1bp samples.jsonl hip  /tmp/p_hip.bin
//   build/parity_fused run  models/Qwen3-0.6B.1bp samples.jsonl vk   /tmp/p_vk.bin
//   build/parity_fused compare /tmp/p_hip.bin /tmp/p_vk.bin 0.05 hip vk
//
// Prompts: JSON-lines of token ids (same format as the ppl-gate samples in
// research/ws00-baseline/samples/).  Uses up to 3 lines, 32 tokens each.
#include "backend.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>
#include <sstream>

extern "C" Backend* create_fused_backend();

namespace {

struct Capture {
    int steps = 0, vocab = 0;
    std::vector<int> argmax;
    std::vector<float> logits;   // steps * vocab
};

bool read_prompts(const char* path, std::vector<std::vector<int>>& out, int max_lines, int max_len) {
    std::ifstream f(path);
    if (!f) { fprintf(stderr, "parity: cannot open %s\n", path); return false; }
    std::string line;
    while (std::getline(f, line) && (int)out.size() < max_lines) {
        if (line.empty() || line[0] != '[') continue;
        for (char& c : line) if (c == ',' || c == '[' || c == ']') c = ' ';
        std::istringstream ss(line);
        std::vector<int> ids; int v;
        while (ss >> v) ids.push_back(v);
        if ((int)ids.size() > 5) {
            if ((int)ids.size() > max_len) ids.resize((size_t)max_len);
            out.push_back(std::move(ids));
        }
    }
    return !out.empty();
}

bool write_capture(const char* path, const Capture& c) {
    FILE* f = fopen(path, "wb");
    if (!f) return false;
    int hdr[3] = {c.steps, c.vocab, 0};
    fwrite(hdr, 4, 3, f);
    fwrite(c.argmax.data(), 4, (size_t)c.steps, f);
    fwrite(c.logits.data(), 4, c.logits.size(), f);
    fclose(f);
    return true;
}

bool read_capture(const char* path, Capture& c) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    int hdr[3] = {0, 0, 0};
    if (fread(hdr, 4, 3, f) != 3) { fclose(f); return false; }
    c.steps = hdr[0]; c.vocab = hdr[1];
    c.argmax.resize((size_t)c.steps);
    c.logits.resize((size_t)c.steps * (size_t)c.vocab);
    if (fread(c.argmax.data(), 4, (size_t)c.steps, f) != (size_t)c.steps ||
        fread(c.logits.data(), 4, c.logits.size(), f) != c.logits.size()) {
        fclose(f); return false;
    }
    fclose(f);
    return true;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: parity_fused run <model> <prompts.jsonl> <gpu|npu|vk> <out.bin>\n"
                        "   or: parity_fused compare <a.bin> <b.bin> [tol] [label_a] [label_b]\n");
        return 2;
    }

    if (std::string(argv[1]) == "compare") {
        if (argc < 4) { fprintf(stderr, "compare needs two captures\n"); return 2; }
        double tol = argc > 4 ? atof(argv[4]) : 0.05;
        const char* la = argc > 5 ? argv[5] : "a";
        const char* lb = argc > 6 ? argv[6] : "b";
        Capture a, b;
        if (!read_capture(argv[2], a) || !read_capture(argv[3], b)) {
            fprintf(stderr, "parity: cannot read captures\n"); return 2;
        }
        if (a.steps != b.steps || a.vocab != b.vocab) {
            fprintf(stderr, "parity: capture mismatch (steps %d/%d vocab %d/%d)\n",
                    a.steps, b.steps, a.vocab, b.vocab);
            return 2;
        }
        int mism = 0; double worst = 0; int worst_step = -1;
        for (int s = 0; s < a.steps; s++) {
            if (a.argmax[s] != b.argmax[s]) { mism++; if (worst_step < 0) worst_step = s; }
            double md = 0;
            for (int i = 0; i < a.vocab; i++) {
                double d = std::fabs((double)a.logits[(size_t)s * a.vocab + i] -
                                     (double)b.logits[(size_t)s * a.vocab + i]);
                if (d > md) md = d;
            }
            fprintf(stderr, "  step %3d: %s=%d %s=%d  max|dlogit|=%.5f\n",
                    s, la, a.argmax[s], lb, b.argmax[s], md);
            if (md > worst) { worst = md; worst_step = s; }
        }
        bool ok = (mism == 0) && (worst <= tol);
        printf("parity %s: tokens %s (%d mism), worst max|dlogit|=%.6f @step %d (tol %.4f)\n",
               ok ? "OK" : "FAIL", mism == 0 ? "match" : "DIVERGE", mism, worst, worst_step, tol);
        return ok ? 0 : 1;
    }

    if (std::string(argv[1]) == "run") {
        if (argc < 6) { fprintf(stderr, "run needs model, prompts, mode, out\n"); return 2; }
        const char* model = argv[2];
        const char* prompts_path = argv[3];
        std::string mode = argv[4];
        const char* out = argv[5];

        // Mode -> env (read by the fused backend at init).  The backend's
        // convention is "var set (any value) = enabled", so UNSET the vars for
        // modes that must not use them — "0" would still enable.  NOTE: the
        // fused backend now defaults to the VK on-pages path, so the HIP mode
        // must FORCE FUSED_HIP_ATTN=1 (unsetting FUSED_VK_ATTN is not enough).
        if (mode == "npu") setenv("USE_NPU_FFN", "1", 1);
        else unsetenv("USE_NPU_FFN");
        if (mode == "hip") setenv("FUSED_HIP_ATTN", "1", 1);
        else unsetenv("FUSED_HIP_ATTN");
        if (mode == "vk") setenv("FUSED_VK_ATTN", "1", 1);
        else unsetenv("FUSED_VK_ATTN");
        if (mode == "vk" && !getenv("VK_ATTN_SHADER_DIR"))
            setenv("VK_ATTN_SHADER_DIR", "build/gpu_attn_vk_shaders", 0);
        fprintf(stderr, "parity: mode=%s USE_NPU_FFN=%s FUSED_HIP_ATTN=%s FUSED_VK_ATTN=%s\n", mode.c_str(),
                getenv("USE_NPU_FFN") ? getenv("USE_NPU_FFN") : "(unset)",
                getenv("FUSED_HIP_ATTN") ? getenv("FUSED_HIP_ATTN") : "(unset)",
                getenv("FUSED_VK_ATTN") ? getenv("FUSED_VK_ATTN") : "(unset)");

        std::vector<std::vector<int>> prompts;
        if (!read_prompts(prompts_path, prompts, 3, 32)) {
            fprintf(stderr, "parity: no prompts in %s\n", prompts_path); return 2;
        }

        Backend* b = create_fused_backend();
        if (!b) { fprintf(stderr, "parity: create_fused_backend failed\n"); return 1; }
        ModelConfig cfg;
        cfg.model_path = model;
        cfg.format = ModelFormat::ONEBP;
        FILE* f = fopen(model, "rb");
        if (!f) { fprintf(stderr, "parity: cannot open %s\n", model); return 1; }
        uint8_t hdr[256];
        if (fread(hdr, 1, 256, f) != 256) { fclose(f); return 1; }
        fclose(f);
        memcpy(&cfg.hidden_size, hdr + 20, 4); memcpy(&cfg.num_layers, hdr + 24, 4);
        cfg.num_heads = 16; cfg.num_kv_heads = 8; cfg.head_dim = 128;
        cfg.intermediate_size = 3072; cfg.vocab_size = 151936;
        if (!b->init(cfg, model)) { fprintf(stderr, "parity: init failed\n"); return 1; }

        Capture cap;
        cap.vocab = cfg.vocab_size;
        std::vector<float> hidden((size_t)cfg.hidden_size), logits((size_t)cfg.vocab_size);
        for (auto& prompt : prompts) {
            b->reset();
            int prev = -1;
            for (int t : prompt) {
                if (!b->forward(t, hidden.data())) { fprintf(stderr, "parity: forward failed\n"); return 1; }
                int am = -1;
                if (!b->lm_head(hidden.data(), logits.data(), &am)) { fprintf(stderr, "parity: lm_head failed\n"); return 1; }
                cap.argmax.push_back(am);
                cap.logits.insert(cap.logits.end(), logits.begin(), logits.end());
                cap.steps++;
                prev = am;
            }
            for (int g = 0; g < 4; g++) {   // generation steps (argmax chain)
                if (!b->forward(prev, hidden.data())) return 1;
                int am = -1;
                if (!b->lm_head(hidden.data(), logits.data(), &am)) return 1;
                cap.argmax.push_back(am);
                cap.logits.insert(cap.logits.end(), logits.begin(), logits.end());
                cap.steps++;
                prev = am;
            }
        }
        if (!write_capture(out, cap)) { fprintf(stderr, "parity: write %s failed\n", out); return 1; }
        fprintf(stderr, "parity: %d steps written to %s\n", cap.steps, out);
        delete b;
        return 0;
    }

    fprintf(stderr, "parity: unknown subcommand %s\n", argv[1]);
    return 2;
}
