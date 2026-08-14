// e2e_safetensors_selfcheck.cpp — pilot #6: real checkpoint end-to-end (v2).
// The Backend interface does NOT expose hidden states (forward(token,out)
// only zeroes out[0]) — so the oracle is next-token argmax via generate().
// Compares the HF-native safetensors load against the GGUF load of the same
// model, plus an independent torch oracle run by the companion python script.
//
// Run:
//   g++ -std=c++17 -O2 -Iinclude -Isrc src/backend_generic.cpp \
//       src/model_discovery.cpp src/gguf_reader.cpp src/q4nx_reader.cpp \
//       src/safetensors_reader.cpp Testing/e2e_safetensors_selfcheck.cpp \
//       -o /tmp/e2e_check && /tmp/e2e_check <dir> <oracle.gguf>
//   python3 Testing/e2e_torch_oracle.py <dir> <vocab_print>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "common.h"
#include "model_discovery.h"
#include "backend.h"

int main(int argc, char** argv) {
    if (argc < 3) { std::printf("usage: %s <model_dir> <oracle.gguf>\n", argv[0]); return 2; }
    const std::string dir = argv[1], gguf_path = argv[2];

    auto models = discover_models(dir);
    ModelConfig st_cfg, gg_cfg;
    for (auto& m : models) {
        if (m.format == ModelFormat::SAFETENSORS) st_cfg = m;
        if (m.model_path == gguf_path) gg_cfg = m;
    }
    if (st_cfg.model_path.empty() || gg_cfg.model_path.empty()) {
        std::printf("FAIL: need both a safetensors and the oracle GGUF in %s\n", dir.c_str());
        return 1;
    }
    std::printf("safetensors: %s arch=%d (%s) layers=%d hidden=%d vocab=%d\n",
                st_cfg.model_name.c_str(), (int)st_cfg.arch, st_cfg.architecture.c_str(),
                st_cfg.n_layers, st_cfg.hidden, st_cfg.vocab);
    std::printf("  st_cfg: heads=%d kv=%d hd=%d ff=%d rope_theta=%g eps=%g max_seq=%d\n",
                st_cfg.n_heads, st_cfg.n_kv_heads, st_cfg.head_dim, st_cfg.n_ff,
                st_cfg.rope_theta, st_cfg.rms_norm_eps, st_cfg.max_seq_len);

    Backend* st = create_generic_backend();
    Backend* gg = create_generic_backend();
    if (!st->init(st_cfg, dir)) { std::printf("FAIL: safetensors init\n"); return 1; }
    std::printf("safetensors: init OK\n");
    if (!gg->init(gg_cfg, gguf_path.substr(0, gguf_path.find_last_of('/')))) {
        std::printf("FAIL: gguf init\n"); return 1;
    }
    std::printf("gguf: init OK\n");

    // ── Next-token predictions: safetensors engine vs GGUF engine ──
    // (If the safetensors load silently fell back to the GGUF, these match
    //  exactly everywhere. Real BF16-vs-Q8_0 loads agree on most tokens.)
    int total = 0, agree = 0, mismatch = 0;
    for (int tok : {5, 42, 99, 1000, 4242, 31337}) {
        // Reset KV state so each seed is a fresh pos-0 prediction — the torch
        // oracle (e2e_torch_oracle.py) evaluates every seed fresh too.
        st->reset(); gg->reset();
        int g_st = st->generate(tok), g_gg = gg->generate(tok);
        bool ok = (g_st == g_gg) && g_st >= 0 && g_st < st_cfg.vocab;
        total++; if (ok) agree++; else mismatch++;
        std::printf("seed %6d: safetensors->%d  gguf->%d  %s\n",
                    tok, g_st, g_gg, g_st == g_gg ? "agree" : "DIFFER");
    }
    std::printf("next-token: %d/%d agree between loaders\n", agree, total);

    // ── 8-step generation chains from token 5 ──
    std::vector<int> chain_st, chain_gg;
    int t = 5;
    for (int i = 0; i < 8; i++) { t = st->generate(t); chain_st.push_back(t); }
    t = 5;
    for (int i = 0; i < 8; i++) { t = gg->generate(t); chain_gg.push_back(t); }
    int same = 0;
    for (size_t i = 0; i < chain_st.size(); i++) if (chain_st[i] == chain_gg[i]) same++;
    std::printf("8-step chain: %zu/%zu identical tokens\n", same, chain_st.size());

    if (mismatch == 0 && same == (int)chain_st.size()) {
        std::printf("SUSPICIOUS: loaders agree everywhere — verify safetensors weights are real (torch oracle)\n");
    } else {
        std::printf("E2E: loaders differ (weights genuinely from different sources) — matching torch decides correctness\n");
    }
    return 0;
}
