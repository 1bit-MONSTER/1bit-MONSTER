// gen_mm_insts_batch.cpp — batch-generate per-shape instruction streams for
// the npu-infer engine (issue #2006).
//
// The amdxdna mm kernel ABI is (opcode, instr_bo, ninstr, bo0..boN) and the
// instruction stream is generated PER GEMM SHAPE (M/K/N/weight_offset are
// baked into the DMA descriptors). A single <xclbin>.bin cannot serve q/k/v/o
// (N=128-decode), gate/up, down (K=3072) and lm_head with one stream.
//
// Kernel shape convention (reverse-engineered from FastFlowLM
// Gemm::generate_seq via libgemm.so, 2026-09-01):
//   out[M, N] = W[M, K] @ act[K, N]
//   M = block output rows  — must be a multiple of 256 (the NPU row grid)
//   K = input (reduction)  — 1024 for q/k/v/o/gate/up, 3072 for down, hidden
//                             for lm_head (tie_word_embeddings => hidden)
//   N = token batch        — must be a multiple of 128; decode N=128 padded
//   weight_offset          — byte offset into the weight BO for this block's
//                             W matrix; the engine keeps one BO per block, so
//                             this is 0 per call (kept in the key for future
//                             shared-BO layouts)
//
// Emits <outdir>/mm_<M>_<K>_<N>_<woff>.bin per shape + <outdir>/manifest.json
// describing the per-projection block shape space.
//
// Build (same libs as gen_mm_insts):
//   g++ -O2 -std=c++17 -include climits -mavx2 gen_mm_insts_batch.cpp -o gen_mm_insts_batch \
//     -I/home/bcloud/amd-oss/fastflowlm/src/include \
//     -I/home/bcloud/amd-oss/fastflowlm/src/include/npu_utils \
//     -L/home/bcloud/amd-oss/fastflowlm/src/lib/xrt -lqwen3_npu -lgemm -lmha -lq4_npu_eXpress \
//     -L/usr/local/lib -laiebu -lxrt_coreutil -lxrt_core \
//     -Wl,-rpath,/home/bcloud/amd-oss/fastflowlm/src/lib/xrt
// Run:
//   ./gen_mm_insts_batch <config.json> <outdir> [n_tokens]
//     n_tokens defaults to 128 (decode pad). Prefill: pass seq (padded up to
//     the next multiple of 128 by the tool).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include "npu_utils/npu_instr_utils.hpp"
#include "modules/gemm.hpp"
#include "lm_config.hpp"

struct Shape {
    uint32_t m, k, n, woff;
    bool operator<(const Shape& o) const {
        if (m != o.m) return m < o.m;
        if (k != o.k) return k < o.k;
        if (n != o.n) return n < o.n;
        return woff < o.woff;
    }
};

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: gen_mm_insts_batch <config.json> <outdir> [n_tokens=128]\n");
        return 1;
    }
    const char* cfg_path = argv[1];
    std::string outdir = argv[2];
    uint32_t n_tokens = (argc > 3) ? (uint32_t)strtoul(argv[3], nullptr, 10) : 128;

    // Pad N (token batch) up to the next multiple of 128 — the kernel's N
    // granularity. The engine always runs a single token per GEMM (decode),
    // so 128 is the minimum; larger batches pad the same way.
    if (n_tokens == 0) n_tokens = 128;
    uint32_t n_pad = ((n_tokens + 127) / 128) * 128;

    LM_Config config;
    std::ifstream f(cfg_path);
    if (!f.is_open()) { fprintf(stderr, "cannot open %s\n", cfg_path); return 1; }
    f >> config._json_config;

    // Pull dims from the config JSON (HF keys).
    auto& j = config._json_config;
    uint32_t hidden   = j.value("hidden_size", 1024u);
    uint32_t interm   = j.value("intermediate_size", 0u);
    uint32_t vocab    = j.value("vocab_size", 0u);
    bool tie_emb      = j.value("tie_word_embeddings", false);
    uint32_t n_heads  = j.value("num_attention_heads", 0u);
    uint32_t n_kv     = j.value("num_key_value_heads", n_heads);
    const uint32_t BLOCK_M = 256;  // kernel M granularity (NPU row grid)
    if (interm == 0 || vocab == 0) {
        fprintf(stderr, "config missing intermediate_size/vocab_size\n");
        return 1;
    }

    // --- Build the per-projection shape space ---------------------------
    // Each projection's weight is split into [BLOCK_M, K] blocks (the engine
    // dequantizes one [256, K] f32 block per BO). One GEMM per block.
    // weight_offset stays 0 (per-block BOs); it is carried in the key so a
    // future shared-BO layout can reuse the cache.
    struct Proj { const char* name; uint32_t k; uint32_t out_rows; };
    std::vector<Proj> projs = {
        {"q_proj",  hidden, hidden},
        {"k_proj",  hidden, hidden},
        {"v_proj",  hidden, hidden},
        {"o_proj",  hidden, hidden},
        {"gate_proj", hidden, interm},
        {"up_proj",  hidden, interm},
        {"down_proj", interm, hidden},
        {"lm_head",  hidden, tie_emb ? hidden : vocab},
    };

    std::map<Shape, std::string> emitted;
    nlohmann::json manifest = nlohmann::json::object();
    manifest["config"] = cfg_path;
    manifest["n_tokens"] = n_tokens;
    manifest["n_pad"] = n_pad;
    manifest["note"] =
        "out[M,N] = W[M,K] @ act[K,N]; M=block rows (mult of 256), "
        "K=reduction, N=token batch (padded to mult of 128), "
        "weight_offset=bytes into the weight BO (0 = per-block BO)";
    manifest["projections"] = nlohmann::json::object();

    Gemm gemm(config);
    std::vector<std::string> files;
    for (auto& p : projs) {
        if (p.k == 0 || p.out_rows == 0) continue;
        uint32_t n_blocks = (p.out_rows + BLOCK_M - 1) / BLOCK_M;
        nlohmann::json jp = nlohmann::json::object();
        jp["k"] = p.k;
        jp["out_rows"] = p.out_rows;
        jp["blocks"] = n_blocks;
        jp["shapes"] = nlohmann::json::array();
        for (uint32_t b = 0; b < n_blocks; b++) {
            uint32_t m = BLOCK_M;  // (last block could be smaller; kernel
                                   // pads — keep full M for simplicity)
            uint32_t woff = 0;
            Shape s{m, p.k, n_pad, woff};
            char fname[256];
            snprintf(fname, sizeof(fname), "mm_%u_%u_%u_%u.bin", s.m, s.k, s.n, s.woff);
            std::string path = outdir + "/" + fname;
            if (emitted.find(s) == emitted.end()) {
                npu_sequence seq;
                gemm.generate_seq(&seq, s.m, s.k, s.n, s.woff, false,
                                  Gemm::NO_Activation, 0);
                seq.cmds2seq();
                seq.write_out_sequence(path);
                emitted[s] = path;
                files.push_back(path);
                fprintf(stderr, "wrote %s (GEMM %ux%ux%u woff=%u)\n",
                        path.c_str(), s.m, s.k, s.n, s.woff);
            }
            jp["shapes"].push_back({{"block", b}, {"file", fname},
                                    {"m", s.m}, {"k", s.k}, {"n", s.n},
                                    {"weight_offset", s.woff}});
        }
        manifest["projections"][p.name] = jp;
    }
    manifest["files"] = files;

    std::string man_path = outdir + "/manifest.json";
    std::ofstream mf(man_path);
    mf << manifest.dump(2);
    mf.close();
    fprintf(stderr, "wrote %s (%zu shapes, %zu unique)\n",
            man_path.c_str(), manifest["projections"].size(), emitted.size());
    return 0;
}
