#include <fstream>
#include <sstream>
#include "models/qwen3/qwen3_npu_sequence.hpp"
#include "npu_utils/npu_instr_utils.hpp"
#include <cstdio>
int main() {
    setvbuf(stderr, NULL, _IONBF, 0);
    LM_Config cfg;
    cfg.model_path = "/tmp/fflm-qwen3";
    cfg.model_name = "Qwen3-0.6B";
    // load the JSON directly (from_pretrained pulls in the path resolver)
    { std::ifstream f("/tmp/fflm-qwen3/config.json"); std::stringstream ss; ss << f.rdbuf();
      cfg._json_config = nlohmann::json::parse(ss.str()); }
    fprintf(stderr, "[1] json loaded: hidden_size=%u layers=%u\n",
            cfg._json_config["hidden_size"].get<uint32_t>(),
            cfg._json_config["num_hidden_layers"].get<uint32_t>());
    fprintf(stderr, "constructing...\n");
    qwen3_npu_sequence seq(cfg, 128);
    fprintf(stderr, "[2] constructed; generating layer 0...\n");
    npu_sequence nseq(device_npu2);   // FIX: initialize header fields (was default-ctor = uninitialized!)
    seq.gen_layer_seq(&nseq, 1);   // layers are 1-indexed (L>0)
    fprintf(stderr, "[3] generated; dumping...\n");
    auto [ptr, n] = nseq.dump();
    nseq.write_out_sequence("/tmp/fflm-layer0.seq");
    fprintf(stderr, "[4] words=%zu header=%08x %08x %08x %08x\n", n, ptr[0], ptr[1], ptr[2], ptr[3]);
    auto pt = nseq.dump_patch_table();
    fprintf(stderr, "[5] patch table: %zu triples\n", pt.size()/3);
    long mn[5] = {1<<30,1<<30,1<<30,1<<30,1<<30}, mx[5] = {0,0,0,0,0};
    for (size_t j = 0; j + 2 < pt.size(); j += 3) {
        unsigned arg = pt[j+1];
        if (arg < 5) { if (pt[j+2] < mn[arg]) mn[arg] = pt[j+2]; if (pt[j+2] > mx[arg]) mx[arg] = pt[j+2]; }
    }
    for (int a = 0; a < 5; a++) fprintf(stderr, "  arg%d: offset span [%ld, %ld] -> ~%ld B\n",
                                        a, mn[a], mx[a], mx[a] - mn[a] + 4096);
    return 0;
    return 0;
}
