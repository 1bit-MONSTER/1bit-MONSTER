// gen_layer_elfs_moe_0946.cpp — regenerate the whole-layer per-ctx MoE ELFs
// from the FastFlowLM **v0.9.46** lib (md5 39a6c36a), whose host lib generates
// the CORRECT GatedDeltaNet sequence (the bf16 GDN NaNs only in the v1.0.x lib,
// md5 51cddb62 — see RESULTS-qwen3.6-moe-parity-2026-09-11.md + addendum 159).
//
// Differences from gen_layer_elfs_moe.cpp (v1.0.x):
//   * gen_layer_seq is 3-arg here (seq, L, is_full) vs 4-arg in v1.0.x.
//   * built against /home/bcloud/.local/flm-v0946 headers + libs (LM_Config
//     and npu_sequence layouts differ between v0.9.x and v1.0.x).
//
// Build:
//   g++ -O2 -std=c++20 -include climits gen_layer_elfs_moe_0946.cpp -o gen_layer_elfs_moe_0946 \
//     -I/home/bcloud/.local/flm-v0946/include \
//     -I/home/bcloud/.local/flm-v0946/include/npu_utils \
//     -L/home/bcloud/.local/flm-v0946/lib/xrt \
//     -lqwen3_6_moe_npu -lq4_npu_eXpress -lgemm -ldequant -lmha -llm_head \
//     -L/usr/local/lib -laiebu -lxrt_coreutil -lxrt_core \
//     -Wl,-rpath,/home/bcloud/.local/flm-v0946/lib/xrt \
//     -DCMAKE_INSTALL_PREFIX="/home/bcloud/.local/flm-v0946" \
//     -DCMAKE_XCLBIN_PREFIX="/home/bcloud/.local/flm-v0946/xclbins"
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>
#include <cstring>
#include <cstdlib>
#include "npu_utils/npu_instr_utils.hpp"
#include "lm_config.hpp"
#include "aiebu/aiebu.h"

// Minimal utils::find_xclbin_path (the only symbol LM_Config::from_pretrained
// needs from utils.cpp). The ELF generator never loads an xclbin; it only
// reads config.json + generates the instruction stream.
namespace utils {
std::string find_xclbin_path() {
    const char* env = std::getenv("FLM_XCLBIN_PATH");
    if (env && *env) return std::string(env);
    return "/home/bcloud/.local/flm-v0946";
}
}

// ---- opaque ABI declarations for the binary-only qwen3_6_moe classes ----
struct qwen3_6_moe_desc {
    unsigned char _pad[0x2000];
    void build(LM_Config& cfg);
};
class qwen3_6_moe_npu_sequence {
public:
    qwen3_6_moe_npu_sequence(qwen3_6_moe_desc& desc, LM_Config config, unsigned int MAX_L);
    // v0.9.46: 3-arg gen_layer_seq (L, is_full)
    void gen_layer_seq(npu_sequence* seq, unsigned int L, bool is_full);
    void gen_lm_head_seq(npu_sequence* seq, int a, int b);
};

int main(int argc, char** argv) {
    std::string model_dir = (argc > 1) ? argv[1]
        : "/home/bcloud/.local/flm-v0946/model/Qwen3.6-35B-A3B-NPU2";
    std::string outdir = (argc > 2) ? argv[2] : ".";
    int L0 = (argc > 3) ? atoi(argv[3]) : 1;
    int L1 = (argc > 4) ? atoi(argv[4]) : 2;

    LM_Config config;
    config.from_pretrained(model_dir);
    fprintf(stderr, "config loaded; building desc...\n");

    const size_t DESC_CAP = 0x20000;
    void* desc_mem = malloc(DESC_CAP);
    if (!desc_mem) { fprintf(stderr, "desc malloc failed\n"); return 1; }
    memset(desc_mem, 0, DESC_CAP);
    qwen3_6_moe_desc* desc = new (desc_mem) qwen3_6_moe_desc;
    desc->build(config);
    fprintf(stderr, "desc built; constructing sequence...\n");

    std::vector<std::string> layer_types;
    if (config._json_config.contains("layer_types") &&
        config._json_config["layer_types"].is_array()) {
        for (auto& e : config._json_config["layer_types"]) layer_types.push_back(e.get<std::string>());
    }
    for (size_t i = 0; i < layer_types.size(); i++)
        fprintf(stderr, "layer %zu: %s\n", i, layer_types[i].c_str());

    const size_t SEQ_CAP = 0x20000;
    void* seq_mem = malloc(SEQ_CAP);
    memset(seq_mem, 0, SEQ_CAP);
    qwen3_6_moe_npu_sequence* qseq =
        new (seq_mem) qwen3_6_moe_npu_sequence(*desc, config, 8192);
    fprintf(stderr, "sequence constructed at %p\n", qseq);

    for (int L = L0; L <= L1; L++) {
        npu_sequence seq(device_npu2);
        bool is_full = false;
        if ((size_t)L < layer_types.size())
            is_full = (layer_types[(size_t)L] == "full_attention");
        fprintf(stderr, "L=%d is_full=%d\n", L, (int)is_full);
        qseq->gen_layer_seq(&seq, L, is_full);
        seq.cmds2seq();
        auto [ptr, nw] = seq.dump();
        char* elf_buf = nullptr;
        uint32_t elf_size = aiebu_assembler_get_elf(
            aiebu_assembler_buffer_type_blob_instr_transaction,
            (const char*)ptr, (size_t)(nw * sizeof(uint32_t)), NULL, 0,
            (void**)&elf_buf, NULL, 0, "", "", NULL, 0);
        if (elf_size == 0) { fprintf(stderr, "L=%d aiebu failed\n", L); continue; }
        {
            char rname[256];
            snprintf(rname, sizeof(rname), "%s/moe_layer_ctx%d.txn", outdir.c_str(), L);
            FILE* fr = fopen(rname, "wb");
            if (fr) { fwrite(ptr, 4, nw, fr); fclose(fr); }
        }
        char fname[256];
        snprintf(fname, sizeof(fname), "%s/moe_layer_ctx%d.elf", outdir.c_str(), L);
        FILE* f = fopen(fname, "wb");
        if (f) { fwrite(elf_buf, 1, elf_size, f); fclose(f); }
        printf("ctx=%d txn_words=%zu elf=%u -> %s\n", L, nw, elf_size, fname);
        free(elf_buf);
    }
    {
        npu_sequence seq(device_npu2);
        qseq->gen_lm_head_seq(&seq, 0, 0);
        seq.cmds2seq();
        auto [ptr, nw] = seq.dump();
        char* elf_buf = nullptr;
        uint32_t elf_size = aiebu_assembler_get_elf(
            aiebu_assembler_buffer_type_blob_instr_transaction,
            (const char*)ptr, (size_t)(nw * sizeof(uint32_t)), NULL, 0,
            (void**)&elf_buf, NULL, 0, "", "", NULL, 0);
        if (elf_size) {
            char fname[256];
            snprintf(fname, sizeof(fname), "%s/moe_lm_head.elf", outdir.c_str());
            FILE* f = fopen(fname, "wb");
            if (f) { fwrite(elf_buf, 1, elf_size, f); fclose(f); }
            printf("lm_head txn_words=%zu elf=%u -> %s\n", nw, elf_size, fname);
        } else {
            fprintf(stderr, "lm_head aiebu failed\n");
        }
        free(elf_buf);
    }
    free(seq_mem);
    free(desc_mem);
    return 0;
}
