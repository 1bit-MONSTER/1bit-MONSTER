// gen_layer_elfs_moe.cpp — build per-context layer ELFs for the hybrid MoE
// (Qwen3.6-35B) using the exported qwen3_6_moe_npu_sequence from
// libqwen3_6_moe_npu.so. The desc is opaque: desc.build(LM_Config&) fills
// the layout from the config (no Q4NX, no load_weights — the engine packs
// the weight BOs itself), and the sequence is heap-allocated generously.
//
// Build (mirror the run_qwen3_6_moe harness link line):
//   g++ -O2 -std=c++20 -include climits gen_layer_elfs_moe.cpp -o gen_layer_elfs_moe \
//     -I/home/bcloud/amd-oss/fastflowlm/src/include \
//     -I/home/bcloud/amd-oss/fastflowlm/src/include/npu_utils \
//     -L/home/bcloud/amd-oss/fastflowlm/src/lib/xrt \
//     -lqwen3_6_moe_npu -lq4_npu_eXpress -lgemm -ldequant -lmha -llm_head \
//     -L/usr/local/lib -laiebu -lxrt_coreutil -lxrt_core \
//     -Wl,-rpath,/home/bcloud/amd-oss/fastflowlm/src/lib/xrt \
//     /home/bcloud/amd-oss/fastflowlm/src/common/utils.cpp \
//     -DCMAKE_INSTALL_PREFIX=\"/home/bcloud/amd-oss/fastflowlm\" \
//     -DCMAKE_XCLBIN_PREFIX=\"/home/bcloud/amd-oss/fastflowlm/src/xclbins\"
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>
#include <cstring>
#include <cstdlib>
#include "npu_utils/npu_instr_utils.hpp"
#include "lm_config.hpp"
#include "aiebu/aiebu.h"

// ---- opaque ABI declarations for the binary-only qwen3_6_moe classes ----
// _ZN16qwen3_6_moe_desc5buildER9LM_Config
struct qwen3_6_moe_desc {
    unsigned char _pad[0x2000];
    void build(LM_Config& cfg);
};
// _ZN24qwen3_6_moe_npu_sequenceC1ER16qwen3_6_moe_desc9LM_Configj
// _ZN24qwen3_6_moe_npu_sequence13gen_layer_seqEP12npu_sequencejbb
class qwen3_6_moe_npu_sequence {
public:
    qwen3_6_moe_npu_sequence(qwen3_6_moe_desc& desc, LM_Config config, unsigned int MAX_L);
    void gen_layer_seq(npu_sequence* seq, unsigned int L, bool a, bool b);
};

int main(int argc, char** argv) {
    std::string model_dir = (argc > 1) ? argv[1]
        : "/tmp/Qwen3.6-35B-A3B-NPU2";
    std::string outdir = (argc > 2) ? argv[2] : ".";
    int L0 = (argc > 3) ? atoi(argv[3]) : 1;
    int L1 = (argc > 4) ? atoi(argv[4]) : 2;

    LM_Config config;
    config.from_pretrained(model_dir);
    fprintf(stderr, "config loaded; building desc...\n");

    qwen3_6_moe_desc desc;
    desc.build(config);
    fprintf(stderr, "desc built; constructing sequence...\n");

    // The gen_layer_seq 3rd arg selects the path: true = full-attn
    // (_gen_sequence), false = linear-attn (_gen_linear_sequence). The 35B
    // alternates by layer (every 4th layer is full_attention), so read the
    // layer_types from the config.
    std::vector<std::string> layer_types =
        config.get<std::vector<std::string>>("layer_types", {});
    for (size_t i = 0; i < layer_types.size(); i++)
        fprintf(stderr, "layer %zu: %s\n", i, layer_types[i].c_str());

    // generously-sized opaque sequence buffer (real size is internal)
    const size_t SEQ_CAP = 0x20000;
    void* seq_mem = malloc(SEQ_CAP);
    memset(seq_mem, 0, SEQ_CAP);
    qwen3_6_moe_npu_sequence* qseq =
        new (seq_mem) qwen3_6_moe_npu_sequence(desc, config, 8192);
    fprintf(stderr, "sequence constructed at %p\n", qseq);

    for (int L = L0; L <= L1; L++) {
        npu_sequence seq(device_npu2);
        bool is_full = false;
        if ((size_t)L < layer_types.size())
            is_full = (layer_types[(size_t)L] == "full_attention");
        fprintf(stderr, "L=%d is_full=%d\n", L, (int)is_full);
        qseq->gen_layer_seq(&seq, L, is_full, false);
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
    free(seq_mem);
    return 0;
}
