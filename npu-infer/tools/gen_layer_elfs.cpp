// gen_layer_elfs.cpp — build per-context layer ELFs exactly like the runtime's
// _setup_kernel: gen_layer_seq(ctx+1) -> aiebu_assembler_get_elf -> ELF file.
// Usage: gen_layer_elfs <model_dir> <outdir> [L_begin] [L_end]
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>
#include <cstring>
#include <cstdlib>
#include "npu_utils/npu_instr_utils.hpp"
#include "models/qwen3/qwen3_npu_sequence.hpp"
#include "lm_config.hpp"
#include "aiebu/aiebu.h"

namespace utils { std::string find_xclbin_path() { return "/home/bcloud/amd-oss/fastflowlm/src/xclbins"; } }

int main(int argc, char** argv) {
    std::string model_dir = (argc > 1) ? argv[1] : "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2";
    std::string outdir = (argc > 2) ? argv[2] : ".";
    int L0 = (argc > 3) ? atoi(argv[3]) : 1;
    int L1 = (argc > 4) ? atoi(argv[4]) : 2;
    LM_Config config;
    config.from_pretrained(model_dir);
    qwen3_npu_sequence qseq(config, 8192);
    for (int L = L0; L <= L1; L++) {
        npu_sequence seq(device_npu2);
        qseq.gen_layer_seq(&seq, L);
        seq.cmds2seq();
        auto [ptr, nw] = seq.dump();
        char* elf_buf = nullptr;
        uint32_t elf_size = aiebu_assembler_get_elf(
            aiebu_assembler_buffer_type_blob_instr_transaction,
            (const char*)ptr, (size_t)(nw * sizeof(uint32_t)), NULL, 0,
            (void**)&elf_buf, NULL, 0, "", "", NULL, 0);
        if (elf_size == 0) { fprintf(stderr, "L=%d aiebu failed\n", L); continue; }
        // raw TXN dump alongside the ELF (for runtime A/B)
        {
            char rname[256];
            snprintf(rname, sizeof(rname), "%s/layer_ctx%d.txn", outdir.c_str(), L);
            FILE* fr = fopen(rname, "wb");
            if (fr) { fwrite(ptr, 4, nw, fr); fclose(fr); }
        }
        char fname[256];
        snprintf(fname, sizeof(fname), "%s/layer_ctx%d.elf", outdir.c_str(), L);
        FILE* f = fopen(fname, "wb");
        if (f) { fwrite(elf_buf, 1, elf_size, f); fclose(f); }
        printf("ctx=%d txn_words=%zu elf=%u -> %s\n", L, nw, elf_size, fname);
        free(elf_buf);
    }
    return 0;
}
