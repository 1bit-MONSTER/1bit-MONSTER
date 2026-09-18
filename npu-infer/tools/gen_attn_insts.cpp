// gen_attn_insts.cpp — generate per-context MHA instruction streams for the
// npu-infer attention kernel (attn.xclbin), issue #2006 decode path.
//
// The attention kernel's instruction stream is CONTEXT-DEPENDENT: the MHA
// engine reads the KV cache up to the current position, so the DMA
// descriptors bake in the context length. Generate one stream per context
// length 1..MAX_CTX: attn_<M>_<K>_<N>_<ctx>_<woff>.bin, matching the
// engine's insts_for key (kernel=attn).
//
// Build (same libs as gen_mm_insts_batch):
//   g++ -O2 -std=c++17 -include climits gen_attn_insts.cpp -o gen_attn_insts \
//     -I/home/bcloud/amd-oss/fastflowlm/src/include \
//     -I/home/bcloud/amd-oss/fastflowlm/src/include/npu_utils \
//     -L/home/bcloud/amd-oss/fastflowlm/src/lib/xrt -lqwen3_npu -lgemm -lmha -lq4_npu_eXpress \
//     -L/usr/local/lib -laiebu -lxrt_coreutil -lxrt_core \
//     -Wl,-rpath,/home/bcloud/amd-oss/fastflowlm/src/lib/xrt
// Run:
//   ./gen_attn_insts <model_dir> <outdir> [max_ctx] [woff]
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>
#include "npu_utils/npu_instr_utils.hpp"
#include "models/qwen3/qwen3_npu_sequence.hpp"
#include "lm_config.hpp"

int main(int argc, char** argv) {
    std::string model_dir = (argc > 1) ? argv[1]
        : "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2";
    std::string outdir = (argc > 2) ? argv[2] : ".";
    int max_ctx = (argc > 3) ? atoi(argv[3]) : 4096;
    int woff = (argc > 4) ? atoi(argv[4]) : 0;
    LM_Config config;
    config.from_pretrained(model_dir);
    qwen3_npu_sequence qseq(config, 4096);
    qseq.set_max_length((uint32_t)max_ctx + 1);
    int n = 0;
    for (int L = 1; L <= max_ctx; L++) {
        npu_sequence seq(device_npu2);
        // single-token attention at position L-1 against the KV [0, L)
        qseq.gen_mha_engine_seq(&seq, L - 1, L);
        seq.cmds2seq();
        auto [ptr, nw] = seq.dump();
        char fname[256];
        snprintf(fname, sizeof(fname), "%s/attn_256_%u_128_%d_%d.bin",
                 outdir.c_str(), config.get<u32>("hidden_size"), L, woff);
        FILE* f = fopen(fname, "wb");
        if (f) { fwrite(ptr, 4, nw, f); fclose(f); n++; }
        if (L % 256 == 0) printf("ctx %d -> %s (%zu words)\n", L, fname, nw);
    }
    printf("generated %d attn streams (ctx 1..%d) -> %s\n", n, max_ctx, outdir.c_str());
    return 0;
}
