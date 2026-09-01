// gen_layer_seq_driver.cpp — generate gen_layer_seq(L) for L=0..27 with the
// runtime's actual parameters and dump the raw TXN words, to match against
// the ELF-captured layer TXNs from cap_interposer.
//
// Build:
//   g++ -O2 -std=c++17 -include climits gen_layer_seq_driver.cpp -o gen_layer_seq_driver \
//     -I/home/bcloud/amd-oss/fastflowlm/src/include \
//     -I/home/bcloud/amd-oss/fastflowlm/src/include/npu_utils \
//     -L/home/bcloud/amd-oss/fastflowlm/src/lib/xrt -lqwen3_npu -lgemm -lmha -lq4_npu_eXpress \
//     -L/usr/local/lib -laiebu -lxrt_coreutil -lxrt_core \
//     -Wl,-rpath,/home/bcloud/amd-oss/fastflowlm/src/lib/xrt
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>
#include <cstring>
#include "npu_utils/npu_instr_utils.hpp"
#include "models/qwen3/qwen3_npu_sequence.hpp"
#include "lm_config.hpp"

int main(int argc, char** argv) {
    std::string model_dir = (argc > 1) ? argv[1]
        : "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2";
    int L0 = (argc > 2) ? atoi(argv[2]) : 0;
    int L1 = (argc > 3) ? atoi(argv[3]) : 27;
    LM_Config config;
    config.from_pretrained(model_dir);
    qwen3_npu_sequence qseq(config, 4096);
    for (int L = L0; L <= L1; L++) {
        npu_sequence seq(device_npu2);
        qseq.gen_layer_seq(&seq, L);
        seq.cmds2seq();
        auto [ptr, nw] = seq.dump();
        char fname[64];
        snprintf(fname, sizeof(fname), "gen_layer_L%02d.bin", L);
        FILE* f = fopen(fname, "wb");
        if (f) { fwrite(ptr, 4, nw, f); fclose(f); }
        printf("L=%02d words=%zu -> %s\n", L, nw, fname);
    }
    return 0;
}
