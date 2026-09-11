#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include "npu_utils/npu_instr_utils.hpp"
#include "models/qwen3/qwen3_npu_sequence.hpp"
#include "lm_config.hpp"
namespace utils { std::string find_xclbin_path() { return "/home/bcloud/amd-oss/fastflowlm/src/xclbins"; } }
int main(int argc, char** argv) {
    std::string md = argc>1?argv[1]:"/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2";
    int npt = argc>2?atoi(argv[2]):9;
    const char* out = argc>3?argv[3]:"mha_dump.bin";
    LM_Config cfg; cfg.from_pretrained(md);
    qwen3_npu_sequence qseq(cfg, 8192);
    npu_sequence seq(device_npu2);
    qseq.gen_mha_engine_seq(&seq, 0, (uint32_t)npt);
    seq.cmds2seq();
    auto [ptr, nw] = seq.dump();
    fprintf(stderr, "mha(0,%d) MAX_L=8192 words=%zu\n", npt, nw);
    FILE* f = fopen(out, "wb"); fwrite(ptr, 4, nw, f); fclose(f);
    return 0;
}
