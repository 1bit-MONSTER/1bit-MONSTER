#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include "npu_utils/npu_instr_utils.hpp"
#include "models/qwen3/qwen3_npu_sequence.hpp"
#include "lm_config.hpp"
namespace utils { std::string find_xclbin_path() { return "/home/bcloud/amd-oss/fastflowlm/src/xclbins"; } }
int main(int argc, char** argv) {
    std::string md = argv[1];
    int npt = atoi(argv[2]);
    int maxl = atoi(argv[3]);
    const char* out = argv[4];
    LM_Config cfg; cfg.from_pretrained(md);
    qwen3_npu_sequence qseq(cfg, maxl);
    npu_sequence seq(device_npu2);
    qseq.gen_mha_engine_seq(&seq, 0, (uint32_t)npt);
    seq.cmds2seq();
    auto [ptr, nw] = seq.dump();
    // count op 0x01 (BLOCKWRITE) in raw words
    int nb = 0, ndp = 0;
    for (size_t i=0;i<nw;i++) if ((ptr[i]&0xFF)==0x01) nb++; else if ((ptr[i]&0xFF)==0x81) ndp++;
    fprintf(stderr, "npt=%d maxl=%d words=%zu BLOCKWRITE~%d DDR_PATCH~%d\n", npt, maxl, nw, nb, ndp);
    FILE* f=fopen(out,"wb"); fwrite(ptr,4,nw,f); fclose(f);
    return 0;
}
