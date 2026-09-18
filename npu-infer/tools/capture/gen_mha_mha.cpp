#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <dlfcn.h>
#include "npu_utils/npu_instr_utils.hpp"
#include "models/qwen3/qwen3_npu_sequence.hpp"
#include "lm_config.hpp"
namespace utils { std::string find_xclbin_path() { return "/home/bcloud/amd-oss/fastflowlm/src/xclbins"; } }
int main(int argc, char** argv) {
    // MHA::generate_mha_sequence(seq, L0, L1, chunk, bool, int)
    typedef void (*genmha_t)(npu_sequence*, uint32_t, uint32_t, uint32_t, bool, int);
    typedef size_t (*chunk_t)();
    void* h = dlopen("/home/bcloud/amd-oss/fastflowlm/src/lib/xrt/libmha.so", RTLD_LAZY|RTLD_GLOBAL);
    auto gen = (genmha_t)dlsym(RTLD_DEFAULT, "_ZN3MHA21generate_mha_sequenceEP12npu_sequencejjjbi");
    auto chunk = (chunk_t)dlsym(RTLD_DEFAULT, "_ZN3MHA14get_chunk_sizeEv");
    if(!gen){fprintf(stderr,"no gen sym\n");return 1;}
    // MHA ctor: MHA(mha_type_t, int)
    fprintf(stderr, "chunk_size=%zu\n", chunk?chunk():0);
    std::string md = "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2";
    LM_Config cfg; cfg.from_pretrained(md);
    // construct MHA via qwen3_npu_sequence? No — MHA needs mha_type_t. Use the
    // qwen3_npu_sequence::gen_mha_engine_seq which wraps it. Instead, try
    // calling MHA::generate_mha_sequence via a stack MHA object.
    // mha_type_t enum unknown; try value 0..3
    for (int t = 0; t < 4; t++) {
        for (uint32_t ch : {64u, 128u, 256u}) {
            // construct MHA on stack (size unknown ~16 bytes)
            char mbuf[64] = {};
            typedef void (*mha_ctor_t)(void*, int, int);
            auto mctor = (mha_ctor_t)dlsym(RTLD_DEFAULT, "_ZN3MHAC1ER10mha_type_ti");
            mctor(mbuf, t, 0);
            npu_sequence seq(device_npu2);
            gen(&seq, 0, 256, ch, false, 0);
            seq.cmds2seq();
            auto [ptr, nw] = seq.dump();
            // count arg_idx pattern quickly
            int c0=0,c1=0,c2=0;
            for(size_t i=0;i+8<nw;i++){ if((ptr[i]&0xFF)==0x81){ uint32_t ai=ptr[i+8]; if(ai==0)c0++; if(ai==1)c1++; if(ai==2)c2++; i+=8;} }
            fprintf(stderr, "type=%d chunk=%u words=%zu argidx {0:%d,1:%d,2:%d}\n", t, ch, nw, c0,c1,c2);
            typedef void (*mha_dtor_t)(void*);
            auto mdtor = (mha_dtor_t)dlsym(RTLD_DEFAULT, "_ZN3MHAD1Ev");
            if(mdtor) mdtor(mbuf);
        }
    }
    return 0;
}
