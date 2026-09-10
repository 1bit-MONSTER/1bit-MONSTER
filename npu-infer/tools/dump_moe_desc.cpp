// dump_moe_desc.cpp — build the qwen3_6_moe_desc (opaque) and hexdump its
// memory so the engine can reproduce the runtime's weight layout offsets.
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>
#include "lm_config.hpp"
#include "npu_utils/npu_instr_utils.hpp"

// _ZN16qwen3_6_moe_desc5buildER9LM_Config
struct qwen3_6_moe_desc {
    unsigned char _pad[0x2000];
    void build(LM_Config& cfg);
};
namespace utils { std::string find_xclbin_path(); }

int main(int argc, char** argv) {
    std::string model_dir = argc > 1 ? argv[1] : "/tmp/Qwen3.6-35B-A3B-NPU2";
    LM_Config config;
    config.from_pretrained(model_dir);
    qwen3_6_moe_desc desc;
    memset(&desc, 0, sizeof(desc));
    desc.build(config);

    // dump the desc memory in 64-bit words (nonzero)
    const uint64_t* w = (const uint64_t*)&desc;
    for (size_t i = 0; i < sizeof(desc) / 8; i++) {
        if (w[i] != 0)
            printf("off=0x%04zx  val=0x%016llx\n", i * 8,
                   (unsigned long long)w[i]);
    }
    return 0;
}
