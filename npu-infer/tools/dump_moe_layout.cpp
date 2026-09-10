#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <string>
#include "lm_config.hpp"
#include "npu_utils/npu_instr_utils.hpp"
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
    const uint64_t* w = (const uint64_t*)&desc;
    uintptr_t table = (uintptr_t)w[0x70/8];
    const uint64_t* tp = (const uint64_t*)table;
    // FULL word dump for layer 0 (all nonzero 4-byte words with context)
    uintptr_t lp = (uintptr_t)tp[0];
    const uint32_t* d4 = (const uint32_t*)lp;
    const uint64_t* d8 = (const uint64_t*)lp;
    printf("layer 0 full (4-byte words, nonzero):\n");
    for (int i = 0; i < 0x500 / 4; i++) {
        uint32_t v = d4[i];
        if (v == 0) continue;
        // name?
        if (i % 2 == 0 && d8[i/2] > 0x500000000000ULL && d8[i/2] < 0x600000000000ULL) {
            const char* s = (const char*)d8[i/2];
            if (isprint((unsigned char)s[0])) { printf("  w[%d] NAME=%s\n", i, s); continue; }
        }
        printf("  w[%d]=0x%x\n", i, v);
    }
    return 0;
}
