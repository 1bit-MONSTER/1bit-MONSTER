#include <cstdio>
#include "npu_utils/npu_instr_utils.hpp"
#include "models/qwen3/qwen3_npu_sequence.hpp"
#include "lm_config.hpp"
int main(int argc, char** argv) {
    std::string md = argc>1?argv[1]:"/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2";
    LM_Config c; c.from_pretrained(md);
    fprintf(stderr, "hidden=%d heads=%d kvheads=%d layers=%d im=%d\n",
        c.get<u32>("hidden_size"), c.get<u32>("num_attention_heads"),
        c.get<u32>("num_key_value_heads"), c.get<u32>("num_hidden_layers"),
        c.get<u32>("intermediate_size"));
    qwen3_npu_sequence q(c, 32768);
    printf("k03=%zu k47=%zu v03=%zu v47=%zu\n",
        q.get_k03_offset(), q.get_k47_offset(), q.get_v03_offset(), q.get_v47_offset());
    return 0;
}
