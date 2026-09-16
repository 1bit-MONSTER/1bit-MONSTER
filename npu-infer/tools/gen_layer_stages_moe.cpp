// gen_stage.cpp — generate a Qwen3.6-35B linear-layer instruction sequence
// STAGE BY STAGE from the runtime's exported qwen3_6_moe_npu_sequence
// generators, so the engine can be rebuilt as a multi-kernel per-layer flow.
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include "npu_utils/npu_instr_utils.hpp"
#include "lm_config.hpp"
#include "aiebu/aiebu.h"

struct qwen3_6_moe_desc { unsigned char _pad[0x2000]; void build(LM_Config&); };

struct weight_desc_t;              // opaque
struct qwen3_6_layer_weright_def;  // opaque

class qwen3_6_moe_npu_sequence {
public:
    qwen3_6_moe_npu_sequence(qwen3_6_moe_desc&, LM_Config, unsigned int);
    void gen_layer_seq(npu_sequence*, unsigned, bool, bool);
    void gen_lm_head_seq(npu_sequence*, int, int);
    void _send_hidden_states(npu_sequence*);
    void _send_rms_weights(npu_sequence*);
    void _send_rope_rms_weights(npu_sequence*);
    void _send_linear_conv_weights(npu_sequence*);
    void _move_alpha_beta_weights(npu_sequence*, unsigned, unsigned, unsigned);
    void gen_seq_conv1d(npu_sequence*, unsigned, unsigned, unsigned, unsigned);
    void gen_mm_seq(npu_sequence*, unsigned, unsigned, unsigned, unsigned, unsigned, int);
    void _gen_linear_sequence(npu_sequence*, bool);
};

static int g_stage = 0;
static void dump_elf(npu_sequence& seq, const char* out) {
    seq.cmds2seq();
    auto [ptr, nw] = seq.dump();
    char* elf = nullptr;
    uint32_t n = aiebu_assembler_get_elf(
        aiebu_assembler_buffer_type_blob_instr_transaction,
        (const char*)ptr, (size_t)(nw*4), NULL, 0, (void**)&elf, NULL, 0, "", "", NULL, 0);
    fprintf(stderr, "stage %d: txn_words=%zu elf=%u\n", g_stage, nw, n);
    if (n) { FILE* f=fopen(out,"wb"); fwrite(elf,1,n,f); fclose(f); }
    { std::string t=std::string(out)+".txn"; FILE* f=fopen(t.c_str(),"wb"); if(f){fwrite(ptr,4,nw,f);fclose(f);} }
    free(elf);
}

int main(int argc, char** argv) {
    std::string model_dir = argc>1?argv[1]:"/home/bcloud/.config/flm/models/Qwen3.6-35B-A3B-NPU2";
    std::string which = argc>2?argv[2]:"hidden";
    std::string out = argc>3?argv[3]:"/tmp/stage.elf";
    int L = argc>4?atoi(argv[4]):0;
    LM_Config config; config.from_pretrained(model_dir);
    void* dm = calloc(1, 0x20000);
    qwen3_6_moe_desc* desc = (qwen3_6_moe_desc*)dm; desc->build(config);
    void* sm = calloc(1, 0x20000);
    qwen3_6_moe_npu_sequence* qs = new (sm) qwen3_6_moe_npu_sequence(*desc, config, 8192);
    npu_sequence seq(device_npu2);
    if (which=="hidden") { qs->_send_hidden_states(&seq); g_stage=1; }
    else if (which=="rms") { qs->_send_rms_weights(&seq); g_stage=2; }
    else if (which=="rope") { qs->_send_rope_rms_weights(&seq); g_stage=3; }
    else if (which=="convw") { qs->_send_linear_conv_weights(&seq); g_stage=4; }
    else if (which=="conv1d") { qs->gen_seq_conv1d(&seq, 4, 2048, 8192, 4); g_stage=5; }
    else if (which=="linear") { qs->_gen_linear_sequence(&seq, false); g_stage=9; }
    else if (which=="layer") { qs->gen_layer_seq(&seq, L, false, false); g_stage=10; }
    else { fprintf(stderr,"unknown stage %s\n", which.c_str()); return 2; }
    dump_elf(seq, out.c_str());
    return 0;
}
