// test_dequant_ref.c — verify Q4NX dequant against Python reference.
// Prints q_proj layer0 block0 first 16 BF16 values (as floats).
#include "model.h"
#include "common.h"
#include <stdio.h>
#include <string.h>

int main(int argc, char** argv) {
    const char* path = (argc > 1) ? argv[1]
        : "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx";
    ModelWeights* mw = model_load(path, QWEN3_0_6B_CONFIG);
    if (!mw) return 1;

    TensorDesc* q = &mw->layers[0].q_proj_weight;
    printf("q_proj: dtype=%s shape=[%lld,%lld] data_base=%llu data_offset=%llu\n",
           q->dtype, (long long)q->shape[0], (long long)q->shape[1],
           (unsigned long long)mw->data_base, (unsigned long long)q->data_offset);
    printf("num_blocks=%d\n", npu_weight_num_blocks(q, &mw->config, (int)mw->config.hidden_size));

    static uint16_t block[256 * 1024];
    const void* data = model_tensor_data(mw, q);
    int n = npu_dequant_block(block, data, q, &mw->config, 0, (int)mw->config.hidden_size);
    printf("block0 wrote %d values\n", n);

    // Also verify embed row 0
    TensorDesc* emb = &mw->embed_tokens;
    const uint16_t* e = (const uint16_t*)model_tensor_data(mw, emb);
    printf("embed dtype=%s\n", emb->dtype);
    for (int i = 0; i < 8; i++) {
        float f; uint32_t b = (uint32_t)e[i] << 16; memcpy(&f, &b, 4);
        printf("emb[%d]=%.5f ", i, f);
    }
    printf("\n");

    printf("q_proj block0 row0 first 16: ");
    for (int i = 0; i < 16; i++) {
        float f; uint32_t b = (uint32_t)block[i] << 16; memcpy(&f, &b, 4);
        printf("%.5f ", f);
    }
    printf("\n");
    printf("q_proj block0 row1 first 16: ");
    for (int i = 0; i < 16; i++) {
        float f; uint32_t b = (uint32_t)block[1024 + i] << 16; memcpy(&f, &b, 4);
        printf("%.5f ", f);
    }
    printf("\n");

    model_free(mw);
    return 0;
}
