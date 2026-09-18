// dump_packed_layer.cpp — write npu_pack_layer_bo()'s output for one layer to a file,
// so it can be diffed against FLM's own weight BO captured under the interposer.
//
// WHY: the four families whose prefill is wrong have had three candidates cleared by control
// (the generated ELFs, the runlist machinery, and the BO's SIZE — the engine's smaller BO is
// demonstrably sufficient for FLM's ELF on two architectures). What remains is the BO's
// CONTENTS: the tile order and offsets inside a correctly-sized BO. A size comparison cannot
// see that, so this tool produces the other half of a byte-level diff.
//
// Usage: dump_packed_layer <model.q4nx> <out.bin> [layer] [H] [NC] [NH] [NKV] [IM] [NV] [HD]
//   defaults: layer 0 and Nanbeige4.1-3B-NPU2 dims (H=2560 NC=32 NH=20 NKV=4 IM=10752 NV=166144 HD=128)
//
// Build: g++ -std=c++17 -O2 -I npu-infer/include -o dump_packed_layer \
//          npu-infer/tools/dump_packed_layer.cpp npu-infer/src/model.c
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>

extern "C" {
#include "model.h"
ModelWeights* model_load(const char* path, ModelConfig config);
void model_free(ModelWeights* mw);
int  npu_layer_bo_bytes(ModelWeights* mw, const ModelConfig* config);
int  npu_pack_layer_bo(uint8_t* bo_buffer, ModelWeights* mw, const ModelConfig* config, int layer_idx);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <model.q4nx> <out.bin> [layer] [H NC NH NKV IM NV HD]\n", argv[0]);
        return 2;
    }
    const char* model = argv[1];
    const char* out   = argv[2];
    int layer = (argc > 3) ? atoi(argv[3]) : 0;
    // Nanbeige4.1-3B-NPU2 defaults: the family the size control flagged.
    int H  = (argc > 4) ? atoi(argv[4]) : 2560;
    int NC = (argc > 5) ? atoi(argv[5]) : 32;
    int NH = (argc > 6) ? atoi(argv[6]) : 20;
    int NKV= (argc > 7) ? atoi(argv[7]) : 4;
    int IM = (argc > 8) ? atoi(argv[8]) : 10752;
    int NV = (argc > 9) ? atoi(argv[9]) : 166144;
    int HD = (argc > 10)? atoi(argv[10]): 128;

    ModelConfig cfg = QWEN3_0_6B_CONFIG;   // start from a full profile, then override
    cfg.hidden_size = H;
    cfg.num_layers = NC;
    cfg.num_attention_heads = NH;
    cfg.num_key_value_heads = NKV;
    cfg.intermediate_size = IM;
    cfg.head_dim = HD;
    cfg.vocab_size = NV;

    ModelWeights* mw = model_load(model, cfg);
    if (!mw) { fprintf(stderr, "model_load failed: %s\n", model); return 1; }
    const int bytes = npu_layer_bo_bytes(mw, &cfg);
    if (bytes <= 0) { fprintf(stderr, "npu_layer_bo_bytes returned %d\n", bytes); model_free(mw); return 1; }
    fprintf(stderr, "layer %d: bo_bytes=%d (%.2f tiles at 5120)\n",
            layer, bytes, bytes / 5120.0);
    uint8_t* bo = (uint8_t*)malloc((size_t)bytes);
    if (!bo) { fprintf(stderr, "alloc failed\n"); model_free(mw); return 1; }
    const int tiles = npu_pack_layer_bo(bo, mw, &cfg, layer);
    fprintf(stderr, "packed tiles=%d\n", tiles);
    FILE* f = fopen(out, "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", out); free(bo); model_free(mw); return 1; }
    fwrite(bo, 1, (size_t)bytes, f);
    fclose(f);
    fprintf(stderr, "wrote %s (%d bytes)\n", out, bytes);
    free(bo);
    model_free(mw);
    return 0;
}
