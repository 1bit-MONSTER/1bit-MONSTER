// npu_runlist_bridge.cpp — single-launch whole-layer per-ctx ELF decode path.
//
// Bridges npu-infer's RuntimeLayerEngine (the FastFlowLM-validated whole-layer
// decode, byte-identical to the runtime) into npu_engine_universal WITHOUT
// dragging npu-infer's ModelConfig (npu-infer/include/common.h) into the
// engine TU — see the header for the name-clash isolation rationale.
//
// This TU is compiled against the npu-infer include tree only (runtime_layer.h
// + model.h + common.h). The engine links model.c (C) + runtime_layer.cpp
// (C++) + this bridge (C++) into npu_engine_universal.
#include "runtime_layer.h"
#include "model.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>

#include <xrt/xrt_device.h>

extern "C" int npu_pack_layer_bo(uint8_t* bo_buffer, void* mw, const void* config, int layer_idx);
extern "C" void npu_layer_tile_offsets(void* mw, int layer_idx, int* off_q, int* off_k, int* off_v, int* off_o, int* off_gu, int* off_d);

// Read whitespace-separated token ids from a file (or stdin for NULL/"-").
static bool read_ids(const char* ids_file, std::vector<int>& ids) {
    if (ids_file && ids_file[0] && strcmp(ids_file, "-") != 0) {
        FILE* f = fopen(ids_file, "r");
        if (!f) { fprintf(stderr, "[runlist] cannot open ids file %s\n", ids_file); return false; }
        int t; while (fscanf(f, "%d", &t) == 1) ids.push_back(t);
        fclose(f);
    } else {
        int t; while (scanf("%d", &t) == 1) ids.push_back(t);
    }
    return !ids.empty();
}

extern "C" int npu_runlist_decode(const char* model_path, int ng, const char* ids_file,
                               int H, int NC, int NH, int NKV, int IM, int NV) {
    // 1) prompt token ids (the engine feeds pre-tokenized ids; no tokenizer here)
    std::vector<int> ids;
    if (!read_ids(ids_file, ids)) { fprintf(stderr, "[runlist] no prompt tokens\n"); return 1; }

    // 2) per-model config (dims from the caller; whole-layer packing is dim-driven)
    ModelConfig cfg = QWEN3_0_6B_CONFIG;
    cfg.hidden_size = H;
    cfg.num_layers = NC;
    cfg.num_attention_heads = NH;
    cfg.num_key_value_heads = NKV;
    cfg.intermediate_size = IM;
    cfg.head_dim = 128;
    cfg.vocab_size = NV;
    cfg.max_position_embeddings = 40960;
    cfg.max_seq_len = 4096;

    // 3) per-model layer.xclbin + per-ctx ELF dir (env overrides first)
    const char* mdir = H == 2048 ? "Qwen3-1.7B-NPU2"
                     : H == 2560 ? "Qwen3-4B-NPU2"
                     : H == 4096 ? "Qwen3-8B-NPU2"
                                 : "Qwen3-0.6B-NPU2";
    const char* elf_default = H == 2048 ? "npu-infer/captures/txn-elfs-1p7b"
                            : H == 2560 ? "npu-infer/captures/txn-elfs-4b"
                            : H == 4096 ? "npu-infer/captures/txn-elfs-8b"
                                        : "npu-infer/captures/txn-elfs";
    if (!getenv("LAYER_XCLBIN")) {
        std::string xb = std::string("/home/bcloud/amd-oss/fastflowlm/src/xclbins/") + mdir + "/layer.xclbin";
        setenv("LAYER_XCLBIN", xb.c_str(), 0);
    }
    const char* env_elf = getenv("NPU_LAYER_ELF_DIR");
    std::string elf_dir = env_elf && env_elf[0] ? env_elf : elf_default;
    std::string lmhead_elf = elf_dir + "/elf_0002_lmhead.bin";

    // 4) model weights via npu-infer's model.c loader
    ModelWeights* mw = model_load(model_path, cfg);
    if (!mw) { fprintf(stderr, "[runlist] model_load failed: %s\n", model_path); return 1; }

    // 5) device + RuntimeLayerEngine (packs per-layer weight BOs + lm_head BO)
    xrt::device dev(0);
    RuntimeLayerEngine rt;
    if (!rt.init(dev, mw, cfg, elf_dir.c_str(), lmhead_elf.c_str())) {
        fprintf(stderr, "[runlist] RuntimeLayerEngine init failed\n");
        model_free(mw);
        return 1;
    }

    // 5) prefill — one whole-layer forward per prompt token (KV accumulates on
    //    device; the per-ctx ELF is regenerated/reused per context length).
    int npt = (int)ids.size();
    printf("=== Prefill %d ===\n", npt); fflush(stdout);
    auto t0 = std::chrono::steady_clock::now();
    int ctx = 0;
    for (int t : ids) {
        if (!rt.embed(t) || !rt.forward(++ctx)) {
            fprintf(stderr, "[runlist] prefill forward ctx=%d failed\n", ctx);
            model_free(mw);
            return 1;
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    double prefill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    printf("Prefill: %.0fms (%.0f ms/tok)\n\n", prefill_ms, prefill_ms / npt);

    // 6) greedy decode — get_logits -> argmax -> emit -> advance.
    auto tgs = std::chrono::steady_clock::now();
    int total = 0;
    for (int i = 0; i < ng; i++) {
        std::vector<float> lg(cfg.vocab_size);
        rt.get_logits(lg.data(), cfg.vocab_size);
        int best = 0;
        for (int v = 1; v < cfg.vocab_size; v++) if (lg[v] > lg[best]) best = v;
        printf("  [%d] %d\n", i + 1, best);
        total++;
        if (i + 1 < ng) {
            if (!rt.embed(best) || !rt.forward(++ctx)) {
                fprintf(stderr, "[runlist] decode forward ctx=%d failed\n", ctx);
                model_free(mw);
                return 1;
            }
        }
    }
    auto tge = std::chrono::steady_clock::now();
    double tts = std::chrono::duration<double>(tge - tgs).count();
    printf("\n=== %.1f ms/tok (%.0f tok/s) | tokens=%d ===\n",
           tts * 1000.0 / ng, ng / tts, total);

    model_free(mw);
    return 0;
}

// ===== bf16 prefill (mm.xclbin dequant + GEMM) support =====
// The engine drives the dequant/GEMM bridge (npu_engine_bf16_mm_bridge) for the
// prefill mm path; this TU packs the per-layer Q4NX weight BOs + tile offsets.
static ModelWeights* g_bf16_mw = nullptr;
static ModelConfig  g_bf16_cfg;

extern "C" int npu_bf16_prefill_init(const char* model_path, int H, int NC, int NH, int NKV, int IM, int NV) {
    g_bf16_cfg = QWEN3_0_6B_CONFIG;
    g_bf16_cfg.hidden_size = H;
    g_bf16_cfg.num_layers = NC;
    g_bf16_cfg.num_attention_heads = NH;
    g_bf16_cfg.num_key_value_heads = NKV;
    g_bf16_cfg.intermediate_size = IM;
    g_bf16_cfg.head_dim = 128;
    g_bf16_cfg.vocab_size = NV;
    g_bf16_cfg.max_position_embeddings = 40960;
    g_bf16_cfg.max_seq_len = 4096;
    g_bf16_mw = model_load(model_path, g_bf16_cfg);
    return g_bf16_mw ? 0 : -1;
}

// Pack layer `layer`'s weight BO into bo (>= 2048*5120 = 10 MB). Returns tiles;
// fills offs[6] = {q,k,v,o,gu,d} tile offsets (for bf16mm_dequant woff = tile*5120).
extern "C" int npu_bf16_pack_layer(int layer, uint8_t* bo, int* offs) {
    if (!g_bf16_mw) return 0;
    int tiles = npu_pack_layer_bo(bo, g_bf16_mw, &g_bf16_cfg, layer);
    npu_layer_tile_offsets(g_bf16_mw, layer, &offs[0], &offs[1], &offs[2], &offs[3], &offs[4], &offs[5]);
    return tiles;
}
