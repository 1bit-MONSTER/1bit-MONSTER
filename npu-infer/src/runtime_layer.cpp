// runtime_layer.cpp — RuntimeLayerEngine: the FastFlowLM runtime's validated
// layer submission path (Round 36). See include/runtime_layer.h for layout
// documentation. All BO layouts here are byte-verified against captured
// runtime BOs (docs/txn-decode-findings.md Round 35/36).
#include "runtime_layer.h"
#include "model.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_xclbin.h>
#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/experimental/xrt_kernel.h>

extern "C" int npu_pack_layer_bo(uint8_t* bo_buffer, void* mw, const void* config, int layer_idx);
extern "C" int npu_pack_lmhead_bo(uint8_t* bo_buffer, void* mw, const void* config);

// ---- model-file layout constants (Qwen3-0.6B q4nx, flm 0.9.22) ----
// Metadata data_offsets are relative to data_base (8-byte len + JSON header);
// absolute file offset = data_base + data_offset. The norm tensors' metadata
// offsets for this model (recovered Round 36 by content search against the
// runtime's captured norm BOs):
static const uint64_t META_L0_ILN_OFF     = 311164928ull;   // L0 input_layernorm
static const uint64_t META_FINAL_NORM_OFF = 311293952ull;   // model.norm.weight
static const int      NORM_BLOCK_BYTES   = 4608;            // ILN+PALN+kn+qn
static const int      LMHEAD_TILES       = 18992;           // 151936/8
static const int      LMHEAD_BO_BYTES    = 98566144;
// Pipeline block order: layer -> position in the physical norm array.
static const int NORM_PIPELINE_ORDER[28] = {
    0, 1, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    2, 20, 21, 22, 23, 24, 25, 26, 27,
    3, 4, 5, 6, 7, 8, 9
};

static bool read_file(const char* path, std::vector<uint8_t>& out) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "RuntimeLayer: cannot open %s\n", path); return false; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    out.resize(sz);
    size_t br = fread(out.data(), 1, sz, f);
    fclose(f);
    return br == (size_t)sz;
}

static inline float bf16_to_f32(uint16_t v) {
    uint32_t bits = (uint32_t)v << 16; float f; memcpy(&f, &bits, 4); return f;
}
static inline uint16_t f32_to_bf16(float v) {
    uint32_t bits; memcpy(&bits, &v, 4);
    uint32_t rb = ((bits >> 16) & 1) + 0x7FFF;
    return (uint16_t)((bits + rb) >> 16);
}

RuntimeLayerEngine::RuntimeLayerEngine() {}
RuntimeLayerEngine::~RuntimeLayerEngine() {}

bool RuntimeLayerEngine::init(xrt::device& dev, ModelWeights* mw, const ModelConfig& cfg,
                              const char* layer_elf_dir, const char* lmhead_elf_path) {
    dev_ = &dev; mw_ = mw; cfg_ = cfg;
    elf_dir_ = layer_elf_dir ? layer_elf_dir : "";
    lmhead_elf_path_ = lmhead_elf_path ? lmhead_elf_path : "";

    const char* xclbin_path = getenv("LAYER_XCLBIN")
        ? getenv("LAYER_XCLBIN")
        : "/home/bcloud/amd-oss/fastflowlm/src/xclbins/Qwen3-0.6B-NPU2/layer.xclbin";
    FILE* f = fopen(xclbin_path, "rb");
    if (!f) { fprintf(stderr, "RuntimeLayer: cannot open %s\n", xclbin_path); return false; }
    fseek(f, 0, SEEK_END); long fsz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<char> raw(fsz);
    fread(raw.data(), 1, fsz, f); fclose(f);
    auto xclbin = std::make_unique<xrt::xclbin>(raw);
    dev.register_xclbin(*xclbin);
    hwctx_ = std::make_unique<xrt::hw_context>(dev, xclbin->get_uuid());
    if (!ensure_layer_kernel(1)) return false;   // eager: like the test
    // ---- shared kv BO (128MB, zero) — matches the byte-verified test ----
    kv_bos_.resize(1);
    kv_bos_[0] = std::make_unique<xrt::ext::bo>(dev, 134217728);
    memset(kv_bos_[0]->map(), 0, 134217728);
    kv_bos_[0]->sync(XCL_BO_SYNC_BO_TO_DEVICE);


    // lm_head kernel (context-independent ELF)
    std::vector<uint8_t> elfh;
    if (!read_file(lmhead_elf_path_.c_str(), elfh)) return false;
    {
        xrt::elf elf((const char*)elfh.data(), elfh.size());
        xrt::module mod(elf);
        kern_lmhead_ = std::make_unique<xrt::ext::kernel>(*hwctx_, mod, "MLIR_AIE");
    }

    // ---- per-layer weight BOs: npu_pack_layer_bo (byte-verified) ----
    weight_bos_.resize(cfg_.num_layers);
    for (int L = 0; L < cfg_.num_layers; L++) {
        weight_bos_[L] = std::make_unique<xrt::ext::bo>(dev, 10485760);
        uint8_t* m = static_cast<uint8_t*>(weight_bos_[L]->map());
        memset(m, 0, 10485760);
        if (!npu_pack_layer_bo(m, mw, &cfg_, L)) {
            fprintf(stderr, "RuntimeLayer: pack layer %d failed\n", L);
            return false;
        }
        weight_bos_[L]->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
    fprintf(stderr, "RuntimeLayer: packed %d layer weight BOs\n", cfg_.num_layers);

    // ---- lm_head weight BO: reorder with G=8 (byte-verified) ----
    if (!pack_lmhead_bo()) return false;

    // ---- norms i5/i6 per layer (physical pipeline blocks) ----
    if (!build_norm_bos()) return false;


    // ---- act / logits / final-norm BOs ----
    bo_act_ = std::make_unique<xrt::ext::bo>(dev, 1048576);
    bo_logits_ = std::make_unique<xrt::ext::bo>(dev, 1048576);
    memset(bo_act_->map(), 0, 1048576);
    memset(bo_logits_->map(), 0, 1048576);
    bo_fnorm_ = std::make_unique<xrt::ext::bo>(dev, 1048576);
    memset(bo_fnorm_->map(), 0, 1048576);
    if (mw_ && mw_->file_data && mw_->file_size >= mw_->data_base + META_FINAL_NORM_OFF + 2048) {
        memcpy(bo_fnorm_->map(), (const uint8_t*)mw_->file_data + mw_->data_base + META_FINAL_NORM_OFF, 2048);
    } else {
        fprintf(stderr, "RuntimeLayer: cannot read final norm from model file\n");
        return false;
    }
    bo_act_->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_logits_->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_fnorm_->sync(XCL_BO_SYNC_BO_TO_DEVICE);

    fprintf(stderr, "RuntimeLayer: init OK (%d layers)\n", cfg_.num_layers);
    return true;
}

bool RuntimeLayerEngine::pack_lmhead_bo() {
    if (npu_pack_lmhead_bo) {
        bo_lmhead_w_ = std::make_unique<xrt::ext::bo>(*dev_, LMHEAD_BO_BYTES);
        uint8_t* m = static_cast<uint8_t*>(bo_lmhead_w_->map());
        memset(m, 0, LMHEAD_BO_BYTES);
        if (npu_pack_lmhead_bo(m, mw_, &cfg_)) {
            bo_lmhead_w_->sync(XCL_BO_SYNC_BO_TO_DEVICE);
            fprintf(stderr, "RuntimeLayer: packed lm_head BO (%d tiles)\n", LMHEAD_TILES);
            return true;
        }
    }
    fprintf(stderr, "RuntimeLayer: npu_pack_lmhead_bo unavailable/failed\n");
    return false;
}

bool RuntimeLayerEngine::build_norm_bos() {
    if (!mw_ || !mw_->file_data || mw_->file_size < mw_->data_base + META_L0_ILN_OFF + 28ull * NORM_BLOCK_BYTES) {
        fprintf(stderr, "RuntimeLayer: model file too small for norm blocks\n");
        return false;
    }
    const uint8_t* base = (const uint8_t*)mw_->file_data + mw_->data_base + META_L0_ILN_OFF;
    i5_bos_.resize(cfg_.num_layers);
    i6_bos_.resize(cfg_.num_layers);
    for (int L = 0; L < cfg_.num_layers; L++) {
        int pos = NORM_PIPELINE_ORDER[L];
        const uint8_t* blk = base + (uint64_t)pos * NORM_BLOCK_BYTES;
        // i5 = ILN(2048) + PALN(2048)
        i5_bos_[L] = std::make_unique<xrt::ext::bo>(*dev_, 1048576);
        uint8_t* m5 = static_cast<uint8_t*>(i5_bos_[L]->map());
        memset(m5, 0, 1048576);
        memcpy(m5, blk, 4096);
        i5_bos_[L]->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        // i6 = [1.875 x64][0 x64][q_norm 128][k_norm 128]  (block tail = [kn][qn])
        i6_bos_[L] = std::make_unique<xrt::ext::bo>(*dev_, 1048576);
        uint8_t* m6 = static_cast<uint8_t*>(i6_bos_[L]->map());
        memset(m6, 0, 1048576);
        uint16_t* w6 = (uint16_t*)m6;
        for (int i = 0; i < 64; i++) w6[i] = f32_to_bf16(1.875f);
        memcpy(m6 + 256, blk + 4352, 256);   // q_norm (block tail second half)
        memcpy(m6 + 512, blk + 4096, 256);   // k_norm (block tail first half)
        i6_bos_[L]->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
    fprintf(stderr, "RuntimeLayer: built %d per-layer norm BOs\n", cfg_.num_layers);
    return true;
}

bool RuntimeLayerEngine::ensure_layer_kernel(int ctx_len) {
    auto it = layer_kernels_.find(ctx_len);
    if (it != layer_kernels_.end()) return true;
    char fname[512];
    snprintf(fname, sizeof(fname), "%s/layer_ctx%d.elf", elf_dir_.c_str(), ctx_len);
    std::vector<uint8_t> elfb;
    if (!read_file(fname, elfb)) {
        fprintf(stderr, "RuntimeLayer: missing layer ELF for ctx=%d (%s)\n", ctx_len, fname);
        return false;
    }
    try {
        xrt::elf elf((const char*)elfb.data(), elfb.size());
        xrt::module mod(elf);
        layer_kernels_[ctx_len] = std::make_unique<xrt::ext::kernel>(*hwctx_, mod, "MLIR_AIE");
    } catch (const std::exception& e) {
        fprintf(stderr, "RuntimeLayer: kernel build ctx=%d failed: %s\n", ctx_len, e.what());
        return false;
    }
    fprintf(stderr, "RuntimeLayer: layer kernel ctx=%d ready\n", ctx_len);
    return true;
}

bool RuntimeLayerEngine::embed(int token) {
    TensorDesc* emb = &mw_->embed_tokens;
    if (emb->ndim != 2 || token < 0 || token >= emb->shape[0]) return false;
    uint64_t off = (uint64_t)token * emb->shape[1] * 2;
    // The runtime reads the embedding at data_base + data_offset (SafeTensors
    // semantics; embed data_offset = 0) — verified byte-exact vs the runtime's
    // act input (preinsts_001_00_i3 == file[data_base + 2048*1000]).
    if (mw_->file_size < mw_->data_base + off + emb->shape[1] * 2) return false;
    uint8_t* m = static_cast<uint8_t*>(bo_act_->map());
    memcpy(m, (const uint8_t*)mw_->file_data + mw_->data_base + off, emb->shape[1] * 2);
    bo_act_->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    return true;
}

bool RuntimeLayerEngine::forward(int ctx_len) {
    if (!ensure_layer_kernel(ctx_len)) return false;
    const char* dbg = getenv("RT_DUMP_ACT_PREFIX");
    if (dbg) {
        char fn[512];
        snprintf(fn, sizeof(fn), "%s_pre.bin", dbg);
        FILE* f = fopen(fn, "wb");
        if (f) {
            bo_act_->sync(XCL_BO_SYNC_BO_FROM_DEVICE, 1048576, 0);
            fwrite(bo_act_->map(), 1, 2048, f);
            fclose(f);
        }
        snprintf(fn, sizeof(fn), "%s_w0.bin", dbg);
        f = fopen(fn, "wb");
        if (f) {
            weight_bos_[0]->sync(XCL_BO_SYNC_BO_FROM_DEVICE, 10485760, 0);
            fwrite(weight_bos_[0]->map(), 1, 9830400, f);
            fclose(f);
        }
        snprintf(fn, sizeof(fn), "%s_w2.bin", dbg);
        f = fopen(fn, "wb");
        if (f) {
            weight_bos_[2]->sync(XCL_BO_SYNC_BO_FROM_DEVICE, 10485760, 0);
            fwrite(weight_bos_[2]->map(), 1, 9830400, f);
            fclose(f);
        }
        snprintf(fn, sizeof(fn), "%s_i5_0.bin", dbg);
        f = fopen(fn, "wb");
        if (f) {
            i5_bos_[0]->sync(XCL_BO_SYNC_BO_FROM_DEVICE, 1048576, 0);
            fwrite(i5_bos_[0]->map(), 1, 4096, f);
            fclose(f);
        }
        snprintf(fn, sizeof(fn), "%s_kv_0.bin", dbg);
        f = fopen(fn, "wb");
        if (f) {
            kv_bos_[0]->sync(XCL_BO_SYNC_BO_FROM_DEVICE, 33554432, 0);
            fwrite(kv_bos_[0]->map(), 1, 4096, f);
            fclose(f);
        }
        snprintf(fn, sizeof(fn), "%s_i6_0.bin", dbg);
        f = fopen(fn, "wb");
        if (f) {
            i6_bos_[0]->sync(XCL_BO_SYNC_BO_FROM_DEVICE, 1048576, 0);
            fwrite(i6_bos_[0]->map(), 1, 768, f);
            fclose(f);
        }
    }
    for (int L = 0; L < cfg_.num_layers; L++) {
        xrt::run run(*layer_kernels_[ctx_len]);
        uint32_t v0 = 3, v1 = 0, v2 = 0;
        run.set_arg(0, (const void*)&v0, sizeof(v0));
        run.set_arg(1, (const void*)&v1, sizeof(v1));
        run.set_arg(2, (const void*)&v2, sizeof(v2));
        run.set_arg(3, (const xrt::bo&)*bo_act_);
        run.set_arg(4, (const xrt::bo&)*weight_bos_[L]);
        run.set_arg(5, (const xrt::bo&)*i5_bos_[L]);
        run.set_arg(6, (const xrt::bo&)*i6_bos_[L]);
        run.set_arg(7, (const xrt::bo&)*kv_bos_[0]);
        try {
            run.start();
            run.wait();
        } catch (const std::exception& e) {
            fprintf(stderr, "RuntimeLayer: layer %d run FAILED: %s\n", L, e.what());
            return false;
        }
        if (dbg && L == 0) {
            char fn0[256];
            snprintf(fn0, sizeof(fn0), "%s_kv_after_L00.bin", dbg);
            FILE* fk0 = fopen(fn0, "wb");
            if (fk0) {
                kv_bos_[0]->sync(XCL_BO_SYNC_BO_FROM_DEVICE, 134217728, 0);
                fwrite(kv_bos_[0]->map(), 1, 134217728, fk0);
                fclose(fk0);
            }
        }
        if (dbg && L == 2) {
            char fn[256];
            snprintf(fn, sizeof(fn), "%s_kv_after_L02.bin", dbg);
            FILE* fk = fopen(fn, "wb");
            if (fk) {
                kv_bos_[0]->sync(XCL_BO_SYNC_BO_FROM_DEVICE, 134217728, 0);
                fwrite(kv_bos_[0]->map(), 1, 134217728, fk);
                fclose(fk);
            }
        }
        if (dbg) {
            char fn[256];
            snprintf(fn, sizeof(fn), "%s_after_L%02d.bin", dbg, L);
            FILE* f = fopen(fn, "wb");
            if (f) {
                bo_act_->sync(XCL_BO_SYNC_BO_FROM_DEVICE, 1048576, 0);
                fwrite(bo_act_->map(), 1, 2048, f);
                fclose(f);
            }
        }
    }
    // lm_head
    {
        xrt::run run(*kern_lmhead_);
        uint32_t v0 = 3, v1 = 0, v2 = 0;
        run.set_arg(0, (const void*)&v0, sizeof(v0));
        run.set_arg(1, (const void*)&v1, sizeof(v1));
        run.set_arg(2, (const void*)&v2, sizeof(v2));
        run.set_arg(3, (const xrt::bo&)*bo_logits_);
        run.set_arg(4, (const xrt::bo&)*bo_lmhead_w_);
        run.set_arg(5, (const xrt::bo&)*bo_act_);
        run.set_arg(6, (const xrt::bo&)*bo_fnorm_);
        try {
            run.start();
            run.wait();
        } catch (const std::exception& e) {
            fprintf(stderr, "RuntimeLayer: lm_head run FAILED: %s\n", e.what());
            return false;
        }
    }
    ctx_len_ = ctx_len;
    return true;
}

bool RuntimeLayerEngine::get_logits(float* out, int vocab) {
    bo_logits_->sync(XCL_BO_SYNC_BO_FROM_DEVICE, 1048576, 0);
    const uint16_t* lg = (const uint16_t*)bo_logits_->map();
    for (int i = 0; i < vocab; i++) out[i] = bf16_to_f32(lg[i]);
    return true;
}

bool RuntimeLayerEngine::dump_act(const char* path, size_t n) {
    bo_act_->sync(XCL_BO_SYNC_BO_FROM_DEVICE, 1048576, 0);
    FILE* f = fopen(path, "wb");
    if (!f) return false;
    fwrite(bo_act_->map(), 1, n, f);
    fclose(f);
    return true;
}

bool RuntimeLayerEngine::dump_logits(const char* path, int vocab) {
    bo_logits_->sync(XCL_BO_SYNC_BO_FROM_DEVICE, 1048576, 0);
    FILE* f = fopen(path, "wb");
    if (!f) return false;
    fwrite(bo_logits_->map(), 2, vocab, f);
    fclose(f);
    return true;
}
