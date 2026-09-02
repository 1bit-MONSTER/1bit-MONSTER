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
// layer -> position in the physical norm array (inverse of PIPELINE_ORDER)
static const int NORM_POS_OF_LAYER[28] = {
    0, 1, 12, 21, 22, 23, 24, 25, 26, 27,
    2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
    13, 14, 15, 16, 17, 18, 19, 20
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
    // ---- per-layer kv BOs (32MB each, zero) — the runtime's design ----
    kv_bos_.resize(cfg_.num_layers);
    for (int L = 0; L < cfg_.num_layers; L++) {
        kv_bos_[L] = std::make_unique<xrt::ext::bo>(dev, 33554432);
        memset(kv_bos_[L]->map(), 0, 33554432);
        kv_bos_[L]->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
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


    // lm_head kernel (context-independent ELF)
    if (getenv("RT_NO_LMHEAD")) { /* disabled for isolation */ }
    else if (!lmhead_elf_path_.empty()) {
        std::vector<uint8_t> elfb;
        if (read_file(lmhead_elf_path_.c_str(), elfb)) {
            try {
                xrt::elf elf((const char*)elfb.data(), elfb.size());
                xrt::module mod(elf);
                kern_lmhead_ = std::make_unique<xrt::ext::kernel>(*hwctx_, mod, "MLIR_AIE");
                fprintf(stderr, "RuntimeLayer: lm_head kernel ready (%s)\n",
                        lmhead_elf_path_.c_str());
            } catch (const std::exception& e) {
                fprintf(stderr, "RuntimeLayer: lm_head kernel build failed: %s\n", e.what());
            }
        } else {
            fprintf(stderr, "RuntimeLayer: cannot read lm_head ELF %s\n",
                    lmhead_elf_path_.c_str());
        }
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
    if (!pack_lmhead_bo()) {
        fprintf(stderr, "RuntimeLayer: lm_head weight BO pack failed\n");
        return false;
    }

    // ---- norms i5/i6 per layer (physical pipeline blocks) ----
    if (!build_norm_bos()) return false;



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
        int pos = NORM_POS_OF_LAYER[L];
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
        for (int i = 0; i < 64; i++) w6[i] = f32_to_bf16(1.0f);
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
        // Lazy on-demand ELF build (Round 38): if the per-ctx ELF is missing,
        // shell out to tools/gen_layer_elfs (0.6ms/ELF — full MAX_L 4096 is
        // ~2.5s). Configure via RT_ELF_GEN=<gen_layer_elfs path> and
        // RT_ELF_MODEL=<model dir>; otherwise this stays a hard error.
        const char* gen = getenv("RT_ELF_GEN");
        const char* mdir = getenv("RT_ELF_MODEL");
        if (gen && mdir && gen[0] && mdir[0]) {
            char cmd[1024];
            snprintf(cmd, sizeof(cmd), "%s %s %s %d %d", gen, mdir, elf_dir_.c_str(),
                     ctx_len, ctx_len);
            fprintf(stderr, "RuntimeLayer: generating missing ELF ctx=%d (%s)\n",
                    ctx_len, cmd);
            int rc = system(cmd);
            if (rc == 0) read_file(fname, elfb);
        }
        if (elfb.empty()) {
            fprintf(stderr, "RuntimeLayer: missing layer ELF for ctx=%d (%s)\n",
                    ctx_len, fname);
            return false;
        }
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

// RoPE cos/sin table for the current token position (pos = ctx_len-1).
// The runtime host-writes this into i6[0:128] before EVERY forward; the layer
// kernel reads it (it does NOT compute RoPE internally).
//
// EXACT runtime formula (reverse-engineered from libqwen3_npu.so, Round 38):
// the runtime keeps a HARDCODED float32 inv_freq[64] table in .rodata
// (NOT 1e6^(-2j/128) in double — the f32 literals are off by up to ~1.5e-5
// relative, which previously caused 1-ULP i6 flips at positions >= 3), then
// computes  phi = inv_freq[j] * (float)pos   (float32 vmulss)
// and calls glibc sincosf(phi), converting the f32 results to bf16 (RNE).
// Reproduced byte-for-byte across all captured positions 0..40 (5248 entries).
// The table below is the exact .rodata dump at 0x152740.
static const float RT_INV_FREQ[64] = {
    1.000000000e+00f, 8.058400154e-01f, 6.493800282e-01f, 5.232999921e-01f,
    4.217000008e-01f, 3.398199975e-01f, 2.738400102e-01f, 2.206699997e-01f,
    1.778299958e-01f, 1.432999969e-01f, 1.154799983e-01f, 9.305699915e-02f,
    7.498899847e-02f, 6.043000147e-02f, 4.869699851e-02f, 3.924199939e-02f,
    3.162299842e-02f, 2.548299916e-02f, 2.053499967e-02f, 1.654800028e-02f,
    1.333499979e-02f, 1.074600033e-02f, 8.659600280e-03f, 6.978299934e-03f,
    5.623400211e-03f, 4.531600047e-03f, 3.651699983e-03f, 2.942699939e-03f,
    2.371399896e-03f, 1.910999999e-03f, 1.539899968e-03f, 1.240900019e-03f,
    1.000000047e-03f, 8.058400126e-04f, 6.493799738e-04f, 5.233000265e-04f,
    4.217000096e-04f, 3.398199915e-04f, 2.738400071e-04f, 2.206700010e-04f,
    1.778300066e-04f, 1.432999998e-04f, 1.154799975e-04f, 9.305700223e-05f,
    7.498900231e-05f, 6.043000030e-05f, 4.869699842e-05f, 3.924200064e-05f,
    3.162299981e-05f, 2.548299926e-05f, 2.053500066e-05f, 1.654799962e-05f,
    1.333500040e-05f, 1.074600004e-05f, 8.659600098e-06f, 6.978300007e-06f,
    5.623399829e-06f, 4.531600098e-06f, 3.651699899e-06f, 2.942699894e-06f,
    2.371399887e-06f, 1.911000027e-06f, 1.539900040e-06f, 1.240900019e-06f,
};

// glibc float32 sincos (the runtime links sincosf@GLIBC_2.2.5)
extern "C" void sincosf(float x, float* s, float* c);

static void update_rope_i6(xrt::ext::bo& i6bo, int pos) {
    uint16_t* w = (uint16_t*)i6bo.map();
    float fpos = (float)pos;
    for (int j = 0; j < 64; j++) {
        float phi = RT_INV_FREQ[j] * fpos;   // float32 multiply (vmulss)
        float s, c;
        sincosf(phi, &s, &c);
        w[j] = f32_to_bf16(c);
        w[64 + j] = f32_to_bf16(s);
    }
    i6bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, 1048576, 0);
}

bool RuntimeLayerEngine::forward(int ctx_len) {
    if (!ensure_layer_kernel(ctx_len)) return false;
    // RoPE table for the current position (pos = ctx_len-1), every layer
    for (int L = 0; L < cfg_.num_layers; L++)
        update_rope_i6(*i6_bos_[L], ctx_len - 1);
    // per-ctx kv dump for the layout diff (RT_KV_DUMP_DIR)
    if (const char* kd = getenv("RT_KV_DUMP_DIR")) {
        char kf[512];
        snprintf(kf, sizeof(kf), "%s/kv_ctx%d.bin", kd, ctx_len);
        FILE* fk = fopen(kf, "wb");
        if (fk) {
            kv_bos_[0]->sync(XCL_BO_SYNC_BO_FROM_DEVICE, 33554432, 0);
            fwrite(kv_bos_[0]->map(), 1, 33554432, fk);  // full 32MB
            fclose(fk);
        }
    }
    if (getenv("RT_DUMP_I6")) {
        FILE* fi6 = fopen(getenv("RT_DUMP_I6"), "wb");
        if (fi6) { i6_bos_[0]->sync(XCL_BO_SYNC_BO_FROM_DEVICE, 1048576, 0); fwrite(i6_bos_[0]->map(), 1, 768, fi6); fclose(fi6); }
        FILE* fi27 = fopen("/tmp/eng_i6_L27.bin", "wb");
        if (fi27) { i6_bos_[27]->sync(XCL_BO_SYNC_BO_FROM_DEVICE, 1048576, 0); fwrite(i6_bos_[27]->map(), 1, 768, fi27); fclose(fi27); }
    }
    if (getenv("RT_DUMP_PREACT")) {
        FILE* fpa = fopen(getenv("RT_DUMP_PREACT"), "wb");
        if (fpa) { bo_act_->sync(XCL_BO_SYNC_BO_FROM_DEVICE, 1048576, 0); fwrite(bo_act_->map(), 1, 2048, fpa); fclose(fpa); }
    }
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
        snprintf(fn, sizeof(fn), "%s_w1.bin", dbg);
        f = fopen(fn, "wb");
        if (f) {
            weight_bos_[1]->sync(XCL_BO_SYNC_BO_FROM_DEVICE, 10485760, 0);
            fwrite(weight_bos_[1]->map(), 1, 9830400, f);
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
        snprintf(fn, sizeof(fn), "%s_i5_1.bin", dbg);
        f = fopen(fn, "wb");
        if (f) {
            i5_bos_[1]->sync(XCL_BO_SYNC_BO_FROM_DEVICE, 1048576, 0);
            fwrite(i5_bos_[1]->map(), 1, 4096, f);
            fclose(f);
        }
        snprintf(fn, sizeof(fn), "%s_i6_1.bin", dbg);
        f = fopen(fn, "wb");
        if (f) {
            i6_bos_[1]->sync(XCL_BO_SYNC_BO_FROM_DEVICE, 1048576, 0);
            fwrite(i6_bos_[1]->map(), 1, 768, f);
            fclose(f);
        }
        snprintf(fn, sizeof(fn), "%s_i6_2.bin", dbg);
        f = fopen(fn, "wb");
        if (f) {
            i6_bos_[2]->sync(XCL_BO_SYNC_BO_FROM_DEVICE, 1048576, 0);
            fwrite(i6_bos_[2]->map(), 1, 768, f);
            fclose(f);
        }
    }
    if (const char* cd = getenv("RT_CLEAN_DUMP")) {
        char cf[512];
        snprintf(cf, sizeof(cf), "%s_pre_ctx%d.bin", cd, ctx_len);
        FILE* fcp = fopen(cf, "wb");
        if (fcp) { bo_act_->sync(XCL_BO_SYNC_BO_FROM_DEVICE, 1048576, 0); fwrite(bo_act_->map(), 1, 2048, fcp); fclose(fcp); }
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
        run.set_arg(7, (const xrt::bo&)*kv_bos_[L]);
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
    // clean per-forward dump (RT_CLEAN_DUMP): act + layer-0 kv after the layers
    if (const char* cd = getenv("RT_CLEAN_DUMP")) {
        char cf[512];
        snprintf(cf, sizeof(cf), "%s_act_ctx%d.bin", cd, ctx_len);
        FILE* fca = fopen(cf, "wb");
        if (fca) { bo_act_->sync(XCL_BO_SYNC_BO_FROM_DEVICE, 1048576, 0); fwrite(bo_act_->map(), 1, 2048, fca); fclose(fca); }
        snprintf(cf, sizeof(cf), "%s_kv_ctx%d.bin", cd, ctx_len);
        FILE* fck = fopen(cf, "wb");
        if (fck) { kv_bos_[0]->sync(XCL_BO_SYNC_BO_FROM_DEVICE, 33554432, 0); fwrite(kv_bos_[0]->map(), 1, 33554432, fck); fclose(fck); }
    }
    // lm_head
    if (kern_lmhead_) {
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
    if (getenv("RT_DUMP_KV")) {
        for (int kk = 0; kk < cfg_.num_layers; kk++) {
            char kfn[64]; snprintf(kfn, sizeof(kfn), "/tmp/engine_kv_L%02d.bin", kk);
            kv_bos_[kk]->sync(XCL_BO_SYNC_BO_FROM_DEVICE, 33554432, 0);
            FILE* fkk=fopen(kfn,"wb");
            if(fkk){ fwrite(kv_bos_[kk]->map(),1,33554432,fkk); fclose(fkk); }
        }
        {
            FILE* fk2=fopen("/tmp/engine_i6_post_fwd1.bin","wb");
            if(fk2){ i6_bos_[0]->sync(XCL_BO_SYNC_BO_FROM_DEVICE, 1048576, 0); fwrite(i6_bos_[0]->map(),1,768,fk2); fclose(fk2); }
        }
        kv_bos_[cfg_.num_layers-1]->sync(XCL_BO_SYNC_BO_FROM_DEVICE, 33554432, 0);
        FILE* fk = fopen(getenv("RT_DUMP_KV"), "wb");
        if (fk) { fwrite(kv_bos_[0]->map(), 1, 134217728, fk); fclose(fk); }
        fprintf(stderr, "kv dumped -> %s\n", getenv("RT_DUMP_KV"));
    }
    return true;
}

const void* RuntimeLayerEngine::map_kv(int layer) const {
    if (layer < 0 || layer >= (int)kv_bos_.size()) return nullptr;
    kv_bos_[layer]->sync(XCL_BO_SYNC_BO_FROM_DEVICE, 33554432, 0);
    return kv_bos_[layer]->map();
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
