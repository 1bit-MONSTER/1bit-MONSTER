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
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/experimental/xrt_ext.h>
#include <xrt/experimental/xrt_xclbin.h>
#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/experimental/xrt_kernel.h>   // xrt::runlist (#2150 single-launch)
#include <xrt/experimental/xrt_kernel.h>

extern "C" int npu_pack_layer_bo(uint8_t* bo_buffer, void* mw, const void* config, int layer_idx);
extern "C" int npu_pack_lmhead_bo(uint8_t* bo_buffer, void* mw, const void* config);
extern "C" int npu_layer_bo_bytes(void* mw, const void* config);

// ---- model-file layout (model-generic) ----
// Metadata data_offsets are relative to data_base (8-byte len + JSON header);
// absolute file offset = data_base + data_offset. The per-layer norm tensors
// (input/post_attention_layernorm, q/k_norm) carry their own data_offsets in
// the metadata (verified equal to the old 0.6B hardcodes), so norms are read
// straight from the metadata — no per-model hardcoded offsets / pipeline order.

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
    // ---- per-layer kv BOs — the runtime's design ----
    // Size = MAX_L * NKV * HD * 4 bytes (k+v, bf16). The ELF MAX_L (from
    // gen_layer_elfs) must match: 128MB = 32768 tokens at NKV=8/HD=128.
    // Hardcoding 32MB here broke 32k contexts (attention walked past the BO
    // -> NaN); use cfg_.npu_kv_cache_bo_size (128MB) with a 32MB floor.
    size_t kv_bo_bytes = cfg_.npu_kv_cache_bo_size > 0
        ? (size_t)cfg_.npu_kv_cache_bo_size : 33554432;
    kv_bos_.resize(cfg_.num_layers);
    for (int L = 0; L < cfg_.num_layers; L++) {
        kv_bos_[L] = std::make_unique<xrt::ext::bo>(dev, kv_bo_bytes);
        memset(kv_bos_[L]->map(), 0, kv_bo_bytes);
        kv_bos_[L]->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }
    // ---- act / logits / final-norm BOs ----
    bo_act_ = std::make_unique<xrt::ext::bo>(dev, 1048576);
    bo_logits_ = std::make_unique<xrt::ext::bo>(dev, 1048576);
    memset(bo_act_->map(), 0, 1048576);
    memset(bo_logits_->map(), 0, 1048576);
    bo_fnorm_ = std::make_unique<xrt::ext::bo>(dev, 1048576);
    memset(bo_fnorm_->map(), 0, 1048576);
    if (mw_ && mw_->norm_weight.ndim == 1) {
        memcpy(bo_fnorm_->map(), model_tensor_data(mw_, &mw_->norm_weight),
               (size_t)cfg_.hidden_size * 2);
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

    // ---- per-layer weight BOs: npu_pack_layer_bo (byte-verified, model-generic) ----
    weight_bos_.resize(cfg_.num_layers);
    int layer_bo_bytes = npu_layer_bo_bytes(mw, &cfg_);
    if (layer_bo_bytes <= 0) {
        fprintf(stderr, "RuntimeLayer: cannot compute per-layer weight BO size\n");
        return false;
    }
    for (int L = 0; L < cfg_.num_layers; L++) {
        weight_bos_[L] = std::make_unique<xrt::ext::bo>(dev, layer_bo_bytes);
        uint8_t* m = static_cast<uint8_t*>(weight_bos_[L]->map());
        memset(m, 0, layer_bo_bytes);
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
    if (npu_pack_lmhead_bo && mw_ && mw_->lm_head_weight.ndim == 2) {
        int tiles = (int)mw_->lm_head_weight.shape[0];
        size_t bo_bytes = (size_t)tiles * 5120;
        bo_lmhead_w_ = std::make_unique<xrt::ext::bo>(*dev_, bo_bytes);
        uint8_t* m = static_cast<uint8_t*>(bo_lmhead_w_->map());
        memset(m, 0, bo_bytes);
        if (npu_pack_lmhead_bo(m, mw_, &cfg_)) {
            bo_lmhead_w_->sync(XCL_BO_SYNC_BO_TO_DEVICE);
            fprintf(stderr, "RuntimeLayer: packed lm_head BO (%d tiles)\n", tiles);
            return true;
        }
    }
    fprintf(stderr, "RuntimeLayer: npu_pack_lmhead_bo unavailable/failed\n");
    return false;
}

bool RuntimeLayerEngine::build_norm_bos() {
    const size_t H2 = (size_t)cfg_.hidden_size * 2;   // one norm tensor in bytes (bf16)
    i5_bos_.resize(cfg_.num_layers);
    i6_bos_.resize(cfg_.num_layers);
    for (int L = 0; L < cfg_.num_layers; L++) {
        LayerWeights* lw = &mw_->layers[L];
        // i5 = ILN(H) + PALN(H)  (bf16, from metadata data_offsets)
        i5_bos_[L] = std::make_unique<xrt::ext::bo>(*dev_, 1048576);
        uint8_t* m5 = static_cast<uint8_t*>(i5_bos_[L]->map());
        memset(m5, 0, 1048576);
        memcpy(m5, model_tensor_data(mw_, &lw->input_layernorm_weight), H2);
        memcpy(m5 + H2, model_tensor_data(mw_, &lw->post_attention_layernorm_weight), H2);
        i5_bos_[L]->sync(XCL_BO_SYNC_BO_TO_DEVICE);
        // i6 = [1.0 x64][0 x64][q_norm 128][k_norm 128]  (kn/qn are HD=128 bf16)
        i6_bos_[L] = std::make_unique<xrt::ext::bo>(*dev_, 1048576);
        uint8_t* m6 = static_cast<uint8_t*>(i6_bos_[L]->map());
        memset(m6, 0, 1048576);
        uint16_t* w6 = (uint16_t*)m6;
        for (int i = 0; i < 64; i++) w6[i] = f32_to_bf16(1.0f);
        memcpy(m6 + 256, model_tensor_data(mw_, &lw->q_norm_weight), 256);
        memcpy(m6 + 512, model_tensor_data(mw_, &lw->k_norm_weight), 256);
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
    // Only the first 128 bf16 (cos/sin for the current position) changed — sync
    // just those 256 bytes, not the full 1MB i6 BO (28 layers x 1MB was ~10% of
    // decode).
    i6bo.sync(XCL_BO_SYNC_BO_TO_DEVICE, 256, 0);
}

bool RuntimeLayerEngine::prefill_batch(const int* tokens, int n) {
    if (n <= 0) return false;
    TensorDesc* emb = &mw_->embed_tokens;
    if (emb->ndim != 2) return false;
    uint8_t* m = static_cast<uint8_t*>(bo_act_->map());
    for (int t = 0; t < n; t++) {
        uint64_t off = (uint64_t)tokens[t] * emb->shape[1] * 2;
        memcpy(m + (size_t)t * emb->shape[1] * 2,
               (const uint8_t*)mw_->file_data + mw_->data_base + off,
               emb->shape[1] * 2);
    }
    bo_act_->sync(XCL_BO_SYNC_BO_TO_DEVICE);
    return forward(n);
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
    // #2150 single-launch: batch every layer run + lm_head into ONE xrt::runlist
    // submit/token. Only when no per-layer debug dump is active — those sync
    // from device mid-stream and cannot coexist with an atomic runlist.
    if (!dbg && !getenv("RT_CLEAN_DUMP") && !getenv("RT_DUMP_KV")) {
        auto t_build0 = std::chrono::steady_clock::now();
        std::vector<xrt::run> runs;
        runs.reserve((size_t)cfg_.num_layers + 1);
        xrt::runlist rl(*hwctx_);
        for (int L = 0; L < cfg_.num_layers; L++) {
            runs.emplace_back(*layer_kernels_[ctx_len]);
            xrt::run& run = runs.back();
            uint32_t v0 = 3, v1 = 0, v2 = 0;
            run.set_arg(0, (const void*)&v0, sizeof(v0));
            run.set_arg(1, (const void*)&v1, sizeof(v1));
            run.set_arg(2, (const void*)&v2, sizeof(v2));
            run.set_arg(3, (const xrt::bo&)*bo_act_);
            run.set_arg(4, (const xrt::bo&)*weight_bos_[L]);
            run.set_arg(5, (const xrt::bo&)*i5_bos_[L]);
            run.set_arg(6, (const xrt::bo&)*i6_bos_[L]);
            run.set_arg(7, (const xrt::bo&)*kv_bos_[L]);
            rl.add(run);
        }
        if (kern_lmhead_) {
            runs.emplace_back(*kern_lmhead_);
            xrt::run& run = runs.back();
            uint32_t v0 = 3, v1 = 0, v2 = 0;
            run.set_arg(0, (const void*)&v0, sizeof(v0));
            run.set_arg(1, (const void*)&v1, sizeof(v1));
            run.set_arg(2, (const void*)&v2, sizeof(v2));
            run.set_arg(3, (const xrt::bo&)*bo_logits_);
            run.set_arg(4, (const xrt::bo&)*bo_lmhead_w_);
            run.set_arg(5, (const xrt::bo&)*bo_act_);
            run.set_arg(6, (const xrt::bo&)*bo_fnorm_);
            rl.add(run);
        }
        try {
            auto t_exec0 = std::chrono::steady_clock::now();
            rl.execute();
            rl.wait();
            if (getenv("NPU_RUNLIST_STATS")) {
                auto t_done = std::chrono::steady_clock::now();
                double bms = std::chrono::duration<double, std::milli>(t_exec0 - t_build0).count();
                double ems = std::chrono::duration<double, std::milli>(t_done - t_exec0).count();
                fprintf(stderr, "[runlist] build=%.2fms exec=%.2fms\n", bms, ems);
            }
        } catch (const std::exception& e) {
            fprintf(stderr, "RuntimeLayer: runlist FAILED: %s\n", e.what());
            return false;
        }
        if (getenv("NPU_RUNLIST_STATS"))
            fprintf(stderr, "[runlist] %zu runs batched -> 1 submit (ctx=%d)\n",
                    runs.size(), ctx_len);
        ctx_len_ = ctx_len;
        return true;
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
    if (!run_lmhead()) return false;
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

bool RuntimeLayerEngine::run_lmhead() {
    if (!kern_lmhead_) return false;
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
        LOG_ERROR("lm_head run FAILED: %s", e.what());
        return false;
    }
    return true;
}

const void* RuntimeLayerEngine::map_kv(int layer) const {
    if (layer < 0 || layer >= (int)kv_bos_.size()) return nullptr;
    kv_bos_[layer]->sync(XCL_BO_SYNC_BO_FROM_DEVICE, 33554432, 0);
    return kv_bos_[layer]->map();
}

bool RuntimeLayerEngine::write_kv(int layer, int token_begin, int n_tokens,
                                  const uint16_t* bf16_kv, int region_stride_u16) {
    if (layer < 0 || layer >= (int)kv_bos_.size() || !bf16_kv) return false;
    if (token_begin < 0 || n_tokens <= 0) return false;
    // Per-token KV footprint within a region: NKV/2 heads x HD dims (bf16).
    const int token_u16 = (cfg_.num_key_value_heads / 2) * cfg_.head_dim;
    if (region_stride_u16 <= 0) region_stride_u16 = token_u16 * 8192;  // MAX_L=8192 -> 8MB
    if ((size_t)(token_begin + n_tokens) * token_u16 > (size_t)region_stride_u16) {
        LOG_ERROR("write_kv: token range %d..%d exceeds region capacity %d",
                token_begin, token_begin + n_tokens - 1, region_stride_u16 / token_u16);
        return false;
    }
    uint16_t* dst = (uint16_t*)kv_bos_[layer]->map();
    for (int region = 0; region < 4; region++) {
        const uint16_t* src = bf16_kv + (size_t)region * region_stride_u16
                                       + (size_t)token_begin * token_u16;
        uint16_t* d = dst + (size_t)region * region_stride_u16
                         + (size_t)token_begin * token_u16;
        memcpy(d, src, (size_t)n_tokens * token_u16 * sizeof(uint16_t));
    }
    kv_bos_[layer]->sync(XCL_BO_SYNC_BO_TO_DEVICE, 33554432, 0);
    return true;
}

bool RuntimeLayerEngine::write_act(const uint16_t* bf16_hidden) {
    if (!bo_act_ || !bf16_hidden) return false;
    uint16_t* dst = (uint16_t*)bo_act_->map();
    memcpy(dst, bf16_hidden, (size_t)cfg_.hidden_size * sizeof(uint16_t));
    bo_act_->sync(XCL_BO_SYNC_BO_TO_DEVICE, 1048576, 0);
    return true;
}

bool RuntimeLayerEngine::get_logits(float* out, int vocab) {
    bo_logits_->sync(XCL_BO_SYNC_BO_FROM_DEVICE, (size_t)vocab * 2, 0);
    const uint16_t* lg = (const uint16_t*)bo_logits_->map();
    for (int i = 0; i < vocab; i++) out[i] = bf16_to_f32(lg[i]);
    return true;
}

int RuntimeLayerEngine::argmax_logits(int vocab) {
    bo_logits_->sync(XCL_BO_SYNC_BO_FROM_DEVICE, (size_t)vocab * 2, 0);
    const uint16_t* lg = (const uint16_t*)bo_logits_->map();
    // bf16 argmax without float conversion: positives (u < 0x8000) beat
    // negatives; among same sign, larger u wins for positive, smaller u
    // (closer to 0) wins for negative.
    //
    // Monotonic key for sign-magnitude bf16: positives must beat negatives, and
    // among negatives the one closest to zero wins. Flipping the sign bit for
    // positives and all bits for negatives gives exactly that ordering, so the
    // argmax becomes a plain unsigned max over 151936 keys (vectorisable).
    int best = 0;
    {
        int nthreads = 1;
#ifdef _OPENMP
        nthreads = omp_get_max_threads();
#endif
        std::vector<uint32_t> lbest((size_t)nthreads, 0);
        std::vector<int> lidx((size_t)nthreads, 0);
#pragma omp parallel num_threads(nthreads)
        {
            int tid = 0;
#ifdef _OPENMP
            tid = omp_get_thread_num();
#endif
            uint32_t bkey = 0;
            int bidx = 0;
#pragma omp for schedule(static) nowait
            for (int i = 0; i < vocab; i++) {
                uint16_t u = lg[i];
                uint32_t key = (uint32_t)(u ^ ((u & 0x8000u) ? 0xFFFFu : 0x8000u));
                if (i == 0 || key > bkey) { bkey = key; bidx = i; }
            }
            lbest[(size_t)tid] = bkey;
            lidx[(size_t)tid] = bidx;
        }
        uint32_t bkey = lbest[0];
        best = lidx[0];
        for (int t = 1; t < nthreads; t++)
            if (lbest[(size_t)t] > bkey) { bkey = lbest[(size_t)t]; best = lidx[(size_t)t]; }
    }
    return best;
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
