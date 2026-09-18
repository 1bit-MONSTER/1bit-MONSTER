#pragma once
// include/prism_engine.h — reusable gfx1151 forward for the Prism ML Bonsai 27B 1BP packs.
//
// Extracted from tests/prism/prism_forward_hip.hip so the HIP backend (plan P3.5) can serve
// the model with exactly the kernels the fork-oracle gate validates. Header-only; include it
// from a HIP translation unit and link the prism_* kernel objects (they are in librocm_cpp).
//
// Contract: init() loads the .1bp, parses __onebp_ext_prism_transform when present (folded
// packs rotate; unfolded packs are plain and apply no perm), and sizes per-layer state from
// max_pos. forward() runs all 64 layers at the engine's current position and returns the
// argmax token id (or -1 on error), incrementing the position. Fails closed: a folded pack
// whose manifest cannot be parsed is rejected in init().

#include "onebp_format.h"
#include "onebp_loader.h"
#include "prism_codec.h"

#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

extern "C" {
int prism_hadamard_fwht_f32(const void*, const int8_t*, void*, int, int, int, void*);
int prism_gemv_f32(const void*, const float*, float*, int, int, int, void*);
int prism_gemv_row4_f32(const void*, const float*, float*, int, int, int, void*);
int prism_gemv_tile_f32(const void*, const float*, float*, int, int, int, void*);
int prism_gemv_dense_f32(const float*, const float*, float*, int, int, void*);
int prism_conv1d_silu_f32(const float*, const float*, float*, float*, int, int, void*);
int prism_l2norm_heads_f32(float*, int, int, void*);
int prism_gbeta_f32(const float*, const float*, const float*, const float*, float*, float*, int, void*);
int prism_gdn_recurrence_f32(const float*, const float*, const float*, const float*, const float*,
                             const float*, const float*, float*, float*, int, int, int, int, void*);
int prism_ssmout_perm_f32(const float*, float*, int, int, int, void*);
int prism_qk_norm_rope_f32(float*, float*, const float*, const float*, int, int, int, int, float, int, int, void*);
int prism_gqa_decode_f32(const float*, const float*, const float*, const float*, float*, int, int, int, int, int, void*);
int prism_rmsnorm_f32(float*, const float*, int, void*);
int prism_add_f32(float*, const float*, int, void*);
int prism_silu_mul_f32(float*, const float*, int, void*);
int prism_split_qgate_f32(const float*, float*, float*, int, int, void*);
}

struct PrismGpuTensor { void* p = nullptr; int nb = 0; int rows = 0; int cols = 0; };

class PrismEngine {
public:
    int H = 5120, NL = 64, NH = 24, NKV = 4, HD = 256, NF = 17408, V = 248320;
    int NK = 16, NV = 48, HK = 128, HV = 128;
    int KD = 2048, VD = 6144, CD = 10240, ROPE = 64, block = 1024;
    float THETA = 1e7f;
    bool has_transform = false;
    int max_pos = 512;
    std::vector<int> linear;

    PrismEngine() = default;
    ~PrismEngine() { for (void* p : allocs_) if (p) hipFree(p); }

    // Loads the model + transform, sizes state/scratch. Returns false (fails closed) if the
    // file is not a Prism 1BP or its transform manifest cannot be parsed.
    bool init(const char* path, int maxpos = 512) {
        if (!m_.load(path)) return false;
        max_pos = maxpos > 0 ? maxpos : 512;
        const OnebpTensor* et = t("__onebp_ext_prism_transform");
        const OnebpTensor* es = t("__onebp_ext_prism_signs");
        has_transform = false;
        if (et && es) {
            if (!onebp_prism_transform_parse(m_.tensor_data(*et), (size_t)et->bytes,
                                             (const int8_t*)m_.tensor_data(*es), (size_t)es->bytes, tv_))
                return false;                       // folded pack we cannot honour -> fail closed
            has_transform = true; block = (int)tv_.hdr->block_size;
        }
        H = m_.header.hidden_size; NL = m_.header.num_layers;
        NH = m_.header.num_attention_heads; NKV = m_.header.num_kv_heads;
        HD = m_.header.head_dim; NF = m_.header.intermediate_size; V = m_.header.vocab_size;
        THETA = m_.header.rope_theta();
        const OnebpTensor* qn = t("blk.0.attn_q_norm.weight");
        if (qn) ROPE = (int)qn->dims[0];
        linear.assign(NL, 1);
        for (int l = 0; l < NL; l++) { char b[64]; std::snprintf(b, sizeof(b), "blk.%d.attn_qkv.weight", l); linear[l] = t(b) ? 1 : 0; }

        conv_state_.assign(NL, nullptr); rec_state_.assign(NL, nullptr);
        k_cache_.assign(NL, nullptr); v_cache_.assign(NL, nullptr);
        for (int l = 0; l < NL; l++) {
            if (linear[l]) {
                alloc(conv_state_[l], (size_t)CD * 3 * 4); HC(hipMemset(conv_state_[l], 0, (size_t)CD * 3 * 4));
                alloc(rec_state_[l], (size_t)NV * HK * HV * 4); HC(hipMemset(rec_state_[l], 0, (size_t)NV * HK * HV * 4));
            } else {
                const size_t kvb = (size_t)max_pos * NKV * HD * 4;
                alloc(k_cache_[l], kvb); HC(hipMemset(k_cache_[l], 0, kvb));
                alloc(v_cache_[l], kvb); HC(hipMemset(v_cache_[l], 0, kvb));
            }
        }
        alloc(d_h_, H * 4); alloc(d_xn_, H * 4); alloc(d_xr_, H * 4);
        alloc(d_qkv_, (size_t)CD * 4); alloc(d_z_, (size_t)VD * 4);
        alloc(d_a_, (size_t)NV * 4); alloc(d_b_, (size_t)NV * 4);
        alloc(d_conv_, (size_t)CD * 4); alloc(d_g_, (size_t)NV * 4); alloc(d_beta_, (size_t)NV * 4);
        alloc(d_core_, (size_t)VD * 4); alloc(d_perm_, (size_t)VD * 4); alloc(d_cr_, (size_t)VD * 4);
        alloc(d_out_, H * 4); alloc(d_yn_, H * 4); alloc(d_yr_, H * 4);
        alloc(d_gate_, (size_t)NF * 4); alloc(d_up_, (size_t)NF * 4); alloc(d_gr_, (size_t)NF * 4);
        alloc(d_down_, H * 4); alloc(d_q_, (size_t)NH * HD * 2 * 4); alloc(d_kb_, (size_t)NKV * HD * 4);
        alloc(d_vb_, (size_t)NKV * HD * 4); alloc(d_attn_, (size_t)NH * HD * 4);
        alloc(d_ar_, (size_t)NH * HD * 4); alloc(d_qs_, (size_t)NH * HD * 4); alloc(d_gt_, (size_t)NH * HD * 4);
        alloc(d_logits_, (size_t)V * 4);
        h_.assign(H, 0.0f); logits_.assign(V, 0.0f);
        sgH_ = sg(H); sgV_ = sg(VD); sgF_ = sg(NF);
        loaded_ = true;
        return true;
    }

    // Runs all layers for one token at the current position. Returns argmax, or -1.
    int forward(int token) {
        if (!loaded_) return -1;
        const OnebpTensor* emb = t("token_embd.weight");
        {
            const uint32_t nb = prism::block_bytes(emb->quant);
            const uint8_t* p = m_.tensor_data(*emb) + (size_t)token * (H / 128) * nb;
            if (!prism::dequant_flat(emb->quant, p, h_.data(), (size_t)H)) return -1;
        }
        HC(hipMemcpy(d_h_, h_.data(), H * 4, hipMemcpyHostToDevice));
        if (has_transform && prism_hadamard_fwht_f32(d_h_, sgH_, d_h_, H, block, 1, nullptr)) return -1;

        for (int l = 0; l < NL; l++) {
            HC(hipMemcpy(d_xn_, d_h_, H * 4, hipMemcpyDeviceToDevice));
            if (prism_rmsnorm_f32(d_xn_, vec(L(l, "blk.%d.attn_norm.weight")), H, nullptr)) return -1;
            rot(d_xn_, sgH_, d_xr_, H, 0);

            if (linear[l]) {
                matvec(L(l, "blk.%d.attn_qkv.weight"), d_xr_, d_qkv_);
                matvec(L(l, "blk.%d.attn_gate.weight"), d_xr_, d_z_);
                matvec(L(l, "blk.%d.ssm_alpha.weight"), d_xr_, d_a_);
                matvec(L(l, "blk.%d.ssm_beta.weight"), d_xr_, d_b_);
                if (prism_conv1d_silu_f32(d_qkv_, vec(L(l, "blk.%d.ssm_conv1d.weight")), conv_state_[l], d_conv_, CD, 3, nullptr)) return -1;
                if (prism_l2norm_heads_f32(d_conv_, NK, HK, nullptr)) return -1;
                if (prism_l2norm_heads_f32(d_conv_ + KD, NK, HK, nullptr)) return -1;
                if (prism_gbeta_f32(d_a_, vec(L(l, "blk.%d.ssm_dt.bias")), vec(L(l, "blk.%d.ssm_a")),
                                    d_b_, d_g_, d_beta_, NV, nullptr)) return -1;
                if (prism_gdn_recurrence_f32(d_conv_, d_conv_ + KD, d_conv_ + 2 * KD, d_g_, d_beta_,
                                             vec(L(l, "blk.%d.ssm_norm.weight")), d_z_, rec_state_[l], d_core_,
                                             NK, NV, HK, HV, nullptr)) return -1;
                const float* core_src = d_core_;
                if (has_transform) { if (prism_ssmout_perm_f32(d_core_, d_perm_, NV, NK, HV, nullptr)) return -1; core_src = d_perm_; }
                rot(core_src, sgV_, d_cr_, VD, 0);
                matvec(L(l, "blk.%d.ssm_out.weight"), d_cr_, d_out_);
                if (prism_add_f32(d_h_, d_out_, H, nullptr)) return -1;
            } else {
                matvec(L(l, "blk.%d.attn_q.weight"), d_xr_, d_q_);
                matvec(L(l, "blk.%d.attn_k.weight"), d_xr_, d_kb_);
                matvec(L(l, "blk.%d.attn_v.weight"), d_xr_, d_vb_);
                if (prism_split_qgate_f32(d_q_, d_qs_, d_gt_, NH, HD, nullptr)) return -1;
                if (prism_qk_norm_rope_f32(d_qs_, d_kb_, vec(L(l, "blk.%d.attn_q_norm.weight")),
                                           vec(L(l, "blk.%d.attn_k_norm.weight")),
                                           NH, NKV, HD, ROPE, THETA, pos_, 0, nullptr)) return -1;
                const size_t kvrow = (size_t)NKV * HD;
                HC(hipMemcpy(k_cache_[l] + (size_t)pos_ * kvrow, d_kb_, kvrow * 4, hipMemcpyDeviceToDevice));
                HC(hipMemcpy(v_cache_[l] + (size_t)pos_ * kvrow, d_vb_, kvrow * 4, hipMemcpyDeviceToDevice));
                if (prism_gqa_decode_f32(d_qs_, k_cache_[l], v_cache_[l], d_gt_, d_attn_,
                                         NH, NKV, HD, pos_ + 1, 0, nullptr)) return -1;
                rot(d_attn_, sgV_, d_ar_, NH * HD, 0);
                matvec(L(l, "blk.%d.attn_output.weight"), d_ar_, d_out_);
                if (prism_add_f32(d_h_, d_out_, H, nullptr)) return -1;
            }

            HC(hipMemcpy(d_yn_, d_h_, H * 4, hipMemcpyDeviceToDevice));
            if (prism_rmsnorm_f32(d_yn_, vec(L(l, "blk.%d.post_attention_norm.weight")), H, nullptr)) return -1;
            rot(d_yn_, sgH_, d_yr_, H, 0);
            matvec(L(l, "blk.%d.ffn_gate.weight"), d_yr_, d_gate_);
            matvec(L(l, "blk.%d.ffn_up.weight"), d_yr_, d_up_);
            if (prism_silu_mul_f32(d_gate_, d_up_, NF, nullptr)) return -1;
            rot(d_gate_, sgF_, d_gr_, NF, 0);
            matvec(L(l, "blk.%d.ffn_down.weight"), d_gr_, d_down_);
            if (prism_add_f32(d_h_, d_down_, H, nullptr)) return -1;
        }

        HC(hipMemcpy(d_yn_, d_h_, H * 4, hipMemcpyDeviceToDevice));
        if (prism_rmsnorm_f32(d_yn_, vec("output_norm.weight"), H, nullptr)) return -1;
        rot(d_yn_, sgH_, d_yr_, H, 0);
        matvec("output.weight", d_yr_, d_logits_);
        HC(hipMemcpy(logits_.data(), d_logits_, (size_t)V * 4, hipMemcpyDeviceToHost));
        int am = 0; float best = logits_[0];
        for (int i = 1; i < V; i++) if (logits_[i] > best) { best = logits_[i]; am = i; }
        pos_++;
        return am;
    }

    void reset() { pos_ = 0; for (int l = 0; l < NL; l++) {
        if (linear[l]) { HC(hipMemset(conv_state_[l], 0, (size_t)CD * 3 * 4)); HC(hipMemset(rec_state_[l], 0, (size_t)NV * HK * HV * 4)); }
        else { HC(hipMemset(k_cache_[l], 0, (size_t)max_pos * NKV * HD * 4)); HC(hipMemset(v_cache_[l], 0, (size_t)max_pos * NKV * HD * 4)); }
    } }
    const std::vector<float>& logits() const { return logits_; }
    int position() const { return pos_; }
    bool ok() const { return loaded_; }

    static std::string L(int l, const char* fmt) { char b[96]; std::snprintf(b, sizeof(b), fmt, l); return std::string(b); }

private:
    static void HC(hipError_t e) { if (e != hipSuccess) { std::fprintf(stderr, "PrismEngine HIP error: %s\n", hipGetErrorString(e)); } }
    template <class T> void alloc(T*& p, size_t bytes) { void* q = nullptr; if (hipMalloc(&q, bytes) != hipSuccess) { std::fprintf(stderr, "PrismEngine: hipMalloc(%zu) failed\n", bytes); std::abort(); } p = (T*)q; allocs_.push_back(q); }

    const OnebpTensor* t(const std::string& n) { for (const auto& x : m_.tensors) if (x.name == n) return &x; return nullptr; }
    int8_t* sg(int width) {
        if (!has_transform) return nullptr;
        auto it = signs_.find((unsigned)width);
        if (it != signs_.end()) return it->second;
        const int8_t* s = tv_.signs_for_width((uint32_t)width);
        if (!s) { std::fprintf(stderr, "PrismEngine: no signs for width %d\n", width); std::abort(); }
        int8_t* d = nullptr; HC(hipMalloc(&d, width)); HC(hipMemcpy(d, s, width, hipMemcpyHostToDevice));
        allocs_.push_back(d); signs_[(unsigned)width] = d; return d;
    }
    void rot(const float* in, int8_t* sgn, float* out, int width, int inverse) {
        if (!has_transform) { if (in != out) HC(hipMemcpy(out, in, (size_t)width * 4, hipMemcpyDeviceToDevice)); return; }
        if (prism_hadamard_fwht_f32(in, sgn, out, width, block, inverse, nullptr)) { std::fprintf(stderr, "PrismEngine: fwht %d failed\n", width); std::abort(); }
    }
    void up(const std::string& n) {
        if (g_.count(n)) return;
        const OnebpTensor* te = t(n);
        if (!te) { std::fprintf(stderr, "PrismEngine: missing %s\n", n.c_str()); std::abort(); }
        PrismGpuTensor gt;
        const int rows = (int)te->dims[0], cols = te->ndim > 1 ? (int)te->dims[1] : 1;
        const uint32_t nb = prism::block_bytes(te->quant);
        if (nb) {
            const size_t bytes = (size_t)rows * (cols / 128) * nb;
            alloc(gt.p, bytes); HC(hipMemcpy(gt.p, m_.tensor_data(*te), bytes, hipMemcpyHostToDevice));
            gt.nb = (int)nb; gt.rows = rows; gt.cols = cols;
        } else if (te->ndim > 1) {
            std::vector<float> out((size_t)rows * cols);
            const uint8_t* p = m_.tensor_data(*te);
            const int ntc = (cols + 255) / 256;
            for (int r = 0; r < rows; r++) for (int c = 0; c < cols; c++) {
                const size_t tile = (size_t)(r / 32) * ntc + (c / 256);
                const size_t off = tile * 32 * 256 * 2 + ((size_t)(r % 32) * 256 + (c % 256)) * 2;
                out[(size_t)r * cols + c] = prism::f16_to_f32((uint16_t)(p[off] | (p[off + 1] << 8)));
            }
            alloc(gt.p, out.size() * 4); HC(hipMemcpy(gt.p, out.data(), out.size() * 4, hipMemcpyHostToDevice));
            gt.nb = 0; gt.rows = rows; gt.cols = cols;
        } else {
            alloc(gt.p, (size_t)rows * 4); HC(hipMemcpy(gt.p, m_.tensor_data(*te), (size_t)rows * 4, hipMemcpyHostToDevice));
            gt.nb = 0; gt.rows = rows; gt.cols = 1;
        }
        g_[n] = gt;
    }
    float* vec(const std::string& n) { up(n); return (float*)g_[n].p; }
    void matvec(const std::string& n, const float* x, float* y) {
        up(n);
        const PrismGpuTensor& G = g_[n];
        if (G.nb) {
            const int rc = (G.nb == 18 || G.nb == 34 || G.nb == 28)
                         ? prism_gemv_tile_f32(G.p, x, y, G.rows, G.cols, G.nb, nullptr)
                         : prism_gemv_row4_f32(G.p, x, y, G.rows, G.cols, G.nb, nullptr);
            if (rc) { std::fprintf(stderr, "PrismEngine: gemv %s failed\n", n.c_str()); std::abort(); }
        } else if (prism_gemv_dense_f32((const float*)G.p, x, y, G.rows, G.cols, nullptr)) {
            std::fprintf(stderr, "PrismEngine: dense %s failed\n", n.c_str()); std::abort();
        }
    }

    OnebpModel m_;
    OnebpPrismTransformView tv_{};
    std::unordered_map<std::string, PrismGpuTensor> g_;
    std::unordered_map<unsigned, int8_t*> signs_;
    std::vector<void*> allocs_;
    std::vector<float*> conv_state_, rec_state_, k_cache_, v_cache_;
    float *d_h_ = nullptr, *d_xn_ = nullptr, *d_xr_ = nullptr, *d_qkv_ = nullptr, *d_z_ = nullptr,
          *d_a_ = nullptr, *d_b_ = nullptr, *d_conv_ = nullptr, *d_g_ = nullptr, *d_beta_ = nullptr,
          *d_core_ = nullptr, *d_perm_ = nullptr, *d_cr_ = nullptr, *d_out_ = nullptr, *d_yn_ = nullptr,
          *d_yr_ = nullptr, *d_gate_ = nullptr, *d_up_ = nullptr, *d_gr_ = nullptr, *d_down_ = nullptr,
          *d_q_ = nullptr, *d_kb_ = nullptr, *d_vb_ = nullptr, *d_attn_ = nullptr, *d_ar_ = nullptr,
          *d_qs_ = nullptr, *d_gt_ = nullptr, *d_logits_ = nullptr;
    int8_t *sgH_ = nullptr, *sgV_ = nullptr, *sgF_ = nullptr;
    std::vector<float> h_, logits_;
    int pos_ = 0;
    bool loaded_ = false;
};
