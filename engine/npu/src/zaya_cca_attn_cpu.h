// zaya_cca_attn_cpu.h — Zaya Cross-Channel Attention, CPU reference port.
//
// Ported 1:1 from the GPU reference kernels:
//   kernels/zaya_cca_prep.hip   (cca_prep_kernel  — conv_qk + qk_means + L2 + RoPE)
//   kernels/zaya_fused_qkv.hip  (RMSNorm + Q/K/V1/V2 projections)
//   src/zaya_engine.cpp         (zaya_forward — per-layer flow, o_proj, residual)
//
// This is the CPU-side attention half of the "NPU FFN ∥ CPU/GPU attention"
// hybrid (research/ws01-npu-attention). It computes the full CCA attention
// block for one token given the dequantized per-layer weights. The Q/K/V and
// o_proj are plain GEMVs; the CCA-specific part is the channel-wise conv_qk +
// qk_means + L2 + partial-RoPE prep before the standard GQA sequence attention.
//
// Zaya1-8B dims (from q4nx manifest top-level + ZayaConfig::zaya1_8b):
//   H=2048  nq=8  nkv=2  hd=128  qd=nq*hd=1024  kd=nkv*hd=256  qkv=qd+kd=1280
//   gc = qkv/(nq+nkv) = 128   nrot = hd/2 = 64   rope_base = 5000000.0
//
// Reference call (zaya_forward, src/zaya_engine.cpp):
//   cca_prep_kernel(..., nq, nkv, hd, hd/2, 5000000.0f, qkv/(nq+nkv));
//   rcpp_kv_cache_attn_decode_fd_prealloc(..., nq, nkv, hd, pos+1, 1/sqrt(hd), ...);
#pragma once

#include <cmath>
#include <cstring>
#include <vector>

namespace zaya_cca {

struct CcaDims {
    int H;      // hidden size (2048)
    int nq;     // q heads (8)
    int nkv;    // kv heads (2)
    int hd;     // head dim (128)
    int qd;     // nq*hd (1024)
    int kd;     // nkv*hd (256)
    int qkv;    // qd+kd (1280)
    int gc;     // group width = qkv/(nq+nkv) (128)
    int nrot;   // partial rotary dims = hd/2 (64)
    float rope_base;  // 5000000.0

    static CcaDims zaya1_8b() {
        CcaDims d;
        d.H = 2048; d.nq = 8; d.nkv = 2; d.hd = 128;
        d.qd = d.nq * d.hd; d.kd = d.nkv * d.hd; d.qkv = d.qd + d.kd;
        d.gc = d.qkv / (d.nq + d.nkv);
        d.nrot = d.hd / 2;
        d.rope_base = 5000000.0f;
        return d;
    }
};

// Per-attention-layer dequantized weights (float). Layouts match the GPU ref:
//   wq[qd*H] wk[kd*H] wv1[(kd/2)*H] wv2[(kd/2)*H] wo[H*qd]
//   cdw[qkv*2] cdb[qkv]  (conv_qk depthwise, 2-tap)
//   cgw[qkv*(gc*2)] cgb[qkv]  (conv_qk grouped)
//   ks[nkv]  (qk_norm temperature, per kv head)
struct CcaWeights {
    std::vector<float> wq, wk, wv1, wv2, wo;
    std::vector<float> cdw, cdb, cgw, cgb, ks;
};

// Persistent per-layer recurrent state (owned by the engine; zeros on reset).
struct CcaState {
    std::vector<float> conv_state;  // [2*qkv]
    std::vector<float> vrec;        // [kd/2] prev token's v_proj_delayed
    void reset(int qkv, int kd2) {
        conv_state.assign((size_t)qkv * 2, 0.0f);
        vrec.assign((size_t)kd2, 0.0f);
    }
};

// CCA prep: q/k/v projections already computed -> q_out/k_out/v_out ready for
// the standard GQA sequence attention + KV-cache store.
//
//   q   [qd]      = wq  @ h_norm
//   k   [kd]      = wk  @ h_norm
//   v_cur [kd/2]  = wv1 @ h_norm
//   v_del [kd/2]  = wv2 @ prev_hs
//
// Ported exactly from cca_prep_kernel (kernels/zaya_cca_prep.hip).
inline void cca_prep(const CcaDims& d, const CcaWeights& w, CcaState& st,
                     const float* q, const float* k,
                     const float* v_cur, const float* v_del,
                     float* q_out, float* k_out, float* v_out,
                     int pos) {
    const int qd = d.qd, kd = d.kd, qkv = d.qkv, nq = d.nq, nkv = d.nkv;
    const int hd = d.hd, gc = d.gc, nrot = d.nrot;
    const int gqa = nq / nkv;

    std::vector<float> sqk(qkv);
    for (int i = 0; i < qkv; i++)
        sqk[i] = (i < qd) ? q[i] : k[i - qd];

    std::vector<float> dw0(qkv), dw1(qkv);
    // Depthwise conv (2-tap) over channels, then state update.
    for (int c = 0; c < qkv; c++) {
        float s0 = st.conv_state[c], s1 = st.conv_state[qkv + c], cur = sqk[c];
        dw0[c] = w.cdw[c * 2 + 0] * s0 + w.cdw[c * 2 + 1] * s1 + w.cdb[c];
        dw1[c] = w.cdw[c * 2 + 0] * s1 + w.cdw[c * 2 + 1] * cur + w.cdb[c];
    }
    for (int c = 0; c < qkv; c++) {
        float old_s1 = st.conv_state[qkv + c];
        st.conv_state[c] = old_s1;
        st.conv_state[qkv + c] = sqk[c];
    }

    // Grouped conv over channels.
    for (int oc = 0; oc < qkv; oc++) {
        int g = oc / gc, base = g * gc;
        const float* cw = &w.cgw[(size_t)oc * (gc * 2)];
        float a = 0.0f;
        for (int j = 0; j < gc; j++)
            a += cw[j * 2 + 0] * dw0[base + j] + cw[j * 2 + 1] * dw1[base + j];
        sqk[oc] = a + w.cgb[oc];
    }

    // qk_means: mix the raw q/k back in (per-head for q, group-averaged for k).
    for (int h = 0; h < nq; h++) {
        int kv = h / gqa;
        for (int dd = 0; dd < hd; dd++)
            sqk[h * hd + dd] += 0.5f * q[h * hd + dd] + 0.5f * k[kv * hd + dd];
    }
    for (int khv = 0; khv < nkv; khv++) {
        for (int dd = 0; dd < hd; dd++) {
            float sm = 0.0f;
            for (int g = 0; g < gqa; g++) sm += q[(khv * gqa + g) * hd + dd];
            sqk[qd + khv * hd + dd] += 0.5f * (sm / (float)gqa) + 0.5f * k[khv * hd + dd];
        }
    }

    // L2 normalize: q heads -> sqrt(hd); kv heads -> sqrt(hd) * ks[khv].
    const float shd = std::sqrt((float)hd);
    for (int h = 0; h < nq; h++) {
        float s = 0.0f;
        for (int dd = 0; dd < hd; dd++) s += sqk[h * hd + dd] * sqk[h * hd + dd];
        float iv = shd / (std::sqrt(s) + 1e-12f);
        for (int dd = 0; dd < hd; dd++) sqk[h * hd + dd] *= iv;
    }
    for (int khv = 0; khv < nkv; khv++) {
        float s = 0.0f;
        for (int dd = 0; dd < hd; dd++) s += sqk[qd + khv * hd + dd] * sqk[qd + khv * hd + dd];
        float iv = shd * w.ks[khv] / (std::sqrt(s) + 1e-12f);
        for (int dd = 0; dd < hd; dd++) sqk[qd + khv * hd + dd] *= iv;
    }

    // Partial RoPE (first nrot dims), half-rotation pairing (d, d+nrot/2).
    {
        std::vector<float> rc(nrot), rs(nrot);
        for (int i = 0; i < nrot / 2; i++) {
            float th = pos * std::pow(d.rope_base, -2.0f * (float)i / (float)nrot);
            rc[i] = std::cos(th); rs[i] = std::sin(th);
            rc[nrot / 2 + i] = rc[i]; rs[nrot / 2 + i] = rs[i];
        }
        for (int hh = 0; hh < nq + nkv; hh++) {
            float* base = (hh < nq) ? &sqk[hh * hd] : &sqk[qd + (hh - nq) * hd];
            for (int dd = 0; dd < nrot; dd++) {
                int d2 = (dd < nrot / 2) ? (dd + nrot / 2) : (dd - nrot / 2);
                float xv = base[dd], xw = base[d2];
                float rh = (dd < nrot / 2) ? -xw : xw;
                base[dd] = xv * rc[dd] + rh * rs[dd];
            }
        }
    }

    for (int i = 0; i < qd; i++) q_out[i] = sqk[i];
    for (int i = 0; i < kd; i++) k_out[i] = sqk[qd + i];

    // V assembly: current (from h_norm) + delayed (from prev token, via vrec).
    for (int i = 0; i < kd / 2; i++) {
        v_out[i] = v_cur[i];
        v_out[kd / 2 + i] = st.vrec[i];
        st.vrec[i] = v_del[i];
    }
}

// Full CCA attention block for one token (CPU). After cca_prep, this runs the
// standard GQA sequence attention over the KV cache, then o_proj and residual
// scaling (ported from zaya_forward, src/zaya_engine.cpp).
//
//   kv_k [seq*nkv*hd], kv_v [seq*nkv*hd]  — layer KV cache (seq includes new token)
//   attn_out [H]
inline void cca_attention(const CcaDims& d, const CcaWeights& w, CcaState& st,
                          const float* h_norm, const float* prev_hs,
                          const float* kv_k, const float* kv_v, int seq,
                          float* attn_out, int pos) {
    const int H = d.H, qd = d.qd, kd = d.kd, nq = d.nq, nkv = d.nkv, hd = d.hd;
    const int hv2 = kd / 2;
    const int gqa = nq / nkv;
    const float scale = 1.0f / std::sqrt((float)hd);

    // Q/K/V projections (GEMV, transposed weights: wq[i*H+j] for output i).
    std::vector<float> q(qd), k(kd), vc(hv2), vd(hv2);
    for (int i = 0; i < qd; i++) { float a = 0; for (int j = 0; j < H; j++) a += w.wq[i * H + j] * h_norm[j]; q[i] = a; }
    for (int i = 0; i < kd; i++) { float a = 0; for (int j = 0; j < H; j++) a += w.wk[i * H + j] * h_norm[j]; k[i] = a; }
    for (int i = 0; i < hv2; i++) { float a = 0; for (int j = 0; j < H; j++) a += w.wv1[i * H + j] * h_norm[j]; vc[i] = a; }
    for (int i = 0; i < hv2; i++) { float a = 0; for (int j = 0; j < H; j++) a += w.wv2[i * H + j] * prev_hs[j]; vd[i] = a; }

    std::vector<float> qo(qd), ko(kd), vo(kd);
    cca_prep(d, w, st, q.data(), k.data(), vc.data(), vd.data(),
             qo.data(), ko.data(), vo.data(), pos);

    // GQA sequence attention: attn[h] = softmax(q[h]·K^T * scale) · V.
    std::vector<float> ao(qd);
    for (int h = 0; h < nq; h++) {
        int kv = h / gqa;
        const float* qh = &qo[h * hd];
        std::vector<float> scores(seq);
        float mx = -1e30f;
        for (int t = 0; t < seq; t++) {
            const float* kt = &kv_k[(size_t)t * nkv * hd + kv * hd];
            float s = 0; for (int dd = 0; dd < hd; dd++) s += qh[dd] * kt[dd];
            s *= scale; scores[t] = s; if (s > mx) mx = s;
        }
        float sum = 0;
        for (int t = 0; t < seq; t++) { scores[t] = std::exp(scores[t] - mx); sum += scores[t]; }
        for (int dd = 0; dd < hd; dd++) {
            float a = 0;
            for (int t = 0; t < seq; t++) a += scores[t] * kv_v[(size_t)t * nkv * hd + kv * hd + dd];
            ao[h * hd + dd] = a / (sum + 1e-12f);
        }
    }

    // o_proj: wo is [H, qd] (attn_output.weight), attn_out = wo @ ao.
    for (int i = 0; i < H; i++) {
        float a = 0; for (int j = 0; j < qd; j++) a += w.wo[i * qd + j] * ao[j];
        attn_out[i] = a;
    }
}

// Running residual update (matches llama.cpp zaya.cpp — Session 14).
// Applied BEFORE the per-layer rmsnorm, using the pre-norm hidden state h:
//   hidden_scaled = (h + hs_bias) * hs_scale
//   residual      = hidden_scaled + (residual + res_bias) * res_scale
// The block then runs on rmsnorm(residual). This supersedes the old post-block
// "attn*hs_s + hs_b + h_old*res_s + res_b" form, which used the wrong
// multiply-before-bias order and a post-normalized hidden state.
inline void residual_scale(int H, const float* h, const float* residual,
                           const float* hs_s, const float* hs_b,
                           const float* res_s, const float* res_b,
                           float* out) {
    for (int i = 0; i < H; i++)
        out[i] = (h[i] + hs_b[i]) * hs_s[i] + (residual[i] + res_b[i]) * res_s[i];
}

} // namespace zaya_cca
