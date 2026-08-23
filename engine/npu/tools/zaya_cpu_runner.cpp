// zaya_cpu_runner.cpp — correctness-first CPU bring-up for ZAYA1-8B (q4nx).
//
// Ties together the existing pieces:
//   engine/npu/src/dequant_q4nx.cpp     (Q4NX int4 tile dequant, verified)
//   engine/npu/src/zaya_cca_attn_cpu.h  (CCA attention port)
//   engine/npu/src/zaya_moe_cpu.h       (EDA router + expert FFN port)
//
// Forward (matches llama.cpp zaya.cpp — alternating layers + running residual):
//   even layer: hidden_scaled=(h+hb)*hs; residual=hidden_scaled+(residual+rb)*rs
//               cur=rmsnorm(residual);  attn=CCA(cur);  h=attn
//   odd  layer: same residual; cur=rmsnorm(residual);  moe=router+FFN;  h=moe
//   final: cur = rmsnorm(h + residual); logits = embed @ cur
//
// NOTE: v_proj_delayed reads the CURRENT hidden state (wv2 @ h); the one-token
// delay is implemented by the vrec state inside cca_prep (v_out = [v_cur, vrec]).

#include "engine/npu/src/model_config.h"
#include "engine/npu/src/dequant_q4nx.h"
#include "engine/npu/src/zaya_cca_attn_cpu.h"
#include "engine/npu/src/zaya_moe_cpu.h"

#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cmath>

static bool get_offsets(const char* js, size_t jl, const char* key,
                        uint64_t* off, uint64_t* size) {
    size_t kl = strlen(key);
    const char* p = js, *e = js + jl;
    while (p < e) {
        auto q = (const char*)memmem(p, e - p, key, kl);
        if (!q) return false;
        if ((q == js || *(q-1) == '"') && *(q + kl) == '"') {
            auto o = strstr(q, "\"data_offsets\"");
            if (o) {
                auto b = strchr(o, '[');
                if (b) {
                    *off  = (uint64_t)strtoull(b + 1, nullptr, 10);
                    auto c = strchr(b + 1, ',');
                    if (c) *size = (uint64_t)strtoull(c + 1, nullptr, 10) - *off;
                    return *size > 0;
                }
            }
        }
        p = q + kl;
    }
    return false;
}

static std::vector<float> load_bf16(const uint8_t* data, uint64_t off, uint64_t size) {
    std::vector<float> v(size / 2);
    // The .q4nx data section starts at an odd file offset (8 + JSON-header
    // size is odd), so (const uint16_t*)(data+off) is a misaligned load —
    // UB flagged by the issue #1775 UBSan run, and a hard fault on
    // ARM/AIE targets. Read the two bytes explicitly (little-endian).
    const uint8_t* p = data + off;
    for (size_t i = 0; i < v.size(); i++) {
        uint32_t bits = (uint32_t)((uint16_t)p[2 * i] | ((uint16_t)p[2 * i + 1] << 8)) << 16;
        float f; memcpy(&f, &bits, 4); v[i] = f;
    }
    return v;
}

// Dequant a Q4NX int4 tile tensor. i8_rows = shape[0] of the packed tensor;
// in_features = the GGUF column dim (the tile's K). Zaya Q4NX stores the GGUF
// [in, out] layout (NOT PyTorch [out, in]); transpose swaps to [out, in].
static std::vector<float> load_i8(const uint8_t* data, uint64_t off, uint64_t size,
                                  int i8_rows, int in_features, bool transpose = false) {
    int rows = 0, cols = 0;
    float* deq = dequant_i8_signed_to_float_ex(data + off, i8_rows, in_features, &rows, &cols);
    std::vector<float> v(deq, deq + (size_t)rows * cols);
    free(deq);
    if (transpose) {
        std::vector<float> t((size_t)rows * cols);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++)
                t[(size_t)c * rows + r] = v[(size_t)r * cols + c];
        return t;
    }
    return v;
}

static void rmsnorm(float* h, const float* w, int n, float eps = 1e-5f) {
    float ss = 0; for (int i = 0; i < n; i++) ss += h[i] * h[i];
    float r = 1.0f / sqrtf(ss / n + eps);
    for (int i = 0; i < n; i++) h[i] = h[i] * r * w[i];
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model.q4nx [token_id]\n", argv[0]); return 1; }
    int token_id = argc > 2 ? atoi(argv[2]) : 0;

    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    struct stat st; fstat(fd, &st);
    uint8_t* md = (uint8_t*)mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    uint64_t hsz; memcpy(&hsz, md, 8);
    const char* js = (const char*)(md + 8);
    size_t jl = (size_t)hsz;
    const uint8_t* D = md + 8 + hsz;

    auto d = zaya_cca::CcaDims::zaya1_8b();
    d.H  = get_top_int(js, jl, "hidden_size");
    int NC = get_top_int(js, jl, "num_hidden_layers");
    int NV = get_top_int(js, jl, "vocab_size");
    d.nq  = get_top_int(js, jl, "num_attention_heads");
    d.nkv = get_top_int(js, jl, "num_key_value_heads");
    d.hd  = get_top_int(js, jl, "head_dim");
    d.qd  = d.nq * d.hd; d.kd = d.nkv * d.hd; d.qkv = d.qd + d.kd;
    d.gc  = d.qkv / (d.nq + d.nkv); d.nrot = d.hd / 2;
    auto m = zaya_moe::MoeDims::zaya1_8b();
    m.H = d.H; m.n_ff = get_top_int(js, jl, "intermediate_size");
    m.n_exp = get_top_int(js, jl, "num_experts"); m.n_exp_t = m.n_exp + 1;
    int rtr_h = 256; m.rtr_h = rtr_h;
    fprintf(stderr, "H=%d NC=%d NV=%d nq=%d nkv=%d hd=%d n_ff=%d n_exp=%d rtr_h=%d\n",
            d.H, NC, NV, d.nq, d.nkv, d.hd, m.n_ff, m.n_exp, rtr_h);

    // embeddings (tied lm_head) + input scale/bias
    uint64_t off, size;
    get_offsets(js, jl, "model.embed_tokens.weight", &off, &size);
    int emb_rows = (int)(size / 5120);
    auto embed = load_i8(D, off, size, emb_rows, d.H);   // [NV, H]
    fprintf(stderr, "embed: %zu floats (%d x %d)\n", embed.size(), NV, d.H);
    uint64_t so, ss; get_offsets(js, jl, "model.input_hidden_states_scale", &so, &ss);
    auto iscale = load_bf16(D, so, ss);
    uint64_t bo, bs; get_offsets(js, jl, "model.input_hidden_states_bias", &bo, &bs);
    auto ibias = load_bf16(D, bo, bs);

    struct Layer {
        zaya_cca::CcaWeights cw; zaya_cca::CcaState cs;
        zaya_moe::RouterWeights rw;
        std::vector<float> gu, dn, nw, pan;
        std::vector<float> pahss, pahsb, parss, parsb, pmhss, pmhsb, pmrss, pmrsb;
    };
    std::vector<Layer> L(NC);
    char key[256];
    for (int l = 0; l < NC; l++) {
        auto& w = L[l];
        w.cs.reset(d.qkv, d.kd / 2);
        #define GET(name, dst) do { uint64_t o_, s_; if (get_offsets(js, jl, name, &o_, &s_)) dst = load_bf16(D, o_, s_); } while(0)
        #define GETI8(name, dst, rows, ifeat) do { uint64_t o_, s_; if (get_offsets(js, jl, name, &o_, &s_)) dst = load_i8(D, o_, s_, rows, ifeat); } while(0)
        #define GETI8T(name, dst, rows, ifeat) do { uint64_t o_, s_; if (get_offsets(js, jl, name, &o_, &s_)) dst = load_i8(D, o_, s_, rows, ifeat, true); } while(0)

        snprintf(key, sizeof key, "model.layers.%d.input_layernorm.weight", l); GET(key, w.nw);
        snprintf(key, sizeof key, "model.layers.%d.post_attention_layernorm.weight", l); GET(key, w.pan);
        // CCA attention. The converter's unpack produces qs[ne1, ne0] = [out, in]
        // (forward layout), quantized over ne0 (in_features). No transpose.
        snprintf(key, sizeof key, "model.layers.%d.self_attn.q_proj.weight", l); GETI8(key, w.cw.wq, 256, d.H);
        snprintf(key, sizeof key, "model.layers.%d.self_attn.k_proj.weight", l); GETI8(key, w.cw.wk, 64, d.H);
        snprintf(key, sizeof key, "model.layers.%d.self_attn.v_proj_current.weight", l); GETI8(key, w.cw.wv1, 32, d.H);
        snprintf(key, sizeof key, "model.layers.%d.self_attn.v_proj_delayed.weight", l); GETI8(key, w.cw.wv2, 32, d.H);
        snprintf(key, sizeof key, "model.layers.%d.self_attn.o_proj.weight", l); GETI8(key, w.cw.wo, 256, d.qd);
        snprintf(key, sizeof key, "model.layers.%d.self_attn.conv_qk_depthwise.weight", l); GET(key, w.cw.cdw);
        snprintf(key, sizeof key, "model.layers.%d.self_attn.conv_qk_depthwise.bias", l); GET(key, w.cw.cdb);
        snprintf(key, sizeof key, "model.layers.%d.self_attn.conv_qk_grouped.weight", l); GET(key, w.cw.cgw);
        snprintf(key, sizeof key, "model.layers.%d.self_attn.conv_qk_grouped.bias", l); GET(key, w.cw.cgb);
        snprintf(key, sizeof key, "model.layers.%d.self_attn.qk_norm.temp", l); GET(key, w.cw.ks);
        snprintf(key, sizeof key, "model.layers.%d.post_attention_residual_scale.hidden_states_scale", l); GET(key, w.pahss);
        snprintf(key, sizeof key, "model.layers.%d.post_attention_residual_scale.hidden_states_bias", l); GET(key, w.pahsb);
        snprintf(key, sizeof key, "model.layers.%d.post_attention_residual_scale.residual_scale", l); GET(key, w.parss);
        snprintf(key, sizeof key, "model.layers.%d.post_attention_residual_scale.residual_bias", l); GET(key, w.parsb);
        snprintf(key, sizeof key, "model.layers.%d.post_mlp_residual_scale.hidden_states_scale", l); GET(key, w.pmhss);
        snprintf(key, sizeof key, "model.layers.%d.post_mlp_residual_scale.hidden_states_bias", l); GET(key, w.pmhsb);
        snprintf(key, sizeof key, "model.layers.%d.post_mlp_residual_scale.residual_scale", l); GET(key, w.pmrss);
        snprintf(key, sizeof key, "model.layers.%d.post_mlp_residual_scale.residual_bias", l); GET(key, w.pmrsb);
        // MoE router
        snprintf(key, sizeof key, "model.layers.%d.mlp.gate.down_proj.weight", l); GET(key, w.rw.gdw);
        snprintf(key, sizeof key, "model.layers.%d.mlp.gate.down_proj.bias", l); GET(key, w.rw.gdb);
        snprintf(key, sizeof key, "model.layers.%d.mlp.gate.router_mlp.norm.weight", l); GET(key, w.rw.rfn);
        snprintf(key, sizeof key, "model.layers.%d.mlp.gate.router_mlp.fc1.weight", l); GET(key, w.rw.rf1);
        snprintf(key, sizeof key, "model.layers.%d.mlp.gate.router_mlp.fc1.bias", l); GET(key, w.rw.rf1b);
        snprintf(key, sizeof key, "model.layers.%d.mlp.gate.router_mlp.fc2.weight", l); GET(key, w.rw.rf2);
        snprintf(key, sizeof key, "model.layers.%d.mlp.gate.router_mlp.fc2.bias", l); GET(key, w.rw.rf2b);
        snprintf(key, sizeof key, "model.layers.%d.mlp.gate.router_mlp.out_proj.weight", l); GET(key, w.rw.rout);
        snprintf(key, sizeof key, "model.layers.%d.mlp.gate.balancing_biases", l); GET(key, w.rw.bb);
        snprintf(key, sizeof key, "model.layers.%d.mlp.gate.router_states_scale", l); GET(key, w.rw.eda);
        // experts. gate_up: [n_exp*2*n_ff, H]; down: [n_exp*H, n_ff].
        snprintf(key, sizeof key, "model.layers.%d.mlp.experts.gate_up_proj.weight", l); GETI8(key, w.gu, (m.n_exp*2*m.n_ff/32)*(d.H/256), d.H);
        snprintf(key, sizeof key, "model.layers.%d.mlp.experts.down_proj.weight", l); GETI8(key, w.dn, (m.n_exp*d.H/32)*(m.n_ff/256), m.n_ff);
        #undef GET
        #undef GETI8
        #undef GETI8T
        if (l == 0)
            fprintf(stderr, "layer0: wq=%zu wk=%zu wv1=%zu wo=%zu cdw=%zu cgw=%zu ks=%zu gu=%zu dn=%zu rf1=%zu rout=%zu\n",
                    w.cw.wq.size(), w.cw.wk.size(), w.cw.wv1.size(), w.cw.wo.size(),
                    w.cw.cdw.size(), w.cw.cgw.size(), w.cw.ks.size(),
                    w.gu.size(), w.dn.size(), w.rw.rf1.size(), w.rw.rout.size());
    }

    // final norm weight (loaded once)
    uint64_t no, ns; get_offsets(js, jl, "model.norm.weight", &no, &ns);
    auto fnw = load_bf16(D, no, ns);

    // ── forward (lambda): one token through 40 layers -> argmax ──
    std::vector<std::vector<float>> kv_k(NC), kv_v(NC);   // per-layer KV cache [seq * nkv * hd]
    std::vector<float> attn_out(d.H), moe_out(d.H), tmp(d.H), h(d.H);
    auto forward = [&](int tok, int pos) -> int {
        // reference embed_lookup_k: (raw + ibias) * iscale
        for (int i = 0; i < d.H; i++) h[i] = (embed[(size_t)tok * d.H + i] + ibias[i]) * iscale[i];
        std::vector<float> residual(d.H, 0.0f);
        bool has_res = false;
        std::vector<float> prev_router;  // recurrent router state (shared across MoE layers)
        for (int l = 0; l < NC; l++) {
            auto& w = L[l];
            auto& lk = kv_k[l];
            auto& lv = kv_v[l];
            // ── residual scaling (current layer's own weights) ──
            // even = attention layer (post_attention scales); odd = MoE (post_mlp scales)
            const float* hs; const float* hb; const float* rs; const float* rb;
            if (l % 2 == 0) { hs = w.pahss.data(); hb = w.pahsb.data(); rs = w.parss.data(); rb = w.parsb.data(); }
            else            { hs = w.pmhss.data(); hb = w.pmhsb.data(); rs = w.pmrss.data(); rb = w.pmrsb.data(); }
            // hidden_scaled = (h + hb) * hs
            for (int i = 0; i < d.H; i++) tmp[i] = (h[i] + hb[i]) * hs[i];
            // residual = hidden_scaled + (residual + rb) * rs
            if (has_res) {
                for (int i = 0; i < d.H; i++) residual[i] = tmp[i] + (residual[i] + rb[i]) * rs[i];
            } else {
                for (int i = 0; i < d.H; i++) residual[i] = tmp[i];
                has_res = true;
            }
            // cur = rmsnorm(residual, input_layernorm)
            rmsnorm(residual.data(), w.nw.data(), d.H);
            if (l % 2 == 0) {
                // ── CCA attention block (even layers) ──
                const int qd = d.qd, kd = d.kd, hv2 = kd/2, H = d.H;
                std::vector<float> q(qd), k(kd), vc(hv2), vd(hv2);
                const int nproj = qd + kd + hv2 + hv2;
                #pragma omp parallel for schedule(static)
                for (int ii = 0; ii < nproj; ii++) {
                    float a = 0;
                    if (ii < qd) {
                        for (int j = 0; j < H; j++) a += w.cw.wq[(size_t)ii * H + j] * residual[j];
                        q[ii] = a;
                    } else if (ii < qd + kd) {
                        int i = ii - qd;
                        for (int j = 0; j < H; j++) a += w.cw.wk[(size_t)i * H + j] * residual[j];
                        k[i] = a;
                    } else if (ii < qd + kd + hv2) {
                        int i = ii - qd - kd;
                        for (int j = 0; j < H; j++) a += w.cw.wv1[(size_t)i * H + j] * residual[j];
                        vc[i] = a;
                    } else {
                        int i = ii - qd - kd - hv2;
                        for (int j = 0; j < H; j++) a += w.cw.wv2[(size_t)i * H + j] * residual[j];
                        vd[i] = a;
                    }
                }
                std::vector<float> qo(qd), ko(kd), vo(kd);
                zaya_cca::cca_prep(d, w.cw, w.cs, q.data(), k.data(), vc.data(), vd.data(),
                                   qo.data(), ko.data(), vo.data(), pos);
                size_t old = lk.size() / (size_t)(d.nkv * d.hd);
                lk.insert(lk.end(), ko.begin(), ko.end());
                lv.insert(lv.end(), vo.begin(), vo.end());
                int seq = (int)old + 1;
                int gqa = d.nq / d.nkv;
                std::vector<float> ao(qd);
                for (int hh = 0; hh < d.nq; hh++) {
                    int kv = hh / gqa;
                    std::vector<float> sc(seq); float mx = -1e30f;
                    for (int t = 0; t < seq; t++) { float s=0; const float* kt=&lk[(size_t)t*d.nkv*d.hd + kv*d.hd]; for (int dd=0;dd<d.hd;dd++) s+=qo[hh*d.hd+dd]*kt[dd]; s*=1.0f/sqrtf((float)d.hd); sc[t]=s; mx=std::max(mx,s); }
                    float sm=0; for (int t=0;t<seq;t++){sc[t]=expf(sc[t]-mx);sm+=sc[t];}
                    for (int dd=0;dd<d.hd;dd++){float a=0; for(int t=0;t<seq;t++)a+=sc[t]*lv[(size_t)t*d.nkv*d.hd+kv*d.hd+dd]; ao[hh*d.hd+dd]=a/(sm+1e-12f);}
                }
                #pragma omp parallel for schedule(static)
                for (int i = 0; i < H; i++) { float a=0; for (int j=0;j<qd;j++) a += w.cw.wo[i*qd+j]*ao[j]; h[i]=a; }
            } else {
                // ── MoE block (odd layers) ──
                std::vector<float> rsv; float wt;
                int e = zaya_moe::router(m, w.rw, residual.data(), prev_router, &wt);
                zaya_moe::expert_ffn(m, e, w.gu, w.dn, residual.data(), moe_out.data());
                for (int i = 0; i < d.H; i++) h[i] = moe_out[i];
            }
        }
        // final: cur = h + residual; rmsnorm(final_norm); lm_head
        for (int i = 0; i < d.H; i++) tmp[i] = h[i] + residual[i];
        rmsnorm(tmp.data(), fnw.data(), d.H);
        std::vector<float> logits(NV);
        #pragma omp parallel for schedule(static)
        for (int v = 0; v < NV; v++) { float a=0; for (int j=0;j<d.H;j++) a += embed[(size_t)v*d.H+j]*tmp[j]; logits[v]=a; }
        if (pos == 0) {
            float mn=1e30, mx=-1e30, ss=0, hm=0, hmx=0;
            for (int v=0; v<NV; v++){ mn=std::min(mn,logits[v]); mx=std::max(mx,logits[v]); ss+=logits[v]*logits[v]; }
            for (int j=0;j<d.H;j++){ hm+=tmp[j]; hmx=std::max(hmx,std::fabs(tmp[j])); }
            float mean=0; for(int v=0;v<NV;v++) mean+=logits[v]; mean/=NV;
            fprintf(stderr, "[dbg] logits: min=%.4f max=%.4f mean=%.4f rms=%.4f | h: mean=%.4f maxabs=%.4f\n", mn, mx, mean, sqrtf(ss/NV), hm/d.H, hmx);
            // top-5 logits
            std::vector<int> idx(NV); for(int v=0;v<NV;v++) idx[v]=v;
            std::partial_sort(idx.begin(), idx.begin()+5, idx.end(), [&](int a,int b){return logits[a]>logits[b];});
            for(int k=0;k<5;k++) fprintf(stderr, "  top%d: tok=%d val=%.4f\n", k, idx[k], logits[idx[k]]);
            fprintf(stderr, "  [tok2=%.4f tok4=%.4f tokeq=%.4f tokBOS=%.4f]\n", logits[236778], logits[236812], logits[236862], logits[2]);
        }
        return (int)(std::max_element(logits.begin(), logits.end()) - logits.begin());
    };

    // prompt tokens from argv (space-separated ids), else single token.
    // BOS token (2) is auto-prepended, matching llama.cpp decode behavior.
    std::vector<int> prompt;
    prompt.push_back(2);  // <bos>
    for (int i = 2; i < argc; i++) prompt.push_back(atoi(argv[i]));
    if (prompt.size() == 1) prompt.push_back(token_id);
    for (int i = 0; i < (int)prompt.size(); i++) forward(prompt[i], i);

    const int N_GEN = 8;
    int cur = prompt.back();
    for (int step = 0; step < N_GEN; step++) {
        int arg = forward(cur, (int)prompt.size() + step);
        printf("%d ", arg);
        fflush(stdout);
        cur = arg;
    }
    printf("\n");
    return 0;
}
