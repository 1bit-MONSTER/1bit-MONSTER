// dump_zaya_f32.cpp — dump the REAL Zaya1-8B q4nx weights as f32, using the
// engine's own loader helpers verbatim (zaya_decode.cpp load preamble).
// Output: <out>.bin with a small header + all per-layer float tensors, so the
// M2 trainer can load the real model. No NPU/XRT touched (CPU only).
//
// Build (strixhalo, from engine/npu/src):
//   g++ -O2 -std=c++20 -I. -I../include -I../generators -o /tmp/dump_zaya \
//       dump_zaya_f32.cpp dequant_q4nx.cpp -fopenmp
// Run: /tmp/dump_zaya /home/bcloud/models/zaya1-8b-fresh.q4nx /home/bcloud/zaya-f32t.bin

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "dequant_q4nx.h"
#include "zaya_cca_attn_cpu.h"
#include "zaya_moe_cpu.h"

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
static int get_top_int(const char* js, size_t jl, const char* key) {
    size_t kl = strlen(key);
    const char* p = js, *e = js + jl;
    while (p < e) {
        auto q = (const char*)memmem(p, e - p, key, kl);
        if (!q) return 0;
        if ((q == js || *(q-1) == '"') && *(q + kl) == '"') {
            auto c = strchr(q + kl, ':');
            if (c) return (int)strtoll(c + 1, nullptr, 10);
        }
        p = q + kl;
    }
    return 0;
}
static std::vector<float> load_bf16(const uint8_t* data, uint64_t off, uint64_t size) {
    std::vector<float> v(size / 2);
    const uint8_t* p = data + off;
    for (size_t i = 0; i < v.size(); i++) {
        uint32_t bits = (uint32_t)((uint16_t)p[2 * i] | ((uint16_t)p[2 * i + 1] << 8)) << 16;
        float f; memcpy(&f, &bits, 4); v[i] = f;
    }
    return v;
}
static std::vector<float> load_i8(const uint8_t* data, uint64_t off, uint64_t size,
                                  int i8_rows, int in_features) {
    int rows = 0, cols = 0;
    float* deq = dequant_i8_signed_to_float_ex(data + off, i8_rows, in_features, &rows, &cols);
    std::vector<float> v(deq, deq + (size_t)rows * cols);
    free(deq);
    return v;
}

struct LayerW {
    zaya_cca::CcaWeights cw;
    zaya_moe::RouterWeights rw;
    std::vector<float> gu, dn, nw;
};
static FILE* g_out = nullptr;
static void write_vec(const std::vector<float>& v) {
    size_t n = v.size();
    fwrite(&n, sizeof n, 1, g_out);
    fwrite(v.data(), sizeof(float), n, g_out);
}

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s model.q4nx out.bin\n", argv[0]); return 1; }
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
    m.rtr_h = 256;
    fprintf(stderr, "H=%d NC=%d NV=%d nq=%d nkv=%d hd=%d n_ff=%d n_exp=%d\n",
            d.H, NC, NV, d.nq, d.nkv, d.hd, m.n_ff, m.n_exp);

    g_out = fopen(argv[2], "wb");
    if (!g_out) { perror("fopen out"); return 1; }
    int dims[9] = {d.H, NC, NV, d.nq, d.nkv, d.hd, m.n_ff, m.n_exp, m.rtr_h};
    fwrite(dims, sizeof dims, 1, g_out);

    uint64_t off, size;
    get_offsets(js, jl, "model.embed_tokens.weight", &off, &size);
    int emb_rows = (int)(size / 5120);
    auto embed = load_i8(D, off, size, emb_rows, d.H);
    write_vec(embed);
    uint64_t no, ns; get_offsets(js, jl, "model.norm.weight", &no, &ns);
    write_vec(load_bf16(D, no, ns));

    std::vector<LayerW> L(NC);
    char key[256];
    #pragma omp parallel for schedule(dynamic) private(key)
    for (int l = 0; l < NC; l++) {
        auto& w = L[l];
        #define GET(name, dst) do { uint64_t o_, s_; if (get_offsets(js, jl, name, &o_, &s_)) dst = load_bf16(D, o_, s_); } while(0)
        #define GETI8(name, dst, rows, ifeat) do { uint64_t o_, s_; if (get_offsets(js, jl, name, &o_, &s_)) dst = load_i8(D, o_, s_, rows, ifeat); } while(0)
        snprintf(key, sizeof key, "model.layers.%d.input_layernorm.weight", l); GET(key, w.nw);
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
        snprintf(key, sizeof key, "model.layers.%d.mlp.gate.down_proj.weight", l); GET(key, w.rw.gdw);
        snprintf(key, sizeof key, "model.layers.%d.mlp.gate.down_proj.bias", l); GET(key, w.rw.gdb);
        snprintf(key, sizeof key, "model.layers.%d.mlp.gate.router_mlp.norm.weight", l); GET(key, w.rw.rfn);
        snprintf(key, sizeof key, "model.layers.%d.mlp.gate.router_mlp.fc1.weight", l); GET(key, w.rw.rf1);
        snprintf(key, sizeof key, "model.layers.%d.mlp.gate.router_mlp.fc1.bias", l); GET(key, w.rw.rf1b);
        snprintf(key, sizeof key, "model.layers.%d.mlp.gate.router_mlp.fc2.weight", l); GET(key, w.rw.rf2);
        snprintf(key, sizeof key, "model.layers.%d.mlp.gate.router_mlp.fc2.bias", l); GET(key, w.rw.rf2b);
        snprintf(key, sizeof key, "model.layers.%d.mlp.gate.router_mlp.out_proj.weight", l); GET(key, w.rw.rout);
        snprintf(key, sizeof key, "model.layers.%d.mlp.gate.router_states_scale", l);
        { uint64_t o_, s_; if (get_offsets(js, jl, key, &o_, &s_)) {
            if (s_ == 2) s_ = (uint64_t)m.rtr_h * 2;
            w.rw.eda = load_bf16(D, o_, s_);
        } }
        if (l % 2 == 1) {
            snprintf(key, sizeof key, "model.layers.%d.mlp.experts.gate_up_proj.weight", l);
            GETI8(key, w.gu, (m.n_exp*2*m.n_ff/32)*(d.H/256), d.H);
            snprintf(key, sizeof key, "model.layers.%d.mlp.experts.down_proj.weight", l);
            GETI8(key, w.dn, (m.n_exp*d.H/32)*(m.n_ff/256), m.n_ff);
        }
        #undef GET
        #undef GETI8
    }
    // write per-layer: nw, cw(10), rw(10), gu, dn (with size sentinels)
    for (int l = 0; l < NC; l++) {
        LayerW& w = L[l];
        int tag = 0x1000 + l;
        fwrite(&tag, sizeof tag, 1, g_out);
        write_vec(w.nw);
        write_vec(w.cw.wq); write_vec(w.cw.wk); write_vec(w.cw.wv1); write_vec(w.cw.wv2);
        write_vec(w.cw.wo); write_vec(w.cw.cdw); write_vec(w.cw.cdb); write_vec(w.cw.cgw);
        write_vec(w.cw.cgb); write_vec(w.cw.ks);
        write_vec(w.rw.gdw); write_vec(w.rw.gdb); write_vec(w.rw.rfn);
        write_vec(w.rw.rf1); write_vec(w.rw.rf1b); write_vec(w.rw.rf2); write_vec(w.rw.rf2b);
        write_vec(w.rw.rout); write_vec(w.rw.bb); write_vec(w.rw.eda);
        write_vec(w.gu); write_vec(w.dn);
    }
    int endtag = 0xFFFF;
    fwrite(&endtag, sizeof endtag, 1, g_out);
    fclose(g_out);
    munmap(md, st.st_size);
    // report sizes
    fprintf(stderr, "embed=%zu fnw=%zu | L0: nw=%zu wq=%zu wk=%zu wv1=%zu wv2=%zu wo=%zu\n",
            embed.size(), (size_t)0,
            L[0].nw.size(), L[0].cw.wq.size(), L[0].cw.wk.size(),
            L[0].cw.wv1.size(), L[0].cw.wv2.size(), L[0].cw.wo.size());
    fprintf(stderr, "L0: cdw=%zu cdb=%zu cgw=%zu cgb=%zu ks=%zu | L1: gu=%zu dn=%zu\n",
            L[0].cw.cdw.size(), L[0].cw.cdb.size(), L[0].cw.cgw.size(), L[0].cw.cgb.size(),
            L[0].cw.ks.size(), L[1].gu.size(), L[1].dn.size());
    fprintf(stderr, "L0 router: gdw=%zu gdb=%zu rf1=%zu rout=%zu eda=%zu\n",
            L[0].rw.gdw.size(), L[0].rw.gdb.size(), L[0].rw.rf1.size(),
            L[0].rw.rout.size(), L[0].rw.eda.size());
    fprintf(stderr, "written %s\n", argv[2]);
    return 0;
}
