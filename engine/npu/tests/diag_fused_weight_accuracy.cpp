// diag_fused_weight_accuracy.cpp — is the fused int4 weight reconstruction
// accurate vs the TRUE 4-bit Q4NX weights? #1934 "resolve once and for all".
//
// For the qwen3-0.6b dense GU (gate/up, each [IM=3072, H=1024]):
//   W_true[r][c]  = q4*scl[r][c/32] + zp[r][c/32]      (exact stored 4-bit weight)
//   W_fused[r][c] = B_shadow[r][c] * S_col[r]           (kernel-effective fused int4)
//   W_lin[r][c]   = q4*(s/S_col)*S_col + zp*(S_col/S_col)?? (see below)
// Reports corr/mae of W_fused vs W_true. If corr ~1.0, the fused int4 faithfully
// reconstructs the 4-bit weights, so the h2/token delta vs the int8 reference is
// pure 4-bit-vs-8-bit quantization (fundamental); if corr is low, there's a
// fixable scale/dequant error in the fused path.
//
// Build: g++ -std=c++23 -O2 -I engine/npu/src -I engine/npu/generators \
//        engine/npu/tests/diag_fused_weight_accuracy.cpp -o /tmp/diag_fwa
// Run:   /tmp/diag_fwa models/FastFlowLM-Qwen3-0.6B-NPU2/model.q4nx [layer]
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include "q4nx_raw.h"
#include "gu_i4_pack.h"

static uint64_t get_off(const char* js, size_t jl, const std::string& name) {
    std::string k = "\"" + name + "\":";
    auto q = strstr(js, k.c_str());
    if (!q || q - js + k.size() >= (long)jl) return ~0ull;
    auto o = strstr(q + k.size(), "data_offsets");
    if (!o) return ~0ull;
    auto b = strchr(o, '['); if (!b) return ~0ull;
    return (uint64_t)strtoull(b + 1, nullptr, 10);
}

static double pear(const std::vector<double>& x, const std::vector<double>& y) {
    int m = (int)x.size(); if (!m) return 0;
    double sx=0,sy=0,sx2=0,sy2=0,sxy=0;
    for (int k=0;k<m;k++){sx+=x[k];sy+=y[k];sx2+=x[k]*x[k];sy2+=y[k]*y[k];sxy+=x[k]*y[k];}
    double mx=sx/m,my=sy/m,cxx=sx2/m-mx*mx,cyy=sy2/m-my*my,cxy=sxy/m-mx*my;
    return (cxx>0&&cyy>0)?cxy/std::sqrt(cxx*cyy):(cxx==0&&cyy==0)?1.0:0.0;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <model.q4nx> [layer]\n", argv[0]); return 2; }
    const int L = argc > 2 ? atoi(argv[2]) : 0;
    const int H = 1024, IM = 3072;

    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) { perror("open"); return 2; }
    struct stat st; fstat(fd, &st);
    uint8_t* md = (uint8_t*)mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (md == MAP_FAILED) { perror("mmap"); return 2; }
    uint64_t hsz; memcpy(&hsz, md, 8);
    uint64_t df = 8 + hsz;
    const char* js = (const char*)(md + 8);
    const int RC = H / 32;
    const int i8_rows = (IM / 32) * (H / 256);   // 384

    // Assemble the interleaved GU [2*IM, H] (gate [0,IM), up [IM,2*IM)) from the
    // gate_proj / up_proj raw tensors (corrected asymmetric reader).
    char key[256];
    snprintf(key, sizeof key, "model.layers.%d.mlp.gate_proj.weight", L);
    uint64_t go = get_off(js, hsz, key) + df;
    snprintf(key, sizeof key, "model.layers.%d.mlp.up_proj.weight", L);
    uint64_t uo = get_off(js, hsz, key) + df;
    auto rg = read_q4nx_raw_asym(md, go, i8_rows, H);
    auto ru = read_q4nx_raw_asym(md, uo, i8_rows, H);
    RawQ4Tensor raw_gu; raw_gu.rows = 2*IM; raw_gu.cols = H;
    raw_gu.q4.assign((size_t)(2*IM)*H, 0);
    raw_gu.scl.assign((size_t)(2*IM)*RC, 0.0f);
    raw_gu.zp.assign((size_t)(2*IM)*RC, 0.0f);
    for (int rr = 0; rr < IM; rr++) {
        memcpy(&raw_gu.q4[(size_t)rr*H], &rg.q4[(size_t)rr*H], sizeof(int8_t)*H);
        memcpy(&raw_gu.q4[(size_t)(IM+rr)*H], &ru.q4[(size_t)rr*H], sizeof(int8_t)*H);
        for (int gg = 0; gg < RC; gg++) {
            raw_gu.scl[(size_t)rr*RC+gg] = rg.scl[(size_t)rr*RC+gg]; raw_gu.zp[(size_t)rr*RC+gg] = rg.zp[(size_t)rr*RC+gg];
            raw_gu.scl[(size_t)(IM+rr)*RC+gg] = ru.scl[(size_t)rr*RC+gg]; raw_gu.zp[(size_t)(IM+rr)*RC+gg] = ru.zp[(size_t)rr*RC+gg];
        }
    }

    GuI4Pack p = pack_gu_fused_i4(raw_gu, 0, H, IM, /*bf16_pair=*/true);
    const size_t N = 2*(size_t)IM;   // output cols

    // Per-column corr + global corr/mae of W_fused (=B_shadow*S_col) vs W_true.
    // W_true uses the SAME bf16-rounded scales the pack uses (srow/zp_cur in
    // pack_gu_fused_i4 bf16_pair branch) so NaN/rounding artifacts are excluded
    // and the reconstruction fidelity is measured directly.
    double sT=0,sF=0,sT2=0,sF2=0,sTF=0; long long cnt=0; double mae=0, maxe=0;
    double cmin=1.0, csum=0; int ccnt=0; long long nanbad=0;
    for (int j = 0; j < (int)N; j++) {
        std::vector<double> x, y;
        // output col j is gate/up INTERLEAVED: col 2p = gate[p], col 2p+1 = up[p].
        // So the raw_gu row for col j is (j&1) ? IM + j/2 : j/2.
        int row_j = (j & 1) ? IM + (j >> 1) : (j >> 1);
        for (int i = 0; i < H; i++) {
            float srow = i4p_bf16_to_f32(f32_to_bf16_impl(raw_gu.scl[(size_t)row_j*RC + i/32]));
            float zp_cur = i4p_bf16_to_f32(f32_to_bf16_impl(raw_gu.zp[(size_t)row_j*RC + i/32]));
            float Wt = (float)raw_gu.q4[(size_t)row_j*H + i]*srow + zp_cur;
            float Wf = (float)p.B_shadow[(size_t)i*N + j] * p.scol[j];
            if (!std::isfinite(Wt) || !std::isfinite(Wf)) { nanbad++; continue; }
            x.push_back(Wt); y.push_back(Wf);
            sT+=Wt; sF+=Wf; sT2+=Wt*Wt; sF2+=Wf*Wf; sTF+=Wt*Wf;
            mae += std::fabs(Wt-Wf); if (std::fabs(Wt-Wf)>maxe) maxe=std::fabs(Wt-Wf);
            cnt++;
        }
        if (x.size() >= 8) { double cj = pear(x, y); if (cj < cmin) cmin = cj; csum += cj; ccnt++; }
    }
    fprintf(stderr, "  non-finite filtered: %lld\n", nanbad);
    int n = (int)cnt;
    double mT=sT/n, mF=sF/n, cxx=sT2/n-mT*mT, cyy=sF2/n-mF*mF, cxy=sTF/n-mT*mF;
    double corr = (cxx>0&&cyy>0)?cxy/std::sqrt(cxx*cyy):(cxx==0&&cyy==0)?1.0:0.0;
    fprintf(stderr, "\n=== qwen3-0.6b L%d dense GU: fused int4 reconstruction vs TRUE 4-bit Q4NX ===\n", L);
    fprintf(stderr, "  N=%ld samples, W_true rms=%.5f, W_fused rms=%.5f\n", cnt, std::sqrt(cxx), std::sqrt(cyy));
    fprintf(stderr, "  GLOBAL corr(W_fused, W_true)=%.6f   MAE=%.6g  max|e|=%.6g\n", corr, mae/n, maxe);
    fprintf(stderr, "  per-col corr (first %d cols): min=%.6f avg=%.6f\n", ccnt, cmin, csum/ccnt);
    // sample a few
    fprintf(stderr, "  [dbg] col0 (gate) W_true/W_fused: ");
    for (int i=0;i<6;i++) fprintf(stderr, "%.4f/%.4f ", (float)raw_gu.q4[0*H+i]*raw_gu.scl[0*RC+i/32]+raw_gu.zp[0*RC+i/32], (float)p.B_shadow[(size_t)i*N+0]*p.scol[0]);
    fprintf(stderr, "\n");
    fprintf(stderr, "%s\n", corr >= 0.999 ? "FUSED INT4 RECONSTRUC. FAITHFUL (gap is 4-bit-vs-8-bit, fundamental)"
                                          : "FUSED INT4 RECONSTRUC. HAS ERROR (fixable scale/dequant bug present)");

    // Isolate GATE (j in [0,IM)) vs UP (j in [IM,2*IM)) columns and detect
    // saturation (B_shadow clamped at ±127).
    for (int which = 0; which < 2; which++) {
        int j0 = which ? IM : 0, j1 = j0 + IM;
        double gT=0,gF=0,gT2=0,gF2=0,gTF=0; long long gc=0; int sat=0; double gmae=0;
        for (int j = j0; j < j1; j++) {
            int row_j = (j & 1) ? IM + (j >> 1) : (j >> 1);
            for (int i = 0; i < H; i++) {
                float srow = raw_gu.scl[(size_t)row_j*RC + i/32];
                float zp_cur = raw_gu.zp[(size_t)row_j*RC + i/32];
                float Wt = (float)raw_gu.q4[(size_t)row_j*H + i]*srow + zp_cur;
                int b = (int)p.B_shadow[(size_t)i*N + j];
                if (b >= 127 || b <= -127) sat++;
                float Wf = (float)b * p.scol[j];
                gT+=Wt; gF+=Wf; gT2+=Wt*Wt; gF2+=Wf*Wf; gTF+=Wt*Wf;
                gmae += std::fabs(Wt-Wf);
                gc++;
            }
        }
        double m1=gT/gc, m2=gF/gc, cx=gT2/gc-m1*m1, cy=gF2/gc-m2*m2, cxy=gTF/gc-m1*m2;
        double cor = (cx>0&&cy>0)?cxy/std::sqrt(cx*cy):0.0;
        fprintf(stderr, "  [%s] corr=%.5f mae=%.6f rms_true=%.5f rms_fused=%.5f satB=%lld/%lld\n",
                which?"UP ":"GATE", cor, gmae/gc, std::sqrt(cx), std::sqrt(cy), sat, gc);
    }
    munmap(md, st.st_size);
    return 0;
}
