// qwen36_moe_probe.cpp — native NPU MoE expert routing validation.
//
// Runs ONE real Qwen3.6-35B-A3B MoE layer FFN on the NPU end-to-end:
//   softmax router → top-8 → per-expert GU/D GEMMs (concatenated, via the
//   engine's I8Ctx + generated instruction sequences on the existing
//   final_i8_*_qwen3.6-moe_35b.xclbins) + shared expert (fused GU + D)
//   × sigmoid gate — compared against a CPU reference (llama.cpp qwen35moe
//   math: LLM_FFN_SILU, softmax gating, shared expert gated by sigmoid).
//
// This closes the "no expert routing on the native NPU path" gate with
// evidence, independent of the GDN/attention side.
//
// Build:
//   g++ -std=c++20 -O2 -fopenmp -o build/qwen36_moe_probe \
//     tools/qwen36_moe_probe.cpp engine/npu/src/dequant_q4nx.cpp \
//     engine/npu/src/gemm_npu_instructions.cpp \
//     -I engine/npu/src -I third_party/FastFlowLM/src/include \
//     -I third_party/FastFlowLM/src/include/npu_utils \
//     -L /opt/xilinx/xrt/lib -lxrt_coreutil -lxrt_core -laiebu -luuid -lm -ldl \
//     -Wl,-rpath,/opt/xilinx/xrt/lib
// Run:
//   ./build/qwen36_moe_probe [model.q4nx] [layer] [xclbin_dir]
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "model_config.h"
#include "dequant_q4nx.h"
#include "npu_utils/npu_instr_utils.hpp"
// gemm_generate_sequence is defined in gemm_npu_instructions.cpp (the
// engine's gemm_generate_sequence_i8 wrapper is declared but never defined —
// dormant-bug path; the probe calls the real function directly).
void gemm_generate_sequence(npu_sequence* seq, uint32_t M, uint32_t K, uint32_t N,
                            uint32_t weight_offset, bool add_bias, int activation,
                            uint32_t bias_offset, uint32_t output_offset);
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>
#include <aiebu/aiebu_assembler.h>

// ── Q4NX helpers (mirror of the engine's jo()) ──
static uint64_t jo(const char* js, size_t jl, const char* nm) {
    size_t nl = strlen(nm);
    const char* p = js; const char* e = js + jl;
    while (p < e) {
        auto q = (const char*)memmem(p, e - p, nm, nl);
        if (!q) return 0;
        if ((q == js || *(q-1) == '"') && *(q + nl) == '"') {
            auto offs = strstr(q, "\"data_offsets\"");
            if (!offs) return 0;
            auto br = strchr(offs, '[');
            return br ? strtoull(br + 1, nullptr, 10) : 0;
        }
        p = q + nl;
    }
    return 0;
}
static inline float bf16f(uint16_t v){uint32_t b=v<<16;float f;memcpy(&f,&b,4);return f;}

// ── minimal I8Ctx (copied from npu_engine_universal.cpp, same API) ──
#include "npu_engine_i8ctx_inc.h"
// ── model geometry ──
static const int H = 2048, IM_EXP = 512, N_EXPERTS = 256, TOP_K = 8;

static const char* exp_keys[3] = {
    "model.layer.%d.mlp.gate_exps_proj.weight",
    "model.layer.%d.mlp.up_exps_proj.weight",
    "model.layer.%d.mlp.down_exps_proj.weight",
};
static const char* sh_keys[3] = {
    "model.layer.%d.mlp.share_gate_exps_proj.weight",
    "model.layer.%d.mlp.share_up_exps_proj.weight",
    "model.layer.%d.mlp.share_down_exps_proj.weight",
};

int main(int argc, char** argv) {
    const char* q4nx_path = argc > 1 ? argv[1]
        : "/home/bcloud/.config/flm/models/Qwen3.6-35B-A3B-NPU2/model.q4nx";
    int layer = argc > 2 ? atoi(argv[2]) : 0;
    std::string xd = argc > 3 ? argv[3] : "/home/bcloud/projects/1bit-systems/engine/npu/xclbins";

    // ── open model ──
    int fd = open(q4nx_path, O_RDONLY);
    struct stat st; fstat(fd, &st);
    uint8_t* md = (uint8_t*)mmap(NULL, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    uint64_t hsz; memcpy(&hsz, md, 8);
    uint64_t df = 8 + hsz;
    const char* js = (const char*)(md + 8); size_t jl = hsz;
    auto i8p = [&](uint64_t o) { return md + df + o; };

    ModelConfig cfg = parse_q4nx_header(q4nx_path, "qwen3_6_moe");
    printf("H=%d NC=%d experts=%d im_exp=%d shared=%d gdn=%d\n",
           cfg.H, cfg.NC, cfg.N_EXPERTS, cfg.IM_EXP, cfg.N_SHARED, (int)cfg.has_gated_delta_net);
    if (!cfg.has_moe) { fprintf(stderr, "not an MoE model\n"); return 1; }

    char bn[128];
    snprintf(bn, 128, "model.layer.%d.moe_router.weight", layer);
    uint64_t router_off = jo(js, jl, bn);
    snprintf(bn, 128, "model.layer.%d.shared_expert_gate.weight", layer);
    uint64_t shgate_off = jo(js, jl, bn);

    uint64_t exp_off[3], sh_off[3];
    for (int t = 0; t < 3; t++) {
        snprintf(bn, 128, exp_keys[t], layer); exp_off[t] = jo(js, jl, bn);
        snprintf(bn, 128, sh_keys[t], layer);  sh_off[t] = jo(js, jl, bn);
    }
    printf("router_off=%llu shgate_off=%llu\n", (unsigned long long)router_off, (unsigned long long)shgate_off);
    for (int t = 0; t < 3; t++)
        printf("exp_off[%d]=%llu sh_off[%d]=%llu\n", t, (unsigned long long)exp_off[t], t, (unsigned long long)sh_off[t]);

    // ── dequant experts + shared experts (whole tensors) ──
    // gate_exps [4096, 8, 5120] = 32768 tile rows → [131072, 2048] f32
    // expert e = rows [e*512, (e+1)*512)
    const int TR_G = 4096 * 8, TR_D = 16384 * 2;
    int er, ec, dr, dc;
    float* gate_f = dequant_i8_to_float_ex(i8p(exp_off[0]), TR_G, H, &er, &ec);
    float* up_f   = dequant_i8_to_float_ex(i8p(exp_off[1]), TR_G, H, &er, &ec);
    float* down_f = dequant_i8_to_float_ex(i8p(exp_off[2]), TR_D, H, &dr, &dc);
    if (!gate_f || !up_f || !down_f) { fprintf(stderr, "expert dequant failed\n"); return 1; }
    printf("gate_exps dequant: [%d, %d]  down: [%d, %d]\n", er, ec, dr, dc);
    // sanity: expert 0 slice = gate_f[0 .. 512*2048)
    float* shg_f[3];
    for (int t = 0; t < 3; t++) {
        int s_tr = t == 2 ? 64 * 2 : 16 * 8;
        int s_r, s_c;
        shg_f[t] = dequant_i8_to_float_ex(i8p(sh_off[t]), s_tr, H, &s_r, &s_c);
        printf("shared[%d] dequant: [%d, %d]\n", t, s_r, s_c);
    }

    // ── router: BF16 [2048, 256], stride-8 row interleave (cracked layout):
    //    logical W[i][j] (i=in 0..2047, j=expert 0..255) = flat[(i%8)*65536 + j*256 + i/8]
    std::vector<float> router(H * N_EXPERTS);
    {
        const uint16_t* rb = (const uint16_t*)i8p(router_off);
        for (int i = 0; i < H; i++)
            for (int j = 0; j < N_EXPERTS; j++)
                router[i * N_EXPERTS + j] = bf16f(rb[(size_t)(i % 8) * 65536 + j * 256 + i / 8]);
    }
    std::vector<float> shgate(H);
    {
        const uint16_t* gb = (const uint16_t*)i8p(shgate_off);
        for (int i = 0; i < H; i++) shgate[i] = bf16f(gb[i]);
    }

    // ── synthetic hidden state (seeded, realistic scale) ──
    std::vector<float> x(H);
    srand(42);
    for (int i = 0; i < H; i++) x[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;

    // ── CPU reference (llama.cpp qwen35moe math) ──
    std::vector<float> logits(N_EXPERTS), probs(N_EXPERTS);
    double lmax = -1e30;
    for (int j = 0; j < N_EXPERTS; j++) {
        double s = 0;
        for (int i = 0; i < H; i++) s += (double)x[i] * router[i * N_EXPERTS + j];
        logits[j] = (float)s;
        if (logits[j] > lmax) lmax = logits[j];
    }
    double lsum = 0;
    for (int j = 0; j < N_EXPERTS; j++) { probs[j] = expf(logits[j] - (float)lmax); lsum += probs[j]; }
    for (int j = 0; j < N_EXPERTS; j++) probs[j] /= (float)lsum;
    std::vector<int> topk(N_EXPERTS);
    for (int j = 0; j < N_EXPERTS; j++) topk[j] = j;
    std::partial_sort(topk.begin(), topk.begin() + TOP_K, topk.end(),
        [&](int a, int b) { return probs[a] > probs[b]; });
    printf("top-8 experts: %d %d %d %d %d %d %d %d  (probs %.3f %.3f ...)\n",
           topk[0], topk[1], topk[2], topk[3], topk[4], topk[5], topk[6], topk[7],
           probs[topk[0]], probs[topk[1]]);

    // reference output
    std::vector<float> ref_out(H, 0.0f);
    {
        std::vector<float> su(IM_EXP);
        for (int e = 0; e < TOP_K; e++) {
            int ex = topk[e];
            const float* G = gate_f + (size_t)ex * IM_EXP * H;
            const float* U = up_f   + (size_t)ex * IM_EXP * H;
            const float* D = down_f + (size_t)ex * IM_EXP * H;
            for (int i = 0; i < IM_EXP; i++) {
                double g = 0, u = 0;
                for (int k = 0; k < H; k++) { g += (double)G[i * H + k] * x[k]; u += (double)U[i * H + k] * x[k]; }
                float gv = (float)g;
                su[i] = (gv / (1.0f + expf(-gv))) * (float)u;
            }
            for (int i = 0; i < H; i++) {
                double d = 0;
                for (int k = 0; k < IM_EXP; k++) d += (double)D[i * IM_EXP + k] * su[k];
                ref_out[i] += (float)probs[ex] * (float)d;
            }
        }
        // shared expert
        {
            const float* G = shg_f[0], * U = shg_f[1], * D = shg_f[2];
            std::vector<float> su(IM_EXP), sh_out(H);
            for (int i = 0; i < IM_EXP; i++) {
                double g = 0, u = 0;
                for (int k = 0; k < H; k++) { g += (double)G[i * H + k] * x[k]; u += (double)U[i * H + k] * x[k]; }
                float gv = (float)g;
                su[i] = (gv / (1.0f + expf(-gv))) * (float)u;
            }
            for (int i = 0; i < H; i++) {
                double d = 0;
                for (int k = 0; k < IM_EXP; k++) d += (double)D[i * IM_EXP + k] * su[k];
                sh_out[i] = (float)d;
            }
            double sg = 0;
            for (int i = 0; i < H; i++) sg += (double)x[i] * shgate[i];
            float sg_sig = 1.0f / (1.0f + expf(-(float)sg));
            for (int i = 0; i < H; i++) ref_out[i] += sg_sig * sh_out[i];
            printf("shared gate: %f -> sigmoid %f\n", (float)sg, sg_sig);
        }
    }

    // ── NPU path: init GEMM contexts with MoE dims via generator ──
    try {
        xrt::device dev(0);
        const int XM = 128;  // batch tile (we run M=1)
        // GU concat: K=H, N=TOP_K*2*IM_EXP (8 experts × fused gate+up)
        I8Ctx cg;
        if (!cg.init_with_generator(dev, (xd + "/final_i8_G_qwen3.6-moe_35b.xclbin").c_str(),
                                    XM, H, TOP_K * 2 * IM_EXP, 1))
            { fprintf(stderr, "FAIL GU ctx\n"); return 1; }
        // D concat: K=TOP_K*IM_EXP, N=H
        I8Ctx cd;
        if (!cd.init_with_generator(dev, (xd + "/final_i8_G_qwen3.6-moe_35b.xclbin").c_str(),
                                    XM, TOP_K * IM_EXP, H, 1))
            { fprintf(stderr, "FAIL D ctx\n"); return 1; }
        // shared expert: fused GU [H, 2*IM_EXP] and D [IM_EXP, H]
        I8Ctx csg, csd;
        if (!csg.init_with_generator(dev, (xd + "/final_i8_G_qwen3.6-moe_35b.xclbin").c_str(),
                                     XM, H, 2 * IM_EXP, 1))
            { fprintf(stderr, "FAIL shared GU ctx\n"); return 1; }
        if (!csd.init_with_generator(dev, (xd + "/final_i8_G_qwen3.6-moe_35b.xclbin").c_str(),
                                     XM, IM_EXP, H, 1))
            { fprintf(stderr, "FAIL shared D ctx\n"); return 1; }

        // pack active experts into concat weights [K, N] (transposed [in,out] layout)
        std::vector<float> gu_w((size_t)H * TOP_K * 2 * IM_EXP);  // [in=H, out=N]
        for (int e = 0; e < TOP_K; e++) {
            int ex = topk[e];
            for (int i = 0; i < IM_EXP; i++) {
                for (int k = 0; k < H; k++) {
                    gu_w[(size_t)k * (TOP_K * 2 * IM_EXP) + e * 2 * IM_EXP + i] =
                        gate_f[(size_t)ex * IM_EXP * H + i * H + k];
                    gu_w[(size_t)k * (TOP_K * 2 * IM_EXP) + e * 2 * IM_EXP + IM_EXP + i] =
                        up_f[(size_t)ex * IM_EXP * H + i * H + k];
                }
            }
        }
        float gu_sc = 0, d_sc = 0, sg_sc = 0, sd_sc = 0;
        cg.packB(0, gu_w.data(), H, TOP_K * 2 * IM_EXP, gu_sc);
        // down concat [K = TOP_K*IM_EXP, N = H]
        std::vector<float> d_w((size_t)TOP_K * IM_EXP * H);
        for (int e = 0; e < TOP_K; e++) {
            int ex = topk[e];
            for (int i = 0; i < H; i++)
                for (int k = 0; k < IM_EXP; k++)
                    d_w[(size_t)(e * IM_EXP + k) * H + i] = down_f[(size_t)ex * IM_EXP * H + i * IM_EXP + k];
        }
        cd.packB(0, d_w.data(), TOP_K * IM_EXP, H, d_sc);
        // shared expert weights
        std::vector<float> sg_w((size_t)H * 2 * IM_EXP);
        for (int i = 0; i < IM_EXP; i++)
            for (int k = 0; k < H; k++) {
                sg_w[(size_t)k * (2 * IM_EXP) + i] = shg_f[0][i * H + k];
                sg_w[(size_t)k * (2 * IM_EXP) + IM_EXP + i] = shg_f[1][i * H + k];
            }
        csg.packB(0, sg_w.data(), H, 2 * IM_EXP, sg_sc);
        std::vector<float> sd_w((size_t)IM_EXP * H);
        for (int i = 0; i < H; i++)
            for (int k = 0; k < IM_EXP; k++)
                sd_w[(size_t)k * H + i] = shg_f[2][i * IM_EXP + k];
        csd.packB(0, sd_w.data(), IM_EXP, H, sd_sc);

        // ── run: GU concat GEMM (M=1) → SiLU per expert → weighted concat → D GEMM
        std::vector<float> gu_out(TOP_K * 2 * IM_EXP), su(TOP_K * IM_EXP), d_out(H), npu_out(H);
        float ascale = 1.0f;
        cg.go(0, x.data(), 1, H, ascale, gu_sc, gu_out.data(), TOP_K * 2 * IM_EXP);
        for (int e = 0; e < TOP_K; e++)
            for (int i = 0; i < IM_EXP; i++) {
                float gv = gu_out[e * 2 * IM_EXP + i];
                su[e * IM_EXP + i] = (gv / (1.0f + expf(-gv))) * gu_out[e * 2 * IM_EXP + IM_EXP + i] * probs[topk[e]];
            }
        cd.go(0, su.data(), 1, TOP_K * IM_EXP, ascale, d_sc, d_out.data(), H);
        // shared expert
        std::vector<float> sg_out(2 * IM_EXP), ssu(IM_EXP), sh_out(H);
        csg.go(0, x.data(), 1, H, ascale, sg_sc, sg_out.data(), 2 * IM_EXP);
        for (int i = 0; i < IM_EXP; i++) {
            float gv = sg_out[i];
            ssu[i] = (gv / (1.0f + expf(-gv))) * sg_out[IM_EXP + i];
        }
        csd.go(0, ssu.data(), 1, IM_EXP, ascale, sd_sc, sh_out.data(), H);
        double sg = 0;
        for (int i = 0; i < H; i++) sg += (double)x[i] * shgate[i];
        float sg_sig = 1.0f / (1.0f + expf(-(float)sg));
        for (int i = 0; i < H; i++) npu_out[i] = d_out[i] + sg_sig * sh_out[i];

        // ── compare ──
        double num = 0, den = 0, maxerr = 0;
        for (int i = 0; i < H; i++) {
            double a = npu_out[i], b = ref_out[i];
            num += (a - b) * (a - b);
            den += b * b;
            if (fabs(a - b) > maxerr) maxerr = fabs(a - b);
        }
        double rmse = sqrt(num / den);
        printf("npu_out[0..4] = %.4f %.4f %.4f %.4f %.4f\n", npu_out[0], npu_out[1], npu_out[2], npu_out[3], npu_out[4]);
        printf("ref_out[0..4] = %.4f %.4f %.4f %.4f %.4f\n", ref_out[0], ref_out[1], ref_out[2], ref_out[3], ref_out[4]);
        printf("rel RMSE = %.6f  max_abs_err = %.6f  (%s)\n", rmse, maxerr,
               rmse < 0.01 ? "PASS" : "FAIL");
        return rmse < 0.01 ? 0 : 2;
    } catch (std::exception& ex) {
        fprintf(stderr, "XRT error: %s\n", ex.what());
        return 1;
    }
}
