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

// ── Q4NX tile dequant matching the 1BP writer (gguf_to_onebp.cpp) ──
// Tile = [32 rows × 256 cols]. Per tile: tr*grps*2 B bf16 scales + same for
// zero-points + tr*tc/2 B packed INT4. Layout (differs from dequant_q4nx.cpp,
// which is group-major/col-major — wrong for 1BP files, was the #1467 bug):
//   scales[row*grps + g], zps[row*grps + g]  (row-major)
//   packed[(row*tc + col)/2], low nibble = even col
// Returns [out_rows, out_cols] row-major f32, caller frees.
static float* dequant_1bp(const uint8_t* data, int i8_rows, int in_features,
                          int* out_rows, int* out_cols) {
    constexpr int TR = 32, TC = 256, GS = 32;
    int ntc = in_features / TC, ntr = i8_rows / ntc;
    *out_rows = ntr * TR; *out_cols = ntc * TC;
    int grps = TC / GS;
    float* out = (float*)calloc((size_t)(*out_rows) * (*out_cols), sizeof(float));
    if (!out) return nullptr;
    for (int ir = 0; ir < i8_rows; ir++) {
        const uint8_t* t = data + (size_t)ir * (TR*grps*2 + TR*grps*2 + TR*TC/2);
        const uint16_t* sc = (const uint16_t*)t;
        const uint16_t* zp = (const uint16_t*)(t + (size_t)TR*grps*2);
        const uint8_t* qd = t + (size_t)TR*grps*4;
        int tr_ = ir / ntc, tc_ = ir % ntc;
        for (int r = 0; r < TR; r++)
            for (int g = 0; g < grps; g++) {
                float s = bf16f(sc[r*grps + g]);
                float z = bf16f(zp[r*grps + g]);
                if (!std::isfinite(s) || std::fabs(s) > 100.0f) s = 0.0f;
                if (!std::isfinite(z) || std::fabs(z) > 100.0f) z = 0.0f;
                for (int i = 0; i < GS; i++) {
                    int col = g*GS + i;
                    uint8_t b = qd[((size_t)r*TC + col) / 2];
                    uint8_t v = (col & 1) ? (b >> 4) : (b & 0x0F);
                    out[((size_t)tr_*TR + r) * (*out_cols) + (size_t)tc_*TC + col] = (float)v*s + z;
                }
            }
    }
    return out;
}

// ── Q8_0 tile dequant (8704 B/row: 512 B bf16 scales + 8192 signed INT8) ──
// Shared-expert and attention-projection tensors use this; layout mirrors
// dequant_q8_0_to_float_ex in dequant_q4nx.cpp.
static float* dequant_q8_0(const uint8_t* data, int i8_rows, int in_features,
                           int* out_rows, int* out_cols) {
    constexpr int TR = 32, TC = 256, Q8_0_ROW_BYTES = 8704;
    int ntc = in_features / TC, ntr = i8_rows / ntc;
    *out_rows = ntr * TR; *out_cols = ntc * TC;
    float* out = (float*)calloc((size_t)(*out_rows) * (*out_cols), sizeof(float));
    if (!out) return nullptr;
    for (int ir = 0; ir < i8_rows; ir++) {
        const uint8_t* t = data + (size_t)ir * Q8_0_ROW_BYTES;
        const uint16_t* sc = (const uint16_t*)t;
        const int8_t* vals = (const int8_t*)(t + 512);
        int tr_ = ir / ntc, tc_ = ir % ntc;
        for (int r = 0; r < TR; r++)
            for (int g = 0; g < TC / 32; g++) {
                float s = bf16f(sc[g*TR + r]);
                if (!std::isfinite(s) || std::fabs(s) > 100.0f) s = 0.0f;
                for (int i = 0; i < 32; i++) {
                    int col = g*32 + i;
                    out[((size_t)tr_*TR + r) * (*out_cols) + (size_t)tc_*TC + col] =
                        (float)vals[r*TC + col] * s;
                }
            }
    }
    return out;
}

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
    // gate/up: per expert [out=IM_EXP, in=H] → 16×8 = 128 i8 tiles/expert × 256 experts
    // down:    per expert [out=H, in=IM_EXP] → 64×2 = 128 i8 tiles/expert × 256 experts
    // dequant's in_features must be the GEMM INPUT dim (K): H for gate/up, IM_EXP for down.
    const int TR_G = 4096 * 8, TR_D = 16384 * 2;
    int er, ec, dr, dc;
    float* gate_f = dequant_1bp(i8p(exp_off[0]), TR_G, H, &er, &ec);
    float* up_f   = dequant_1bp(i8p(exp_off[1]), TR_G, H, &er, &ec);
    float* down_f = dequant_1bp(i8p(exp_off[2]), TR_D, IM_EXP, &dr, &dc);
    if (!gate_f || !up_f || !down_f) { fprintf(stderr, "expert dequant failed\n"); return 1; }
    printf("gate_exps dequant: [%d, %d]  down: [%d, %d]\n", er, ec, dr, dc);
    float* shg_f[3];
    for (int t = 0; t < 3; t++) {
        int s_tr = t == 2 ? 64 * 2 : 16 * 8;         // down is 64×2 tiles, gate/up 16×8
        int s_r, s_c;
        // Shared expert tensors are Q8_0 (8704 B/row), not Q4NX — see header shapes.
        shg_f[t] = dequant_q8_0(i8p(sh_off[t]), s_tr,
                                t == 2 ? IM_EXP : H, &s_r, &s_c);
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

    // ── NPU path: init GEMM contexts with MoE dims ──
    // Uses each xclbin's OWN instruction stream (aiecc --aie-generate-npu-insts
    // sidecar). Host-generated streams (gemm_generate_sequence_i8) don't match
    // the aiecc-compiled partition format — the DPU never starts (#1466).
    try {
        xrt::device dev(0);
        const int XM = 128;  // batch tile (we run M=1)
        auto init_ctx = [&](I8Ctx& c, const char* xp, const char* ip,
                            int K, int N) -> bool {
            c.MD = XM; c.KD = K; c.ND = N;
            return c.init(dev, xp, ip, 4, 1);
        };
        // GU concat: K=H, N=TOP_K*2*IM_EXP (8 experts × fused gate+up)
        I8Ctx cg;
        if (!init_ctx(cg, (xd + "/final_i8_MOE_GU_qwen3.6-moe_35b.xclbin").c_str(),
                      (xd + "/insts_i8_MOE_GU_qwen3.6-moe_35b.txt").c_str(),
                      H, TOP_K * 2 * IM_EXP))
            { fprintf(stderr, "FAIL GU ctx\n"); return 1; }
        // D concat: K=TOP_K*IM_EXP, N=H
        I8Ctx cd;
        if (!init_ctx(cd, (xd + "/final_i8_MOE_D_qwen3.6-moe_35b.xclbin").c_str(),
                      (xd + "/insts_i8_MOE_D_qwen3.6-moe_35b.txt").c_str(),
                      TOP_K * IM_EXP, H))
            { fprintf(stderr, "FAIL D ctx\n"); return 1; }
        // shared expert: fused GU [H, 2*IM_EXP] and D [IM_EXP, H]
        I8Ctx csg, csd;
        if (!init_ctx(csg, (xd + "/final_i8_MOE_SGU_qwen3.6-moe_35b.xclbin").c_str(),
                      (xd + "/insts_i8_MOE_SGU_qwen3.6-moe_35b.txt").c_str(),
                      H, 2 * IM_EXP))
            { fprintf(stderr, "FAIL shared GU ctx\n"); return 1; }
        if (!init_ctx(csd, (xd + "/final_i8_MOE_SD_qwen3.6-moe_35b.xclbin").c_str(),
                      (xd + "/insts_i8_MOE_SD_qwen3.6-moe_35b.txt").c_str(),
                      IM_EXP, H))
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
        // Each GEMM gets a dynamic activation scale (engine's dynamic_ascale);
        // a fixed 1.0 zeroed the D input (su ~0.05 → int8 0) and gave the
        // GU stage ~18% quant noise.
        auto dyn_ascale = [](const float* x, int n) {
            float amax = 0;
            for (int i = 0; i < n; i++) { float a = fabsf(x[i]); if (std::isfinite(a) && a > amax) amax = a; }
            return (amax < 1e-12f ? 1.0f : amax) / 127.0f;
        };
        std::vector<float> gu_out(TOP_K * 2 * IM_EXP), su(TOP_K * IM_EXP), d_out(H), npu_out(H);
        float ag = dyn_ascale(x.data(), H);
        cg.go(0, x.data(), 1, H, ag, gu_sc, gu_out.data(), TOP_K * 2 * IM_EXP);
        for (int e = 0; e < TOP_K; e++)
            for (int i = 0; i < IM_EXP; i++) {
                float gv = gu_out[e * 2 * IM_EXP + i];
                su[e * IM_EXP + i] = (gv / (1.0f + expf(-gv))) * gu_out[e * 2 * IM_EXP + IM_EXP + i] * probs[topk[e]];
            }
        float asu = dyn_ascale(su.data(), TOP_K * IM_EXP);
        cd.go(0, su.data(), 1, TOP_K * IM_EXP, asu, d_sc, d_out.data(), H);
        // shared expert
        std::vector<float> sg_out(2 * IM_EXP), ssu(IM_EXP), sh_out(H);
        float asg = dyn_ascale(x.data(), H);
        csg.go(0, x.data(), 1, H, asg, sg_sc, sg_out.data(), 2 * IM_EXP);
        for (int i = 0; i < IM_EXP; i++) {
            float gv = sg_out[i];
            ssu[i] = (gv / (1.0f + expf(-gv))) * sg_out[IM_EXP + i];
        }
        float assu = dyn_ascale(ssu.data(), IM_EXP);
        csd.go(0, ssu.data(), 1, IM_EXP, assu, sd_sc, sh_out.data(), H);
        double sg = 0;
        for (int i = 0; i < H; i++) sg += (double)x[i] * shgate[i];
        float sg_sig = 1.0f / (1.0f + expf(-(float)sg));
        for (int i = 0; i < H; i++) npu_out[i] = d_out[i] + sg_sig * sh_out[i];

        // ── CPU INT8-pipeline simulation (the honest gate) ──
        // Replicates exactly what the NPU computes: packB-quantized weights
        // (read back from the BOs), quantize_async activations, int32 dot
        // products, mean-group-scale dequant. If the NPU matches this, it is
        // doing the intended int8 math; residual error vs the f32 reference
        // is quantization, documented by the sim-vs-ref correlation.
        {
            auto sim_gemm = [](const int8_t* Aq, int K, const int8_t* Bq,
                               const std::vector<float>& gsc, float ascale,
                               float* out, int N) {
                float ssum = 0; for (float s : gsc) ssum += s;
                float bscale = ssum / gsc.size();
                for (int n = 0; n < N; n++) {
                    int64_t acc = 0;
                    for (int k = 0; k < K; k++) acc += (int64_t)Aq[k] * Bq[(size_t)k * N + n];
                    out[n] = (float)acc * ascale * bscale;
                }
            };
            auto quant_a = [](const float* x, int n, float ascale) {
                std::vector<int8_t> q(n);
                float is = 1.0f / ascale;
                for (int i = 0; i < n; i++) {
                    int v = (int)roundf(x[i] * is);
                    q[i] = (int8_t)(v > 127 ? 127 : v < -127 ? -127 : v);
                }
                return q;
            };
            // GU stage
            std::vector<int8_t> xq = quant_a(x.data(), H, ag);
            const int8_t* guB = (const int8_t*)cg.layerB[0]->map();
            std::vector<float> gu_sim(TOP_K * 2 * IM_EXP), su_sim(TOP_K * IM_EXP);
            sim_gemm(xq.data(), H, guB, cg.group_scales[0], ag, gu_sim.data(), TOP_K * 2 * IM_EXP);
            for (int e = 0; e < TOP_K; e++)
                for (int i = 0; i < IM_EXP; i++) {
                    float gv = gu_sim[e * 2 * IM_EXP + i];
                    su_sim[e * IM_EXP + i] = (gv / (1.0f + expf(-gv))) * gu_sim[e * 2 * IM_EXP + IM_EXP + i] * probs[topk[e]];
                }
            // D stage
            std::vector<int8_t> suq = quant_a(su_sim.data(), TOP_K * IM_EXP, asu);
            const int8_t* dB = (const int8_t*)cd.layerB[0]->map();
            std::vector<float> d_sim(H);
            sim_gemm(suq.data(), TOP_K * IM_EXP, dB, cd.group_scales[0], asu, d_sim.data(), H);
            // shared stages
            const int8_t* sgB = (const int8_t*)csg.layerB[0]->map();
            std::vector<float> sg_sim(2 * IM_EXP), ssu_sim(IM_EXP);
            sim_gemm(xq.data(), H, sgB, csg.group_scales[0], asg, sg_sim.data(), 2 * IM_EXP);
            for (int i = 0; i < IM_EXP; i++) {
                float gv = sg_sim[i];
                ssu_sim[i] = (gv / (1.0f + expf(-gv))) * sg_sim[IM_EXP + i];
            }
            std::vector<int8_t> ssuq = quant_a(ssu_sim.data(), IM_EXP, assu);
            const int8_t* sdB = (const int8_t*)csd.layerB[0]->map();
            std::vector<float> sh_sim(H);
            sim_gemm(ssuq.data(), IM_EXP, sdB, csd.group_scales[0], assu, sh_sim.data(), H);
            std::vector<float> npu_sim(H);
            for (int i = 0; i < H; i++) npu_sim[i] = d_sim[i] + sg_sig * sh_sim[i];
            double snum = 0, sden = 0;
            for (int i = 0; i < H; i++) { double d = npu_out[i] - npu_sim[i]; snum += d*d; sden += (double)npu_sim[i]*npu_sim[i]; }
            double sim_rmse = sqrt(snum / sden);
            // sim vs f32 ref correlation (documents quantization error)
            double sa = 0, sb = 0, sab = 0, saa = 0, sbb = 0;
            for (int i = 0; i < H; i++) { sa += npu_sim[i]; sb += ref_out[i]; }
            sa /= H; sb /= H;
            for (int i = 0; i < H; i++) { double da = npu_sim[i]-sa, db = ref_out[i]-sb; saa += da*da; sbb += db*db; sab += da*db; }
            double sim_corr = sab / sqrt(saa * sbb);
            printf("npu_out[0..4]  = %.4f %.4f %.4f %.4f %.4f\n", npu_out[0], npu_out[1], npu_out[2], npu_out[3], npu_out[4]);
            printf("npu_sim[0..4]  = %.4f %.4f %.4f %.4f %.4f\n", npu_sim[0], npu_sim[1], npu_sim[2], npu_sim[3], npu_sim[4]);
            printf("ref_out[0..4]  = %.4f %.4f %.4f %.4f %.4f\n", ref_out[0], ref_out[1], ref_out[2], ref_out[3], ref_out[4]);
            printf("NPU vs CPU-int8-sim: rel RMSE = %.6f  (%s)\n", sim_rmse,
                   sim_rmse < 0.05 ? "PASS — NPU matches the int8 pipeline" : "FAIL — NPU deviates from int8 math");
            printf("sim vs f32 ref: corr = %.4f  (quantization error; 1.0 = lossless)\n", sim_corr);
            return sim_rmse < 0.05 ? 0 : 2;
        }

    } catch (std::exception& ex) {
        fprintf(stderr, "XRT error: %s\n", ex.what());
        return 1;
    }
}
