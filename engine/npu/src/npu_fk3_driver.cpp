// npu_fk3_driver.cpp — see npu_fk3_driver.h.
//
// Buffer map, taken from the two verified benches (bench_ngrr_bf16 for launch A,
// bench_fk3_layer for launch B). group_id(n) is XRT's per-argument memory bank.
//
//   launch A   bI(1, cacheable)  bA(3) (M+1,H) f32   bW(4) (H,NQKV) bf16
//              bAN(5) (M,H) bf16 (unused, zeroed)     bC(6) (M,NQKV) bf16
//   launch B   bI(1, cacheable)  bA(3) (M+1,H) f32   bW(4) (H,NQKV) bf16 (unused)
//              bAN(5)  bQ(6) (M,NQKV) bf16   bO(7) (NH,M,HD) bf16
//              bWO(8) (NH*HD,NO) bf16         bC(9) (M,NO) bf16
//              bA2(10) (M+1,H) f32 (rows 0..M-1 are OUTPUT: the O-proj's f32 o;
//                                   only row M is host-provided = FFN norm gamma)
//              bAN2(11) bW2(12) (H,2*IM) bf16  bC2(13) (M,2*IM) bf16
//              bSL(14) (M,IM) bf16  bWD(15) (IM+H,H) bf16  = [W_D ; I]
//              bCD(16) (M,H) bf16 = LAYER OUTPUT   bHBF(17) (M,H) bf16
#include "npu_fk3_driver.h"
#include "npu_fk3_rope.h"

#include <xrt/xrt_bo.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_kernel.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// The engine's own dequantizers (npu_engine_bf16_mm_bridge.cpp). Using these is what
// makes dequant parity a construction rather than a verification: launch A and B see
// exactly the weights the per-op path would have used.
extern "C" void bf16mm_dequant(uint16_t* wout, const uint8_t* q4nx, uint32_t D_in,
                               uint32_t D_out, uint32_t q4nx_weight_offset);
extern "C" void bf16mm_dequant_mode(uint16_t* wout, const uint8_t* q4nx, uint32_t D_in,
                                   uint32_t D_out, uint32_t q4nx_weight_offset, int mode);

namespace fk3 {

namespace {

std::vector<uint32_t> read_words(const char* path) {
    std::vector<uint32_t> v;
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[fk3] cannot open %s\n", path); return v; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0 || (sz % 4) != 0) { fclose(f); return v; }
    v.resize((size_t)sz / 4);
    if (fread(v.data(), 4, v.size(), f) != v.size()) v.clear();
    fclose(f);
    return v;
}

std::vector<char> read_file(const char* path) {
    std::vector<char> v;
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[fk3] cannot open %s\n", path); return v; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0) { fclose(f); return v; }
    v.resize((size_t)sz);
    if (fread(v.data(), 1, v.size(), f) != v.size()) v.clear();
    fclose(f);
    return v;
}

}  // namespace

struct FusedLayer::Impl {
    // geometry
    int M = 0, H = 0, NH = 0, NKV = 0, HD = 0, IM = 0, NC = 0;
    int NQKV = 0, KOFF = 0, VOFF = 0, qout = 0, NO = 0, N2 = 0, NI = 0, ND = 0;
    int KO = 64, n_k = 0;

    xrt::device dev;

    // launch A
    xrt::kernel krA;
    xrt::bo iA, aA, wA, anA, cA;
    std::unique_ptr<xrt::hw_context> hwA;
    size_t insA_words = 0;

    // launch B
    xrt::kernel krB;
    xrt::bo iB, aB, wB, anB, qB, oB, woB, cB, a2B, an2B, w2B, c2B, slB, wdB, cdB, hbfB;
    std::unique_ptr<xrt::hw_context> hwB;
    size_t insB_words = 0;

    // per-layer weights (launch A needs W_QKV; launch B needs W_O, W2, WD)
    std::vector<xrt::bo> wQKV, wO, w2, wd;
    std::vector<uint8_t> wQKV_ready, wO_ready, w2_ready, wd_ready;
};

FusedLayer::FusedLayer() : p(new Impl) {}
FusedLayer::~FusedLayer() = default;
int FusedLayer::M() const { return p->M; }

bool FusedLayer::init(int device_index, const char* xclbinA, const char* instsA,
                      const char* xclbinB, const char* instsB, int M, int H, int NH,
                      int NKV, int HD, int IM, int NC) {
    Impl& s = *p;
    s.M = M; s.H = H; s.NH = NH; s.NKV = NKV; s.HD = HD; s.IM = IM; s.NC = NC;
    s.qout = NH * HD;
    s.KOFF = s.qout;
    s.VOFF = s.qout + NKV * HD;
    s.NQKV = s.qout + 2 * NKV * HD;
    s.NO = H;                 // O-proj output width
    s.NI = IM;                // SiLU width
    s.N2 = 2 * IM;            // GU width (gate then up)
    s.ND = H;                 // D output width = H (the identity block's height)
    s.n_k = H / s.KO;
    if (H % s.KO) { fprintf(stderr, "[fk3] H=%d not a multiple of KO=%d\n", H, s.KO); return false; }

    auto ins_a = read_words(instsA);
    auto ins_b = read_words(instsB);
    auto xb_a = read_file(xclbinA);
    auto xb_b = read_file(xclbinB);
    if (ins_a.empty() || ins_b.empty() || xb_a.empty() || xb_b.empty()) return false;
    s.insA_words = ins_a.size();
    s.insB_words = ins_b.size();

    s.dev = xrt::device(device_index);

    // ---- launch A -------------------------------------------------------------
    {
        xrt::xclbin xc{xb_a};
        s.dev.register_xclbin(xc);
        s.hwA.reset(new xrt::hw_context(s.dev, xc.get_uuid()));
        s.krA = xrt::kernel(*s.hwA, "MLIR_AIE");
        s.iA = xrt::bo(s.dev, ins_a.size() * 4, XCL_BO_FLAGS_CACHEABLE, s.krA.group_id(1));
        s.aA = xrt::bo(s.dev, (size_t)(s.M + 1) * s.H * 4, XRT_BO_FLAGS_HOST_ONLY, s.krA.group_id(3));
        s.wA = xrt::bo(s.dev, (size_t)s.H * s.NQKV * 2, XRT_BO_FLAGS_HOST_ONLY, s.krA.group_id(4));
        s.anA = xrt::bo(s.dev, (size_t)s.M * s.H * 2, XRT_BO_FLAGS_HOST_ONLY, s.krA.group_id(5));
        s.cA = xrt::bo(s.dev, (size_t)s.M * s.NQKV * 2, XRT_BO_FLAGS_HOST_ONLY, s.krA.group_id(6));
        memcpy(s.iA.map(), ins_a.data(), ins_a.size() * 4);
        s.iA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        memset(s.anA.map(), 0, (size_t)s.M * s.H * 2);
        s.anA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }

    // ---- launch B -------------------------------------------------------------
    {
        xrt::xclbin xc{xb_b};
        s.dev.register_xclbin(xc);
        s.hwB.reset(new xrt::hw_context(s.dev, xc.get_uuid()));
        s.krB = xrt::kernel(*s.hwB, "MLIR_AIE");
        const size_t sAN = (size_t)s.n_k * s.M * s.KO * 2;
        s.iB = xrt::bo(s.dev, ins_b.size() * 4, XCL_BO_FLAGS_CACHEABLE, s.krB.group_id(1));
        s.aB = xrt::bo(s.dev, (size_t)(s.M + 1) * s.H * 4, XRT_BO_FLAGS_HOST_ONLY, s.krB.group_id(3));
        s.wB = xrt::bo(s.dev, (size_t)s.H * s.NQKV * 2, XRT_BO_FLAGS_HOST_ONLY, s.krB.group_id(4));
        s.anB = xrt::bo(s.dev, sAN, XRT_BO_FLAGS_HOST_ONLY, s.krB.group_id(5));
        s.qB = xrt::bo(s.dev, (size_t)s.M * s.NQKV * 2, XRT_BO_FLAGS_HOST_ONLY, s.krB.group_id(6));
        s.oB = xrt::bo(s.dev, (size_t)s.NH * s.M * s.HD * 2, XRT_BO_FLAGS_HOST_ONLY, s.krB.group_id(7));
        s.woB = xrt::bo(s.dev, (size_t)s.qout * s.NO * 2, XRT_BO_FLAGS_HOST_ONLY, s.krB.group_id(8));
        s.cB = xrt::bo(s.dev, (size_t)s.M * s.NO * 2, XRT_BO_FLAGS_HOST_ONLY, s.krB.group_id(9));
        s.a2B = xrt::bo(s.dev, (size_t)(s.M + 1) * s.H * 4, XRT_BO_FLAGS_HOST_ONLY, s.krB.group_id(10));
        s.an2B = xrt::bo(s.dev, sAN, XRT_BO_FLAGS_HOST_ONLY, s.krB.group_id(11));
        s.w2B = xrt::bo(s.dev, (size_t)s.H * s.N2 * 2, XRT_BO_FLAGS_HOST_ONLY, s.krB.group_id(12));
        s.c2B = xrt::bo(s.dev, (size_t)s.M * s.N2 * 2, XRT_BO_FLAGS_HOST_ONLY, s.krB.group_id(13));
        s.slB = xrt::bo(s.dev, (size_t)s.M * s.NI * 2, XRT_BO_FLAGS_HOST_ONLY, s.krB.group_id(14));
        s.wdB = xrt::bo(s.dev, (size_t)(s.NI + s.H) * s.ND * 2, XRT_BO_FLAGS_HOST_ONLY, s.krB.group_id(15));
        s.cdB = xrt::bo(s.dev, (size_t)s.M * s.ND * 2, XRT_BO_FLAGS_HOST_ONLY, s.krB.group_id(16));
        s.hbfB = xrt::bo(s.dev, (size_t)s.M * s.H * 2, XRT_BO_FLAGS_HOST_ONLY, s.krB.group_id(17));
        memcpy(s.iB.map(), ins_b.data(), ins_b.size() * 4);
        s.iB.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        // launch B is built with NOQKV=1, so its QKV phases do not exist: bW/bAN are
        // never read and are left zeroed rather than uploaded per layer (this is what
        // keeps the resident weight footprint at ~0.9 GB instead of ~1.1 GB).
        memset(s.wB.map(), 0, (size_t)s.H * s.NQKV * 2);
        memset(s.anB.map(), 0, sAN);
        memset(s.an2B.map(), 0, sAN);
        s.wB.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        s.anB.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        s.an2B.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }

    // ---- per-layer weight BOs --------------------------------------------------
    s.wQKV.resize(NC); s.wO.resize(NC); s.w2.resize(NC); s.wd.resize(NC);
    s.wQKV_ready.assign(NC, 0); s.wO_ready.assign(NC, 0);
    s.w2_ready.assign(NC, 0); s.wd_ready.assign(NC, 0);
    for (int l = 0; l < NC; l++) {
        s.wQKV[l] = xrt::bo(s.dev, (size_t)s.H * s.NQKV * 2, XRT_BO_FLAGS_HOST_ONLY, s.krA.group_id(4));
        s.wO[l] = xrt::bo(s.dev, (size_t)s.qout * s.NO * 2, XRT_BO_FLAGS_HOST_ONLY, s.krB.group_id(8));
        s.w2[l] = xrt::bo(s.dev, (size_t)s.H * s.N2 * 2, XRT_BO_FLAGS_HOST_ONLY, s.krB.group_id(12));
        s.wd[l] = xrt::bo(s.dev, (size_t)(s.NI + s.H) * s.ND * 2, XRT_BO_FLAGS_HOST_ONLY, s.krB.group_id(15));
    }

    fprintf(stderr, "[fk3] init ok: M=%d H=%d NH=%d NKV=%d HD=%d IM=%d NC=%d "
                    "NQKV=%d KOFF=%d VOFF=%d (launch A %zu words, launch B %zu words)\n",
            s.M, s.H, s.NH, s.NKV, s.HD, s.IM, s.NC, s.NQKV, s.KOFF, s.VOFF,
            s.insA_words, s.insB_words);
    return true;
}

bool FusedLayer::prepare_layer(int l, const WeightSource& src) {
    Impl& s = *p;
    if (l < 0 || l >= s.NC || !src.bo || !src.offs) return false;
    const uint32_t row = src.byte_row;
    auto off = [&](int i) { return (uint32_t)src.offs[i] * row; };

    // QKV: (H, NQKV), the engine's own Wqkv.
    {
        std::vector<uint16_t> w((size_t)s.H * s.NQKV);
        bf16mm_dequant(w.data(), src.bo, (uint32_t)s.H, (uint32_t)s.NQKV, off(0));
        memcpy(s.wQKV[l].map(), w.data(), w.size() * 2);
        s.wQKV[l].sync(XCL_BO_SYNC_BO_TO_DEVICE);
        s.wQKV_ready[l] = 1;
    }
    // O-proj: (qout, H), the engine's own Wo.
    {
        std::vector<uint16_t> w((size_t)s.qout * s.NO);
        bf16mm_dequant(w.data(), src.bo, (uint32_t)s.qout, (uint32_t)s.NO, off(3));
        memcpy(s.wO[l].map(), w.data(), w.size() * 2);
        s.wO[l].sync(XCL_BO_SYNC_BO_TO_DEVICE);
        s.wO_ready[l] = 1;
    }
    // GU: the engine dequantizes GATE then UP into one (H, 2*IM) buffer, and that is
    // exactly the layout W2 wants — gate in columns [0,IM), up in [IM,2*IM).
    {
        std::vector<uint16_t> g((size_t)s.H * 2 * s.IM);
        bf16mm_dequant_mode(g.data(), src.bo, (uint32_t)s.H, (uint32_t)s.IM, off(4), 2);              // gate
        bf16mm_dequant_mode(g.data() + (size_t)s.H * s.IM, src.bo, (uint32_t)s.H, (uint32_t)s.IM, off(4), 1);  // up
        memcpy(s.w2[l].map(), g.data(), g.size() * 2);
        s.w2[l].sync(XCL_BO_SYNC_BO_TO_DEVICE);
        s.w2_ready[l] = 1;
    }
    // D: the engine's Wd is (IM, H); the fused kernel wants [W_D ; I] = (IM+H, H),
    // and that identity block IS residual 2 (A_D = [silu | h], so C_D = silu*W_D + h).
    {
        std::vector<uint16_t> w((size_t)(s.NI + s.H) * s.ND);
        bf16mm_dequant(w.data(), src.bo, (uint32_t)s.NI, (uint32_t)s.ND, off(5));
        for (int r = 0; r < s.H; r++)
            for (int n = 0; n < s.ND; n++)
                w[(size_t)(s.NI + r) * s.ND + n] = (uint16_t)(r == n ? 0x3F80 : 0x0000);  // bf16 1.0 / 0.0
        memcpy(s.wd[l].map(), w.data(), w.size() * 2);
        s.wd[l].sync(XCL_BO_SYNC_BO_TO_DEVICE);
        s.wd_ready[l] = 1;
    }
    return true;
}

bool FusedLayer::run(int l, const float* x, const float* gamma_in, const float* gamma_ffn,
                     int pos0, uint16_t* bKv, int kv_region, int v_add, float* out) {
    Impl& s = *p;
    if (l < 0 || l >= s.NC || !s.wQKV_ready[l] || !s.wO_ready[l] || !s.w2_ready[l] || !s.wd_ready[l]) {
        fprintf(stderr, "[fk3] layer %d not prepared\n", l);
        return false;
    }
    const int M = s.M;

    // ---- inputs ---------------------------------------------------------------
    // A rows 0..M-1 = x, row M = the input norm's gamma (the fused norm reads it as
    // a plain f32 row rather than holding it in the kernel).
    {
        float* a = (float*)s.aA.map();
        memcpy(a, x, (size_t)M * s.H * 4);
        memcpy(a + (size_t)M * s.H, gamma_in, (size_t)s.H * 4);
        s.aA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        float* a2 = (float*)s.a2B.map();
        memcpy(a2 + (size_t)M * s.H, gamma_ffn, (size_t)s.H * 4);   // rows 0..M-1 are outputs
        s.a2B.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }

    // ---- launch A: fused RMSNorm(input) + QKV --------------------------------
    {
        auto r = s.krA((unsigned)3, s.iA, (unsigned)s.insA_words, s.aA, s.wQKV[l], s.anA, s.cA);
        r.wait();
        s.cA.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        memcpy(s.qB.map(), s.cA.map(), (size_t)M * s.NQKV * 2);
        s.qB.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }

    // ---- host: RoPE on Q and K in place, then scatter K/V into the KV cache ----
    {
        uint16_t* q = (uint16_t*)s.qB.map();
        // Q at head stride HD from 0; K at KOFF + kh*HD; V untouched.
        fk3::rope_qk_bf16(q, M, s.NH, s.NKV, s.HD, 1e6f, pos0);
        s.qB.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        // Same layout qk_norm_pi writes: region = kvh<4?0:1, lh = kvh&3, slot = 4*HD.
        // K/V are already rotated here (the engine's cache holds rotated keys and the
        // decode kernel depends on that).
        const size_t slot = (size_t)4 * s.HD;
        for (int pi = 0; pi < M; pi++) {
            const uint16_t* row = q + (size_t)pi * s.NQKV;
            for (int kvh = 0; kvh < s.NKV; kvh++) {
                const int region = kvh < 4 ? 0 : 1, lh = kvh & 3;
                const uint16_t* k = row + s.KOFF + (size_t)kvh * s.HD;
                const uint16_t* v = row + s.VOFF + (size_t)kvh * s.HD;
                uint16_t* kd = bKv + (size_t)region * kv_region + (size_t)pi * slot + (size_t)lh * s.HD;
                uint16_t* vd = bKv + (size_t)(region + v_add) * kv_region + (size_t)pi * slot + (size_t)lh * s.HD;
                memcpy(kd, k, (size_t)s.HD * 2);
                memcpy(vd, v, (size_t)s.HD * 2);
            }
        }
    }

    // ---- launch B: attention + O-proj + FFN norm + GU + SiLU + D -------------
    {
        auto r = s.krB((unsigned)3, s.iB, (unsigned)s.insB_words, s.aB, s.wB, s.anB, s.qB, s.oB,
                       s.woB, s.cB, s.a2B, s.an2B, s.w2B, s.c2B, s.slB, s.wdB, s.cdB, s.hbfB);
        r.wait();
        s.cdB.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    }

    // ---- layer output: bf16 -> f32 for the next layer -------------------------
    {
        const uint16_t* cd = (const uint16_t*)s.cdB.map();
        for (size_t i = 0; i < (size_t)M * s.H; i++) {
            uint32_t v = (uint32_t)cd[i] << 16;
            float f;
            memcpy(&f, &v, 4);
            out[i] = f;
        }
    }
    return true;
}

}  // namespace fk3
