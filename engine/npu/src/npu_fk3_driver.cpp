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
#include <chrono>
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

    // The xclbin objects MUST outlive the hw_contexts and kernels built from them:
    // xrt::kernel/hw_context reference the xclbin (and the axlf buffer it owns), so a
    // local xclbin destroyed at the end of init() leaves them pointing at freed memory.
    // The working bench keeps its xclbin alive for the whole program, which is one of
    // the few things it does that this driver did not. Declared FIRST so it is destroyed
    // LAST, after the kernels and BOs that depend on it.
    std::unique_ptr<xrt::xclbin> xcA, xcB;
    // And the RAW BYTES the xclbins were parsed from must outlive them too: xrt::xclbin
    // is constructed from a byte buffer and holds a reference into it in this XRT, so
    // reading the file into a local vector in init() and letting it die on return leaves
    // every kernel built from that xclbin pointing at freed memory. That is what made
    // launch B return zeros (and occasionally hang) while the identical xclbin driven by
    // bench_fk3_layer - whose xb lives for the whole program - worked perfectly.
    std::vector<char> xbA_bytes, xbB_bytes;

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
    s.xbA_bytes = read_file(xclbinA);
    s.xbB_bytes = read_file(xclbinB);
    const std::vector<char>& xb_a = s.xbA_bytes;
    const std::vector<char>& xb_b = s.xbB_bytes;
    if (ins_a.empty() || ins_b.empty() || xb_a.empty() || xb_b.empty()) return false;
    s.insA_words = ins_a.size();
    s.insB_words = ins_b.size();

    s.dev = xrt::device(device_index);

    // ---- launch A -------------------------------------------------------------
    // Skipped entirely under NPU_FK3_SKIP_A so a run can isolate whether merely
    // having a SECOND hw_context (and a second registered xclbin) is what breaks
    // launch B's dispatch.
    if (!getenv("NPU_FK3_SKIP_A")) {
        s.xcA.reset(new xrt::xclbin(xb_a));
        s.dev.register_xclbin(*s.xcA);
        s.hwA.reset(new xrt::hw_context(s.dev, s.xcA->get_uuid()));
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
        s.xcB.reset(new xrt::xclbin(xb_b));
        s.dev.register_xclbin(*s.xcB);
        s.hwB.reset(new xrt::hw_context(s.dev, s.xcB->get_uuid()));
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
        // bench_fk3_layer's exact initial state for the rest: everything zeroed, and the
        // QKV weight filled with pseudo-random values even though the emitted MLIR calls
        // no QKV phase (nq_acc_mac x3 = O-proj + GU + D). If the kernel consumes this BO
        // anyway, a zeroed one would silently zero the whole layer; the bench never
        // leaves it zero, which is one difference this driver introduced on its own.
        memset(s.oB.map(), 0, (size_t)s.NH * s.M * s.HD * 2);
        memset(s.cB.map(), 0, (size_t)s.M * s.NO * 2);
        memset(s.c2B.map(), 0, (size_t)s.M * s.N2 * 2);
        memset(s.slB.map(), 0, (size_t)s.M * s.NI * 2);
        memset(s.cdB.map(), 0, (size_t)s.M * s.ND * 2);
        memset(s.hbfB.map(), 0, (size_t)s.M * s.H * 2);
        memset(s.qB.map(), 0, (size_t)s.M * s.NQKV * 2);
        if (getenv("NPU_FK3_RANDOM_WB")) {
            uint16_t* wb = (uint16_t*)s.wB.map();
            for (size_t i = 0; i < (size_t)s.H * s.NQKV; i++) {
                float v = (float)((int)(i % 13) - 6) * 0.05f;
                uint32_t u; memcpy(&u, &v, 4);
                wb[i] = (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1)) >> 16);
            }
        }
        s.wB.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        s.anB.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        s.an2B.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        s.oB.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        s.cB.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        s.c2B.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        s.slB.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        s.cdB.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        s.hbfB.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        s.qB.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }

    // ---- per-layer weight BOs --------------------------------------------------
    s.wQKV.resize(NC); s.wO.resize(NC); s.w2.resize(NC); s.wd.resize(NC);
    s.wQKV_ready.assign(NC, 0); s.wO_ready.assign(NC, 0);
    s.w2_ready.assign(NC, 0); s.wd_ready.assign(NC, 0);
    for (int l = 0; l < NC; l++) {
        if (!getenv("NPU_FK3_SKIP_A"))
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


// Upload verification: compares what map() shows against the host array that was just
// copied in. This is the only trustworthy check here - a probe that reinterprets bytes
// can mislead, but memcmp of the exact bytes cannot.
static void verify_upload(const char* name, xrt::bo& b, const void* host, size_t bytes) {
    int rc = memcmp(b.map(), host, bytes);
    fprintf(stderr, "[fk3] upload %s: %zu bytes, memcmp=%s\n", name, bytes, rc == 0 ? "MATCH" : "DIFFER");
}

bool FusedLayer::prepare_random(int l) {
    Impl& s = *p;
    if (l < 0 || l >= s.NC) return false;
    auto rne = [](float f) -> uint16_t {
        uint32_t u; memcpy(&u, &f, 4);
        return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1)) >> 16);
    };
    // Same formulas as bench_fk3_layer.cpp, element index i over each buffer.
    {
        std::vector<uint16_t> w((size_t)s.qout * s.NO);
        for (size_t i = 0; i < w.size(); i++) w[i] = rne((float)((int)(i % 11) - 5) * 0.05f);   // (int) is load-bearing: size_t would wrap
        memcpy(s.wO[l].map(), w.data(), w.size() * 2);
        s.wO[l].sync(XCL_BO_SYNC_BO_TO_DEVICE);
        s.wO_ready[l] = 1;
    }
    {
        std::vector<uint16_t> w((size_t)s.H * s.N2);
        for (size_t i = 0; i < w.size(); i++) w[i] = rne((float)((int)(i % 13) - 6) * 0.05f);   // (int) is load-bearing: size_t would wrap
        memcpy(s.w2[l].map(), w.data(), w.size() * 2);
        s.w2[l].sync(XCL_BO_SYNC_BO_TO_DEVICE);
        if (getenv("NPU_FK3_DUMP")) verify_upload("w2", s.w2[l], w.data(), w.size() * 2);
        s.w2_ready[l] = 1;
    }
    {
        std::vector<uint16_t> w((size_t)(s.NI + s.H) * s.ND);
        for (size_t i = 0; i < (size_t)s.NI * s.ND; i++) w[i] = rne((float)((int)(i % 9) - 4) * 0.05f);    // (int) is load-bearing: size_t would wrap
        for (int r = 0; r < s.H; r++)
            for (int n = 0; n < s.ND; n++)
                w[(size_t)(s.NI + r) * s.ND + n] = (uint16_t)(r == n ? 0x3F80 : 0x0000);
        memcpy(s.wd[l].map(), w.data(), w.size() * 2);
        s.wd[l].sync(XCL_BO_SYNC_BO_TO_DEVICE);
        s.wd_ready[l] = 1;
    }
    s.wQKV_ready[l] = 1;   // never consumed by launch B (QKV phases are dropped)
    return true;
}

// Empirical weight override. The engine's upload reorders every weight it is handed, so the
// dequantized array is not what its GEMM multiplies by (measured: the effective weight is a
// permutation of the raw one - A @ W_eff matches the engine's own QKV buffer to 0.43% while
// A @ W_raw misses by 129%). A weight recovered by the calibration solve is fed here.
// See FUSED-RMSNORM-QKV-DESIGN.md. `count` limits how many leading elements are replaced, so a
// weight that has extra structure appended after the dequant (W_D's identity block) keeps it.
// Returns true only on a full-length read.
static bool fk3_maybe_override(std::vector<uint16_t>& w, size_t count,
                               const char* envname, const char* label, int l) {
    const char* path = getenv(envname);
    if (!path) return false;
    if (count > w.size()) count = w.size();
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "[fk3] %s: cannot open %s\n", label, path); return false; }
    size_t got = fread(w.data(), 2, count, f);
    fclose(f);
    if (got != count) {
        fprintf(stderr, "[fk3] %s: SHORT read %zu of %zu from %s - ignoring\n", label, got, count, path);
        return false;
    }
    if (l == 0)
        fprintf(stderr, "[fk3] %s override: loaded %zu bf16 from %s\n", label, got, path);
    return true;
}

bool FusedLayer::prepare_layer(int l, const WeightSource& src) {
    Impl& s = *p;
    if (l < 0 || l >= s.NC || !src.bo || !src.offs) return false;
    const uint32_t row = src.byte_row;
    auto off = [&](int i) { return (uint32_t)src.offs[i] * row; };

    // QKV: (H, NQKV), the engine's own Wqkv.
    if (getenv("NPU_FK3_SKIP_A")) {
        s.wQKV_ready[l] = 1;   // not built and never read - launch B drops the QKV phases
    } else {
        std::vector<uint16_t> w((size_t)s.H * s.NQKV);
        bf16mm_dequant(w.data(), src.bo, (uint32_t)s.H, (uint32_t)s.NQKV, off(0));
        fk3_maybe_override(w, w.size(), "NPU_FK3_WQKV_FROM", "WQKV", l);
        if (l == 0 && getenv("NPU_FK3_DUMP")) {
            FILE* f = fopen("/tmp/fk3_w_wqkv.bin", "wb");
            if (f) { fwrite(w.data(), 2, w.size(), f); fclose(f); }
        }
        memcpy(s.wQKV[l].map(), w.data(), w.size() * 2);
        s.wQKV[l].sync(XCL_BO_SYNC_BO_TO_DEVICE);
        s.wQKV_ready[l] = 1;
    }
    // O-proj: (qout, H), the engine's own Wo.
    {
        std::vector<uint16_t> w((size_t)s.qout * s.NO);
        bf16mm_dequant(w.data(), src.bo, (uint32_t)s.qout, (uint32_t)s.NO, off(3));
        fk3_maybe_override(w, w.size(), "NPU_FK3_WO_FROM", "WO", l);
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
        fk3_maybe_override(g, g.size(), "NPU_FK3_WGU_FROM", "WGU", l);
        memcpy(s.w2[l].map(), g.data(), g.size() * 2);
        s.w2[l].sync(XCL_BO_SYNC_BO_TO_DEVICE);
        if (l == 0 && getenv("NPU_FK3_DUMP")) {
            FILE* f = fopen("/tmp/fk3_w_w2.bin", "wb");
            if (f) { fwrite(g.data(), 2, g.size(), f); fclose(f); }
            FILE* f2 = fopen("/tmp/fk3_w_wo.bin", "wb");
            if (f2) { fwrite(s.wO[l].map(), 2, (size_t)s.qout * s.NO, f2); fclose(f2); }
        }
        s.w2_ready[l] = 1;
    }
    // D: the engine's Wd is (IM, H); the fused kernel wants [W_D ; I] = (IM+H, H),
    // and that identity block IS residual 2 (A_D = [silu | h], so C_D = silu*W_D + h).
    {
        std::vector<uint16_t> w((size_t)(s.NI + s.H) * s.ND);
        bf16mm_dequant(w.data(), src.bo, (uint32_t)s.NI, (uint32_t)s.ND, off(5));
        fk3_maybe_override(w, (size_t)s.NI * s.ND, "NPU_FK3_WD_FROM", "WD", l);
        for (int r = 0; r < s.H; r++)
            for (int n = 0; n < s.ND; n++)
                w[(size_t)(s.NI + r) * s.ND + n] = (uint16_t)(r == n ? 0x3F80 : 0x0000);  // bf16 1.0 / 0.0
        if (l == 0 && getenv("NPU_FK3_DUMP")) {
            FILE* f = fopen("/tmp/fk3_w_wd.bin", "wb");
            if (f) { fwrite(w.data(), 2, w.size(), f); fclose(f); }
        }
        memcpy(s.wd[l].map(), w.data(), w.size() * 2);
        s.wd[l].sync(XCL_BO_SYNC_BO_TO_DEVICE);
        s.wd_ready[l] = 1;
    }
    return true;
}

bool FusedLayer::run(int l, const float* x, const float* gamma_in, const float* gamma_ffn,
                     int nrow, int pos0, uint16_t* bKv, int kv_region, int v_add, float* out,
                     float* kvf_k, float* kvf_v, const float* qn, const float* kn) {
    Impl& s = *p;
    if (l < 0 || l >= s.NC || !s.wQKV_ready[l] || !s.wO_ready[l] || !s.w2_ready[l] || !s.wd_ready[l]) {
        fprintf(stderr, "[fk3] layer %d not prepared\n", l);
        return false;
    }
    const int M = s.M;
    if (l == 0 && getenv("NPU_FK3_DUMP")) {
        // If two BOs map to the same host address, they are the same memory and every
        // write to one clobbers the other - which is what identical garbage values in
        // different buffers (W2 and WD sharing the exact same maxabs) would mean.
        fprintf(stderr, "[fk3] BO maps: aB=%p a2B=%p wB=%p wO[0]=%p w2[0]=%p wd[0]=%p\n",
                (void*)s.aB.map(), (void*)s.a2B.map(), (void*)s.wB.map(),
                (void*)s.wO[l].map(), (void*)s.w2[l].map(), (void*)s.wd[l].map());
        // Is map() stable across calls? The bench calls it once per BO and keeps the
        // pointer; this driver calls it inline at every write. If the address moves, every
        // write lands in one mapping while sync() transfers another - which is exactly the
        // symptom (host mapping unchanged, device full of garbage, a2B's one-shot write
        // being the only one that survived).
        fprintf(stderr, "[fk3] map() repeat: aB %p %p %p | w2 %p %p %p\n",
                (void*)s.aB.map(), (void*)s.aB.map(), (void*)s.aB.map(),
                (void*)s.w2[l].map(), (void*)s.w2[l].map(), (void*)s.w2[l].map());
        fprintf(stderr, "[fk3] BO sizes: aB=%llu a2B=%llu w2=%llu wd=%llu\n",
                (unsigned long long)s.aB.size(), (unsigned long long)s.a2B.size(),
                (unsigned long long)s.w2[l].size(), (unsigned long long)s.wd[l].size());
        fflush(stderr);
    }
    if (nrow <= 0 || nrow > M) nrow = M;

    // ---- inputs ---------------------------------------------------------------
    // BOTH launches have their own A (group 3). Launch A's A is the layer input x +
    // the input norm's gamma in row M. Launch B's A is the SAME x (its add-aware FFN
    // norm reduces x against A2 and has no separate x fifo) plus a gamma row that the
    // FFN norm does not read (it takes its gamma from A2's row M). Forgetting launch
    // B's A is a silent all-zero layer: silu=0, h=0, so D = 0*W_D + h = 0.
    {
        // Launch A's A only exists when launch A does: under NPU_FK3_SKIP_A its BOs are
        // never created, and calling map() on a default-constructed xrt::bo is UB (this
        // is what made the standalone isolation run hang rather than report).
        if (!getenv("NPU_FK3_SKIP_A")) {
            float* a = (float*)s.aA.map();
            memcpy(a, x, (size_t)M * s.H * 4);
            memcpy(a + (size_t)M * s.H, gamma_in, (size_t)s.H * 4);
            s.aA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }
        memcpy(s.aB.map(), (const void*)x, (size_t)M * s.H * 4);
        memcpy((char*)s.aB.map() + (size_t)M * s.H * 4, gamma_in, (size_t)s.H * 4);
        s.aB.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        float* a2 = (float*)s.a2B.map();
        memcpy(a2 + (size_t)M * s.H, gamma_ffn, (size_t)s.H * 4);   // rows 0..M-1 are outputs
        s.a2B.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }

    if (getenv("NPU_FK3_SKIP_A")) {
        // Decisive isolation: drive launch B exactly as the bench does - its own
        // pseudo-random A (non-zero), gamma rows 1.0, pseudo-random QKV - and skip
        // launch A entirely. Built into a LOCAL buffer because launch A's BOs do not
        // exist in this mode.
        std::vector<float> a((size_t)(M + 1) * s.H);
        for (size_t i = 0; i < (size_t)M * s.H; i++) a[i] = (float)((int)(i % 61) - 30) * 0.02f;       // (int) is load-bearing: size_t would wrap
        for (int i = 0; i < s.H; i++) a[(size_t)M * s.H + i] = 1.0f;
        memcpy(s.aB.map(), a.data(), a.size() * sizeof(float));
        s.aB.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        if (getenv("NPU_FK3_DUMP")) verify_upload("aB(skipA)", s.aB, a.data(), a.size() * sizeof(float));
        float* a2 = (float*)s.a2B.map();
        for (size_t i = 0; i < (size_t)M * s.H; i++) a2[i] = 0.0f;
        for (int i = 0; i < s.H; i++) a2[(size_t)M * s.H + i] = 1.0f;
        s.a2B.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        uint16_t* q = (uint16_t*)s.qB.map();
        if (getenv("NPU_FK3_ZERO_Q")) {
            // The bench never fills bQ: it memsets it, so its attention runs on Q=0 and
            // it STILL produces 98.4% correct layer output via the residual path. Running
            // the driver under the same condition is the tightest A/B available: if CD is
            // non-zero here the invocation is sound and only the Q-dependent path differs;
            // if CD is still zero, the invocation itself is the problem.
            memset(q, 0, (size_t)M * s.NQKV * 2);
        } else {
            for (size_t i = 0; i < (size_t)M * s.NQKV; i++) {
                float v = (float)((int)(i % 13) - 6) * 0.05f;
                uint32_t u; memcpy(&u, &v, 4);
                q[i] = (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1)) >> 16);
            }
        }
        s.qB.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }

    // ---- launch A: fused RMSNorm(input) + QKV --------------------------------
    if (!getenv("NPU_FK3_SKIP_A")) {
        if (l == 0 && getenv("NPU_FK3_DUMP")) {
            // Verify what LAUNCH A actually consumes. Every probe so far measured launch B's
            // A (s.aB) - the buffer launch A reads (s.aA, including its gamma row M) has
            // never been checked, and that is precisely the pair whose output is 2.51x off
            // versus the per-op path. Sync first: a pre-launch read without sync sees a
            // stale staging buffer, which has already misled me twice.
            s.aA.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            const float* a = (const float*)s.aA.map();
            auto st = [](const char* n, const float* v, int n_el) {
                double mx = 0, sum = 0;
                for (int i = 0; i < n_el; i++) { double q = v[i] < 0 ? -v[i] : v[i]; if (q > mx) mx = q; sum += q; }
                fprintf(stderr, "[fk3] %-22s maxabs=%.5f meanabs=%.5f first4=[%.4f %.4f %.4f %.4f]\n",
                        n, mx, sum / n_el, v[0], v[1], v[2], v[3]);
            };
            st("aA data (all M rows)", a, M * s.H);
            st("aA row M (gamma)", a + (size_t)M * s.H, s.H);
            FILE* fa = fopen("/tmp/fk3_drv_aA.bin", "wb");
            if (fa) { fwrite(s.aA.map(), 1, (size_t)(M + 1) * s.H * 4, fa); fclose(fa); }
            fprintf(stderr, "[fk3] aA bytes=%llu  (want %llu)\n",
                    (unsigned long long)s.aA.size(), (unsigned long long)((size_t)(M + 1) * s.H * 4));
        }
        auto t0 = std::chrono::steady_clock::now();
        auto r = s.krA((unsigned)3, s.iA, (unsigned)s.insA_words, s.aA, s.wQKV[l], s.anA, s.cA);
        r.wait();
        auto t1 = std::chrono::steady_clock::now();
        s.cA.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        if (l == 0 && getenv("NPU_FK3_TIMING"))
            fprintf(stderr, "[fk3] launch A (l=0): %.2f ms\n",
                    std::chrono::duration<double, std::milli>(t1 - t0).count());
        if (l == 0 && getenv("NPU_FK3_DUMP")) {
            FILE* f = fopen("/tmp/fk3_drv_A.bin", "wb");
            if (f) { fwrite(s.cA.map(), 2, (size_t)M * s.NQKV, f); fclose(f); }
        }
        memcpy(s.qB.map(), s.cA.map(), (size_t)M * s.NQKV * 2);
        s.qB.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    }

    // ---- host: RoPE on Q and K in place, then scatter K/V into the KV cache ----
    if (!getenv("NPU_FK3_SKIP_A")) {
        uint16_t* q = (uint16_t*)s.qB.map();
        // Q at head stride HD from 0; K at KOFF + kh*HD; V untouched.
        fk3::rope_qk_bf16(q, nrow, s.NH, s.NKV, s.HD, 1e6f, pos0, -1, qn, kn);
        s.qB.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        // Same layout qk_norm_pi writes: region = kvh<4?0:1, lh = kvh&3, slot = 4*HD.
        // K/V are already rotated here (the engine's cache holds rotated keys and the
        // decode kernel depends on that).
        const size_t slot = (size_t)4 * s.HD;
        for (int pi = 0; pi < nrow; pi++) {
            const uint16_t* row = q + (size_t)pi * s.NQKV;
            for (int kvh = 0; kvh < s.NKV; kvh++) {
                const int region = kvh < 4 ? 0 : 1, lh = kvh & 3;
                const uint16_t* k = row + s.KOFF + (size_t)kvh * s.HD;
                const uint16_t* v = row + s.VOFF + (size_t)kvh * s.HD;
                uint16_t* kd = bKv + (size_t)region * kv_region + (size_t)pi * slot + (size_t)lh * s.HD;
                uint16_t* vd = bKv + (size_t)(region + v_add) * kv_region + (size_t)pi * slot + (size_t)lh * s.HD;
                memcpy(kd, k, (size_t)s.HD * 2);
                memcpy(vd, v, (size_t)s.HD * 2);
                if (kvf_k && kvf_v) {
                    // Also fill the ENGINE's f32 KV cache - the buffer the decode reads.
                    // Same index the per-op path uses: (pos)*NKV*HD + kvh*HD.
                    float* fk = kvf_k + (size_t)(pos0 + pi) * s.NKV * s.HD + (size_t)kvh * s.HD;
                    float* fv = kvf_v + (size_t)(pos0 + pi) * s.NKV * s.HD + (size_t)kvh * s.HD;
                    for (int d = 0; d < s.HD; d++) {
                        uint32_t uk = (uint32_t)k[d] << 16, uv = (uint32_t)v[d] << 16;
                        float a, b; memcpy(&a, &uk, 4); memcpy(&b, &uv, 4);
                        fk[d] = a; fv[d] = b;
                    }
                }
            }
        }
    }

    // ---- launch B: attention + O-proj + FFN norm + GU + SiLU + D -------------
    {
        if (l == 0 && getenv("NPU_FK3_DUMP")) {
            // BEFORE the launch: distinguishes "my upload never reached the device" from
            // "the kernel overwrote its own input". Same probe runs after the launch.
            auto stat = [&](const char* n, xrt::bo& b, size_t elems, bool is_f32, bool do_sync) {
                if (do_sync) b.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                const float* fp = (const float*)b.map();
                const uint16_t* hp = (const uint16_t*)b.map();
                long nz = 0; double mx = 0.0;
                for (size_t i = 0; i < elems; i++) {
                    double v;
                    if (is_f32) v = fp[i];
                    else { uint32_t u = (uint32_t)hp[i] << 16; float t; memcpy(&t, &u, 4); v = t; }
                    if (v != 0.0) nz++;
                    if (v < 0) v = -v;
                    if (v > mx) mx = v;
                }
                fprintf(stderr, "[fk3] PRE  %-18s nonzero=%.4f maxabs=%.4f\n", n, (double)nz / (double)elems, mx);
            };
            stat("A  (launch B in)", s.aB, (size_t)(M + 1) * s.H, true, false);
            stat("A2 (O-proj f32)", s.a2B, (size_t)(M + 1) * s.H, true, false);
            stat("W2 (GU weight)", s.w2[l], (size_t)s.H * s.N2, false, false);
            stat("WD (D weight)", s.wd[l], (size_t)(s.NI + s.H) * s.ND, false, false);
            stat("WO (O weight)", s.wO[l], (size_t)s.qout * s.NO, false, false);
            fflush(stderr);
        }
        // NOTE: the PER-LAYER weight BOs (wO[l]/w2[l]/wd[l]) are the ones prepare_layer()
        // and prepare_random() fill. Passing the shared woB/w2B/wdB here instead - as this
        // driver did - hands the kernel three zeroed weight buffers: the GU GEMM emits
        // zeros, SiLU emits zeros, the add-aware norm divides by a zero variance, and the
        // layer output is zero. Launch A escaped it because its weight is already
        // per-layer (wQKV[l]), which is exactly the A-works/B-doesn't asymmetry observed.
        if (l == 0 && getenv("NPU_FK3_DUMP")) {
            // PRE-launch input dump to files: the post-launch dump cannot distinguish "I
            // uploaded the wrong thing" from "the kernel overwrote its own input".
            auto pdump = [](const char* n, const void* q, size_t bytes) {
                char path[256]; snprintf(path, sizeof path, "/tmp/drvPRE_%s.bin", n);
                FILE* f = fopen(path, "wb"); if (f) { fwrite(q, 1, bytes, f); fclose(f); }
            };
            pdump("aB",  s.aB.map(),  (size_t)(M + 1) * s.H * 4);
            pdump("a2B", s.a2B.map(), (size_t)(M + 1) * s.H * 4);
            pdump("qB",  s.qB.map(),  (size_t)M * s.NQKV * 2);
            pdump("w2",  s.w2[l].map(), (size_t)s.H * s.N2 * 2);
            pdump("wd",  s.wd[l].map(), (size_t)(s.NI + s.H) * s.ND * 2);
            pdump("wo",  s.wO[l].map(), (size_t)s.qout * s.NO * 2);
        }
        auto t0 = std::chrono::steady_clock::now();
        auto r = s.krB((unsigned)3, s.iB, (unsigned)s.insB_words, s.aB, s.wB, s.anB, s.qB, s.oB,
                       s.wO[l], s.cB, s.a2B, s.an2B, s.w2[l], s.c2B, s.slB, s.wd[l], s.cdB, s.hbfB);
        r.wait();
        auto t1 = std::chrono::steady_clock::now();
        s.cdB.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        if (l == 0 && getenv("NPU_FK3_TIMING"))
            fprintf(stderr, "[fk3] launch B (l=0): %.2f ms\n",
                    std::chrono::duration<double, std::milli>(t1 - t0).count());
        if (l == 0 && getenv("NPU_FK3_DUMP")) {
            FILE* f = fopen("/tmp/fk3_drv_B.bin", "wb");
            if (f) { fwrite(s.cdB.map(), 2, (size_t)M * s.H, f); fclose(f); }
            FILE* g = fopen("/tmp/fk3_drv_Q.bin", "wb");
            if (g) { fwrite(s.qB.map(), 2, (size_t)M * s.NQKV, g); fclose(g); }
            // Full stage dump, matching bench_fk3_layer's BENCH_DUMP_STAGES names, so the
            // two invocations can be compared byte-for-byte and the first differing buffer
            // identifies where they part company.
            auto dump = [](const char* n, const void* ptr, size_t bytes) {
                char path[256]; snprintf(path, sizeof path, "/tmp/drv_%s.bin", n);
                FILE* f = fopen(path, "wb"); if (f) { fwrite(ptr, 1, bytes, f); fclose(f); }
            };
            dump("aB",  s.aB.map(),  (size_t)(M + 1) * s.H * 4);
            dump("a2B", s.a2B.map(), (size_t)(M + 1) * s.H * 4);
            dump("qB",  s.qB.map(),  (size_t)M * s.NQKV * 2);
            dump("oB",  s.oB.map(),  (size_t)s.NH * M * s.HD * 2);
            dump("cB",  s.cB.map(),  (size_t)M * s.NO * 2);
            dump("c2B", s.c2B.map(), (size_t)M * s.N2 * 2);
            dump("slB", s.slB.map(), (size_t)M * s.NI * 2);
            dump("hbfB",s.hbfB.map(),(size_t)M * s.H * 2);
            dump("cdB", s.cdB.map(), (size_t)M * s.ND * 2);
            dump("w2",  s.w2[l].map(), (size_t)s.H * s.N2 * 2);
            dump("wd",  s.wd[l].map(), (size_t)(s.NI + s.H) * s.ND * 2);
            dump("wo",  s.wO[l].map(), (size_t)s.qout * s.NO * 2);
            // Which of launch B's outputs does the kernel actually touch? If some are
            // non-zero and CD is not, the schedule ran and the failure is confined to the
            // final D output; if ALL are zero, the kernel wrote nothing at all.
            struct OutProbe { const char* n; xrt::bo* b; size_t elems; bool is_f32; };
            OutProbe outs[] = {
                {"A  (launch B in)", &s.aB, (size_t)(M + 1) * s.H, true},
                {"A2 (O-proj f32)", &s.a2B, (size_t)(M + 1) * s.H, true},
                {"W2 (GU weight)",   &s.w2[l], (size_t)s.H * s.N2, false},
                {"WD (D weight)",    &s.wd[l], (size_t)(s.NI + s.H) * s.ND, false},
                {"O  (attention)",  &s.oB,  (size_t)s.NH * M * s.HD, false},
                {"C  (O-proj bf16)",&s.cB,  (size_t)M * s.NO, false},
                {"C2 (GU)",         &s.c2B, (size_t)M * s.N2, false},
                {"SL (SiLU)",       &s.slB, (size_t)M * s.NI, false},
                {"HBF (resid 1)",   &s.hbfB,(size_t)M * s.H, false},
                {"CD (LAYER OUT)",  &s.cdB, (size_t)M * s.ND, false},
            };
            for (auto& o : outs) {
                o.b->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                const float* fp = (const float*)o.b->map();
                const uint16_t* hp = (const uint16_t*)o.b->map();
                long nz = 0; double mx = 0.0;
                for (size_t i = 0; i < o.elems; i++) {
                    double v;
                    if (o.is_f32) { v = fp[i]; }
                    else { uint32_t u = (uint32_t)hp[i] << 16; float t; memcpy(&t, &u, 4); v = t; }
                    if (v != 0.0) nz++;
                    if (v < 0) v = -v;
                    if (v > mx) mx = v;
                }
                fprintf(stderr, "[fk3]   %-18s nonzero=%.4f maxabs=%.4f\n", o.n, (double)nz / (double)o.elems, mx);
            }
            fflush(stderr);
        }
    }

    // ---- layer output: bf16 -> f32 for the next layer -------------------------
    {
        const uint16_t* cd = (const uint16_t*)s.cdB.map();
        for (size_t i = 0; i < (size_t)nrow * s.H; i++) {
            uint32_t v = (uint32_t)cd[i] << 16;
            float f;
            memcpy(&f, &v, 4);
            out[i] = f;
        }
    }
    return true;
}

}  // namespace fk3
