// silu_quant.h — the fused GU→SiLU→D on-core arithmetic (issue #1759).
//
// Compiled into BOTH the AIE kernel (mm_kernel_reference.cc, via the AIE2P
// Peano toolchain) AND the host-side CPU reference (engine/npu/tests/
// test_fused_silu.cpp) so the exact bit-level contract is verified on x86
// BEFORE the NPU round-trip. NO AIE intrinsics here — plain scalar C.
//
// ── Contract (per MoE layer, ONE launch; interleaved GU weight pack) ──
//   C1[j]    = Σ_i A[i]·B_gu[i][j]        int8×int8 → int32 (B_gu col 2p =
//              gate[p], col 2p+1 = up[p] — interleaved, per-column scales)
//   gate_f   = C1[2p]   · gs'[2p]         (gs' = ag·gs_g,   host-folded)
//   up_f·qn  = C1[2p+1] · gs'[2p+1]       (gs' = ag·qn_s·gs_u, host-folded)
//   h2[p]    = silu(gate_f) · up_f·qn_s   (silu(x) = x·sigmoid_lut(x))
//   A2[p]    = sat8(round(h2[p]))         (D GEMM's int8 A operand)
//   C2[j]    = Σ_p A2[p]·B_d[p][j]        int8×int8 → int32
//   out[j]   = C2[j] · (gs_d[j] / qn_s)   (host dequant; ag cancels)
//
// The per-token qn_s = 127/max|h2| is computed HOST-side from the same int8
// GU GEMM (integer accumulation is order-independent, so the host's c1 equals
// the NPU's exactly) and folded into the per-column header. This reproduces
// the current two-launch path's per-token adaptation — measured corr
// 0.9993–0.9996 vs float on zaya1-8b.q4nx, argmax parity.
//
// LUT: 256-entry sigmoid over [-XLUT, XLUT]; XLUT=4 covers the measured
// gate_f range [-3.4, 3.4] with saturation headroom.
#pragma once

#include <cstdint>

// No <cmath>/libm: this header is compiled into the AIE2P bare-metal kernel
// (Peano toolchain) where roundf/expf/fabsf are unavailable. All helpers are
// plain float/int ops that the AIE scalar unit lowers to hardware instructions.
#define SILU_XLUT 4.0f
#define SILU_LUT_N 256

static inline float silu_absf(float x) { return x < 0.0f ? -x : x; }

// round-half-away-from-zero (identical to roundf for |x| < 2^23).
static inline int silu_roundf(float x) {
    return x >= 0.0f ? (int)(x + 0.5f) : (int)(x - 0.5f);
}

// sigmoid LUT over [-4, 4] (256 entries; generated, matches the CPU reference).
static const float silu_sigmoid_lut[SILU_LUT_N] = {
    0.017986210f, 0.018548795f, 0.019128635f, 0.019726236f, 0.020342120f, 0.020976821f, 0.021630888f, 0.022304885f,
    0.022999389f, 0.023714994f, 0.024452306f, 0.025211950f, 0.025994565f, 0.026800805f, 0.027631342f, 0.028486863f,
    0.029368073f, 0.030275692f, 0.031210459f, 0.032173131f, 0.033164478f, 0.034185293f, 0.035236384f, 0.036318578f,
    0.037432718f, 0.038579669f, 0.039760311f, 0.040975544f, 0.042226286f, 0.043513473f, 0.044838062f, 0.046201024f,
    0.047603351f, 0.049046055f, 0.050530162f, 0.052056720f, 0.053626791f, 0.055241457f, 0.056901816f, 0.058608984f,
    0.060364092f, 0.062168288f, 0.064022734f, 0.065928609f, 0.067887104f, 0.069899426f, 0.071966791f, 0.074090430f,
    0.076271585f, 0.078511506f, 0.080811454f, 0.083172696f, 0.085596507f, 0.088084167f, 0.090636957f, 0.093256166f,
    0.095943077f, 0.098698978f, 0.101525151f, 0.104422873f, 0.107393415f, 0.110438041f, 0.113558002f, 0.116754535f,
    0.120028864f, 0.123382192f, 0.126815703f, 0.130330557f, 0.133927888f, 0.137608800f, 0.141374365f, 0.145225620f,
    0.149163563f, 0.153189150f, 0.157303293f, 0.161506854f, 0.165800645f, 0.170185421f, 0.174661877f, 0.179230647f,
    0.183892299f, 0.188647329f, 0.193496162f, 0.198439143f, 0.203476538f, 0.208608527f, 0.213835205f, 0.219156573f,
    0.224572536f, 0.230082905f, 0.235687387f, 0.241385585f, 0.247176995f, 0.253061003f, 0.259036883f, 0.265103795f,
    0.271260781f, 0.277506765f, 0.283840551f, 0.290260821f, 0.296766135f, 0.303354930f, 0.310025519f, 0.316776091f,
    0.323604713f, 0.330509327f, 0.337487757f, 0.344537703f, 0.351656750f, 0.358842363f, 0.366091897f, 0.373402594f,
    0.380771590f, 0.388195915f, 0.395672502f, 0.403198188f, 0.410769719f, 0.418383757f, 0.426036883f, 0.433725606f,
    0.441446365f, 0.449195540f, 0.456969455f, 0.464764386f, 0.472576568f, 0.480402202f, 0.488237465f, 0.496078512f,
    0.503921488f, 0.511762535f, 0.519597798f, 0.527423432f, 0.535235614f, 0.543030545f, 0.550804460f, 0.558553635f,
    0.566274394f, 0.573963117f, 0.581616243f, 0.589230281f, 0.596801812f, 0.604327498f, 0.611804085f, 0.619228410f,
    0.626597406f, 0.633908103f, 0.641157637f, 0.648343250f, 0.655462297f, 0.662512243f, 0.669490673f, 0.676395287f,
    0.683223909f, 0.689974481f, 0.696645070f, 0.703233865f, 0.709739179f, 0.716159449f, 0.722493235f, 0.728739219f,
    0.734896205f, 0.740963117f, 0.746938997f, 0.752823005f, 0.758614415f, 0.764312613f, 0.769917095f, 0.775427464f,
    0.780843427f, 0.786164795f, 0.791391473f, 0.796523462f, 0.801560857f, 0.806503838f, 0.811352671f, 0.816107701f,
    0.820769353f, 0.825338123f, 0.829814579f, 0.834199355f, 0.838493146f, 0.842696707f, 0.846810850f, 0.850836437f,
    0.854774380f, 0.858625635f, 0.862391200f, 0.866072112f, 0.869669443f, 0.873184297f, 0.876617808f, 0.879971136f,
    0.883245465f, 0.886441998f, 0.889561959f, 0.892606585f, 0.895577127f, 0.898474849f, 0.901301022f, 0.904056923f,
    0.906743834f, 0.909363043f, 0.911915833f, 0.914403493f, 0.916827304f, 0.919188546f, 0.921488494f, 0.923728415f,
    0.925909570f, 0.928033209f, 0.930100574f, 0.932112896f, 0.934071391f, 0.935977266f, 0.937831712f, 0.939635908f,
    0.941391016f, 0.943098184f, 0.944758543f, 0.946373209f, 0.947943280f, 0.949469838f, 0.950953945f, 0.952396649f,
    0.953798976f, 0.955161938f, 0.956486527f, 0.957773714f, 0.959024456f, 0.960239689f, 0.961420331f, 0.962567282f,
    0.963681422f, 0.964763616f, 0.965814707f, 0.966835522f, 0.967826869f, 0.968789541f, 0.969724308f, 0.970631927f,
    0.971513137f, 0.972368658f, 0.973199195f, 0.974005435f, 0.974788050f, 0.975547694f, 0.976285006f, 0.977000611f,
    0.977695115f, 0.978369112f, 0.979023179f, 0.979657880f, 0.980273764f, 0.980871365f, 0.981451205f, 0.982013790f,
};

// silu(x) = x·σ(x) via the LUT (clamped + quantized index).
static inline float silu_lut(float x) {
    float t = x < -SILU_XLUT ? -SILU_XLUT : (x > SILU_XLUT ? SILU_XLUT : x);
    int idx = silu_roundf((t + SILU_XLUT) * (255.0f / (2.0f * SILU_XLUT)));
    if (idx < 0) idx = 0;
    if (idx > 255) idx = 255;
    return x * silu_sigmoid_lut[idx];
}

static inline int8_t silu_sat8(int x) {
    return (int8_t)(x > 127 ? 127 : x < -127 ? -127 : x);
}

// The fused kernel's per-tile SiLU+quant step (issue #1759).
//
//   c1    : int32 GU accumulator tile, row-major [n_cols], cols 2p/2p+1 =
//           (gate, up) pair p (interleaved pack). Only row 0 is valid for
//           decode M=1; rows 1-7 are zero.
//   gs    : per-column folded header [n_cols] — gs'[2p]=ag·gs_g, gs'[2p+1]=
//           ag·qn_s·gs_u (host-written per token).
//   h2    : int8 output [n_pairs], the D GEMM's A operand chunk.
//   n_pairs : number of (gate, up) pairs (= n_cols/2); M=8 tiles carry 64.
static inline void silu_quant_i8(const int32_t* c1, const float* gs,
                                 int8_t* h2, int n_pairs) {
    for (int p = 0; p < n_pairs; p++) {
        float g = (float)c1[2 * p]     * gs[2 * p];       // gate_f
        float u = (float)c1[2 * p + 1] * gs[2 * p + 1];   // up_f·qn_s
        float h = silu_lut(g) * u;                        // h2·qn_s
        h2[p] = silu_sat8(silu_roundf(h));
    }
}

// ── Fixed-point Q22 silu for the int4 fused path (issue #1769, #1844) ──────
// The aie2p backend mis-compiles the float silu loop (correct g/u, wrong h
// for p>=1 — #1836) and int64 math (#1843), so the on-core silu is PURE
// int32. v50/v51 used gQ22 = c1*fold (single Q22 fold): it overflowed int32
// for |g|>512 / |u|>512 (the reported "host h2=12 -> NPU 0" zero pairs) and
// the fixed Q22 fold rounded small per-column scales to zero. The v59
// contract (CPU-gated by test_i4_silu_q22.cpp, corr 0.99997 vs the float
// silu_quant reference on realistic data):
//
//   fold[j]   : int32 = round(S'[j] * 2^Q), Q per TILE from the tile's MIN
//               |S'| (Q = 22 - s, s = max(0, 15+ceil(log2(minS')))) so small
//               scales keep >= ~64 bits of fold (no rounding to zero).
//   boundG[j] : int32 = (2^31-1) / |fold[j]|      -> |c1g| <= boundG keeps
//               gQ = c1g*fold[j] overflow-free.
//   boundU[j] : int32 = 4*((2^31-1)/|fold[j]|) + 3 -> uQ = (c1u>>2)*fold[j]
//               = u*2^(Q-2) stays overflow-free for |u| <= 2^(33-Q)
//               ((c1u>>2) <= (2^31-1)/fold  <=>  c1u <= 4*((2^31-1)/fold)+3).
//   h2        : sat8(round(silu(g)*u)) with g = c1g*S'[2p],
//               u = c1u*S'[2p+1] — the FLOAT reference's exact semantics.
//
// Arithmetic (all int32, no division, no libcalls — AIE2P-safe):
//   gQ  = c1g*foldg                  (|gQ|  <= 2^31)
//   uQ  = (c1u>>2)*foldu             (|uQ|  <= 2^31; u*2^(Q-2))
//   gc  = clamp(gQ, +-4*2^Q)         (LUT clamp)
//   idx = round((gc + 4*2^Q)*255 / 2^(Q+3))   -> [0,255]
//   siluQ  = silu(g)*2^Q  = (gQ*sigma_lut[idx]) via an 11-bit split so the
//            small-gQ pairs keep full precision (no truncate-to-zero)
//   siluF  = silu*2^11 ; uF = u*2^5   -> hQ = siluF*uF = h*2^16
//   hQ saturating: clamp |uF| to the largest power of two with
//            siluF*cap <= 2^31; clamped pairs ALWAYS have h >= 2^14 -> the
//            sat8 output is exactly the float reference's (both 127).
static const int32_t silu_sigmoid_q22[256] = {
    75440, 77799, 80231, 82738, 85321, 87983, 90727, 93553,
    96466, 99468, 102560, 105747, 109029, 112411, 115894, 119483,
    123179, 126985, 130906, 134944, 139102, 143384, 147792, 152331,
    157004, 161815, 166767, 171864, 177110, 182509, 188064, 193781,
    199663, 205714, 211939, 218342, 224927, 231699, 238664, 245824,
    253185, 260753, 268531, 276525, 284739, 293179, 301851, 310758,
    319906, 329301, 338948, 348852, 359018, 369452, 380159, 391145,
    402414, 413974, 425827, 437981, 450441, 463211, 476297, 489704,
    503438, 517502, 531904, 546646, 561734, 577173, 592967, 609120,
    625637, 642522, 659778, 677409, 695418, 713809, 732585, 751748,
    771300, 791244, 811582, 832314, 853442, 874968, 896890, 919209,
    941925, 965038, 988545, 1012445, 1036735, 1061415, 1086479, 1111926,
    1137750, 1163948, 1190514, 1217442, 1244727, 1272363, 1300341, 1328655,
    1357297, 1386257, 1415526, 1445096, 1474955, 1505094, 1535501, 1566164,
    1597072, 1628212, 1659571, 1691136, 1722893, 1754829, 1786928, 1819177,
    1851560, 1884063, 1916669, 1949363, 1982130, 2014953, 2047816, 2080704,
    2113600, 2146488, 2179351, 2212174, 2244941, 2277635, 2310241, 2342744,
    2375127, 2407376, 2439475, 2471411, 2503168, 2534733, 2566092, 2597232,
    2628140, 2658803, 2689210, 2719349, 2749208, 2778778, 2808047, 2837007,
    2865649, 2893963, 2921941, 2949577, 2976862, 3003790, 3030356, 3056554,
    3082378, 3107825, 3132889, 3157569, 3181859, 3205759, 3229266, 3252379,
    3275095, 3297414, 3319336, 3340862, 3361990, 3382722, 3403060, 3423004,
    3442556, 3461719, 3480495, 3498886, 3516895, 3534526, 3551782, 3568667,
    3585184, 3601337, 3617131, 3632570, 3647658, 3662400, 3676802, 3690866,
    3704600, 3718007, 3731093, 3743863, 3756323, 3768477, 3780330, 3791890,
    3803159, 3814145, 3824852, 3835286, 3845452, 3855356, 3865003, 3874398,
    3883546, 3892453, 3901125, 3909565, 3917779, 3925773, 3933551, 3941119,
    3948480, 3955640, 3962605, 3969377, 3975962, 3982365, 3988590, 3994641,
    4000523, 4006240, 4011795, 4017194, 4022440, 4027537, 4032489, 4037300,
    4041973, 4046512, 4050920, 4055202, 4059360, 4063398, 4067319, 4071125,
    4074821, 4078410, 4081893, 4085275, 4088557, 4091744, 4094836, 4097838,
    4100751, 4103577, 4106321, 4108983, 4111566, 4114073, 4116505, 4118864,
};

// One (gate, up) pair of the fixed-point silu (v59). CPU-emulated
// bit-exactly by test_i4_silu_q22.cpp.
// v60: shG (= Q-11) and shU (= Q-7) are HOST-PRECOMPUTED and passed in —
// the aie2p backend miscompiles register-computed shift counts (measured
// 2026-08-24: siluQ >> (Q-11) compiled to shift-by-garbage, corr ~0.017;
// memory-loaded shift counts are honored). The CPU gate passes the same
// values so the contract is unchanged.
static inline int8_t silu_pair_q22(int32_t c1g, int32_t c1u,
                                   int32_t foldg, int32_t foldu,
                                   int32_t boundg, int32_t boundu, int Q,
                                   int shG, int shU) {
    int ag_ = c1g < 0 ? -c1g : c1g;
    if (ag_ > boundg) ag_ = boundg;
    c1g = c1g < 0 ? -ag_ : ag_;
    int au_ = c1u < 0 ? -c1u : c1u;
    if (au_ > boundu) au_ = boundu;
    c1u = c1u < 0 ? -au_ : au_;
    int gQ = c1g * foldg;                  // g*2^Q, |gQ| <= 2^31
    int uQ = (c1u >> 2) * foldu;           // u*2^(Q-2), |uQ| <= 2^31
    int CL = 4 << Q;
    int gc = gQ < -CL ? -CL : (gQ > CL ? CL : gQ);
    int sh = Q - 7;
    int base = sh >= 0 ? ((gc + CL) >> sh) : ((gc + CL) << (-sh));
    int idx = ((base * 255) + 512) >> 10;
    idx = idx < 0 ? 0 : (idx > 255 ? 255 : idx);
    int sig = silu_sigmoid_q22[idx];
    // |gQ| < 2^20: gQ*(sig>>11) fits int32 ((2^20-1)*(2^11-1) < 2^31) and
    // keeps full gate precision — the >>11 split (gQ>>11)*(sig>>11) loses the
    // low bits of g*2^(Q-11) and measured ~5% LOW on the real zaya weights
    // for gQ in [2048, 2^20) (the old v51 threshold was 2048). |gQ| >= 2^20:
    // the split's relative truncation error is <= ~2^-10, negligible.
    int siluQ = (gQ > -(1 << 20) && gQ < (1 << 20)) ? ((gQ * (sig >> 11)) >> 11)
                                                    : ((gQ >> 11) * (sig >> 11));
    if (Q < 11) {
        int lim = 1 << (20 + Q);
        if (siluQ > lim) siluQ = lim;
        else if (siluQ < -lim) siluQ = -lim;
    }
    int siluF = shG >= 0 ? (siluQ >> shG) : (siluQ << (-shG));   // shG = Q-11, host-precomputed
    if (Q < 7) {
        int lim = 1 << (24 + Q);
        if (uQ > lim) uQ = lim;
        else if (uQ < -lim) uQ = -lim;
    }
    int uF = shU >= 0 ? (uQ >> shU) : (uQ << (-shU));           // shU = Q-7, host-precomputed
    int as = siluF < 0 ? -siluF : siluF;
    int aus = uF < 0 ? -uF : uF;
    if (as > 1) {
        int v = as; int cap = 1 << 30;
        while (v > 1) { v >>= 1; cap >>= 1; }
        if (aus > cap) aus = cap;
    }
    uF = uF < 0 ? -aus : aus;
    int hQ = siluF * uF;                   // h*2^16, |hQ| <= 2^31
    int ha = hQ < 0 ? -hQ : hQ;
    int h = (ha + (1 << 15)) >> 16;        // round-half-away
    h = hQ < 0 ? -h : h;
    return silu_sat8(h);
}
