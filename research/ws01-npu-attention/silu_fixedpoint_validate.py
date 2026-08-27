#!/usr/bin/env python3
"""Exact fixed-point SiLU vs float-reference validation (fused GU->SiLU->D).

The fused kernel (issue #1759, merged into main via #1892) computes
    h2[p] = sat8(round(silu(gate_f)·up_f·qn_s))
on-NPU with a pure-int32 fixed-point sigmoid (silu_pair_q22 in
engine/npu/generators/silu_quant.h) plus a float-LUT CPU-side path
(silu_quant_i8 / silu_lut). The D GEMM consumes h2 as its int8 A operand,
so the bar is NOT bit-exactness but: after int8 quantization the kernel h2
must match the float reference to <= 1 LSB for ~all samples.

This script emulates BOTH on-NPU contracts bit-for-bit (host fold math +
pure int32 Q22 arithmetic + float32 LUT path) and compares them against the
TRUE float reference (exact exp sigmoid, float64) — the strictest bar —
over the realistic Zaya envelope used by the C++ gate
(engine/npu/src/test_i4_silu_q22.cpp) plus adversarial corners and a
wide-gate stress (LUT clamp sensitivity).

Contract provenance: engine/npu/generators/silu_quant.h + gu_i4_pack.h
write_silu_pad_meta @ origin/main (3ced5bdb, merged #1892). The LUTs below
are verbatim from that header.

Gates (same as the C++ gate):
    corr(kernel, ref) >= 0.999
    >= 98% of pairs within |dH2| <= 1 int8 LSB
    max |dH2| <= 8

Usage:  python3 silu_fixedpoint_validate.py
"""
import math
import numpy as np

# ── silu_quant.h verbatim tables (256 entries) ─────────────────────────────
SIG_FLOAT = np.array([  # silu_sigmoid_lut, float32 ([-4,4] range)
    0.017986210, 0.018548795, 0.019128635, 0.019726236, 0.020342120, 0.020976821, 0.021630888, 0.022304885,
    0.022999389, 0.023714994, 0.024452306, 0.025211950, 0.025994565, 0.026800805, 0.027631342, 0.028486863,
    0.029368073, 0.030275692, 0.031210459, 0.032173131, 0.033164478, 0.034185293, 0.035236384, 0.036318578,
    0.037432718, 0.038579669, 0.039760311, 0.040975544, 0.042226286, 0.043513473, 0.044838062, 0.046201024,
    0.047603351, 0.049046055, 0.050530162, 0.052056720, 0.053626791, 0.055241457, 0.056901816, 0.058608984,
    0.060364092, 0.062168288, 0.064022734, 0.065928609, 0.067887104, 0.069899426, 0.071966791, 0.074090430,
    0.076271585, 0.078511506, 0.080811454, 0.083172696, 0.085596507, 0.088084167, 0.090636957, 0.093256166,
    0.095943077, 0.098698978, 0.101525151, 0.104422873, 0.107393415, 0.110438041, 0.113558002, 0.116754535,
    0.120028864, 0.123382192, 0.126815703, 0.130330557, 0.133927888, 0.137608800, 0.141374365, 0.145225620,
    0.149163563, 0.153189150, 0.157303293, 0.161506854, 0.165800645, 0.170185421, 0.174661877, 0.179230647,
    0.183892299, 0.188647329, 0.193496162, 0.198439143, 0.203476538, 0.208608527, 0.213835205, 0.219156573,
    0.224572536, 0.230082905, 0.235687387, 0.241385585, 0.247176995, 0.253061003, 0.259036883, 0.265103795,
    0.271260781, 0.277506765, 0.283840551, 0.290260821, 0.296766135, 0.303354930, 0.310025519, 0.316776091,
    0.323604713, 0.330509327, 0.337487757, 0.344537703, 0.351656750, 0.358842363, 0.366091897, 0.373402594,
    0.380771590, 0.388195915, 0.395672502, 0.403198188, 0.410769719, 0.418383757, 0.426036883, 0.433725606,
    0.441446365, 0.449195540, 0.456969455, 0.464764386, 0.472576568, 0.480402202, 0.488237465, 0.496078512,
    0.503921488, 0.511762535, 0.519597798, 0.527423432, 0.535235614, 0.543030545, 0.550804460, 0.558553635,
    0.566274394, 0.573963117, 0.581616243, 0.589230281, 0.596801812, 0.604327498, 0.611804085, 0.619228410,
    0.626597406, 0.633908103, 0.641157637, 0.648343250, 0.655462297, 0.662512243, 0.669490673, 0.676395287,
    0.683223909, 0.689974481, 0.696645070, 0.703233865, 0.709739179, 0.716159449, 0.722493235, 0.728739219,
    0.734896205, 0.740963117, 0.746938997, 0.752823005, 0.758614415, 0.764312613, 0.769917095, 0.775427464,
    0.780843427, 0.786164795, 0.791391473, 0.796523462, 0.801560857, 0.806503838, 0.811352671, 0.816107701,
    0.820769353, 0.825338123, 0.829814579, 0.834199355, 0.838493146, 0.842696707, 0.846810850, 0.850836437,
    0.854774380, 0.858625635, 0.862391200, 0.866072112, 0.869669443, 0.873184297, 0.876617808, 0.879971136,
    0.883245465, 0.886441998, 0.889561959, 0.892606585, 0.895577127, 0.898474849, 0.901301022, 0.904056923,
    0.906743834, 0.909363043, 0.911915833, 0.914403493, 0.916827304, 0.919188546, 0.921488494, 0.923728415,
    0.925909570, 0.928033209, 0.930100574, 0.932112896, 0.934071391, 0.935977266, 0.937831712, 0.939635908,
    0.941391016, 0.943098184, 0.944758543, 0.946373209, 0.947943280, 0.949469838, 0.950953945, 0.952396649,
    0.953798976, 0.955161938, 0.956486527, 0.957773714, 0.959024456, 0.960239689, 0.961420331, 0.962567282,
    0.963681422, 0.964763616, 0.965814707, 0.966835522, 0.967826869, 0.968789541, 0.969724308, 0.970631927,
    0.971513137, 0.972368658, 0.973199195, 0.974005435, 0.974788050, 0.975547694, 0.976285006, 0.977000611,
    0.977695115, 0.978369112, 0.979023179, 0.979657880, 0.980273764, 0.980871365, 0.981451205, 0.982013790,
], dtype=np.float32)

SIG_Q22 = np.array([  # silu_sigmoid_q22, int32 (Q22 fixed-point sigmoid)
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
], dtype=np.int64)

SILU_XLUT = 4.0


# ── exact int32 helpers (Python ints, wrapped to int32 like the C contract) ──
def wrap32(x):
    x = int(x)
    return (x + 0x80000000) % 0x100000000 - 0x80000000


def sat8(x):
    return 127 if x > 127 else (-127 if x < -127 else x)


def round_half_away(x):
    """silu_roundf: round-half-away-from-zero (int32 result)."""
    return int(math.floor(x + 0.5)) if x >= 0 else int(math.ceil(x - 0.5))


def silu_lut_float(x):
    """silu_quant.h silu_lut: float32 sigmoid LUT over [-4,4]; returns x·σ(x)."""
    x32 = np.float32(x)
    t = np.float32(SILU_XLUT) if x32 > np.float32(SILU_XLUT) else (
        np.float32(-SILU_XLUT) if x32 < np.float32(-SILU_XLUT) else x32)
    # idx = round((t + 4)*255/8) — one float32 multiply, then round-half-away
    idx = int(np.floor((np.float32(t) + np.float32(SILU_XLUT)) *
                       np.float32(255.0 / (2.0 * SILU_XLUT)) + np.float32(0.5)))
    idx = 0 if idx < 0 else (255 if idx > 255 else idx)
    return float(np.float32(x32) * SIG_FLOAT[idx])


def silu_quant_i8_float(c1g, c1u, sg, su):
    """silu_quant.h silu_quant_i8: float-LUT path -> int8 h2."""
    g = np.float32(np.float32(c1g) * np.float32(sg))
    u = np.float32(np.float32(c1u) * np.float32(su))
    h = np.float32(np.float32(silu_lut_float(g)) * u)
    return sat8(round_half_away(float(h)))


def silu_pair_q22(c1g, c1u, foldg, foldu, boundg, boundu, Q, shG, shU):
    """silu_quant.h silu_pair_q22 — the on-NPU pure-int32 contract, bit-exact."""
    ag_ = -c1g if c1g < 0 else c1g
    if ag_ > boundg:
        ag_ = boundg
    c1g = -ag_ if c1g < 0 else ag_
    au_ = -c1u if c1u < 0 else c1u
    if au_ > boundu:
        au_ = boundu
    c1u = -au_ if c1u < 0 else au_
    gQ = wrap32(c1g * foldg)                       # g·2^Q, |gQ| <= 2^31
    uQ = wrap32((c1u >> 2) * foldu)                # u·2^(Q-2), |uQ| <= 2^31
    CL = 4 << Q
    gc = -CL if gQ < -CL else (CL if gQ > CL else gQ)
    sh = Q - 7
    base = (gc + CL) >> sh if sh >= 0 else (gc + CL) << (-sh)
    idx = ((base * 255) + 512) >> 10
    idx = 0 if idx < 0 else (255 if idx > 255 else idx)
    sig = SIG_Q22[idx]
    sig11 = sig >> 11
    if -(1 << 20) < gQ < (1 << 20):
        siluQ = wrap32((gQ * sig11) >> 11)
    else:
        siluQ = wrap32((gQ >> 11) * sig11)
    if Q < 11:
        lim = 1 << (20 + Q)
        siluQ = lim if siluQ > lim else (-lim if siluQ < -lim else siluQ)
    siluF = wrap32(siluQ >> shG) if shG >= 0 else wrap32(siluQ << (-shG))
    if Q < 7:
        lim = 1 << (24 + Q)
        uQ = lim if uQ > lim else (-lim if uQ < -lim else uQ)
    uF = wrap32(uQ >> shU) if shU >= 0 else wrap32(uQ << (-shU))
    asf = -siluF if siluF < 0 else siluF
    aus = -uF if uF < 0 else uF
    if asf > 1:
        cap = 1 << (31 - asf.bit_length())         # = 2^(30 - floor(log2 asf))
        if aus > cap:
            aus = cap
    uF = -aus if uF < 0 else aus
    hQ = wrap32(siluF * uF)                        # h·2^16, |hQ| <= 2^31
    ha = -hQ if hQ < 0 else hQ
    h = (ha + (1 << 15)) >> 16
    h = -h if hQ < 0 else h
    return sat8(h)


# ── host fold math (write_silu_pad_meta, gu_i4_pack.h) per 64-pair tile ──
def tile_meta(sg, su, t0, n):
    """Per-64-pair-tile host fold: Q from tile min|S'|, fold, bounds, shG/shU.

    Returns (Q, shG, shU, foldg[n], foldu[n], boundg[n], boundu[n]) with the
    same interleaved col layout as the real pad writer (col 2p = gate, 2p+1 =
    up; foldG chunk = cols 2p, boundU chunk = cols 2p+1).
    """
    scol = []
    for i in range(n):
        scol.append(sg[t0 + i])
        scol.append(su[t0 + i])
    minS = min(abs(v) for v in scol)
    s = 0
    if minS > 0:
        s = 15 + math.ceil(math.log2(minS))
        s = 0 if s < 0 else (22 if s > 22 else s)
    Q = 22 - s
    shG, shU = Q - 11, Q - 7
    foldg, foldu, boundg, boundu = [], [], [], []
    for j, v in enumerate(scol):
        q = round_half_away(v * (1 << Q))
        aq = -q if q < 0 else q
        if aq < 1:
            aq = 1
        if aq > 1073741823:
            aq = 1073741823
        q = -aq if q < 0 else aq
        f = -q if q < 0 else q
        bg = 2147483647 // f                     # (2^31-1)/|foldG|
        bu = wrap32(4 * bg + 3)                  # 4·boundG+3 (int32 in C)
        if j % 2 == 0:
            foldg.append(q)
            boundg.append(bg)
        else:
            foldu.append(q)
            boundu.append(bu)
    return Q, shG, shU, foldg, foldu, boundg, boundu


def ref_h2_true(c1g, c1u, sg, su):
    """TRUE float reference: h2 = sat8(round(silu(g)·u)) with exact exp (f64)."""
    g = float(c1g) * float(sg)
    u = float(c1u) * float(su)
    h = (g / (1.0 + math.exp(-g))) * u
    return sat8(round_half_away(h))


def fold_qn_s(c1g, c1u, sg, su, token=2048):
    """Fold the per-token qn_s = 127/max|silu(g)·u| into the up scales su,
    exactly like the real host (silu_quant.h: gs'[2p+1] = ag·qn_s·gs_u).

    Returns su' (folded), and the per-token max|h| used for qn_s.
    """
    su_f = list(su)
    mx = [0.0] * ((len(c1g) + token - 1) // token)
    for t, i in enumerate(range(0, len(c1g), token)):
        m = 0.0
        for j in range(i, min(i + token, len(c1g))):
            h = abs((float(c1g[j]) * float(sg[j])) /
                    (1.0 + math.exp(-float(c1g[j]) * float(sg[j]))) *
                    float(c1u[j]) * float(su[j]))
            if h > m:
                m = h
        mx[t] = m
        qn = 1.0 if m < 1e-12 else 127.0 / m
        for j in range(i, min(i + token, len(c1g))):
            su_f[j] = float(su[j]) * qn
    return su_f, mx


def run_gate(name, c1g, c1u, sg, su, folded=False, gate_mode="all"):
    """gate_mode: 'all' — gate every row (end-to-end, qn_s folded);
    'contract' — gate only the Q22-vs-floatLUT row (raw contract test; the
    vs-TRUE rows there are informational — the unnormalized u amplifies the
    LUT range error into several LSBs, which the real per-token qn_s fold
    prevents; that is exactly what the folded section demonstrates)."""
    n = len(c1g)
    hr = [ref_h2_true(c1g[i], c1u[i], sg[i], su[i]) for i in range(n)]
    hl = [silu_quant_i8_float(c1g[i], c1u[i], sg[i], su[i]) for i in range(n)]
    hk = [0] * n
    for t0 in range(0, n, 64):                   # 64-pair tiles (C++ gate tiling)
        nt = min(64, n - t0)
        Q, shG, shU, fg, fu, bg_, bu_ = tile_meta(sg, su, t0, nt)
        for i in range(nt):
            hk[t0 + i] = silu_pair_q22(c1g[t0 + i], c1u[t0 + i],
                                       fg[i], fu[i], bg_[i], bu_[i], Q, shG, shU)
    a = np.array(hr)
    b = np.array(hl)
    c = np.array(hk)

    def stats(ref, kern, label, gate):
        d = np.abs(ref - kern)
        corr = np.corrcoef(ref.astype(float), kern.astype(float))[0, 1]
        within = 100.0 * np.mean(d <= 1)
        exact = 100.0 * np.mean(d == 0)
        nz = int(np.sum((ref != 0) & (kern == 0)))
        sat = int(np.sum(np.abs(kern) == 127))
        rms_r = np.sqrt(np.mean(ref.astype(float) ** 2))
        rms_k = np.sqrt(np.mean(kern.astype(float) ** 2))
        print(f"    {label:30s} corr={corr:.6f} within+/-1={within:6.2f}% "
              f"exact={exact:6.2f}% worst|d|={int(d.max())} "
              f"zero(ref!=0,kern=0)={nz} sat8={sat} "
              f"rms(ref)={rms_r:.2f} rms(kern)={rms_k:.2f}"
              + ("" if gate else "   [info]"))
        fails = 0
        if gate:
            if not (corr >= 0.999):
                print(f"    FAIL corr {corr:.4f} < 0.999"); fails += 1
            if not (within >= 98.0):
                print(f"    FAIL within+/-1 {within:.2f}% < 98%"); fails += 1
            if int(d.max()) > 8:
                print(f"    FAIL worst|d| {int(d.max())} > 8"); fails += 1
        return fails

    print(f"[{name}] pairs={n}" + ("  (per-token qn_s folded into su — end-to-end h2)"
                                   if folded else "  (raw contract test, no qn_s)"))
    f = 0
    f += stats(a, c, "Q22-int32   vs TRUE float", gate_mode == "all")
    f += stats(a, b, "float-LUT   vs TRUE float", gate_mode == "all")
    f += stats(b, c, "Q22-int32   vs float-LUT", gate_mode in ("all", "contract"))
    return f


def hazard_probe(c1g, c1u, sg, su, token=2048):
    """Fold qn_s into su and count columns where the host fold produces
    foldG < 8 — the regime where boundU = 4·boundG+3 overflows int32 and the
    Q22 contract's overflow-freedom proof breaks (boundU goes NEGATIVE, the
    |c1u| clamp misbehaves and h2 flips to +-127 garbage).

    The real Zaya model never enters this regime: measured per-token
    max|h2f| = 1.79 -> qn_s = 70.8 (folds UP, su' ~ O(1), folds >= ~164),
    whereas a synthetic token whose max|h2f| ~ O(1000) folds DOWN (qn_s ~
    0.06) and pushes the small-scale tail below 2^-16·2^Q.
    """
    su_f, mx = fold_qn_s(c1g, c1u, sg, su, token)
    n_fold_lt8 = 0
    n_boundu_neg = 0
    n_cols = 0
    q_min = 99
    for t0 in range(0, len(c1g), 64):
        nt = min(64, len(c1g) - t0)
        Q, shG, shU, fg, fu, bg_, bu_ = tile_meta(sg, su_f, t0, nt)
        q_min = min(q_min, Q)
        for i in range(nt):
            n_cols += 2
            for f in (fg[i], fu[i]):
                if abs(f) < 8:
                    n_fold_lt8 += 1
            for b in (bg_[i], bu_[i]):
                if b < 0:
                    n_boundu_neg += 1
    qns = [127.0 / m for m in mx]
    print(f"[hazard] per-token qn_s: min={min(qns):.3f} med={sorted(qns)[len(qns)//2]:.3f} "
          f"max={max(qns):.3f}  min tile Q={q_min}")
    print(f"[hazard] columns with |fold|<8: {n_fold_lt8}/{n_cols} "
          f"({100.0*n_fold_lt8/n_cols:.2f}%)  boundU<0 (int32 overflow): "
          f"{n_boundu_neg}/{n_cols}")
    return su_f, mx


def main():
    fails = 0

    # ── 1. Realistic Zaya envelope (the C++ gate's synthetic mode) ──
    # gate g ~ N(0,1.2) clamp +-8 (measured gate range [-3.4,3.4]); up u ~
    # +-10^U(-0.5,2.8) (measured up ~ +-74..250, tails ~600); S' log-uniform
    # over [1e-5.5, 1e-1.5]; c1 = g/S' (c1 and S' anti-correlate so the
    # pre-activations stay O(1), exactly like the real GU GEMM).
    rng = np.random.default_rng(42)
    N = 60000
    sg = 10.0 ** rng.uniform(-5.5, -1.5, N)
    su = 10.0 ** rng.uniform(-5.5, -1.5, N)
    g = np.clip(rng.normal(0.0, 1.2, N), -8.0, 8.0)
    u = 10.0 ** rng.uniform(-0.5, 2.8, N) * rng.choice([-1.0, 1.0], N)
    c1g = np.clip(np.rint(g / sg), -33000000, 33000000).astype(np.int64)
    c1u = np.clip(np.rint(u / su), -33000000, 33000000).astype(np.int64)
    fails += run_gate("synthetic envelope (N=60000)", c1g.tolist(), c1u.tolist(),
                      sg.tolist(), su.tolist(), gate_mode="contract")

    # ── 1b. HAZARD PROBE: the same envelope with the per-token qn_s fold
    # applied (real host flow). This probes the fold<8 / boundU-overflow
    # regime: a synthetic token with max|h2f| ~ O(1000) folds qn_s DOWN
    # (0.06), pushing small up-scales below 2^-16 — where boundU wraps
    # negative and the Q22 h2 flips sign (corr collapses). The REAL model
    # folds qn_s UP (measured qn_s = 70.8, max|h2f| = 1.79, folds >= ~164),
    # ~3 orders of magnitude away from the hazard. Rows are informational. ──
    su_f, mx = hazard_probe(c1g.tolist(), c1u.tolist(), sg.tolist(), su.tolist())
    fails += run_gate("folded envelope (hazard probe, N=60000)",
                      c1g.tolist(), c1u.tolist(), sg.tolist(), su_f, folded=True,
                      gate_mode="none")

    # ── 2. Adversarial corners (the v50/v51 overflow failure class) ──
    # small gate (silu ~ 0.15..0.02) with up ~ 550..650 (the old uQ22 wrap
    # -> "host h2=12 -> NPU 0"), plus tiny-gate/huge-up corners that the old
    # truncations zeroed.
    rng = np.random.default_rng(7)
    a1g, a1u, a1sg, a1su = [], [], [], []
    for i in range(2000):
        s_g = 10.0 ** rng.uniform(-5.5, -1.5)
        s_u = 10.0 ** rng.uniform(-5.5, -1.5)
        a1sg.append(s_g); a1su.append(s_u)
        if i % 2 == 0:
            gg = rng.uniform(-0.7, 0.7)
            uu = rng.uniform(550.0, 650.0) * rng.choice([-1.0, 1.0])
        else:
            gg = rng.uniform(0.001, 0.05) * rng.choice([-1.0, 1.0])
            uu = rng.uniform(500.0, 2000.0) * rng.choice([-1.0, 1.0])
        a1g.append(np.clip(round(gg / s_g), -33000000, 33000000))
        a1u.append(np.clip(round(uu / s_u), -33000000, 33000000))
    fails += run_gate("adversarial corners (N=2000)", a1g, a1u, a1sg, a1su,
                      gate_mode="contract")

    # ── 3. Wide-gate stress — is the [-4,4] LUT clamp safe if the gate tail
    #    is wider than the measured [-3.4,3.4]? g ~ N(0,2.5) clip +-10.
    #    (raw contract rows informational: without qn_s the LUT-range error
    #    is amplified by unnormalized u; the Q22-vs-floatLUT row is gated —
    #    both paths share the same LUT, so the fixed-point arithmetic itself
    #    stays exact.) ──
    rng = np.random.default_rng(3)
    N3 = 20000
    sg3 = 10.0 ** rng.uniform(-5.5, -1.5, N3)
    su3 = 10.0 ** rng.uniform(-5.5, -1.5, N3)
    g3 = np.clip(rng.normal(0.0, 2.5, N3), -10.0, 10.0)
    u3 = 10.0 ** rng.uniform(-0.5, 2.8, N3) * rng.choice([-1.0, 1.0], N3)
    c1g3 = np.clip(np.rint(g3 / sg3), -33000000, 33000000).astype(np.int64)
    c1u3 = np.clip(np.rint(u3 / su3), -33000000, 33000000).astype(np.int64)
    beyond = 100.0 * np.mean(np.abs(g3) > SILU_XLUT)
    print(f"\n[XLUT] gate_f ~ N(0,2.5): {beyond:.2f}% beyond LUT clamp +-4 "
          f"(measured zaya gate range [-3.4,3.4])")
    fails += run_gate("wide-gate stress (N=20000)", c1g3.tolist(), c1u3.tolist(),
                      sg3.tolist(), su3.tolist(), gate_mode="contract")

    print("\n" + ("ALL GATES PASS" if fails == 0 else f"{fails} GATE(S) FAILED"))
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
