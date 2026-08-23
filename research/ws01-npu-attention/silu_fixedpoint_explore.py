#!/usr/bin/env python3
# Explore fixed-point SiLU approximations for the fused GU->SiLU->D kernel.
#
# The fused kernel must compute, on the NPU in int32/int8 fixed point:
#   silu_f = silu(gate_f) * up_f
#   silu_i8 = round(silu_f / Q)
# where gate_f = gate_i32 * S, up_f = up_i32 * S (single scale S for now),
# and Q is the D GEMM's input quantization scale (amax(silu_f)/127).
#
# Goal: find a sigmoid approximation that, after int8 quantization, matches the
# float reference to <= 1 LSB for ~all samples. We don't need a perfect sigmoid
# because the downstream D GEMM quantizes to int8 anyway.
import numpy as np

rng = np.random.default_rng(0)

# Realistic gate/up int32 accumulators: int8 activations x int8 weights summed
# over K=2048. std ~ sqrt(2048) * (128/3) * (128/3) ~ 2000 (rough); values are
# large int32. We model gate_i32, up_i32 directly.
n = 1_000_000
gate_i32 = rng.normal(0, 2000, n).astype(np.int64)
up_i32 = rng.normal(0, 2000, n).astype(np.int64)

# Scale S maps int32 -> float. Typical: weight scale ~ 0.01/127, act scale ~
# 1/127, so S ~ 1e-6 -> gate_f ~ gate_i32 * 1e-6 ~ +-0.002 ... that's too small.
# In practice the scales are set so gate_f is O(1). We sweep S so gate_f spans
# a sane range and the sigmoid is exercised.
for S in [2e-4, 5e-4, 1e-3]:
    gate_f = gate_i32 * S
    up_f = up_i32 * S
    # float reference
    silu_f = (gate_f / (1.0 + np.exp(-gate_f))) * up_f
    Q = np.max(np.abs(silu_f)) / 127.0
    silu_i8_ref = np.clip(np.round(silu_f / Q), -127, 127).astype(np.int32)

    # Candidate A: piecewise-linear sigmoid on the scaled int32.
    # sigmoid(x) ~ 0.5 + 0.25*x for |x|<=2 (clamped), which is a hard-ish sigmoid.
    # Work directly in int32 with a fixed-point scale.
    # Represent gate_f in Q12: g = gate_i32 * S * 4096. sigmoid(g) = 0.5 + 0.25*x
    # is a poor approx. Try the "fast sigmoid" x/(1+|x|):
    #   sigmoid(x) ~ 0.5 + 0.5 * x/(1+|x|)
    # in fixed point: x = gate_f (float). Compute in float first to gauge the
    # LUT-vs-poly error, then design the int32 version.
    for name, sig_approx in [
        ("0.5+0.5*x/(1+|x|)", lambda x: 0.5 + 0.5 * x / (1.0 + np.abs(x))),
        ("piecewise 0.5+0.25x clamp", lambda x: np.clip(0.5 + 0.25 * x, 0.0, 1.0)),
        ("poly x^3 (small)", lambda x: np.clip(0.5 + 0.25 * x - x**3 / 48.0, 0.0, 1.0)),
    ]:
        sig = sig_approx(gate_f)
        silu_approx = (gate_f * sig) * up_f
        silu_i8_app = np.clip(np.round(silu_approx / Q), -127, 127).astype(np.int32)
        match = np.mean(np.abs(silu_i8_app - silu_i8_ref) <= 1)
        print(f"S={S:.0e} {name:28s} within-1-LSB={match*100:.3f}%  "
              f"gate range=[{gate_f.min():.2f},{gate_f.max():.2f}]")

# The real fixed-point target: sigmoid as a function of a Q12 fixed-point gate.
# Show what a 256-entry LUT would give (this is the cleanest on-NPU option).
print("\n--- LUT (256-entry, indexed by clamped gate_i32 in a fixed range) ---")
for S in [2e-4, 5e-4, 1e-3]:
    gate_f = gate_i32 * S
    up_f = up_i32 * S
    silu_f = (gate_f / (1.0 + np.exp(-gate_f))) * up_f
    Q = np.max(np.abs(silu_f)) / 127.0
    silu_i8_ref = np.clip(np.round(silu_f / Q), -127, 127).astype(np.int32)
    # 256-entry LUT over gate_f in [-8, 8], index = clip(round((gate_f+8)/16*256))
    lut_x = np.linspace(-8, 8, 256)
    lut_sig = 1.0 / (1.0 + np.exp(-lut_x))
    idx = np.clip(((gate_f + 8.0) / 16.0 * 255.0).astype(np.int32), 0, 255)
    sig = lut_sig[idx]
    silu_lut = (gate_f * sig) * up_f
    silu_i8_lut = np.clip(np.round(silu_lut / Q), -127, 127).astype(np.int32)
    match = np.mean(np.abs(silu_i8_lut - silu_i8_ref) <= 1)
    print(f"S={S:.0e} LUT256 within-1-LSB={match*100:.3f}%  gate_f range=[{gate_f.min():.2f},{gate_f.max():.2f}]")
