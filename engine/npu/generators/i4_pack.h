// i4_pack.h — symmetric int4 (Q4NX-style) pack/unpack for the fused decode
//             weight path (issue #1769).
//
// Compiled into BOTH the AIE kernel (mm_kernel_reference.cc / the future
// int4 dequant stage, via the AIE2P Peano toolchain) AND the host-side CPU
// reference (engine/npu/tests/test_i4_pack.cpp) so the exact bit-level
// contract is pinned on x86 BEFORE the NPU round-trip — same dual-compile
// discipline as silu_quant.h. NO AIE intrinsics, no <cmath>/libm — plain
// scalar C the AIE2P scalar unit lowers to hardware instructions.
//
// ── Contract (Phase-1 MVP from the #1769 plan) ────────────────────────────
//   Codes:       symmetric two's-complement int4, q ∈ [-8, 7].
//   Pack:        q4 = clamp(round(x / 16), -8, 7); nibble = q4 & 0xF.
//   Unpack:      int8 = sign-extend(nibble) << 4   (the ×16 is FOLDED into
//                the existing per-column/per-group scales, so the kernel's
//                dequant stage is a pure nibble→int8 expansion — byte-
//                identical pipeline, zero extra metadata).
//   Layout:      nibble pairs along the innermost (contiguous) axis;
//                one byte holds elements (2j, 2j+1) with the EVEN element
//                in the LOW nibble. For the fused B tiles this is the
//                col-in-8 mmul axis (byte s = i0*1024 + i1*64 + i2*8 + i3,
//                i3 = col-in-8): 8 elements → 4 bytes, exactly half the
//                int8 streamed bytes (~12.6 MB → ~6.3 MB/layer).
//
//   round(x/16): x + (x >= 0 ? 8 : -8) then truncating /16 — symmetric
//                round-half-away-from-zero, no float involved (int8→int4
//                must not depend on FP rounding in the kernel).
#pragma once

#include <cstdint>
#include <cstddef>

// Pack n int8 values into n/2 bytes (n even). Consecutive pairs (x[2j],
// x[2j+1]); x[2j] → low nibble, x[2j+1] → high nibble.
static inline void pack_i8_to_i4(const int8_t* __restrict src, size_t n,
                                 uint8_t* __restrict dst) {
    for (size_t j = 0; j < n / 2; j++) {
        int8_t a = src[2 * j];
        int8_t b = src[2 * j + 1];
        // clamp(round(x/16), -8, 7): round-half-away-from-zero in integers.
        int qa = (a >= 0 ? (a + 8) : (a - 8)) / 16;
        int qb = (b >= 0 ? (b + 8) : (b - 8)) / 16;
        if (qa > 7) qa = 7; else if (qa < -8) qa = -8;
        if (qb > 7) qb = 7; else if (qb < -8) qb = -8;
        dst[j] = (uint8_t)((uint8_t)(qb & 0x0F) << 4) | (uint8_t)(qa & 0x0F);
    }
}

// Unpack n/2 bytes back into n int8 values (n even). The ×16 scale fold
// means unpacked = sign-extend(nibble) << 4 — the kernel consumes these
// int8 values directly; the per-column scale absorbs the 1/16.
static inline void unpack_i4_to_i8(const uint8_t* __restrict src, size_t n,
                                   int8_t* __restrict dst) {
    for (size_t j = 0; j < n / 2; j++) {
        uint8_t byte = src[j];
        int8_t a = (int8_t)((byte & 0x0F) << 4) >> 4;  // sign-extend low nibble
        int8_t b = (int8_t)((byte >> 4) << 4) >> 4;    // sign-extend high nibble
        dst[2 * j]     = (int8_t)(a << 4);
        dst[2 * j + 1] = (int8_t)(b << 4);
    }
}
