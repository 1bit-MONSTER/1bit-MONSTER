// tq2_graph.h — minimal ADF graph: single tile running the TQ2 GEMV kernel.
// Windows (bytes): A = 32*64 int8, B = 128*16 packed TQ2, S = 128*2 bf16,
// C = 32*128 bf16. PLIO data files are hex words (hex=true).
#pragma once
#include <adf.h>

void tq2_gemv_adf(input_window_int8 *, input_window_uint8 *,
                  input_window_uint16 *, output_window_uint16 *);

constexpr int M = 32, K = 64, N = 128;
constexpr int A_BYTES = M * K;        // 2048
constexpr int B_BYTES = N * K / 4;    // 2048 (16 bytes/row, 2-bit codes)
constexpr int S_BYTES = N * 2 * 2;    // 512
constexpr int C_BYTES = M * N * 2;    // 8192

class Tq2Graph : public adf::graph {
public:
    adf::input_plio inA, inB, inS;
    adf::output_plio outC;
    adf::kernel k;

    Tq2Graph() {
        k = adf::kernel::create(tq2_gemv_adf);
        adf::source(k) = "tq2_kernel_adf.cc";
        adf::runtime<adf::ratio>(k) = 0.9;

        inA = adf::input_plio::create("inA", adf::plio_32_bits, "data/inA.txt", 0.0, true);
        inB = adf::input_plio::create("inB", adf::plio_32_bits, "data/inB.txt", 0.0, true);
        inS = adf::input_plio::create("inS", adf::plio_32_bits, "data/inS.txt", 0.0, true);
        outC = adf::output_plio::create("outC", adf::plio_32_bits, "outC.txt", 0.0, true);

        adf::connect<adf::window<A_BYTES>>(inA.out[0], k.in[0]);
        adf::connect<adf::window<B_BYTES>>(inB.out[0], k.in[1]);
        adf::connect<adf::window<S_BYTES>>(inS.out[0], k.in[2]);
        adf::connect<adf::window<C_BYTES>>(k.out[0], outC.in[0]);
    }
};
