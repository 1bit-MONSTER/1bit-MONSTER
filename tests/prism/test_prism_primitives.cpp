// test_prism_primitives.cpp — Prism numerics primitives (plan P2, first step).
//
// A. include/prism_codec.h::dequant_block() is cross-checked against the already
//    verified GGUF reader (src/gguf_reader.cpp + include/gguf_reader.h, whose
//    dequant was itself validated against Prism's runtime/codec.py) on **real file
//    bytes** for all three packings.
// B. The Hadamard contract (forward = signs then H, inverse = H then signs, block
//    size 1024, normalized) is checked for round-trip, orthonormality, block
//    independence, a hand-computed 2-point case, and rejection of bad geometry.
// C. `--dump <width> <block>` prints the forward transform of a deterministic input
//    so tests/prism/dump_prism_fwht.py can diff it against an independent Python
//    implementation that builds the Hadamard matrix explicitly (naive O(B^2), so it
//    shares no code path with the butterfly).
//
// Build:
//   g++ -O2 -std=c++17 -I include -I src tests/prism/test_prism_primitives.cpp \
//       src/gguf_reader.cpp -o /tmp/tpp
// Run:
//   /tmp/tpp <model.gguf>          # parts A+B
//   /tmp/tpp --dump 2048 1024      # part C
#include "prism_codec.h"
#include "gguf_reader.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

static int failures = 0;

static void check(const char* what, bool ok, const char* detail = "") {
    std::printf("  %s %s%s%s\n", ok ? "ok  " : "FAIL", what,
                detail[0] ? " — " : "", detail);
    if (!ok) failures++;
}

// Deterministic test input shared with the Python dumper. Deliberately STATELESS:
// a mutable generator's call order is easy to get subtly different across languages
// (we hit exactly that), and a mismatch would then look like a transform bug.
//   x[i] = (i % 7 - 3) / 3     — exact rationals, identical in both languages
//   s[i] = +1 / -1 from an integer mix of i
inline float test_value(int i) { return (float)((i % 7) - 3) / 3.0f; }
inline int8_t test_sign(int i) {
    const uint32_t h = (uint32_t)i * 2654435761u;
    return ((h >> 16) & 1u) ? (int8_t)1 : (int8_t)-1;
}

static void partA(const char* gguf) {
    std::printf("A. dequant_block vs gguf_reader on real bytes\n");
    GgufReader r;
    if (!r.open(gguf)) { check("open model", false, gguf); return; }
    const uint32_t dtypes[3] = {GGUF_DTYPE_Q1_0, GGUF_DTYPE_PQ2_0, GGUF_DTYPE_PTQ1_0};
    const uint32_t quants[3] = {11, 12, 13};
    for (int k = 0; k < 3; k++) {
        const auto* inf = (const GgufTensorInfo*)nullptr;
        std::string name;
        for (const auto& n : r.tensor_names()) {
            const GgufTensorInfo* i = r.tensor_info(n);
            if (i && i->dtype == dtypes[k] && i->shape.size() == 2) { name = n; inf = i; break; }
        }
        if (!inf) { std::printf("     (no tensor of GGUF dtype %u)\n", dtypes[k]); continue; }
        const uint32_t nb = prism::block_bytes(quants[k]);
        std::vector<uint8_t> raw;
        if (!r.get_tensor_raw(name, 128, (int)nb, raw, nullptr)) {
            check("get_tensor_raw", false, name.c_str());
            continue;
        }
        // two blocks: the first and the last of row 0
        const int width = (int)inf->shape[0];
        const size_t blocks_per_row = (size_t)width / 128;
        std::vector<float> a(128), b(128);
        bool all_ok = true;
        for (size_t blk : {(size_t)0, blocks_per_row - 1}) {
            prism::dequant_block(quants[k], raw.data() + blk * nb, a.data());
            gguf_dequant(dtypes[k], raw.data() + blk * nb, b.data(), 128);
            if (std::memcmp(a.data(), b.data(), 128 * sizeof(float)) != 0) all_ok = false;
        }
        char detail[160];
        std::snprintf(detail, sizeof(detail), "%s (%s, %zu blocks/row)",
                      name.c_str(), quants[k] == 11 ? "Q1_0" : quants[k] == 12 ? "PQ2_0" : "PTQ1_0",
                      blocks_per_row);
        check("verbatim dequant matches the reader", all_ok, detail);
    }
}

static void partB() {
    std::printf("B. Hadamard contract\n");
    const int W = 5120, B = 1024;
    std::vector<float> x(W), orig(W);
    std::vector<int8_t> signs(W);
    for (int i = 0; i < W; i++) { x[i] = test_value(i); signs[i] = test_sign(i); }
    orig = x;

    check("forward rejects a non-power-of-two block", !prism::hadamard_forward(x.data(), W, 1000, signs.data()));
    check("forward rejects a width that is not a multiple of the block",
          !prism::hadamard_forward(x.data(), W + 1, B, signs.data()));

    prism::hadamard_forward(x.data(), W, B, signs.data());
    // per-block L2 norm is preserved (H/sqrt(B) is orthonormal)
    bool norm_ok = true;
    for (int base = 0; base < W; base += B) {
        double n0 = 0, n1 = 0;
        for (int i = 0; i < B; i++) { n0 += (double)orig[base + i] * orig[base + i]; n1 += (double)x[base + i] * x[base + i]; }
        if (std::fabs(n0 - n1) > 1e-3 * (1.0 + n0)) norm_ok = false;
    }
    check("forward preserves the per-block norm", norm_ok);

    // block independence: block 1 must not move when block 0 changes
    std::vector<float> y = orig;
    y[0] += 1000.0f;
    prism::hadamard_forward(y.data(), W, B, signs.data());
    bool indep = true;
    for (int i = B; i < W; i++) if (std::fabs(y[i] - x[i]) > 1e-4f) indep = false;
    check("blocks are independent", indep);

    // round-trip: inverse(forward(x)) == x
    prism::hadamard_inverse(x.data(), W, B, signs.data());
    float worst = 0.0f;
    for (int i = 0; i < W; i++) worst = std::fmax(worst, std::fabs(x[i] - orig[i]));
    check("inverse undoes forward", worst < 1e-4f, "max abs error printed below");
    if (worst >= 1e-4f) std::printf("     max abs error = %.3e\n", (double)worst);

    // hand-computed 2-point case: H = [[1,1],[1,-1]], scale 1/sqrt(2), no signs
    float two[2] = {1.0f, 0.0f};
    prism::hadamard_forward(two, 2, 2, nullptr);
    const float inv = (float)(1.0 / std::sqrt(2.0));
    check("2-point transform matches [[1,1],[1,-1]]/sqrt(2)",
          std::fabs(two[0] - inv) < 1e-6f && std::fabs(two[1] - inv) < 1e-6f);

    // sign placement: forward must apply signs BEFORE the transform
    float p[2] = {1.0f, 1.0f};
    const int8_t sgn[2] = {-1, 1};
    prism::hadamard_forward(p, 2, 2, sgn);      // [-1,1] -> H/sqrt2 -> [0,-sqrt2]
    check("forward applies signs before the transform",
          std::fabs(p[0]) < 1e-6f && std::fabs(p[1] + (float)std::sqrt(2.0)) < 1e-5f);
}

static void partC(int width, int block) {
    std::vector<float> x(width);
    std::vector<int8_t> signs(width);
    for (int i = 0; i < width; i++) { x[i] = test_value(i); signs[i] = test_sign(i); }
    prism::hadamard_forward(x.data(), width, block, signs.data());
    std::printf("width=%d block=%d\n", width, block);
    for (int i = 0; i < width; i++) std::printf("%.9e\n", (double)x[i]);
}

int main(int argc, char** argv) {
    if (argc >= 4 && std::strcmp(argv[1], "--dump") == 0) {
        partC(std::atoi(argv[2]), std::atoi(argv[3]));
        return 0;
    }
    std::printf("test_prism_primitives\n");
    if (argc < 2) { std::fprintf(stderr, "usage: %s <model.gguf> | --dump <width> <block>\n", argv[0]); return 2; }
    partA(argv[1]);
    partB();
    std::printf("%s (%d failure%s)\n", failures ? "FAILED" : "ALL CHECKS PASSED",
                failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
