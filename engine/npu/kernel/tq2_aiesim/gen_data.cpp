// gen_data.cpp — generate PLIO data files + golden output for the TQ2 sim.
// Same encode as test_tq2_gemv_ref.cc (seed 42 → identical vectors).
//
//   ./gen_data          writes data/in{A,B,S}.txt + golden.txt (LSB-first words)
//   ./gen_data msb      same, but bytes packed MSB-first per 32-bit word
//   ./gen_data check    compares outC.txt (sim) vs golden.txt
//
// PLIO 32-bit hex format: one 8-hex-digit word per line.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

constexpr int M = 32, K = 64, N = 128;
constexpr int GROUP = 32, NGROUPS = 2;
constexpr int BYTES_PER_ROW = K / 4;  // 16 (4 × 2-bit codes per byte)

static bool g_msb = false;
static void write_words(const char *path, const uint8_t *buf, size_t nbytes) {
    FILE *f = fopen(path, "w");
    for (size_t i = 0; i < nbytes; i += 4) {
        uint32_t w = 0;
        for (int j = 0; j < 4 && i + j < nbytes; j++)
            w |= (uint32_t)buf[i + j] << (8 * (g_msb ? 3 - j : j));
        fprintf(f, "%08x\n", w);
    }
    fclose(f);
}

static float bf16_to_float(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }
static uint16_t float_to_bf16(float f) { uint32_t u; memcpy(&u, &f, 4); return (uint16_t)(u >> 16); }

int main(int argc, char **argv) {
    if (argc > 1 && std::string(argv[1]) == "check") {
        // outC.txt lines: "0xLLLL 0xHHHH" (two bf16 halves, low first)
        FILE *fg = fopen("golden.txt", "r"), *fo = fopen("outC.txt", "r");
        if (!fg || !fo) { printf("missing golden.txt or outC.txt\n"); return 1; }
        std::vector<uint16_t> out;
        char t1[16], t2[16];
        while (fscanf(fo, "%15s %15s", t1, t2) == 2) {
            out.push_back((uint16_t)strtol(t1, nullptr, 16));
            out.push_back((uint16_t)strtol(t2, nullptr, 16));
        }
        double maxerr = 0; int idx = 0; float g;
        while (fscanf(fg, "%f", &g) == 1 && idx < M * N && idx < (int)out.size()) {
            maxerr = std::max(maxerr, std::abs((double)g - bf16_to_float(out[idx])));
            idx++;
        }
        printf("compared %d values, max abs error: %.6f\n", idx, maxerr);
        bool ok = idx == M * N && maxerr < 0.1;
        printf("%s\n", ok ? "PASS" : "FAIL");
        return ok ? 0 : 1;
    }

    g_msb = argc > 1 && std::string(argv[1]) == "msb";
    srand(42);
    std::vector<int8_t> A(M * K), W(N * K);
    std::vector<uint8_t> B(N * BYTES_PER_ROW);
    std::vector<uint16_t> Sb(N * NGROUPS);
    std::vector<float> S(N * NGROUPS);
    for (auto &v : A) v = (int8_t)(rand() % 21 - 10);
    for (int i = 0; i < N * NGROUPS; i++) {
        S[i] = (float)(rand() % 100) / 50.0f - 1.0f;
        Sb[i] = float_to_bf16(S[i]);
    }
    for (int n = 0; n < N; n++) {
        for (int k = 0; k < K; k++) {
            int c = rand() & 3;              // code 0=-1, 1=0, 2=+1, 3=0
            W[n * K + k] = c == 0 ? -1 : (c == 2 ? 1 : 0);
        }
        for (int i = 0; i < BYTES_PER_ROW; i++) {
            uint8_t b = 0;
            for (int p = 0; p < 4; p++) {
                int c = (W[n * K + 4 * i + p] == -1) ? 0 : (W[n * K + 4 * i + p] == 1 ? 2 : 1);
                b |= (uint8_t)c << (2 * p);
            }
            B[n * BYTES_PER_ROW + i] = b;
        }
    }

    if (system("mkdir -p data") != 0) return 1;
    write_words("data/inA.txt", (uint8_t *)A.data(), A.size());
    write_words("data/inB.txt", B.data(), B.size());
    write_words("data/inS.txt", (uint8_t *)Sb.data(), Sb.size() * 2);

    // golden: C[m][n] = sum_k A[m][k] * S[n][k/GROUP] * W[n][k]
    FILE *fg = fopen("golden.txt", "w");
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float sum = 0;
            for (int k = 0; k < K; k++)
                sum += (float)A[m * K + k] * S[n * NGROUPS + k / GROUP] * W[n * K + k];
            fprintf(fg, "%.6f\n", sum);
        }
    fclose(fg);
    printf("data written (A=%d B=%d S=%d, golden %d values)\n",
           (int)A.size(), (int)B.size(), (int)S.size(), M * N);
    return 0;
}
