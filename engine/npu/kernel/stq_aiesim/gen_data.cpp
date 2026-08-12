// gen_data.cpp — generate PLIO data files + golden output for the STQ sim.
// Same encode as test_stq_gemv_ref.cc (seed 42 → identical vectors).
//
//   ./gen_data          writes data/in{A,B,S}.txt + golden.txt (LSB-first words)
//   ./gen_data msb      same, but bytes packed MSB-first per 32-bit word
//   ./gen_data check    compares outC.txt (sim) vs golden.txt
//
// PLIO 32-bit hex format: one 8-hex-digit word per line.
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

constexpr int M = 32, K = 64, N = 128;
constexpr int GROUP = 32, NGROUPS = 2;
constexpr int BLOCKS_PER_ROW = K / 4;              // 16
constexpr int BYTES_PER_ROW = BLOCKS_PER_ROW * 5 / 8; // 10

static uint8_t stq_encode(const int8_t w[4]) {
    int zero_pos = -1, nz = 0;
    for (int p = 0; p < 4; p++) {
        if (w[p] == 0) { assert(zero_pos < 0); zero_pos = p; }
        else { assert(w[p] == 1 || w[p] == -1); nz++; }
    }
    assert(zero_pos >= 0 && nz == 3);
    uint8_t signs = 0; int si = 0;
    for (int p = 0; p < 4; p++)
        if (p != zero_pos) { if (w[p] == 1) signs |= 1 << si; si++; }
    return (zero_pos << 3) | signs;
}

static void stq_pack_row(const int8_t *w, uint8_t *out) {
    for (int i = 0; i < BYTES_PER_ROW; i++) out[i] = 0;
    for (int b = 0; b < BLOCKS_PER_ROW; b++) {
        uint8_t code = stq_encode(w + 4 * b);
        int bit = 5 * b;
        out[bit / 8] |= code << (bit % 8);
        out[bit / 8 + 1] |= code >> (8 - bit % 8);
    }
}

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


    if (argc > 2 && std::string(argv[1]) == "probe") {
        int kp = atoi(argv[2]);
        srand(42);
        std::vector<int8_t> A(M * K, 0), W(N * K);
        std::vector<uint8_t> B(N * BYTES_PER_ROW);
        std::vector<uint16_t> Sb(N * NGROUPS);
        bool fingerprint = getenv("FINGERPRINT") != nullptr;
        for (int n = 0; n < N; n++) for (int g = 0; g < NGROUPS; g++)
            Sb[n * NGROUPS + g] = float_to_bf16(fingerprint ? (float)(n + 1) : 1.0f);
        const char* pm = getenv("PROBE_M");
        if (pm) { A[atoi(pm) * K + kp] = 1; }
        else for (int m = 0; m < M; m++) A[m * K + kp] = 1;
        // consume rand in same order as normal mode for identical W
        for (int i = 0; i < M * K; i++) rand();
        for (int i = 0; i < N * NGROUPS; i++) rand();
        for (int n = 0; n < N; n++) {
            for (int b = 0; b < BLOCKS_PER_ROW; b++) {
                int zp = rand() % 4;
                for (int p = 0; p < 4; p++)
                    W[n * K + 4 * b + p] = p == zp ? 0 : (rand() & 1 ? 1 : -1);
            }
            stq_pack_row(&W[n * K], &B[n * BYTES_PER_ROW]);
        }
        if (system("mkdir -p data") != 0) return 1;
        write_words("data/inA.txt", (uint8_t *)A.data(), A.size());
        write_words("data/inB.txt", B.data(), B.size());
        write_words("data/inS.txt", (uint8_t *)Sb.data(), Sb.size() * 2);
        { FILE *fw = fopen("W_dump.txt","w"); for (int n=0;n<N;n++){ for(int k=0;k<K;k++) fprintf(fw,"%d ",W[n*K+k]); fprintf(fw,"\n"); } fclose(fw); }
        // expected: C[m][n] = W[n][kp]
        FILE *fg = fopen("probe_expect.txt", "w");
        for (int n = 0; n < N; n++) fprintf(fg, "%d\n", W[n * K + kp]);
        fclose(fg);
        printf("probe k=%d written\n", kp);
        return 0;
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
        for (int b = 0; b < BLOCKS_PER_ROW; b++) {
            int zp = rand() % 4;
            for (int p = 0; p < 4; p++)
                W[n * K + 4 * b + p] = p == zp ? 0 : (rand() & 1 ? 1 : -1);
        }
        stq_pack_row(&W[n * K], &B[n * BYTES_PER_ROW]);
    }

    if (system("mkdir -p data") != 0) return 1;
    write_words("data/inA.txt", (uint8_t *)A.data(), A.size());
    write_words("data/inB.txt", B.data(), B.size());
    write_words("data/inS.txt", (uint8_t *)Sb.data(), Sb.size() * 2);

    FILE *fg = fopen("golden.txt", "w");
    for (int m = 0; m < M; m++)
        for (int n = 0; n < N; n++) {
            float sum = 0;
            for (int k = 0; k < K; k++)
                sum += (float)A[m * K + k] * W[n * K + k] * S[n * NGROUPS + k / GROUP];
            fprintf(fg, "%f\n", bf16_to_float(float_to_bf16(sum)));
        }
    fclose(fg);
    printf("data written (%s): inA %zuB inB %zuB inS %zuB, golden %d vals\n",
           g_msb ? "MSB-first" : "LSB-first", A.size(), B.size(), Sb.size() * 2, M * N);
    return 0;
}
