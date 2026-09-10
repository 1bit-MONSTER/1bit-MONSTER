// verify_bf16_qkv.cpp — end-to-end verification: real Qwen3-0.6B layer-0 BO
// (npu_pack_layer_bo) → dequant.xclbin (QKV bf16 W) → mm.xclbin 2-batch Q GEMM,
// compared against a CPU bf16 reference (A_sparse × Q_W).
#include <climits>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include "model.h"
#include "common.h"

extern "C" int npu_pack_layer_bo(uint8_t* bo_buffer, void* mw, const void* config, int layer_idx);

// bridge
extern "C" int bf16mm_init(const char* model_dir, const char* xclbin_dir);
extern "C" void bf16mm_dequant(uint16_t* wout, const uint8_t* q4nx, uint32_t D_in, uint32_t D_out, uint32_t q4nx_weight_offset);
extern "C" void bf16mm_gemm_2batch(uint16_t* C, const uint16_t* A, const uint16_t* W, uint32_t K, uint32_t N, uint32_t woff_elements);

static inline float b2f(uint16_t h){ uint32_t u=((uint32_t)h)<<16; float f; memcpy(&f,&u,4); return f; }
static inline uint16_t f2b(float f){ uint32_t u; memcpy(&u,&f,4); return (uint16_t)((u+0x7FFF+((u>>16)&1))>>16); }

int main(int argc, char** argv) {
    const char* model_path = (argc > 1) ? argv[1] : "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx";
    const char* model_dir  = (argc > 2) ? argv[2] : "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2";
    const char* xclbin_dir = (argc > 3) ? argv[3] : "/home/bcloud/amd-oss/fastflowlm/src/xclbins/Qwen3-0.6B-NPU2";

    // 1. load weights + pack layer 0 BO (10 MB)
    ModelConfig cfg = QWEN3_0_6B_CONFIG;
    ModelWeights* mw = model_load(model_path, cfg);
    if (!mw) { fprintf(stderr, "model_load failed\n"); return 1; }
    const size_t BO_BYTES = 2048 * 5120;   // 10 MB fixed layer BO
    std::vector<uint8_t> layer_bo(BO_BYTES, 0);
    int tiles = npu_pack_layer_bo(layer_bo.data(), mw, &cfg, 0);
    fprintf(stderr, "packed layer 0: %d tiles\n", tiles);
    if (tiles <= 0) return 1;

    // 2. init bf16mm (mm.xclbin + dequant.xclbin)
    if (!bf16mm_init(model_dir, xclbin_dir)) { fprintf(stderr, "bf16mm_init failed\n"); return 1; }
    fprintf(stderr, "bf16mm init ok\n");

    // 3. dequant QKV W (1024×4096 → 8 MB)
    const int D_in = 1024, D_qkv = 4096;
    std::vector<uint16_t> Wqkv((size_t)D_in * D_qkv, 0);
    bf16mm_dequant(Wqkv.data(), layer_bo.data(), D_in, D_qkv, 0);
    // sanity: Wqkv non-zero?
    int wnz = 0; for (size_t i = 0; i < Wqkv.size(); i++) if (Wqkv[i]) wnz++;
    fprintf(stderr, "QKV W dequant: %d/%zu non-zero\n", wnz, Wqkv.size());
    if (wnz == 0) { fprintf(stderr, "FAIL: W all zero\n"); return 1; }

    // 4. synthetic sparse A: A[k][0] = k+1 (markers 1..256, exact bf16)
    const int M = 256, K = 1024, N = 2048;
    std::vector<uint16_t> A((size_t)M * K, 0), Q((size_t)M * N, 0);
    for (int k = 0; k < M; k++) A[k * K] = f2b((float)(k + 1));

    // 5. 2-batch Q GEMM
    bf16mm_gemm_2batch(Q.data(), A.data(), Wqkv.data(), K, N, 0);

    // 6. CPU reference: C[k][n] = A[k][0] * Q_W[0][n] = (k+1) * Wqkv[n] (n<2048)
    int bad = 0, tol_bad = 0;
    for (int k = 0; k < M; k++) {
        float mk = (float)(k + 1);
        for (int n = 0; n < N; n++) {
            float w = b2f(Wqkv[n]);            // Q_W[0][n] = Wqkv[0*2048 + n]
            float ref = b2f(f2b(mk * w));
            float got = b2f(Q[k * N + n]);
            float d = got - ref;
            if (d != 0.0f) {
                if (fabsf(d) <= 2.0f) tol_bad++; else bad++;
                if (bad + tol_bad < 20)
                    fprintf(stderr, "k=%d n=%d got=%.4f ref=%.4f (w=%.6f)\n", k, n, got, ref, w);
            }
        }
    }
    fprintf(stderr, "Q GEMM vs CPU ref: exact-mismatch=%d tol(<=2)=%d / %d elems\n", bad, tol_bad, M * N);
    // summary: per-row check of first 4 cols
    fprintf(stderr, "row 0 cols0-3: %.4f %.4f %.4f %.4f\n", b2f(Q[0]), b2f(Q[1]), b2f(Q[2]), b2f(Q[3]));
    fprintf(stderr, "row 127 cols0-3: %.4f %.4f %.4f %.4f\n", b2f(Q[127*N]), b2f(Q[127*N+1]), b2f(Q[127*N+2]), b2f(Q[127*N+3]));
    fprintf(stderr, "row 128 cols0-3: %.4f %.4f %.4f %.4f\n", b2f(Q[128*N]), b2f(Q[128*N+1]), b2f(Q[128*N+2]), b2f(Q[128*N+3]));
    fprintf(stderr, "row 255 cols0-3: %.4f %.4f %.4f %.4f\n", b2f(Q[255*N]), b2f(Q[255*N+1]), b2f(Q[255*N+2]), b2f(Q[255*N+3]));
    model_free(mw);
    return (bad == 0) ? 0 : 1;
}
