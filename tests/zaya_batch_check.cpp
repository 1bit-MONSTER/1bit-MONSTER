// zaya_batch_check.cpp — issues #1712/#1713 regression: zaya_forward_batch
// must match zaya_forward (single-token) at pos=0.
//
// The batch path is a single-position evaluator: attention is concat(V1, 0)
// (no KV cache, no conv state, no vrec delay line — exact at pos=0 because
// the single-token flash-decode softmaxes over one cached element, output=V)
// and the router scores the same 2 slots as the single-token EDA router
// (eda_router_gpu_kernel router_top_k=2, reference: tests/zaya_gpu_decode.cpp).
//
// Skips when the model file is absent (like issue1527_loader_check). A model
// needs real ZAYA attention + MoE layers to exercise the path — e.g.
// ZAYA1-8B.1bp: ./zaya_batch_check models/ZAYA1-8B.1bp
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <sys/stat.h>
#include "zaya_engine.h"
#include "onebp_format.h"

static bool file_exists(const char* p) { struct stat st; return stat(p, &st) == 0; }

// Build the runtime config from the .1bp header (same struct the engine reads).
static ZayaConfig cfg_from_header(const char* path) {
    FILE* f = fopen(path, "rb");
    OnebpHeader h;
    if (!f || fread(&h, 1, sizeof(h), f) != sizeof(h) || !h.valid()) {
        if (f) fclose(f);
        return ZayaConfig{};
    }
    fclose(f);
    int n_ff = h.n_ff_exp > 0 ? (int)h.n_ff_exp : (int)h.intermediate_size;
    return ZayaConfig::from_model(h.hidden_size, h.num_layers, h.num_attention_heads,
                                  h.num_kv_heads, h.head_dim, h.vocab_size,
                                  (int)h.num_experts, n_ff, 256,
                                  h.max_seq_len > 0 ? (int)h.max_seq_len : 4096);
}

// Compare B batch logits against B single-token references; returns 0 if
// every token matches within tolerance (fp16 logits, equivalent-but-distinct
// kernels: mm_k vs moe_tiled_gemv, wmma vs serial expert FFN, batched vs
// tiled lm_head — so bitwise equality is not expected).
static int compare(const char* tag, const std::vector<float>& batch,
                   const std::vector<float>* ref, int B, int vocab) {
    float worst = 0.f;
    int worst_tok = -1, big = 0, nan = 0;
    for (int b = 0; b < B; b++) {
        for (int v = 0; v < vocab; v++) {
            float d = fabsf(batch[(size_t)b * vocab + v] - ref[b][v]);
            if (std::isnan(d)) { nan++; worst = NAN; continue; }  // NaN is a mismatch, not a pass
            if (d > worst) { worst = d; worst_tok = b; }
            if (d > 0.1f) big++;
        }
    }
    printf("  %-6s max|diff|=%.5f (token %d)  logits>0.1 apart: %d  NaN pairs: %d\n",
           tag, worst, worst_tok, big, nan);
    return (worst < 0.05f && nan == 0) ? 0 : 1;
}

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "models/ZAYA1-8B.1bp";
    if (!file_exists(path)) {
        fprintf(stderr, "SKIP: %s not present\n", path);
        return 0;
    }
    ZayaConfig cfg = cfg_from_header(path);
    if (cfg.h == 0 || cfg.vocab == 0) {
        fprintf(stderr, "FAIL: unreadable .1bp header %s\n", path);
        return 1;
    }
    ZayaState* s = zaya_init_onebp(path, &cfg);
    if (!s) {
        fprintf(stderr, "FAIL: zaya_init_onebp(%s)\n", path);
        return 1;
    }
    const int vocab = cfg.vocab;
    const int seed_tokens[] = {2, 7, 42, 1234};
    int tokens[4];
    for (int i = 0; i < 4; i++) tokens[i] = seed_tokens[i] % vocab;  // in-range, distinct for vocab > 42
    const int Bs[] = {1, 2, 4};             // 1/2 = fused path, 4 = sorted path
    int fails = 0;

    for (int B : Bs) {
        // Single-token references: fresh reset per token (pos=0, vrec=0).
        std::vector<std::vector<float>> ref(B, std::vector<float>(vocab));
        for (int b = 0; b < B; b++) {
            zaya_reset(s);
            zaya_forward(s, tokens[b], ref[b].data());
        }
        // Batch: all B tokens at pos=0 in one call.
        zaya_reset(s);
        std::vector<float> batch((size_t)B * vocab);
        zaya_forward_batch(s, tokens, batch.data(), B);

        char tag[16];
        snprintf(tag, sizeof(tag), "B=%d", B);
        fails += compare(tag, batch, ref.data(), B, vocab);
    }

    zaya_destroy(s);
    if (fails) { fprintf(stderr, "FAIL: %d batch-vs-single-token mismatch(es)\n", fails); return 1; }
    printf("PASS: zaya_forward_batch matches zaya_forward at pos=0 (B=1,2,4)\n");
    return 0;
}
