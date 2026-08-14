// deepseek_v3_gating_selfcheck.cpp — synthetic validation of the DeepSeek-V3
// MoE gating path in src/deepseek.cpp (#1637). No checkpoint needed: the
// gating math is deterministic — given known router logits, the expected
// selection is hand-computable. Covers sigmoid scoring, the correction bias
// (selection only), group-limited greedy (top-2-per-group sum), norm_topk_prob,
// and routed_scaling.
//
// Run:
//   g++ -std=c++17 -Iinclude -Isrc Testing/deepseek_v3_gating_selfcheck.cpp \
//       src/deepseek.cpp src/gguf_reader.cpp -o /tmp/dsv3_check && /tmp/dsv3_check
#include <cstdio>
#include <cmath>
#include <vector>
#include <cstring>
#include "deepseek.h"

static int fails = 0, total = 0;
static void ok(bool cond, const char* label) {
    ++total;
    if (!cond) { std::printf("FAIL %s\n", label); ++fails; }
}

int main() {
    using namespace deepseek_math;

    // ── Case 1: V2-Lite softmax path (regression — the validated path) ────
    // logits -> softmax -> greedy top-k with raw weights (norm_topk_prob=0).
    {
        DeepSeekConfig cfg;
        cfg.score_func = 0; cfg.n_expert_groups = 1; cfg.n_limited_groups = 1;
        cfg.routed_scaling = 1.0f; cfg.norm_topk_prob = 0;
        float logits[8] = {0.1f, 0.5f, 0.2f, 0.9f, 0.05f, 0.7f, 0.3f, 0.6f};
        int ids[8]; float wts[8];
        select_experts(logits, 8, 4, cfg, nullptr, ids, wts);
        // softmax: the top-4 by logit are 3 (0.9), 5 (0.7), 7 (0.6), 1 (0.5)
        ok(ids[0] == 3 && ids[1] == 5 && ids[2] == 7 && ids[3] == 1, "V2 softmax top-4 order");
        // weights are the raw softmax probs
        float p[8]; memcpy(p, logits, sizeof(p)); softmax_inplace(p, 8);
        ok(fabsf(wts[0] - p[3]) < 1e-6f, "V2 weight = raw softmax prob");
    }

    // ── Case 2: V3 sigmoid + bias + group-limited + norm + scaling ─────────
    // 8 experts, 4 groups of 2, top_k=4, n_limited_groups=2, routed_scaling=2.5,
    // norm_topk_prob=1. Bias flips the top group selection.
    {
        DeepSeekConfig cfg;
        cfg.score_func = 1; cfg.n_expert_groups = 4; cfg.n_limited_groups = 2;
        cfg.routed_scaling = 2.5f; cfg.norm_topk_prob = 1;
        float logits[8] = {2.0f, 1.0f, 0.5f, 0.2f, 1.5f, 0.1f, 0.3f, 0.4f};
        // sigmoid(logits): 0.881 0.731 0.622 0.550 0.818 0.525 0.574 0.599
        // bias: push group 3's experts up so groups {0,3} win over {0,2}
        float bias[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f};
        // sel = sigmoid + bias:
        //   g0: 0.881, 0.731 -> group score 1.612
        //   g1: 0.622, 0.550 -> 1.172
        //   g2: 0.818, 0.525 -> 1.343
        //   g3: 1.574, 1.599 -> 3.173   (bias lifted g3 to the top)
        // top-2 groups: g3 (3.173), g0 (1.612). g1, g2 masked.
        // top-4 experts within {g0, g3}: 6 (1.574), 7 (1.599), 0 (0.881), 1 (0.731)
        int ids[8]; float wts[8];
        select_experts(logits, 8, 4, cfg, bias, ids, wts);
        ok(ids[0] == 7 && ids[1] == 6 && ids[2] == 0 && ids[3] == 1, "V3 group-limited selection order");
        // weights = raw sigmoid of the selected (unbiased), normalized, x 2.5
        float s0 = 1.0f/(1.0f+expf(-2.0f));  // expert 0
        float s1 = 1.0f/(1.0f+expf(-1.0f));  // expert 1
        float s6 = 1.0f/(1.0f+expf(-0.3f));  // expert 6
        float s7 = 1.0f/(1.0f+expf(-0.4f));  // expert 7
        float sum = s0 + s1 + s6 + s7;
        ok(fabsf(wts[0] - (s7/sum)*2.5f) < 1e-5f, "V3 weight = sigmoid normalized x routed_scaling");
    }

    // ── Case 3: group-limited WITHOUT bias (V3 no-bias fallback) ──────────
    {
        DeepSeekConfig cfg;
        cfg.score_func = 1; cfg.n_expert_groups = 4; cfg.n_limited_groups = 2;
        cfg.routed_scaling = 1.0f; cfg.norm_topk_prob = 0;
        float logits[8] = {2.0f, 1.0f, 0.5f, 0.2f, 1.5f, 0.1f, 0.3f, 0.4f};
        int ids[8]; float wts[8];
        select_experts(logits, 8, 4, cfg, nullptr, ids, wts);
        // no bias: g0 (1.612), g2 (1.343) are the top-2 groups; g3 (1.173), g1 (1.172)
        // top-4 within {g0, g2}: experts 0 (0.881), 4 (0.818), 1 (0.731), 5 (0.525)
        ok(ids[0] == 0 && ids[1] == 4 && ids[2] == 1 && ids[3] == 5, "V3 group-limited without bias");
    }

    std::printf("DSV3 GATING SELFCHECK: %d/%d %s\n", total - fails, total, fails ? "FAILED" : "passed");
    return fails ? 1 : 0;
}
