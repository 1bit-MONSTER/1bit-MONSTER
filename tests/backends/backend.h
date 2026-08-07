// backend.h — Inference backend interface (tests/ version)
//
// Uses the canonical BackendType and ModelConfig from include/common.h.
// This file provides the simplified InferenceBackend interface used by
// zaya_server. The canonical src::Backend interface is in src/backend.h.
//
// Key difference:
//   InferenceBackend::forward() = fuse forward+lm_head -> returns token_id
//   src::Backend::forward()     = returns hidden state, separate lm_head+generate
//
#pragma once
#include "../../include/common.h"
#include <vector>
#include <string>
#include <cstdio>
#include <random>
#include <algorithm>

struct InferenceResult {
    std::vector<int> tokens;
    std::string text;
    int prompt_tokens = 0;
    int completion_tokens = 0;
    float gen_ms = 0;
    float tok_s = 0;
};

class InferenceBackend {
public:
    virtual ~InferenceBackend() = default;
    virtual BackendType type() const = 0;
    virtual const char* name() const = 0;
    virtual bool is_available() = 0;
    virtual bool load_model(const ModelConfig& cfg) = 0;
    virtual void unload_model() = 0;
    virtual int forward(int token_id, int pos) = 0;
    virtual void reset_state() = 0;
    virtual float estimated_tok_s() const { return 0; }
    virtual bool is_coherent() const { return true; }

    // Logits-based sampling (temperature / top-p / repetition penalty).
    // Default implementation is greedy forward(); backends with logits
    // access (via lm_head) override this to sample properly — argmax
    // decoding makes small models loop ("The the capital\n...\n...").
    virtual int sample_token(int token_id, int pos, float temperature, float top_p,
                             float repeat_penalty, const std::vector<int>& recent) {
        (void)temperature; (void)top_p; (void)repeat_penalty; (void)recent;
        return forward(token_id, pos);
    }
};

// ── CPU sampling helpers (shared by backends that expose logits) ──
inline int sample_from_logits(std::vector<float>& logits, int argmax,
                              float temperature, float top_p, float repeat_penalty,
                              const std::vector<int>& recent, unsigned seed) {
    const int V = (int)logits.size();
    if (temperature <= 0.0f) return argmax;  // greedy
    // repetition penalty (standard: divide positive logits, multiply negative)
    if (repeat_penalty > 0.0f && repeat_penalty != 1.0f) {
        for (int id : recent) {
            if (id >= 0 && id < V) {
                float l = logits[id];
                logits[id] = l > 0.0f ? l / repeat_penalty : l * repeat_penalty;
            }
        }
    }
    // temperature
    float inv_t = 1.0f / temperature;
    for (int i = 0; i < V; i++) logits[i] *= inv_t;
    // top-p (nucleus): keep the smallest set whose cumulative prob >= top_p.
    // The tail beyond the 512 top candidates is untouched (only relevant for
    // near-uniform distributions, where any sample is fine).
    if (top_p > 0.0f && top_p < 1.0f) {
        std::vector<int> idx(V);
        for (int i = 0; i < V; i++) idx[i] = i;
        std::partial_sort(idx.begin(), idx.begin() + std::min(V, 512), idx.end(),
                          [&](int a, int b) { return logits[a] > logits[b]; });
        float max_l = logits[idx[0]];
        double sum = 0.0;
        int cut = V;
        for (int k = 0; k < V; k++) {
            sum += expf((double)(logits[idx[k]] - max_l));
            if (sum >= top_p) { cut = k + 1; break; }
        }
        float thr = (cut < V) ? logits[idx[cut-1]] : -1e30f;
        for (int i = 0; i < V; i++) if (logits[i] < thr) logits[i] = -1e30f;
    }
    // softmax + sample
    float mx = logits[0];
    for (int i = 1; i < V; i++) if (logits[i] > mx) mx = logits[i];
    double sum = 0.0;
    std::vector<float> probs(V);
    for (int i = 0; i < V; i++) { probs[i] = expf(logits[i] - mx); sum += probs[i]; }
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    double r = dist(rng) * sum;
    double acc = 0.0;
    for (int i = 0; i < V; i++) { acc += probs[i]; if (acc >= r) return i; }
    return argmax;
}

// Detect all available backends on this hardware
std::vector<InferenceBackend*> detect_backends();

// Pick the fastest available backend
InferenceBackend* select_best_backend(std::vector<InferenceBackend*>* existing = nullptr);

// Simple JSON helpers shared across backends
namespace json_helpers {
    inline std::string get_str(const std::string& j, const std::string& k) {
        auto p = j.find("\"" + k + "\"");
        if (p == std::string::npos) return "";
        p = j.find(':', p);
        if (p == std::string::npos) return "";
        p = j.find_first_of("\"", p);
        if (p == std::string::npos || j[p] != '\"') {
            auto ns = j.find_first_of("-0123456789", p + 1);
            if (ns != std::string::npos) {
                auto ne = j.find_first_not_of("0123456789.e-+", ns);
                return j.substr(ns, ne - ns);
            }
            return "";
        }
        auto e = j.find('\"', p + 1);
        if (e == std::string::npos) return "";
        return j.substr(p + 1, e - p - 1);
    }
    inline int get_int(const std::string& j, const std::string& k, int d = 0) {
        auto s = get_str(j, k);
        if (s.empty()) return d;
        return atoi(s.c_str());
    }
    inline float get_float(const std::string& j, const std::string& k, float d = 0.0f) {
        auto s = get_str(j, k);
        if (s.empty()) return d;
        return (float)atof(s.c_str());
    }
}
