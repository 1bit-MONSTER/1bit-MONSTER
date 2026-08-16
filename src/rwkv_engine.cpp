// rwkv_engine.cpp — RWKV-4/5/6 linear-attention decoder (time-mixing + channel
// mixing, WKV recurrence), CPU. Mirrors transformers modeling_rwkv.py EXACTLY
// (validated against Testing/e2e_numpy_ref_rwkv.py: exact argmax chain
// match on a random-init tiny checkpoint, corr 1.0000 vs torch).
//
// Per-token recurrence, no KV cache — five [H] (or [AH]) state vectors per
// layer: attn-shift, ffn-shift, WKV numerator, WKV denominator, WKV max
// (the max state starts at -1e30, i.e. "no previous context").
// Attention: token shift (time_mix blend) -> key/value/receptance linear
// (sigmoid receptance) -> WKV recurrence (num/den/max per channel, output =
// num/den with time_first) -> receptance * out -> output projection.
// Feed-forward: token shift -> relu2(key) -> value projection -> sigmoid
// receptance gate. LayerNorm is mean-centered (weight+bias), NOT RMSNorm.
// Inference weight rescale (transformers _rescale_layers): block output
// projections divided by 2^int(l/rescale_every) at load time.

#include "backend.h"
#include "safetensors_reader.h"
#include <cmath>
#include <ctime>
#include <cstring>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

namespace {

struct RwkvLayer {
    size_t ln1_w = SIZE_MAX, ln1_b = SIZE_MAX, ln2_w = SIZE_MAX, ln2_b = SIZE_MAX;
    // attention
    size_t tm_k = SIZE_MAX, tm_v = SIZE_MAX, tm_r = SIZE_MAX;
    size_t key_w = SIZE_MAX, value_w = SIZE_MAX, rec_w = SIZE_MAX, out_w = SIZE_MAX;
    size_t time_decay = SIZE_MAX, time_first = SIZE_MAX;
    // feed forward
    size_t ff_tm_k = SIZE_MAX, ff_tm_r = SIZE_MAX;
    size_t ff_key_w = SIZE_MAX, ff_value_w = SIZE_MAX, ff_rec_w = SIZE_MAX;
    int shift = 0;  // weights_[0] index of per-layer attn-shift buffer
    int shift2 = 0; // ffn-shift buffer
    int num = 0, den = 0, mx = 0;  // WKV states (in float state vectors)
};

static void layernorm(const float* x, const float* w, const float* b, int n, float eps, float* out) {
    float mean = 0, var = 0;
    for (int i = 0; i < n; i++) mean += x[i];
    mean /= n;
    for (int i = 0; i < n; i++) { float d = x[i] - mean; var += d * d; }
    var /= n;
    float r = 1.0f / std::sqrt(var + eps);
    for (int i = 0; i < n; i++) out[i] = (x[i] - mean) * r * w[i] + b[i];
}

}  // namespace

class RwkvBackend : public Backend {
public:
    RwkvBackend() { type = BackendType::GENERIC; name = "rwkv_cpu"; }

    bool init(const ModelConfig& cfg_, const std::string& dir) override {
        cfg = cfg_;
        SafetensorsWeightReader rdr;
        std::string single = dir + "/model.safetensors";
        bool ok = rdr.open(single);
        if (!ok) ok = rdr.open_dir(dir);
        if (!ok) ok = rdr.open(dir);
        if (!ok) { fprintf(stderr, "[rwkv] open failed\n"); return false; }
        w_ = std::move(rdr);
        if (!load_config(dir)) return false;
        if (!load_weights()) return false;
        // per-layer state: attn-shift [H], ffn-shift [H], num/den/mx [AH]
        state_.assign((size_t)L * (2 * H + 3 * AH), 0.0f);
        // WKV max starts at -1e30 ("no context yet")
        for (int l = 0; l < L; l++)
            for (int i = 0; i < AH; i++) state_[(size_t)l * (2 * H + 3 * AH) + 2 * H + 2 * AH + i] = -1e30f;
        return true;
    }

    bool reset() override {
        std::fill(state_.begin(), state_.end(), 0.0f);
        for (int l = 0; l < L; l++)
            for (int i = 0; i < AH; i++) state_[(size_t)l * (2 * H + 3 * AH) + 2 * H + 2 * AH + i] = -1e30f;
        return true;
    }

    int generate(int token_id) override {
        std::vector<float> x(H);
        embed(token_id, x.data());
        step(x.data());
        return argmax(x.data());
    }

    bool forward(int token_id, float* hidden_out) override {
        embed(token_id, hidden_out);
        step(hidden_out);
        return true;
    }

    bool lm_head(const float* hidden, float* logits, int* argmax) override {
        const float* W = wt(head_w);
        for (int i = 0; i < V; i++) {
            float s = 0;
            for (int j = 0; j < H; j++) s += W[(size_t)i * H + j] * hidden[j];
            logits[i] = s;
        }
        if (argmax) { *argmax = 0; for (int i = 1; i < V; i++) if (logits[i] > logits[*argmax]) *argmax = i; }
        return true;
    }

    void destroy() override { delete this; }
    float benchmark(int tokens = 10) override {
        std::vector<float> x(H, 0.0f);
        long long t0 = clock();
        for (int i = 0; i < tokens; i++) step(x.data());
        return (float)(clock() - t0) / CLOCKS_PER_SEC * 1000.0f / tokens;
    }

private:
    SafetensorsWeightReader w_;
    std::vector<float> weights_, state_;
    int H = 0, L = 0, V = 0, AH = 0, FF = 0;
    int rescale_every = 6;
    float eps = 1e-5f;
    size_t embed_w = SIZE_MAX, pre_ln_w = SIZE_MAX, pre_ln_b = SIZE_MAX;
    size_t ln_out_w = SIZE_MAX, ln_out_b = SIZE_MAX, head_w = SIZE_MAX;
    std::vector<RwkvLayer> layers;

    const float* wt(size_t i) const { return i == SIZE_MAX ? nullptr : weights_.data() + i; }
    size_t store(std::vector<float>&& v) { size_t at = weights_.size(); weights_.insert(weights_.end(), v.begin(), v.end()); return at; }
    size_t store_t(const std::string& n, int rows, int cols = 1) {
        std::vector<float> v;
        if (!w_.get_tensor_f32(n, v) || (int)v.size() != rows * cols) {
            fprintf(stderr, "[rwkv] missing/misized %s (%zu want %d)\n", n.c_str(), v.size(), rows * cols);
            return SIZE_MAX;
        }
        return store(std::move(v));
    }
    void mm(size_t W, const float* x, int in, int out, float* y) {
        const float* Wd = wt(W);
        for (int i = 0; i < out; i++) {
            float s = 0;
            for (int j = 0; j < in; j++) s += Wd[(size_t)i * in + j] * x[j];
            y[i] = s;
        }
    }
    void embed(int tok, float* out) {
        const float* W = wt(embed_w);
        for (int j = 0; j < H; j++) out[j] = W[(size_t)tok * H + j];
    }
    int argmax(const float* x) {
        const float* W = wt(head_w);
        int best = 0; float bv = -1e30f;
        for (int i = 0; i < V; i++) {
            float s = 0;
            for (int j = 0; j < H; j++) s += W[(size_t)i * H + j] * x[j];
            if (s > bv) { bv = s; best = i; }
        }
        return best;
    }

    bool load_config(const std::string& dir) {
        std::string txt;
        { std::ifstream f(dir + "/config.json"); if (f) txt.assign(std::istreambuf_iterator<char>(f), {}); }
        auto find_int = [&](const char* k, int& o) {
            size_t p = txt.find(k); if (p == std::string::npos) return false;
            p = txt.find(':', p); o = atoi(txt.c_str() + p + 1); return true;
        };
        auto find_float = [&](const char* k, float& o) {
            size_t p = txt.find(k); if (p == std::string::npos) return false;
            p = txt.find(':', p); o = (float)atof(txt.c_str() + p + 1); return true;
        };
        find_int("hidden_size", H);
        find_int("vocab_size", V);
        find_int("num_hidden_layers", L);
        find_int("attention_hidden_size", AH);
        find_int("intermediate_size", FF);
        find_int("rescale_every", rescale_every);
        find_float("layer_norm_epsilon", eps);
        if (AH <= 0) AH = H;   // attention_hidden_size defaults to hidden_size
        if (FF <= 0) FF = 4 * H;
        layers.assign(L, {});
        return L > 0;
    }

    bool load_weights() {
        embed_w = store_t("rwkv.embeddings.weight", V, H);
        pre_ln_w = store_t("rwkv.blocks.0.pre_ln.weight", H);
        pre_ln_b = store_t("rwkv.blocks.0.pre_ln.bias", H);
        ln_out_w = store_t("rwkv.ln_out.weight", H);
        ln_out_b = store_t("rwkv.ln_out.bias", H);
        head_w = store_t("head.weight", V, H);
        for (int l = 0; l < L; l++) {
            auto& ly = layers[l];
            char b[256];
            snprintf(b, sizeof b, "rwkv.blocks.%d.", l);
            std::string pfx = b;
            ly.ln1_w = store_t(pfx + "ln1.weight", H);
            ly.ln1_b = store_t(pfx + "ln1.bias", H);
            ly.ln2_w = store_t(pfx + "ln2.weight", H);
            ly.ln2_b = store_t(pfx + "ln2.bias", H);
            std::string apfx = pfx + "attention.";
            ly.tm_k = store_t(apfx + "time_mix_key", H);
            ly.tm_v = store_t(apfx + "time_mix_value", H);
            ly.tm_r = store_t(apfx + "time_mix_receptance", H);
            ly.key_w = store_t(apfx + "key.weight", AH, H);
            ly.value_w = store_t(apfx + "value.weight", AH, H);
            ly.rec_w = store_t(apfx + "receptance.weight", AH, H);
            ly.out_w = store_t(apfx + "output.weight", H, AH);
            ly.time_decay = store_t(apfx + "time_decay", AH);
            ly.time_first = store_t(apfx + "time_first", AH);
            std::string fpfx = pfx + "feed_forward.";
            ly.ff_tm_k = store_t(fpfx + "time_mix_key", H);
            ly.ff_tm_r = store_t(fpfx + "time_mix_receptance", H);
            ly.ff_key_w = store_t(fpfx + "key.weight", FF, H);
            ly.ff_value_w = store_t(fpfx + "value.weight", H, FF);
            ly.ff_rec_w = store_t(fpfx + "receptance.weight", H, H);
            // inference weight rescale: divide output projections by 2^int(l/rescale_every)
            if (rescale_every > 0) {
                int k = l / rescale_every;
                if (k > 0) {
                    float div = std::exp2((float)k);
                    for (size_t i = ly.out_w; i < ly.out_w + (size_t)H * AH; i++) weights_[i] /= div;
                    for (size_t i = ly.ff_value_w; i < ly.ff_value_w + (size_t)H * FF; i++) weights_[i] /= div;
                }
            }
        }
        return true;
    }

    // WKV single-token recurrence; state in/out: num, den, mx (AH each).
    // out receives the gated output (num/den with time_first) for this token.
    void wkv(const float* time_decay_p, const float* time_first_p,
             const float* key, const float* value,
             float* num, float* den, float* mx, float* out) {
        for (int i = 0; i < AH; i++) {
            float td = -std::exp(time_decay_p[i]);
            float tf = time_first_p[i];
            // output from current state
            float max_for_output = std::max(mx[i], key[i] + tf);
            float e1 = std::exp(mx[i] - max_for_output);
            float e2 = std::exp(key[i] + tf - max_for_output);
            float num_out = e1 * num[i] + e2 * value[i];
            float den_out = e1 * den[i] + e2;
            out[i] = num_out / den_out;
            // state update uses the PRE-update num/den (torch keeps separate)
            float max_for_state = std::max(mx[i] + td, key[i]);
            e1 = std::exp(mx[i] + td - max_for_state);
            e2 = std::exp(key[i] - max_for_state);
            num[i] = e1 * num[i] + e2 * value[i];
            den[i] = e1 * den[i] + e2;
            mx[i] = max_for_state;
        }
    }

    void step(float* x) {
        std::vector<float> pre(H), xn(H), att(3 * H), att_out(H), ff(2 * H), ff_val(H);
        std::vector<float> keyv(AH), valv(AH), recv(AH), wkv_out(AH);
        std::vector<float> ff_key(FF);
        for (int l = 0; l < L; l++) {
            auto& ly = layers[l];
            float* st = state_.data() + (size_t)l * (2 * H + 3 * AH);
            float* shift = st;            // [H] attn shift (prev normed input)
            float* shift2 = st + H;       // [H] ffn shift
            float* num = st + 2 * H;      // [AH]
            float* den = st + 2 * H + AH; // [AH]
            float* mx = st + 2 * H + 2 * AH; // [AH]

            // pre_ln only on layer 0 (result kept separate from ln1 output)
            float* in = x;
            if (l == 0) { layernorm(x, wt(pre_ln_w), wt(pre_ln_b), H, eps, pre.data()); in = pre.data(); }

            // ---- attention (time mixing) ----
            layernorm(in, wt(ly.ln1_w), wt(ly.ln1_b), H, eps, xn.data());
            const float* tm_k = wt(ly.tm_k), * tm_v = wt(ly.tm_v), * tm_r = wt(ly.tm_r);
            for (int i = 0; i < H; i++) {
                att[i] = xn[i] * tm_k[i] + shift[i] * (1 - tm_k[i]);
                att[H + i] = xn[i] * tm_v[i] + shift[i] * (1 - tm_v[i]);
                att[2 * H + i] = xn[i] * tm_r[i] + shift[i] * (1 - tm_r[i]);
            }
            for (int i = 0; i < H; i++) shift[i] = xn[i];  // shift = ln1-normed input
            mm(ly.key_w, att.data(), H, AH, keyv.data());
            mm(ly.value_w, att.data() + H, H, AH, valv.data());
            mm(ly.rec_w, att.data() + 2 * H, H, AH, recv.data());
            for (int i = 0; i < AH; i++) recv[i] = 1.0f / (1.0f + std::exp(-recv[i]));
            wkv(wt(ly.time_decay), wt(ly.time_first), keyv.data(), valv.data(), num, den, mx, wkv_out.data());
            for (int i = 0; i < AH; i++) keyv[i] = recv[i] * wkv_out[i];
            mm(ly.out_w, keyv.data(), AH, H, att_out.data());
            for (int i = 0; i < H; i++) x[i] = in[i] + att_out[i];

            // ---- feed forward (channel mixing) ----
            layernorm(x, wt(ly.ln2_w), wt(ly.ln2_b), H, eps, xn.data());
            const float* ftk = wt(ly.ff_tm_k), * ftr = wt(ly.ff_tm_r);
            for (int i = 0; i < H; i++) {
                ff[i] = xn[i] * ftk[i] + shift2[i] * (1 - ftk[i]);
                ff[H + i] = xn[i] * ftr[i] + shift2[i] * (1 - ftr[i]);
            }
            for (int i = 0; i < H; i++) shift2[i] = xn[i];  // shift2 = ln2-normed input
            mm(ly.ff_key_w, ff.data(), H, FF, ff_key.data());
            for (int i = 0; i < FF; i++) { float k = ff_key[i]; ff_key[i] = (k > 0 ? k : 0); ff_key[i] *= ff_key[i]; }  // relu2
            mm(ly.ff_value_w, ff_key.data(), FF, H, ff_val.data());
            mm(ly.ff_rec_w, ff.data() + H, H, H, ff.data());  // receptance
            for (int i = 0; i < H; i++) ff_val[i] *= 1.0f / (1.0f + std::exp(-ff[i]));
            for (int i = 0; i < H; i++) x[i] += ff_val[i];
        }
        layernorm(x, wt(ln_out_w), wt(ln_out_b), H, eps, x);
    }
};

extern "C" Backend* create_rwkv_backend() { return new RwkvBackend(); }
