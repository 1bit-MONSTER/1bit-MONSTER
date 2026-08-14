// openelm.cpp — Apple OpenELM bespoke backend (per-layer heterogeneous dims).
// Follows the deepseek.cpp template. Loads the HF safetensors via the repo's
// SafetensorsWeightReader; forward matches modeling_openelm.py exactly
// (validated vs Testing/e2e_numpy_ref_openelm.py, the numpy port).
#include "openelm.h"
#include "safetensors_reader.h"
#include <cstdio>
#include <fstream>
#include <sstream>

namespace openelm_math {

// Minimal config.json array reader: extract an integer array like
// "num_query_heads": [12, 12, ...]. The OpenELM config has per-layer lists.
static std::vector<int> json_int_array(const std::string& text, const std::string& key) {
    std::vector<int> out;
    auto pos = text.find("\"" + key + "\"");
    if (pos == std::string::npos) return out;
    auto lb = text.find('[', pos), rb = text.find(']', lb);
    if (lb == std::string::npos || rb == std::string::npos) return out;
    std::string inner = text.substr(lb + 1, rb - lb - 1);
    std::stringstream ss(inner);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        try { out.push_back(std::stoi(tok)); } catch (...) {}
    }
    return out;
}
static int json_int(const std::string& text, const std::string& key, int def) {
    auto pos = text.find("\"" + key + "\"");
    if (pos == std::string::npos) return def;
    auto col = text.find(':', pos);
    if (col == std::string::npos) return def;
    return std::atoi(text.c_str() + col + 1);
}
static float json_float(const std::string& text, const std::string& key, float def) {
    auto pos = text.find("\"" + key + "\"");
    if (pos == std::string::npos) return def;
    auto col = text.find(':', pos);
    if (col == std::string::npos) return def;
    return (float)std::atof(text.c_str() + col + 1);
}
static int make_divisible(float v, int divisor) {
    return (int)(v + divisor / 2.0f) / divisor * divisor;
}

bool OpenELMModel::load(const std::string& dir, const std::string& st_path) {
    std::ifstream cf(dir + "/config.json");
    if (!cf) { fprintf(stderr, "[openelm] FAIL: no config.json in %s\n", dir.c_str()); return false; }
    std::stringstream css; css << cf.rdbuf();
    std::string cfgtext = css.str();

    cfg.model_dim    = json_int(cfgtext, "model_dim", 1280);
    cfg.num_layers   = json_int(cfgtext, "num_transformer_layers", 16);
    cfg.head_dim     = json_int(cfgtext, "head_dim", 64);
    cfg.vocab_size   = json_int(cfgtext, "vocab_size", 32000);
    cfg.rms_eps      = json_float(cfgtext, "rms_norm_eps", 1e-6f);
    cfg.q_heads      = json_int_array(cfgtext, "num_query_heads");
    cfg.kv_heads     = json_int_array(cfgtext, "num_kv_heads");
    std::vector<int> fm; { // ffn_multipliers are floats in the config
        auto pos = cfgtext.find("\"ffn_multipliers\"");
        if (pos != std::string::npos) {
            auto lb = cfgtext.find('[', pos), rb = cfgtext.find(']', lb);
            std::string inner = cfgtext.substr(lb + 1, rb - lb - 1);
            std::stringstream ss(inner); std::string tok;
            while (std::getline(ss, tok, ',')) fm.push_back((int)(std::atof(tok.c_str()) * 1000));
        }
    }
    int divisor = json_int(cfgtext, "ffn_dim_divisor", 256);
    if ((int)cfg.q_heads.size() != cfg.num_layers || (int)cfg.kv_heads.size() != cfg.num_layers) {
        fprintf(stderr, "[openelm] FAIL: per-layer head lists missing/short\n"); return false;
    }
    cfg.intermediate.resize(cfg.num_layers);
    for (int i = 0; i < cfg.num_layers; i++)
        cfg.intermediate[i] = make_divisible((fm.size() > (size_t)i ? fm[i] / 1000.0f : 1.0f) * cfg.model_dim, divisor);

    SafetensorsWeightReader r;
    if (!r.open(st_path)) { fprintf(stderr, "[openelm] FAIL: %s\n", st_path.c_str()); return false; }
    if (!r.get_tensor_f32("transformer.token_embeddings.weight", token_emb)) return false;
    if (!r.get_tensor_f32("transformer.norm.weight", final_norm)) return false;
    layers.resize(cfg.num_layers);
    for (int il = 0; il < cfg.num_layers; il++) {
        std::string p = "transformer.layers." + std::to_string(il) + ".";
        auto& l = layers[il];
        bool ok = true;
        ok &= r.get_tensor_f32(p + "attn_norm.weight", l.attn_norm);
        ok &= r.get_tensor_f32(p + "attn.qkv_proj.weight", l.qkv_w);
        ok &= r.get_tensor_f32(p + "attn.q_norm.weight", l.q_norm);
        ok &= r.get_tensor_f32(p + "attn.k_norm.weight", l.k_norm);
        ok &= r.get_tensor_f32(p + "attn.out_proj.weight", l.out_proj);
        ok &= r.get_tensor_f32(p + "ffn_norm.weight", l.ffn_norm);
        ok &= r.get_tensor_f32(p + "ffn.proj_1.weight", l.proj_1);
        ok &= r.get_tensor_f32(p + "ffn.proj_2.weight", l.proj_2);
        if (!ok) { fprintf(stderr, "[openelm] FAIL: layer %d tensors\n", il); return false; }
    }
    fprintf(stderr, "[openelm] loaded: %s (%d layers, dim=%d, hd=%d, q=%d..%d kv=%d..%d)\n",
            dir.c_str(), cfg.num_layers, cfg.model_dim, cfg.head_dim,
            cfg.q_heads[0], cfg.q_heads[cfg.num_layers - 1], cfg.kv_heads[0], cfg.kv_heads[cfg.num_layers - 1]);
    return true;
}

// forward: one token. kv_cache[il] holds [pos][kv_h * hd] k rows then v rows
// interleaved per position: for each past position t, k_t is [kv_h*hd] and v_t
// is [kv_h*hd]. We store per position: cache[t*2*kv_h*hd + ...].
std::vector<float> openelm_forward(
    const OpenELMModel& model, int token_id,
    std::vector<std::vector<float>>& kv_cache, int& pos)
{
    const auto& cfg = model.cfg;
    int D = cfg.model_dim, HD = cfg.head_dim;
    std::vector<float> x(D);
    if (token_id >= 0 && token_id < cfg.vocab_size)
        for (int i = 0; i < D; i++) x[i] = model.token_emb[(size_t)token_id * D + i];

    std::vector<float> norm(D), qkv, attn_out(D), ffn_out;
    for (int il = 0; il < cfg.num_layers; il++) {
        const auto& l = model.layers[il];
        int qh = cfg.q_heads[il], kh = cfg.kv_heads[il], inter = cfg.intermediate[il];
        int qdim = qh * HD, kdim = kh * HD;
        int total = qdim + 2 * kdim;
        qkv.assign(total, 0.0f);
        // attention (pre-norm)
        rmsnorm(norm.data(), x.data(), l.attn_norm.data(), D, cfg.rms_eps);
        // qkv = norm @ qkv_w.T  (qkv_w [total, D], row split [q|k|v])
        for (int i = 0; i < total; i++) {
            float s = 0;
            const float* wr = l.qkv_w.data() + (size_t)i * D;
            for (int j = 0; j < D; j++) s += norm[j] * wr[j];
            qkv[i] = s;
        }
        // per-head q/k RMSNorm + rope
        std::vector<float> kcache_row(kdim), vcache_row(kdim);
        for (int h = 0; h < qh; h++) {
            float* qp = qkv.data() + (size_t)h * HD;
            rmsnorm(qp, qp, l.q_norm.data(), HD, cfg.rms_eps);
            rope_half(qp, HD, pos);
        }
        for (int h = 0; h < kh; h++) {
            float* kp = qkv.data() + (size_t)qdim + (size_t)h * HD;
            rmsnorm(kp, kp, l.k_norm.data(), HD, cfg.rms_eps);
            rope_half(kp, HD, pos);
            for (int d = 0; d < HD; d++) kcache_row[(size_t)h * HD + d] = kp[d];
        }
        for (int h = 0; h < kh; h++) {
            const float* vp = qkv.data() + (size_t)qdim + (size_t)kdim + (size_t)h * HD;
            for (int d = 0; d < HD; d++) vcache_row[(size_t)h * HD + d] = vp[d];
        }
        // append to the layer cache (k then v per position)
        auto& kc = kv_cache[il];
        for (float f : kcache_row) kc.push_back(f);
        for (float f : vcache_row) kc.push_back(f);
        // attention over the cache: for each query head, attend over all cached
        // positions. GQA: kv head = query_head / (qh/kh).
        int g = qh / kh;
        int npos = pos + 1;
        std::fill(attn_out.begin(), attn_out.end(), 0.0f);
        std::vector<float> scores(npos);
        for (int h = 0; h < qh; h++) {
            int kvh = h / g;
            const float* qp = qkv.data() + (size_t)h * HD;
            for (int t = 0; t < npos; t++) {
                const float* kp = kc.data() + (size_t)t * 2 * kdim + (size_t)kvh * HD;
                float s = 0;
                for (int d = 0; d < HD; d++) s += qp[d] * kp[d];
                scores[t] = s / sqrtf((float)HD);
            }
            // softmax
            float mx = scores[0]; for (int t = 1; t < npos; t++) mx = std::max(mx, scores[t]);
            float sum = 0; for (int t = 0; t < npos; t++) { scores[t] = expf(scores[t] - mx); sum += scores[t]; }
            for (int t = 0; t < npos; t++) scores[t] /= sum;
            for (int d = 0; d < HD; d++) {
                float acc = 0;
                for (int t = 0; t < npos; t++)
                    acc += scores[t] * kc[(size_t)t * 2 * kdim + (size_t)kdim + (size_t)kvh * HD + d];
                attn_out[(size_t)h * HD + d] = acc;
            }
        }
        // out_proj [D, qdim]: x += attn_out @ out_proj.T
        for (int i = 0; i < D; i++) {
            float s = 0;
            const float* wr = l.out_proj.data() + (size_t)i * qdim;
            for (int j = 0; j < qdim; j++) s += attn_out[j] * wr[j];
            x[i] += s;
        }
        // FFN (pre-norm): proj_1 -> [y1|y2] -> swish(y1)*y2 -> proj_2
        rmsnorm(norm.data(), x.data(), l.ffn_norm.data(), D, cfg.rms_eps);
        ffn_out.assign(inter, 0.0f);
        std::vector<float> y1(inter), y2(inter);
        for (int i = 0; i < inter; i++) {
            float s1 = 0, s2 = 0;
            const float* w1 = l.proj_1.data() + (size_t)i * D;
            const float* w2 = l.proj_1.data() + (size_t)(inter + i) * D;
            for (int j = 0; j < D; j++) { s1 += norm[j] * w1[j]; s2 += norm[j] * w2[j]; }
            y1[i] = s1; y2[i] = s2;
        }
        for (int i = 0; i < inter; i++) ffn_out[i] = silu(y1[i]) * y2[i];
        for (int i = 0; i < D; i++) {
            float s = 0;
            const float* wr = l.proj_2.data() + (size_t)i * inter;
            for (int j = 0; j < inter; j++) s += ffn_out[j] * wr[j];
            x[i] += s;
        }
    }
    // final norm + tied lm_head
    rmsnorm(norm.data(), x.data(), model.final_norm.data(), D, cfg.rms_eps);
    std::vector<float> logits(cfg.vocab_size);
    for (int v = 0; v < cfg.vocab_size; v++) {
        const float* wr = model.token_emb.data() + (size_t)v * D;
        float s = 0;
        for (int j = 0; j < D; j++) s += norm[j] * wr[j];
        logits[v] = s;
    }
    pos++;
    return logits;
}

}  // namespace openelm_math
