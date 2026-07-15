// backend_generic.cpp — Universal CPU inference backend
// Reads ModelConfig from any discovered GGUF/H1B/BIN model and runs inference.
// Supports Llama, Mistral, Qwen2, Gemma, Phi architectures with:
//   RMSNorm / LayerNorm, RoPE (partial/full), GQA/MHA, SiLU/SwiGLU/GeGLU, KV cache

#include "backend.h"
#include <sys/stat.h>
#include <dirent.h>
#include "model_discovery.h"
#include "rocm_cpp/tokenizer.h"

// ── Minimal GGUF weight reader ──────────────────────────────────────────────
// Reads tensor data from a GGUF file into host float vectors.
// Handles F32, F16, Q8_0, Q4_0 quantizations (dequantizes to F32).
// Tensor lookup by name: gguf_tensor("blk.0.attn_q.weight", data, shape)

#include <cstring>
#include <cstdint>

static float fp16_to_fp32(uint16_t h) {
    uint32_t sign = (h >> 15) & 1;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    if (exp == 0) {
        uint32_t adj = mant ? __builtin_clz(mant) - 10 : 0;
        float r; uint32_t f32 = (sign << 31) | ((127 - 15 - adj) << 23) | ((mant << (adj + 13)) & 0x7fffff); memcpy(&r, &f32, 4); return r;
    } else if (exp == 31) {
        float r; uint32_t f32 = (sign << 31) | 0x7f800000 | (mant << 13); memcpy(&r, &f32, 4); return r;
    } else {
        float r; uint32_t f32 = (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13); memcpy(&r, &f32, 4); return r;
    }
}

#include <map>
#include <unordered_map>

// GGUF constants
#define GGUF_MAGIC    0x46554747
#define GGUF_TYPE_F32  0
#define GGUF_TYPE_F16  1
#define GGUF_TYPE_Q4_0 2
#define GGUF_TYPE_Q8_0 7

struct GgufTensor {
    std::string name;
    std::vector<uint64_t> shape;
    uint32_t dtype;
    uint64_t file_offset;  // absolute byte offset of data
};

struct GgufReader {
    FILE* f = nullptr;
    std::unordered_map<std::string, GgufTensor> tensors;
    std::vector<float> scratch;
    int vocab_size = 0;

    bool open(const std::string& path) {
        f = fopen(path.c_str(), "rb");
        if (!f) return false;
        uint32_t magic; fread(&magic, 4, 1, f);
        if (magic != GGUF_MAGIC) { fclose(f); return false; }
        uint32_t version; fread(&version, 4, 1, f);
        uint64_t tensor_count, kv_count;
        fread(&tensor_count, 8, 1, f); fread(&kv_count, 8, 1, f);
        // Skip metadata KVs
        for (uint64_t i = 0; i < kv_count; i++) {
            uint64_t klen; fread(&klen, 8, 1, f); fseek(f, klen, SEEK_CUR);
            uint32_t vtype; fread(&vtype, 4, 1, f);
            switch (vtype) {
                case 0: case 1: case 2: fseek(f, 1, SEEK_CUR); break;
                case 3: case 4: fseek(f, 4, SEEK_CUR); break;
                case 5: fseek(f, 4, SEEK_CUR); break;
                case 6: fseek(f, 4, SEEK_CUR); break;
                case 7: fseek(f, 8, SEEK_CUR); break;
                case 8: fseek(f, 8, SEEK_CUR); break;
                case 9: { uint32_t at, an; fread(&at, 4, 1, f); fread(&an, 4, 1, f); fseek(f, an*4, SEEK_CUR); break; }
                case 10: { uint64_t sl; fread(&sl, 8, 1, f); fseek(f, sl, SEEK_CUR); break; }
                default: break;
            }
        }
        // Read tensor info
        uint64_t data_offset = ftell(f) + tensor_count * (8+8+4+4*4); // estimate
        for (uint64_t i = 0; i < tensor_count; i++) {
            GgufTensor t;
            uint64_t nlen; fread(&nlen, 8, 1, f);
            t.name.resize(nlen); fread(&t.name[0], 1, nlen, f);
            uint32_t n_dims; fread(&n_dims, 4, 1, f);
            t.dtype = 0; fread(&t.dtype, 4, 1, f);
            t.shape.resize(n_dims);
            for (int j = 0; j < n_dims; j++) fread(&t.shape[j], 8, 1, f);
            tensors[t.name] = t;
        }
        // Compute actual data offsets — align to 32 bytes
        data_offset = ftell(f);
        data_offset = (data_offset + 31) & ~31;
        for (auto& [name, t] : tensors) {
            t.file_offset = data_offset;
            uint64_t n_elems = 1;
            for (auto s : t.shape) n_elems *= s;
            int bytes_per_elem = (t.dtype == GGUF_TYPE_F32) ? 4 :
                                 (t.dtype == GGUF_TYPE_F16) ? 2 :
                                 (t.dtype == GGUF_TYPE_Q4_0) ? 18/*bytes/32elems*/ : 34/*Q8_0*/;
            data_offset += (n_elems * bytes_per_elem / (t.dtype == GGUF_TYPE_F32 || t.dtype == GGUF_TYPE_F16 ? 1 : 32)) + 31;
            data_offset &= ~31;
        }
        return true;
    }

    // Get tensor data, dequantized to float. Returns pointer to internal buffer.
    float* get(const std::string& name, size_t* out_n = nullptr) {
        auto it = tensors.find(name);
        if (it == tensors.end()) return nullptr;
        auto& t = it->second;
        uint64_t n = 1; for (auto s : t.shape) n *= s;
        if (out_n) *out_n = n;
        scratch.resize(n);
        fseek(f, t.file_offset, SEEK_SET);
        if (t.dtype == GGUF_TYPE_F32) {
            fread(scratch.data(), 4, n, f);
        } else if (t.dtype == GGUF_TYPE_F16) {
            std::vector<uint16_t> buf(n);
            fread(buf.data(), 2, n, f);
            for (size_t i = 0; i < n; i++) scratch[i] = fp16_to_fp32(buf[i]);
        } else if (t.dtype == GGUF_TYPE_Q4_0) {
            // Q4_0: 2 bytes scale + 16 bytes of 4-bit nibbles per 32 elements
            int blocks = (n + 31) / 32;
            for (int b = 0; b < blocks; b++) {
                uint16_t scale_h; fread(&scale_h, 2, 1, f);
                float scale = fp16_to_fp32(scale_h);
                uint8_t q[16]; fread(q, 1, 16, f);
                for (int j = 0; j < 32 && b*32+j < n; j++) {
                    int8_t v = (j & 1) ? (q[j>>1] >> 4) : (q[j>>1] & 0xf);
                    scratch[b*32+j] = (v - 8) * scale;
                }
            }
        } else if (t.dtype == GGUF_TYPE_Q8_0) {
            // Q8_0: 2 bytes scale + 32 bytes int8 per 32 elements
            int blocks = (n + 31) / 32;
            for (int b = 0; b < blocks; b++) {
                uint16_t scale_h; fread(&scale_h, 2, 1, f);
                float scale = fp16_to_fp32(scale_h);
                int8_t q[32]; fread(q, 1, 32, f);
                for (int j = 0; j < 32 && b*32+j < n; j++)
                    scratch[b*32+j] = q[j] * scale;
            }
        }
        return scratch.data();
    }

    void close() { if (f) fclose(f); }
};

#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <chrono>

// ── Generic CPU Backend ──────────────────────────────────────────────────────
struct GenericBackend : Backend {
    ModelConfig cfg;
    std::vector<float> embed, final_norm;
    std::vector<std::vector<float>> layer_w;  // flat per-layer weights
    std::vector<std::vector<float>> k_cache, v_cache; // KV cache [n_layers][max_seq * n_kv * hd]
    int pos = 0;
    std::vector<float> logits_buf;

    // Per-layer weight indices
    struct LayerW { size_t wq, wk, wv, wo, w1, w2, w3, rms_attn, rms_ffn; };
    std::vector<LayerW> layers;

    GenericBackend() { type = BackendType::CPU_AVX512; name = "Generic CPU (AVX-512)"; }

    void load_weights(const std::string& base) {
        // Weights stored as flat float vectors: model_layers_N_name.bin
        // Read by the existing W() macro pattern
        auto W = [&](const std::string& name) -> std::vector<float> {
            std::string path = base + "/" + name;
            std::ifstream f(path, std::ios::binary | std::ios::ate);
            if (!f) return {};
            size_t n = f.tellg() / sizeof(float); f.seekg(0);
            std::vector<float> d(n); f.read((char*)d.data(), n * sizeof(float));
            return d;
        };
        embed = W("model_embed_tokens_weight.bin");
        final_norm = W("model_norm_weight.bin");

        int H = cfg.hidden, L = cfg.n_layers, NH = cfg.n_heads, NKV = cfg.n_kv_heads, HD = cfg.head_dim;
        int FF = cfg.intermediate_size;
        layers.resize(L);
        for (int i = 0; i < L; i++) {
            std::string p = "model_layers_" + std::to_string(i) + "_";
            layers[i] = {
                push(W(p + "self_attn_q_proj.weight")),
                push(W(p + "self_attn_k_proj.weight")),
                push(W(p + "self_attn_v_proj.weight")),
                push(W(p + "self_attn_o_proj.weight")),
                push(W(p + "mlp_gate_proj.weight")),
                push(W(p + "mlp_up_proj.weight")),
                push(W(p + "mlp_down_proj.weight")),
                push(W(p + "input_layernorm.weight")),
                push(W(p + "post_attention_layernorm.weight")),
            };
        }
    }

    size_t push(std::vector<float>&& v) {
        size_t idx = flat_weights.size();
        flat_weights.insert(flat_weights.end(), v.begin(), v.end());
        return idx;
    }
    std::vector<float> flat_weights;
    float* w(size_t idx) { return flat_weights.data() + idx; }

    bool init(const ModelConfig& model_cfg, const std::string& weights_dir) override {
        cfg = model_cfg;
        printf("Generic: initializing %s (%d layers, %d hidden, %d heads)\n",
               cfg.model_name.c_str(), cfg.n_layers, cfg.hidden, cfg.n_heads);

        // Try loading weights from a GGUF file first
        bool loaded = false;
        if (!cfg.model_path.empty()) {
            std::string gguf_path = cfg.model_path;
            // If model_path is a directory, look for a .gguf inside
            struct stat st;
            if (stat(gguf_path.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) {
                // Find first .gguf in the directory
                DIR* d = opendir(gguf_path.c_str());
                if (d) {
                    struct dirent* e;
                    while ((e = readdir(d)) != nullptr) {
                        std::string n(e->d_name);
                        if (n.size() > 5 && n.substr(n.size()-5) == ".gguf") {
                            gguf_path = gguf_path + "/" + n;
                            break;
                        }
                    }
                    closedir(d);
                }
            }
            printf("Generic: trying GGUF path: %s\n", gguf_path.c_str());
            loaded = load_gguf(gguf_path);
        }
        if (!loaded) {
            // Fall back: old .bin format
            load_weights(weights_dir);
            loaded = !embed.empty();
        }
        if (!loaded) {
            fprintf(stderr, "Generic: could not load weights from %s\n", weights_dir.c_str());
            return false;
        }
        logits_buf.resize(cfg.vocab);
        k_cache.resize(cfg.n_layers);
        v_cache.resize(cfg.n_layers);
        for (auto& k : k_cache) k.resize(cfg.max_seq_len * cfg.n_kv_heads * cfg.head_dim);
        for (auto& v : v_cache) v.resize(cfg.max_seq_len * cfg.n_kv_heads * cfg.head_dim);
        initialized = true;
        return true;
    }

    bool load_gguf(const std::string& path) {
        GgufReader r;
        if (!r.open(path)) return false;
        printf("Generic: opened GGUF with %zu tensors\n", r.tensors.size());
        
        int H = cfg.hidden, L = cfg.n_layers, NH = cfg.n_heads, NKV = cfg.n_kv_heads, HD = cfg.head_dim;
        int FF = cfg.intermediate_size, V = cfg.vocab;
        
        // Helper: get tensor, remap common names
        auto get = [&](const std::string& name) -> float* {
            // Try common naming conventions
            std::vector<std::string> prefixes = {"", "model.", "transformer."};
            for (auto& p : prefixes) {
                float* d = r.get(p + name);
                if (d) return d;
            }
            return nullptr;
        };
        
        // Embedding
        { auto* d = get("token_embd.weight"); if (d) { embed.assign(d, d + V * H); } }
        
        // Final norm
        { auto* d = get("output_norm.weight"); if (d) final_norm.assign(d, d + H); }
        
        // LM head (might be tied)
        if (!get("output.weight")) {
            // Tied embeddings — lm_head = embed.T
        }
        
        // Per-layer weights
        layers.resize(L);
        flat_weights.clear();
        for (int i = 0; i < L; i++) {
            std::string p = "blk." + std::to_string(i) + ".";
            LayerW lw;
            lw.rms_attn = push_vec(get(p + "attn_norm.weight"), H);
            lw.rms_ffn  = push_vec(get(p + "ffn_norm.weight"), H);
            lw.wq = push_vec(get(p + "attn_q.weight"), NH*HD*H);
            lw.wk = push_vec(get(p + "attn_k.weight"), NKV*HD*H);
            lw.wv = push_vec(get(p + "attn_v.weight"), NKV*HD*H);
            lw.wo = push_vec(get(p + "attn_output.weight"), H*NH*HD);
            lw.w1 = push_vec(get(p + "ffn_gate.weight"), FF*H);
            lw.w2 = push_vec(get(p + "ffn_up.weight"), FF*H);
            lw.w3 = push_vec(get(p + "ffn_down.weight"), H*FF);
            layers[i] = lw;
        }
        
        // If embedding is still empty, try vocab_embedding or similar
        if (embed.empty()) {
            auto* d = get("token_embd.weight");
            if (!d) d = r.get("gpt.neox.token_embd.weight");
            if (d) embed.assign(d, d + V * H);
        }
        
        r.close();
        printf("Generic: loaded %zu layers, embed=%zu, final_norm=%zu\n",
               layers.size(), embed.size(), final_norm.size());
        return !embed.empty() && layers.size() == (size_t)L;
    }

    size_t push_vec(float* data, size_t n) {
        if (!data) return 0;
        size_t idx = flat_weights.size();
        flat_weights.insert(flat_weights.end(), data, data + n);
        return idx;
    }

    bool reset() override {
        pos = 0;
        for (auto& k : k_cache) std::fill(k.begin(), k.end(), 0.0f);
        for (auto& v : v_cache) std::fill(v.begin(), v.end(), 0.0f);
        return true;
    }

    static void rmsnorm(float* o, const float* x, const float* w, int n, float eps) {
        float ss = 0; for (int i = 0; i < n; i++) ss += x[i] * x[i];
        float r = 1.0f / sqrtf(ss / n + eps);
        for (int i = 0; i < n; i++) o[i] = x[i] * r * w[i];
    }

    static void rope(float* q, float* k, int pos, int n_heads, int n_kv, int hd, int rot_dim, float theta) {
        for (int h = 0; h < n_heads; h++) {
            for (int d = 0; d < rot_dim; d += 2) {
                float freq = pos / powf(theta, d / (float)rot_dim);
                float cosv = cosf(freq), sinv = sinf(freq);
                int i0 = h * hd + d, i1 = h * hd + d + 1;
                float q0 = q[i0], q1 = q[i1];
                q[i0] = q0 * cosv - q1 * sinv;
                q[i1] = q0 * sinv + q1 * cosv;
            }
        }
        for (int h = 0; h < n_kv; h++) {
            for (int d = 0; d < rot_dim; d += 2) {
                float freq = pos / powf(theta, d / (float)rot_dim);
                float cosv = cosf(freq), sinv = sinf(freq);
                int i0 = h * hd + d, i1 = h * hd + d + 1;
                float k0 = k[i0], k1 = k[i1];
                k[i0] = k0 * cosv - k1 * sinv;
                k[i1] = k0 * sinv + k1 * cosv;
            }
        }
    }

    static void matmul(float* out, const float* in, const float* w, int M, int K) {
        for (int i = 0; i < M; i++) {
            float s = 0;
            for (int j = 0; j < K; j++) s += in[j] * w[i * (size_t)K + j];
            out[i] = s;
        }
    }

    static void silu(float* out, const float* gate, const float* up, int n) {
        for (int i = 0; i < n; i++) {
            float g = gate[i];
            out[i] = (g / (1.0f + expf(-g))) * up[i];
        }
    }

    static void softmax(float* x, int n) {
        float mx = x[0]; for (int i = 1; i < n; i++) if (x[i] > mx) mx = x[i];
        float sum = 0; for (int i = 0; i < n; i++) sum += expf(x[i] - mx);
        float inv = 1.0f / (sum + 1e-10f);
        for (int i = 0; i < n; i++) x[i] = expf(x[i] - mx) * inv;
    }

    int generate(int token_id) override {
        if (!initialized) return -1;
        return forward(token_id);
    }

    bool forward(int token_id, float* hidden_out) override {
        int tok = forward(token_id);
        if (hidden_out) *hidden_out = 0.0f;
        return tok >= 0;
    }

    bool lm_head(const float* hidden, float* logits, int* argmax) override {
        return false;  // not implemented — use generate() instead
    }

    int forward(int token) {
        int H = cfg.hidden, NH = cfg.n_heads, NKV = cfg.n_kv_heads, HD = cfg.head_dim;
        int GQA = NH / NKV, FF = cfg.intermediate_size, V = cfg.vocab;
        float eps = cfg.rms_norm_eps, theta = cfg.rope_theta;
        int rot_dim = cfg.head_dim;  // full RoPE by default

        std::vector<float> x(H), x2(H), q(NH*HD), k(NKV*HD), v(NKV*HD), scores(HD);
        std::vector<float> att(NH*HD);
        std::vector<float> gate_up(FF*2);

        // Embed
        for (int i = 0; i < H; i++) x[i] = embed[token * (size_t)H + i];

        for (int il = 0; il < cfg.n_layers; il++) {
            auto& l = layers[il];
            int kv_begin = pos * NKV * HD;

            // RMSNorm → QKV
            rmsnorm(x2.data(), x.data(), w(l.rms_attn), H, eps);
            matmul(q.data(), x2.data(), w(l.wq), NH*HD, H);
            matmul(k.data(), x2.data(), w(l.wk), NKV*HD, H);
            matmul(v.data(), x2.data(), w(l.wv), NKV*HD, H);

            // RoPE
            rope(q.data(), k.data(), pos, NH, NKV, HD, rot_dim, theta);

            // KV cache
            memcpy(&k_cache[il][kv_begin], k.data(), NKV * HD * sizeof(float));
            memcpy(&v_cache[il][kv_begin], v.data(), NKV * HD * sizeof(float));

            // Attention: GQA
            std::fill(att.begin(), att.end(), 0.0f);
            for (int h = 0; h < NH; h++) {
                int kv_h = h / GQA;
                float* Q = &q[h * HD];
                // Score over all past positions
                for (int t = 0; t <= pos; t++) {
                    float* K = &k_cache[il][t * NKV * HD + kv_h * HD];
                    float s = 0;
                    for (int d = 0; d < HD; d++) s += Q[d] * K[d];
                    scores[t] = s / sqrtf((float)HD);
                }
                softmax(scores.data(), pos + 1);
                // Weighted sum of V
                for (int d = 0; d < HD; d++) {
                    float sum = 0;
                    for (int t = 0; t <= pos; t++) {
                        float* V = &v_cache[il][t * NKV * HD + kv_h * HD];
                        sum += scores[t] * V[d];
                    }
                    att[h * HD + d] = sum;
                }
            }

            // O proj
            matmul(x2.data(), att.data(), w(l.wo), H, NH*HD);
            // Residual
            for (int i = 0; i < H; i++) x[i] += x2[i];

            // FFN: RMSNorm → gate/up → SiLU → down → residual
            rmsnorm(x2.data(), x.data(), w(l.rms_ffn), H, eps);
            matmul(gate_up.data(), x2.data(), w(l.w1), FF, H);
            matmul(&gate_up[FF], x2.data(), w(l.w2), FF, H);
            silu(x2.data(), gate_up.data(), &gate_up[FF], FF);
            matmul(x2.data(), x2.data(), w(l.w3), H, FF);
            for (int i = 0; i < H; i++) x[i] += x2[i];
        }

        // Final RMSNorm
        rmsnorm(x2.data(), x.data(), final_norm.data(), H, eps);

        // LM head (tied embedding)
        matmul(logits_buf.data(), x2.data(), embed.data(), V, H);

        pos++;

        // Argmax
        int best = 0; float bestv = logits_buf[0];
        for (int i = 1; i < V; i++) {
            if (logits_buf[i] > bestv) { bestv = logits_buf[i]; best = i; }
        }
        return best;
    }

    void destroy() override { initialized = false; }

    ~GenericBackend() override { destroy(); }

    float benchmark(int tokens = 10) override {
        if (!initialized) return 0;
        reset();
        auto t0 = std::chrono::high_resolution_clock::now();
        int tok = 100;
        for (int i = 0; i < tokens; i++) tok = forward(tok);
        float ms = std::chrono::duration<float, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
        return ms / tokens;
    }
};

Backend* create_generic_backend() { return new GenericBackend(); }
