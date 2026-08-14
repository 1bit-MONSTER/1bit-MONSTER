// backend_generic.cpp — Universal CPU inference backend
// Reads ModelConfig from any discovered GGUF/H1B/BIN model and runs inference.
// Supports Llama, Mistral, Qwen2, Qwen3, Gemma, Phi architectures with:
//   RMSNorm / LayerNorm, RoPE (partial/full), GQA/MHA, SiLU/SwiGLU/GeGLU, KV cache,
//   optional QKV bias, optional per-head Q/K-norm (Qwen3), MoE FFN routing
//   (top-k softmax gating over "_exps" stacked expert tensors — Qwen2-MoE/
//   Qwen3-MoE/Mixtral convention; no shared-expert or expert-bias support).

#include "backend.h"
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>
#include "model_discovery.h"
#include "safetensors_reader.h"
#include "rocm_cpp/tokenizer.h"

// 1BP loader with full TQ2/Q4NX tile dequant (unity build)
#include "../engine/npu/src/onebp_loader.cpp"

#include <cstdio>
#include <cstring>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <algorithm>
#include <vector>
#include <string>
#include <fstream>
#include <memory>
#include <immintrin.h>

// ── Generic CPU Backend ──────────────────────────────────────────────────────

// #1240: the packed TQ2 GEMV uses BMI2 (_pext_u32) + AVX-512 (_mm512_*). Gate
// it at runtime so CPUs without that ISA (pre-Haswell Intel, Haswell has BMI2
// but no AVX-512, older AMD, future non-x86 ports) fall back to the fp32 path
// instead of SIGILL. This is the cross-platform CPU backend — it must never
// crash on the packed path.
static bool cpu_has_packed_isa() {
#if defined(__x86_64__) || defined(__i386__)
    return __builtin_cpu_supports("bmi2") && __builtin_cpu_supports("avx512f");
#else
    return false;
#endif
}

// #1346: the AVX-512 kernels (gemv_row_avx512 / silu_avx512 below) are always
// compiled via target attribute and selected at runtime, so a binary built
// with -march=native on an AVX-512 machine keeps a portable fallback and
// never SIGILLs on older hosts. Same rationale as cpu_has_packed_isa() (#1240):
// this is the cross-platform CPU backend — it must never crash on the dense
// GEMV/SiLU path either.
static bool cpu_has_avx512() {
#if defined(__x86_64__) || defined(__i386__)
    return __builtin_cpu_supports("avx512f");
#else
    return false;
#endif
}

struct GenericBackend : Backend {
    std::vector<float> embed, final_norm, output_weight;
    std::vector<std::vector<float>> layer_w;  // flat per-layer weights
    // Packed TQ2 path (WS-04): keep the 1BP mmap alive and run multiplication-free
    // GEMV on raw 2-bit tiles instead of dequantizing to fp32. Used only when
    // packed_ is true (1BP TQ2 models); all other formats use the fp32 path.
    std::unique_ptr<NpuOnebpModel> tq2_;
    bool packed_ = false;
    int tr_ = 32, tc_ = 256, gs_ = 32;
    struct PackedW { const uint8_t* base = nullptr; int N = 0, K = 0; };
    std::vector<PackedW> packed_w_;
    int pk_lm_ = -1;
    std::vector<std::vector<float>> k_cache, v_cache; // KV cache [n_layers][max_seq * n_kv * hd]
    int pos = 0;
    std::vector<float> logits_buf;

    // Pre-allocated scratch buffers (avoid per-token heap allocations in forward_embed)
    std::vector<float> scratch_x, scratch_x2, scratch_x3, scratch_q, scratch_k, scratch_v;
    std::vector<float> scratch_scores, scratch_att, scratch_gate_up, scratch_silu_buf;
    std::vector<float> scratch_moe_router_probs, scratch_moe_ffn_acc;
    std::vector<float> scratch_moe_gate, scratch_moe_up, scratch_moe_silu, scratch_moe_down;
    std::vector<int> scratch_moe_idx;  // expert index for partial_sort
    std::vector<float> rope_freqs;     // precomputed RoPE frequencies (1/theta^(2i/rot_dim))
    std::vector<float> rope_freqs_local; // gemma3 hybrid: local-layer theta table (0 = same)
    float inv_sqrt_hd_ = 0.0f;         // cached 1/sqrt(head_dim)
    bool scratch_allocated_ = false;
    int cached_debug_ops_ = -1;  // cached getenv("CPU_DEBUG_OPS") — checked once
    int cached_num_layers_ = -1;  // cached CPU_NUM_LAYERS

    // Per-layer weight indices. bq/bk/bv are optional QKV biases (Qwen2 and
    // some other architectures use biased attention projections, unlike
    // Llama) — SIZE_MAX means "not present in this model", distinct from a
    // legitimate index 0 into flat_weights.
    // q_norm/k_norm: optional per-head QK RMSNorm (Qwen3 and others), applied
    // to each head's head_dim-sized slice with a shared [head_dim] weight,
    // right after QKV bias and before RoPE.
    // moe_*: present only on MoE layers (cfg.n_experts > 0). w1/w2/w3 are
    // unused for such layers — the FFN routes through the expert tensors
    // instead. moe_gate_exps/moe_up_exps/moe_down_exps are flat [n_expert *
    // n_ff * hidden]-sized blocks (GGUF's stacked "_exps" 3D tensors); expert
    // e's slice starts at index e * n_ff * hidden and is row-major [n_ff,
    // hidden] — i.e. the exact same layout matmul() already expects for the
    // dense case, just offset per-expert.
    struct LayerW {
        size_t wq = SIZE_MAX, wk = SIZE_MAX, wv = SIZE_MAX, wo = SIZE_MAX;
        size_t rms_attn = SIZE_MAX, rms_ffn = SIZE_MAX;
        size_t w1 = SIZE_MAX, w2 = SIZE_MAX, w3 = SIZE_MAX;
        size_t bq = SIZE_MAX, bk = SIZE_MAX, bv = SIZE_MAX;
        size_t bo = SIZE_MAX;              // attention-output bias (GPT-2 c_proj.bias)
        size_t w1_b = SIZE_MAX, w3_b = SIZE_MAX;  // FFN up/down biases (GPT-2 c_fc/c_proj)
        size_t rms_attn_b = SIZE_MAX, rms_ffn_b = SIZE_MAX;  // LayerNorm biases (GPT-2 ln_1/ln_2)
        size_t q_norm = SIZE_MAX, k_norm = SIZE_MAX;
        size_t post_attn_norm = SIZE_MAX, post_ffn_norm = SIZE_MAX;
        size_t moe_gate_inp = SIZE_MAX, moe_gate_exps = SIZE_MAX, moe_up_exps = SIZE_MAX, moe_down_exps = SIZE_MAX;
        int pk_q = -1, pk_k = -1, pk_v = -1, pk_o = -1, pk_w1 = -1, pk_w2 = -1, pk_w3 = -1;  // packed TQ2 slots
    };
    std::vector<LayerW> layers;
    // GPT-2: learned position-embedding table (wpe) + final LayerNorm bias.
    std::vector<float> pos_embed, final_norm_bias;
    // Gemma-2/3/4 logit soft-capping (attn cap 50, final cap 30). Keyed on
    // the arch STRING, not the enum — RCPP_ARCH_GEMMA also covers Granite
    // (Llama-style: no caps, rms eps 1e-5) and Gemma-1 (no caps).
    bool gemma_softcap_ = false;
    float attn_cap_ = 0.0f;    // attention-logit tanh cap (gemma-2/3, 0 = none)
    float final_cap_ = 0.0f;   // LM-head-logit tanh cap (gemma-2/3, 0 = none)
    // Gemma-family embeddings are stored unscaled; the model expects them
    // scaled by sqrt(hidden) at the input (gemma_pytorch / llama.cpp
    // gemma/gemma2/gemma3 graphs: inpL = scale(inpL, sqrtf(n_embd))).
    // Without this the residual stream's composition after layer 0 is wrong
    // and the model runs out of distribution (caught by the #1243 ppl gate:
    // Gemma-3-1B ppl 6e8 with correct tokenization).
    bool gemma_emb_scale_ = false;

    GenericBackend() { type = BackendType::GENERIC; name = "Generic CPU (GGUF)"; }

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

        int L = cfg.n_layers;
        layers.resize(L);
        // Missing .bin files must fail the load, not silently leave zero
        // weights: push(empty) returned a valid-looking idx pointing at the
        // end of flat_weights, so forward() read past the buffer.
        auto Ld = [&](std::vector<float> v) -> size_t {
            if (v.empty()) return SIZE_MAX;
            return push(std::move(v));
        };
        for (int i = 0; i < L; i++) {
            std::string p = "model_layers_" + std::to_string(i) + "_";
            // LayerW order: wq, wk, wv, wo, rms_attn, rms_ffn, w1, w2, w3
            layers[i] = {
                Ld(W(p + "self_attn_q_proj.weight")),          // wq
                Ld(W(p + "self_attn_k_proj.weight")),          // wk
                Ld(W(p + "self_attn_v_proj.weight")),          // wv
                Ld(W(p + "self_attn_o_proj.weight")),          // wo
                Ld(W(p + "input_layernorm.weight")),            // rms_attn
                Ld(W(p + "post_attention_layernorm.weight")),   // rms_ffn
                Ld(W(p + "mlp_gate_proj.weight")),              // w1
                Ld(W(p + "mlp_up_proj.weight")),                // w2
                Ld(W(p + "mlp_down_proj.weight")),              // w3
            };
            if (layers[i].wq == SIZE_MAX || layers[i].wk == SIZE_MAX ||
                layers[i].wv == SIZE_MAX || layers[i].wo == SIZE_MAX ||
                layers[i].rms_attn == SIZE_MAX || layers[i].rms_ffn == SIZE_MAX ||
                layers[i].w1 == SIZE_MAX || layers[i].w2 == SIZE_MAX ||
                layers[i].w3 == SIZE_MAX) {
                fprintf(stderr, "Generic: .bin weights incomplete at layer %d — ABORTING\n", i);
                layers.resize(0);
                flat_weights.clear();
                return;
            }
        }
    }

    size_t push(std::vector<float>&& v) {
        size_t idx = flat_weights.size();
        flat_weights.insert(flat_weights.end(), v.begin(), v.end());
        return idx;
    }
    std::vector<float> flat_weights;
    float* w(size_t idx) {

        if (idx >= flat_weights.size()) {
            fprintf(stderr, "[generic] FATAL: weight index %zu out of range (size=%zu)\n",
                    idx, flat_weights.size());
            static float g_zero = 0.0f;
            return &g_zero;
        }
        return flat_weights.data() + idx;
    }

    bool init(const ModelConfig& model_cfg, const std::string& weights_dir) override {
        cfg = model_cfg;
        if (!cfg.sane()) {
            fprintf(stderr, "Generic: REFUSING implausible config (hidden=%d layers=%d heads=%d kv=%d seq=%d)\n",
                    cfg.hidden_size, cfg.num_layers, cfg.num_heads, cfg.num_kv_heads, cfg.max_seq_len);
            return false;
        }
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
            FILE* mf = fopen(gguf_path.c_str(), "rb");
            bool is_gguf = false;
            if (mf) {
                char magic[4] = {0};
                if (fread(magic, 1, 4, mf) == 4) is_gguf = (memcmp(magic, "GGUF", 4) == 0);
                fclose(mf);
            }
            if (is_gguf) {
                printf("Generic: trying GGUF path: %s\n", gguf_path.c_str());
                loaded = load_gguf(gguf_path);
            }
        }
        if (!loaded && cfg.format == ModelFormat::ONEBP && !cfg.model_path.empty()) {
            // Try loading from 1BP format (shared with NPU engine)
            loaded = load_1bp(cfg.model_path);
        }
        if (!loaded && cfg.format == ModelFormat::SAFETENSORS && !cfg.model_path.empty()) {
            // HF-native: load weights directly from a .safetensors checkpoint.
            loaded = load_safetensors(cfg.model_path);
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
        for (auto& k : k_cache) k.resize((size_t)cfg.max_seq_len * cfg.n_kv_heads * cfg.head_dim);
        for (auto& v : v_cache) v.resize((size_t)cfg.max_seq_len * cfg.n_kv_heads * cfg.head_dim);

        // Pre-allocate scratch buffers (avoid per-token heap allocations)
        int H = cfg.hidden, NH = cfg.n_heads, NKV = cfg.n_kv_heads, HD = cfg.head_dim, FF = cfg.intermediate_size;
        size_t score_sz = (size_t)cfg.max_seq_len > (size_t)HD ? (size_t)cfg.max_seq_len : (size_t)HD;
        scratch_x.resize(H); scratch_x2.resize(H); scratch_x3.resize(H);
        scratch_q.resize((size_t)NH * HD); scratch_k.resize((size_t)NKV * HD); scratch_v.resize((size_t)NKV * HD);
        scratch_scores.resize(score_sz); scratch_att.resize((size_t)NH * HD);
        scratch_gate_up.resize(FF * 2); scratch_silu_buf.resize(FF);
        if (cfg.n_experts > 0) {
            scratch_moe_router_probs.resize(cfg.n_experts);
            scratch_moe_ffn_acc.resize(H);
            scratch_moe_gate.resize(FF); scratch_moe_up.resize(FF);
            scratch_moe_silu.resize(FF);  // #1342: silu output is FF-sized — must NOT share the H-sized down buffer
            scratch_moe_down.resize(H);
            scratch_moe_idx.resize(cfg.n_experts);
        }
        scratch_allocated_ = true;

        // Precompute RoPE frequencies (avoid powf+cosf+sinf per token)
        int rot_dim = cfg.rope_dim > 0 ? cfg.rope_dim : cfg.head_dim, half = rot_dim / 2;
        rope_freqs.resize(half);
        float theta = cfg.rope_theta > 0 ? cfg.rope_theta : 10000.0f;
        for (int i = 0; i < half; i++)
            rope_freqs[i] = 1.0f / powf(theta, (2.0f * i) / (float)rot_dim);
        // Gemma-3 hybrid: local layers use rope_local_base_freq (10000), full
        // layers use rope_theta (1e6). Precompute both tables.
        if (cfg.sliding_window_pattern > 0 && cfg.rope_local_base_freq > 0.0f) {
            rope_freqs_local.resize(half);
            for (int i = 0; i < half; i++)
                rope_freqs_local[i] = 1.0f / powf(cfg.rope_local_base_freq, (2.0f * i) / (float)rot_dim);
        }
        inv_sqrt_hd_ = 1.0f / sqrtf((float)cfg.head_dim);

        // Cache getenv results checked in the hot path
        cached_debug_ops_ = getenv("CPU_DEBUG_OPS") ? 1 : 0;
        {
            const char* nl = getenv("CPU_NUM_LAYERS");
            cached_num_layers_ = (nl && nl[0]) ? atoi(nl) : cfg.n_layers;
        }

        // Gemma-2/3/4: logit soft-capping + rms eps 1e-6. Caps come from the
        // config when present (gemma2: 50/30; gemma3-1b: none), with the old
        // arch-string defaults as fallback for config-less GGUF loads.
        gemma_softcap_ = cfg.architecture.rfind("gemma", 0) == 0 && cfg.architecture != "gemma";
        attn_cap_ = cfg.attn_logit_softcapping > 0.0f ? cfg.attn_logit_softcapping
                  : (gemma_softcap_ ? 50.0f : 0.0f);
        final_cap_ = cfg.final_logit_softcapping > 0.0f ? cfg.final_logit_softcapping
                   : (gemma_softcap_ ? 30.0f : 0.0f);
        if (cfg.architecture.rfind("gemma", 0) == 0) cfg.rms_norm_eps = 1e-6f;
        // Gemma-family input embedding scaling (sqrt(hidden)); Granite shares
        // the enum but is Llama-style and must NOT be scaled.
        // Gemma-2/3 input embedding scaling: the HF path multiplies token
        // embeddings by sqrt(hidden) (verified empirically for gemma3: lookup
        // output norm 32.46 = 0.956 x sqrt(1152)); llama.cpp's gemma graphs
        // bake the same. Granite shares the enum but is Llama-style.
        gemma_emb_scale_ = cfg.architecture.rfind("gemma", 0) == 0;

        initialized = true;
        return true;
    }

    bool load_1bp(const std::string& path) {
        printf("Generic: loading 1BP: %s\n", path.c_str());
        tq2_ = std::make_unique<NpuOnebpModel>();
        if (!tq2_->open(path.c_str())) {
            fprintf(stderr, "Generic: failed to open 1BP\n");
            return false;
        }
        NpuOnebpModel& model = *tq2_;
        auto& h = model.header();
        packed_ = (h.quant == ONEBP_TQ2) && !getenv("GENERIC_NO_PACKED") && cpu_has_packed_isa();
        if (h.quant == ONEBP_TQ2 && !packed_)
            printf("Generic: TQ2 packed path OFF (%s) — using fp32 path\n",
                   cpu_has_packed_isa() ? "GENERIC_NO_PACKED set" : "CPU lacks BMI2+AVX-512");
        if (packed_) {
            tr_ = h.tile_rows; tc_ = h.tile_cols; gs_ = h.group_size;
            printf("Generic: TQ2 packed path ON (tile %dx%d gs=%d)\n", tr_, tc_, gs_);
        }
        if (cfg.hidden == 0 || cfg.format == ModelFormat::ONEBP) {   // header is authoritative for 1BP
            cfg.hidden = cfg.hidden_size = h.hidden_size;
            cfg.n_layers = cfg.num_layers = h.num_layers;
            cfg.n_heads = cfg.num_heads = cfg.num_attention_heads = h.num_attention_heads;
            cfg.n_kv_heads = cfg.num_kv_heads = h.num_kv_heads ? h.num_kv_heads : h.num_attention_heads;
            cfg.head_dim = h.head_dim ? h.head_dim : 128;
            cfg.n_ff = cfg.intermediate_size = h.intermediate_size;
            cfg.vocab = cfg.vocab_size = h.vocab_size;
        }
        printf("Generic: 1BP H=%d NC=%d NH=%d NKV=%d HD=%d IM=%d V=%d quant=%u\n",
               cfg.hidden, cfg.n_layers, cfg.n_heads, cfg.n_kv_heads,
               cfg.head_dim, cfg.n_ff, cfg.vocab, h.quant);

        // Load embedding
        if (!model.get_tensor_f32("token_embd.weight", embed)) {
            fprintf(stderr, "Generic: missing token_embd.weight\n");
            return false;
        }
        if (getenv("CPU_DEBUG_OPS")) {
            double s = 0, mx = 0;
            for (size_t i = 0; i < embed.size(); i++) { s += fabs(embed[i]); if (fabs(embed[i]) > mx) mx = fabs(embed[i]); }
            fprintf(stderr, "[cpu] embed mean|.|=%g max|.|=%g\n", s / embed.size(), mx);
        }

        // Load final norm + output weight
        if (!model.get_tensor_f32("token_embd_norm.weight", final_norm))
            if (!model.get_tensor_f32("model.norm.weight", final_norm))
                model.get_tensor_f32("output_norm.weight", final_norm);   // 1BP writer convention
        if (!model.get_tensor_f32("output.weight", output_weight))
            model.get_tensor_f32("lm_head.weight", output_weight);
        if (packed_) {
            if (const uint8_t* b = model.get_tile_ptr("output.weight", 0, 0)) {
                packed_w_.push_back({b, cfg.vocab, cfg.hidden});
                pk_lm_ = (int)packed_w_.size() - 1;
            } else if (const uint8_t* b = model.get_tile_ptr("lm_head.weight", 0, 0)) {
                packed_w_.push_back({b, cfg.vocab, cfg.hidden});
                pk_lm_ = (int)packed_w_.size() - 1;
            }
        }

        // Per-layer weights
        layers.resize(cfg.n_layers);
        char buf[128];
        for (int l = 0; l < cfg.n_layers; l++) {
            auto& lw = layers[l];
            auto load = [&](const char* blk, const char* legacy, size_t& idx,
                            int rows, int cols) {
                std::vector<float> w;
                snprintf(buf, sizeof(buf), "blk.%d.%s", l, blk);
                if (!model.get_tensor_f32(buf, w)) {
                    snprintf(buf, sizeof(buf), "model.layers.%d.%s", l, legacy);
                    model.get_tensor_f32(buf, w);
                }
                if ((int)w.size() == rows * cols)
                    idx = push(std::move(w));
                else if (getenv("CPU_DEBUG_OPS"))
                    fprintf(stderr, "[generic] load fail: %s (%d elems, want %d)\n",
                            buf, (int)w.size(), rows * cols);
            };
            int H = cfg.hidden, NH = cfg.n_heads, NKV = cfg.n_kv_heads;
            int HD = cfg.head_dim, IM = cfg.n_ff;
            load("attn_q.weight", "self_attn.q_proj.weight", lw.wq, H, NH*HD);
            load("attn_k.weight", "self_attn.k_proj.weight", lw.wk, H, NKV*HD);
            load("attn_v.weight", "self_attn.v_proj.weight", lw.wv, H, NKV*HD);
            load("attn_output.weight", "self_attn.o_proj.weight", lw.wo, NH*HD, H);
            load("ffn_gate.weight", "mlp.gate_proj.weight", lw.w1, H, IM);
            load("ffn_up.weight", "mlp.up_proj.weight", lw.w2, H, IM);
            load("ffn_down.weight", "mlp.down_proj.weight", lw.w3, IM, H);
            if (packed_) {
                auto pk = [&](const char* blk, const char* legacy, int& idx, int M, int K) {
                    char b2[128];
                    snprintf(b2, sizeof(b2), "blk.%d.%s", l, blk);
                    const uint8_t* base = model.get_tile_ptr(b2, 0, 0);
                    if (!base) {
                        snprintf(b2, sizeof(b2), "model.layers.%d.%s", l, legacy);
                        base = model.get_tile_ptr(b2, 0, 0);
                    }
                    if (base) { packed_w_.push_back({base, M, K}); idx = (int)packed_w_.size() - 1; }
                };
                pk("attn_q.weight", "self_attn.q_proj.weight", lw.pk_q, NH*HD, H);
                pk("attn_k.weight", "self_attn.k_proj.weight", lw.pk_k, NKV*HD, H);
                pk("attn_v.weight", "self_attn.v_proj.weight", lw.pk_v, NKV*HD, H);
                pk("attn_output.weight", "self_attn.o_proj.weight", lw.pk_o, H, NH*HD);
                pk("ffn_gate.weight", "mlp.gate_proj.weight", lw.pk_w1, IM, H);
                pk("ffn_up.weight", "mlp.up_proj.weight", lw.pk_w2, IM, H);
                pk("ffn_down.weight", "mlp.down_proj.weight", lw.pk_w3, H, IM);
            }

            // RMS norm weights
            load("input_norm.weight", "input_layernorm.weight", lw.rms_attn, H, 1);
            load("attn_norm.weight", "input_layernorm.weight", lw.rms_attn, H, 1);   // 1BP writer convention
            load("post_attention_norm.weight", "post_attention_layernorm.weight", lw.rms_ffn, H, 1);
            load("ffn_norm.weight", "post_attention_layernorm.weight", lw.rms_ffn, H, 1); // 1BP writer convention
            // Gemma-2/3 post-norms: RMSNorm on the attn/FFN outputs before
            // the residual adds (see issue #1243 ppl gate). Distinct from the
            // pre-FFN rms_ffn above; absent on Llama-class models -> SIZE_MAX.
            load("post_attention_norm.weight", "post_attention_layernorm.weight", lw.post_attn_norm, H, 1);
            load("post_ffw_norm.weight", "post_ffw_norm.weight", lw.post_ffn_norm, H, 1);
            // Qwen3 per-head QK-norm (raw fp32 vectors in 1BP); required for
            // attention stability — without it Qwen3 logits collapse (flat).
            load("attn_q_norm.weight", "self_attn.q_norm.weight", lw.q_norm, HD, 1);
            load("attn_k_norm.weight", "self_attn.k_norm.weight", lw.k_norm, HD, 1);
            // Required tensors must all be present with the exact expected
            // size; anything else is a corrupt/mismatched 1BP file that would
            // silently produce garbage (or read past g_zero via w(SIZE_MAX)).
            if (lw.wq == SIZE_MAX || lw.wk == SIZE_MAX || lw.wv == SIZE_MAX ||
                lw.wo == SIZE_MAX || lw.rms_attn == SIZE_MAX || lw.rms_ffn == SIZE_MAX ||
                lw.w1 == SIZE_MAX || lw.w2 == SIZE_MAX || lw.w3 == SIZE_MAX) {
                fprintf(stderr, "Generic: 1BP layer %d: missing required tensor — ABORTING LOAD\n", l);
                return false;
            }
        }
        printf("Generic: 1BP loaded — %d layers, %.1fM params\n",
               cfg.n_layers, (double)embed.size() / 1e6);
        if (!packed_) tq2_.reset();  // Q4NX: fp32 pool is self-contained
        return true;
    }

    // Inverse of llama.cpp's convert-time RoPE pre-rotation: GGUF stores
    // attn_q/k with dims interleaved [0, hd/2, 1, hd/2+1, ...]; the engine's
    // half-split pairing expects natural order (corrected 2026-08-13).
    void unrotate_rope_rows(std::vector<float>& flat, size_t idx, int heads, int dim, int in) {
        if (dim < 2 || (dim & 1)) return;
        std::vector<float> tmp(&flat[idx], &flat[idx] + (size_t)heads * dim * in);
        std::vector<float> out(tmp.size());
        for (int h = 0; h < heads; h++) {
            for (int p = 0; p < dim; p++) {
                int d = (p % 2) * (dim / 2) + (p / 2);   // inverse of the interleave
                memcpy(&out[(size_t)(h * dim + d) * in], &tmp[(size_t)(h * dim + p) * in],
                       (size_t)in * sizeof(float));
            }
        }
        memcpy(&flat[idx], out.data(), out.size() * sizeof(float));
    }


    // HF-native weight loading: read a .safetensors checkpoint directly into
    // the same flat-weights structure the GGUF/.bin loaders use. Weights are
    // NATURAL order (no RoPE pre-rotation) — the engine's half-split rope
    // pairing matches transformers exactly (corrected 2026-08-13).
    bool load_safetensors(const std::string& path) {
        std::string f = path;
        struct stat st;
        if (stat(f.c_str(), &st) == 0 && S_ISDIR(st.st_mode)) {
            DIR* d = opendir(f.c_str());
            if (d) {
                struct dirent* e;
                while ((e = readdir(d)) != nullptr) {
                    std::string n(e->d_name);
                    if (n.size() > 12 && n.substr(n.size() - 12) == ".safetensors") { f = f + "/" + n; break; }
                }
                closedir(d);
            }
        }
        SafetensorsWeightReader r;
        bool opened = false;
        // Sharded checkpoint? A sibling model.safetensors.index.json means the
        // real tensors span shards — the index MUST take precedence over a
        // single-shard open.
        std::string dir = f;
        size_t slash = dir.find_last_of('/');
        if (slash != std::string::npos) dir = dir.substr(0, slash);
        else dir = ".";
        struct stat idx_st;
        if (stat((dir + "/model.safetensors.index.json").c_str(), &idx_st) == 0 && S_ISREG(idx_st.st_mode))
            opened = r.open_dir(dir);
        if (!opened && stat(f.c_str(), &st) == 0 && S_ISREG(st.st_mode))
            opened = r.open(f);
        if (!opened) {
            fprintf(stderr, "Generic: safetensors: cannot read %s (%s)\n", f.c_str(), r.error().c_str());
            return false;
        }

        // Gemma2/3/4 RMSNorm convention: HF stores gamma and the model computes
        // x * (1 + gamma); llama.cpp's GGUF bakes the +1. The safetensors
        // loader adds +1 to every norm weight for gemma2/3/4.
        const bool gemma_post_norms = (cfg.architecture == "gemma2" ||
                                       cfg.architecture == "gemma3" ||
                                       cfg.architecture == "gemma4");

        // Arch guard — same refusal list as load_gguf + unknown.
        if (cfg.arch == RCPP_ARCH_UNKNOWN ||
            cfg.arch == RCPP_ARCH_ZAYA || cfg.arch == RCPP_ARCH_ZAMBA2 ||
            cfg.arch == RCPP_ARCH_ZAMBA || cfg.arch == RCPP_ARCH_MAMBA ||
            cfg.arch == RCPP_ARCH_QWEN35) {
            fprintf(stderr, "  [generic] Refusing to load %s (arch=%d%s) via safetensors\n",
                    f.c_str(), (int)cfg.arch,
                    cfg.arch == RCPP_ARCH_UNKNOWN ? " UNKNOWN — add an arch mapping" : "");
            return false;
        }

        // OLMo (allenai OLMo-1B-0724-hf): LayerNorm has NO learnable weights
        // (OlmoLayerNorm: mean/var only, eps 1e-5) and attention clips QKV to
        // [-8, 8] (config clip_qkv: 8.0). Verified against modeling_olmo.py
        // 2026-08-14. Also uses RoPE theta 10000 (natural order).
        if (cfg.arch == RCPP_ARCH_OLMO) {
            cfg.norm_is_layernorm = true;
            cfg.clip_qkv = 8.0f;
            cfg.rms_norm_eps = 1e-5f;
        }
        // GPT-2: learned position embeddings (wpe), LayerNorm with affine
        // weight+bias, no RoPE, non-gated gelu FFN. Names are h.N.* (no
        // "model." prefix), wte/wpe/ln_f under "transformer.".
        if (cfg.arch == RCPP_ARCH_GPT2) {
            cfg.norm_is_layernorm = true;
            cfg.use_learned_pos = true;
            cfg.no_rope = true;
            cfg.rms_norm_eps = 1e-5f;
        }
        // Falcon (old arch): nn.LayerNorm with weight+bias, PARALLEL
        // attention+FFN on one norm, fused MQA qkv, erf-gelu non-gated FFN,
        // RoPE theta 10000 (natural). Names: transformer.h.N.*.
        if (cfg.arch == RCPP_ARCH_FALCON) {
            cfg.norm_is_layernorm = true;
            cfg.parallel_attn_ffn = true;
            cfg.rms_norm_eps = 1e-5f;
        }
        // GPT-NeoX/Pythia: nn.LayerNorm weight+bias, PARALLEL attn+FFN on
        // one norm (use_parallel_residual), fused query_key_value, no-GQA,
        // non-gated erf-gelu FFN with biases everywhere, rotary theta from
        // rotary_emb_base. Names: gpt_neox.layers.N.*.
        if (cfg.arch == RCPP_ARCH_GPTNEOX) {
            cfg.norm_is_layernorm = true;
            cfg.parallel_attn_ffn = true;
            cfg.rms_norm_eps = 1e-5f;
        }
        // OPT: learned position embeddings (embed_positions), nn.LayerNorm
        // weight+bias, projection biases everywhere, NON-GATED RELU FFN
        // (fc1/fc2), no rotary, sequential structure. Names: model.decoder.*.
        if (cfg.arch == RCPP_ARCH_OPT) {
            cfg.norm_is_layernorm = true;
            cfg.use_learned_pos = true;
            cfg.no_rope = true;
            cfg.pos_offset = 2;  // OPT pads 2 positions (offset=2 in embed_positions)
            cfg.rms_norm_eps = 1e-5f;
        }

        if (!r.get_tensor_f32("model.embed_tokens.weight", embed) &&
            !r.get_tensor_f32("token_embd.weight", embed) &&
            !r.get_tensor_f32("transformer.wte.weight", embed) &&
            !r.get_tensor_f32("wte.weight", embed) &&
            !r.get_tensor_f32("transformer.word_embeddings.weight", embed) &&
            !r.get_tensor_f32("model.tok_embeddings.weight", embed) &&
            !r.get_tensor_f32("gpt_neox.embed_in.weight", embed) &&
            !r.get_tensor_f32("model.decoder.embed_tokens.weight", embed)) {
            fprintf(stderr, "Generic: safetensors: missing embedding\n");
            return false;
        }
        if (cfg.hidden > 0 && embed.size() % (size_t)cfg.hidden == 0 && !embed.empty())
            cfg.vocab = cfg.vocab_size = (int)(embed.size() / (size_t)cfg.hidden);

        if (cfg.use_learned_pos) {
            // Learned position table — authoritative for max_seq_len (gpt2 wpe,
            // OPT embed_positions).
            if (!r.get_tensor_f32("transformer.wpe.weight", pos_embed) &&
                !r.get_tensor_f32("wpe.weight", pos_embed) &&
                !r.get_tensor_f32("model.decoder.embed_positions.weight", pos_embed)) {
                fprintf(stderr, "Generic: safetensors: missing position table\n");
                return false;
            }
            if (pos_embed.empty() || pos_embed.size() % (size_t)cfg.hidden != 0) {
                fprintf(stderr, "Generic: safetensors: position table misized\n");
                return false;
            }
            cfg.max_seq_len = (int)(pos_embed.size() / (size_t)cfg.hidden);
            if (!r.get_tensor_f32("transformer.ln_f.weight", final_norm))
                r.get_tensor_f32("ln_f.weight", final_norm);
            if (!r.get_tensor_f32("transformer.ln_f.bias", final_norm_bias))
                r.get_tensor_f32("ln_f.bias", final_norm_bias);
        }

        if (!r.get_tensor_f32("model.norm.weight", final_norm))
            if (!r.get_tensor_f32("token_embd_norm.weight", final_norm))
                if (!r.get_tensor_f32("transformer.ln_f.weight", final_norm))
                    if (!r.get_tensor_f32("gpt_neox.final_layer_norm.weight", final_norm))
                        if (!r.get_tensor_f32("model.decoder.final_layer_norm.weight", final_norm))
                            r.get_tensor_f32("ln_f.weight", final_norm);
        if (!r.get_tensor_f32("transformer.ln_f.bias", final_norm_bias))
            if (!r.get_tensor_f32("gpt_neox.final_layer_norm.bias", final_norm_bias))
                if (!r.get_tensor_f32("model.decoder.final_layer_norm.bias", final_norm_bias))
                    r.get_tensor_f32("ln_f.bias", final_norm_bias);
        if (gemma_post_norms && !final_norm.empty())
            for (auto& v : final_norm) v += 1.0f;
        if (!r.get_tensor_f32("output.weight", output_weight))
            if (!r.get_tensor_f32("lm_head.weight", output_weight))
                r.get_tensor_f32("embed_out.weight", output_weight);  // GPT-NeoX untied lm_head

        layers.resize(cfg.n_layers);
        char buf[128];
        for (int l = 0; l < cfg.n_layers; l++) {
            auto& lw = layers[l];
            auto load = [&](const char* hf_name, size_t& idx, int rows, int cols) {
                std::vector<float> w;
                snprintf(buf, sizeof(buf), "model.layers.%d.%s", l, hf_name);
                if (!r.get_tensor_f32(buf, w)) return;
                if ((int)w.size() == rows * cols) idx = push(std::move(w));
                else fprintf(stderr, "Generic: safetensors %s: %d elems, want %d\n",
                             buf, (int)w.size(), rows * cols);
            };
            int H = cfg.hidden, NH = cfg.n_heads, NKV = cfg.n_kv_heads;
            int HD = cfg.head_dim, IM = cfg.n_ff;
            if (cfg.arch == RCPP_ARCH_PHI) {
                // Phi-3 (and Phi-2 fused variants): self_attn.qkv_proj.weight
                // is Q+K+V stacked row-wise [Q(NH*HD) | K(NKV*HD) | V(NKV*HD)];
                // mlp.gate_up_proj.weight is [gate(IM) | up(IM)].
                std::vector<float> qkv;
                snprintf(buf, sizeof(buf), "model.layers.%d.self_attn.qkv_proj.weight", l);
                if (!r.get_tensor_f32(buf, qkv) ||
                    (int)qkv.size() != (size_t)(NH + 2 * NKV) * HD * H) {
                    fprintf(stderr, "Generic: safetensors %s: missing/misized qkv_proj\n", buf);
                    return false;
                }
                std::vector<float> q, k, v;
                q.reserve((size_t)NH * HD * H); k.reserve((size_t)NKV * HD * H); v.reserve((size_t)NKV * HD * H);
                size_t qr = (size_t)NH * HD, kr = (size_t)NKV * HD;
                for (size_t r = 0; r < qr; r++) q.insert(q.end(), qkv.begin() + r * H, qkv.begin() + (r + 1) * H);
                for (size_t r = 0; r < kr; r++) k.insert(k.end(), qkv.begin() + (qr + r) * H, qkv.begin() + (qr + r + 1) * H);
                for (size_t r = 0; r < kr; r++) v.insert(v.end(), qkv.begin() + (qr + kr + r) * H, qkv.begin() + (qr + kr + r + 1) * H);
                lw.wq = push(std::move(q)); lw.wk = push(std::move(k)); lw.wv = push(std::move(v));
                load("self_attn.o_proj.weight", lw.wo, NH * HD, H);
                std::vector<float> gu;
                snprintf(buf, sizeof(buf), "model.layers.%d.mlp.gate_up_proj.weight", l);
                if (!r.get_tensor_f32(buf, gu) || (int)gu.size() != (size_t)2 * IM * H) {
                    fprintf(stderr, "Generic: safetensors %s: missing/misized gate_up_proj\n", buf);
                    return false;
                }
                std::vector<float> gate, up;
                gate.reserve((size_t)IM * H); up.reserve((size_t)IM * H);
                for (size_t r = 0; r < (size_t)IM; r++) gate.insert(gate.end(), gu.begin() + r * H, gu.begin() + (r + 1) * H);
                for (size_t r = 0; r < (size_t)IM; r++) up.insert(up.end(), gu.begin() + ((size_t)IM + r) * H, gu.begin() + ((size_t)IM + r + 1) * H);
                lw.w1 = push(std::move(gate)); lw.w2 = push(std::move(up));
                load("mlp.down_proj.weight", lw.w3, IM, H);
            } else if (cfg.arch == RCPP_ARCH_GPT2) {
                // GPT-2: h.N.* names (no "model." prefix), fused c_attn, biases
                // everywhere, LayerNorm with weight+bias (ln_1/ln_2), non-gated
                // gelu FFN (c_fc/c_proj). lw.w2 stays SIZE_MAX (no gate).
                // CRITICAL: GPT-2 Conv1D stores weights [in, out] (NOT the
                // HF-linear [out, in]) — every projection must be transposed.
                auto push_t = [&](const std::vector<float>& w, int rows, int cols) {
                    // w is [rows, cols]; return transposed [cols, rows] index
                    std::vector<float> t((size_t)rows * cols);
                    for (int r = 0; r < rows; r++)
                        for (int c = 0; c < cols; c++)
                            t[(size_t)c * rows + r] = w[(size_t)r * cols + c];
                    return push(std::move(t));
                };
                auto load2 = [&](const char* tname, size_t& idx, int in_rows, int out_cols) {
                    std::vector<float> w;
                    snprintf(buf, sizeof(buf), "h.%d.%s", l, tname);
                    if (!r.get_tensor_f32(buf, w)) return;
                    if ((int)w.size() == in_rows * out_cols) idx = push_t(w, in_rows, out_cols);
                    else fprintf(stderr, "Generic: safetensors %s: %d elems, want %d\n", buf, (int)w.size(), in_rows * out_cols);
                };
                // c_attn.weight [H, 3H] (in, out) — split COLUMNS into
                // q/k/v [H, H] each, then transpose to engine [out, in].
                std::vector<float> qkv;
                snprintf(buf, sizeof(buf), "h.%d.attn.c_attn.weight", l);
                if (!r.get_tensor_f32(buf, qkv) || (int)qkv.size() != H * 3 * H) {
                    fprintf(stderr, "Generic: safetensors %s: missing/misized c_attn\n", buf);
                    return false;
                }
                {
                    std::vector<float> q((size_t)H * H), k((size_t)H * H), v((size_t)H * H);
                    for (int r = 0; r < H; r++) {
                        for (int c = 0; c < H; c++) {
                            q[(size_t)c * H + r] = qkv[(size_t)r * 3 * H + c];
                            k[(size_t)c * H + r] = qkv[(size_t)r * 3 * H + H + c];
                            v[(size_t)c * H + r] = qkv[(size_t)r * 3 * H + 2 * H + c];
                        }
                    }
                    lw.wq = push(std::move(q)); lw.wk = push(std::move(k)); lw.wv = push(std::move(v));
                }
                load2("attn.c_proj.weight", lw.wo, H, H);
                load2("mlp.c_fc.weight", lw.w1, H, IM);
                load2("mlp.c_proj.weight", lw.w3, IM, H);
                // c_attn.bias [3H] (out dim) → split q/k/v head biases.
                std::vector<float> qkvb;
                snprintf(buf, sizeof(buf), "h.%d.attn.c_attn.bias", l);
                if (r.get_tensor_f32(buf, qkvb) && (int)qkvb.size() == 3 * NH * HD) {
                    std::vector<float> bq(qkvb.begin(), qkvb.begin() + NH * HD);
                    std::vector<float> bk(qkvb.begin() + NH * HD, qkvb.begin() + 2 * NH * HD);
                    std::vector<float> bv(qkvb.begin() + 2 * NH * HD, qkvb.end());
                    lw.bq = push(std::move(bq)); lw.bk = push(std::move(bk)); lw.bv = push(std::move(bv));
                }
                load2("attn.c_proj.bias", lw.bo, 1, NH * HD);
                load2("mlp.c_fc.bias", lw.w1_b, 1, IM);
                load2("mlp.c_proj.bias", lw.w3_b, 1, H);
                load2("ln_1.weight", lw.rms_attn, 1, H);
                load2("ln_1.bias", lw.rms_attn_b, 1, H);
                load2("ln_2.weight", lw.rms_ffn, 1, H);
                load2("ln_2.bias", lw.rms_ffn_b, 1, H);
            } else if (cfg.arch == RCPP_ARCH_FALCON) {
                // Falcon (old arch): transformer.h.N.* names. Fused MQA
                // query_key_value [H + 2*HD, H] (config: multi_query=True):
                // q rows [0,H), k rows [H,H+HD), v rows [H+HD, H+2HD) — one
                // kv head. nn.Linear [out,in] orientation (no transpose).
                // Parallel attn+FFN: rms_ffn stays SIZE_MAX (single
                // input_layernorm feeds both); w2 stays SIZE_MAX (erf-gelu).
                std::vector<float> qkv;
                snprintf(buf, sizeof(buf), "transformer.h.%d.self_attention.query_key_value.weight", l);
                size_t qkv_rows = (size_t)NH * HD + 2 * (size_t)HD;
                if (!r.get_tensor_f32(buf, qkv) || (int)qkv.size() != (int)(qkv_rows * H)) {
                    fprintf(stderr, "Generic: safetensors %s: missing/misized qkv\n", buf);
                    return false;
                }
                size_t qrows = (size_t)NH * HD, krows = (size_t)HD;
                std::vector<float> q(qrows * H), k(krows * H), v(krows * H);
                for (size_t row = 0; row < qrows; row++)
                    memcpy(q.data() + row * H, qkv.data() + row * H, H * sizeof(float));
                for (size_t row = 0; row < krows; row++) {
                    memcpy(k.data() + row * H, qkv.data() + (qrows + row) * H, H * sizeof(float));
                    memcpy(v.data() + row * H, qkv.data() + (qrows + krows + row) * H, H * sizeof(float));
                }
                lw.wq = push(std::move(q)); lw.wk = push(std::move(k)); lw.wv = push(std::move(v));
                auto load2 = [&](const char* tname, size_t& idx, int rows, int cols) {
                    std::vector<float> w;
                    snprintf(buf, sizeof(buf), "transformer.h.%d.%s", l, tname);
                    if (!r.get_tensor_f32(buf, w)) return;
                    if ((int)w.size() == rows * cols) idx = push(std::move(w));
                    else fprintf(stderr, "Generic: safetensors %s: %d elems, want %d\n", buf, (int)w.size(), rows * cols);
                };
                load2("self_attention.dense.weight", lw.wo, NH * HD, H);
                load2("mlp.dense_h_to_4h.weight", lw.w1, H, IM);
                load2("mlp.dense_4h_to_h.weight", lw.w3, IM, H);
                load2("input_layernorm.weight", lw.rms_attn, H, 1);
                load2("input_layernorm.bias", lw.rms_attn_b, H, 1);
            } else if (cfg.arch == RCPP_ARCH_LLAMA && cfg.architecture == "internlm2") {
                // InternLM2: fused wqkv [(NH+2·NKV)·HD, H] with HEAD-INTERLEAVED
                // layout (llama.cpp conversion/ internlm.py): each kv-group is
                // [q(×qpk) | k | v] rows — NOT [all-q | all-k | all-v].
                // Names: attention.wqkv/wo, attention_norm/ffn_norm (RMSNorm),
                // feed_forward.w1/w2/w3. Llama orientation [out,in].
                std::vector<float> qkv;
                snprintf(buf, sizeof(buf), "model.layers.%d.attention.wqkv.weight", l);
                size_t qr2 = (size_t)NH * HD, kr2 = (size_t)NKV * HD;
                size_t qkv_rows = qr2 + 2 * kr2;
                if (!r.get_tensor_f32(buf, qkv) || (int)qkv.size() != (int)(qkv_rows * H)) {
                    fprintf(stderr, "Generic: safetensors %s: missing/misized wqkv\n", buf);
                    return false;
                }
                {
                    size_t qpk = (size_t)NH / NKV;          // q heads per kv group
                    size_t gsize = (qpk + 2) * (size_t)HD;  // rows per group
                    std::vector<float> q(qr2 * H), k(kr2 * H), v(kr2 * H);
                    for (size_t g = 0; g < (size_t)NKV; g++) {
                        for (size_t qh = 0; qh < qpk; qh++) {
                            memcpy(q.data() + ((g * qpk + qh) * HD) * H,
                                   qkv.data() + ((g * gsize + qh * HD)) * H, (size_t)HD * H * sizeof(float));
                        }
                        memcpy(k.data() + (g * HD) * H,
                               qkv.data() + ((g * gsize + qpk * HD)) * H, (size_t)HD * H * sizeof(float));
                        memcpy(v.data() + (g * HD) * H,
                               qkv.data() + ((g * gsize + (qpk + 1) * HD)) * H, (size_t)HD * H * sizeof(float));
                    }
                    lw.wq = push(std::move(q)); lw.wk = push(std::move(k)); lw.wv = push(std::move(v));
                }
                auto load2 = [&](const char* tname, size_t& idx, int rows, int cols) {
                    std::vector<float> w;
                    snprintf(buf, sizeof(buf), "model.layers.%d.%s", l, tname);
                    if (!r.get_tensor_f32(buf, w)) return;
                    if ((int)w.size() == rows * cols) idx = push(std::move(w));
                    else fprintf(stderr, "Generic: safetensors %s: %d elems, want %d\n", buf, (int)w.size(), rows * cols);
                };
                // InternLM2 MLP (modeling_internlm2.py):
                //   down = w2( silu(w1 x) * (w3 x) )  — w1 AND w3 are the
                //   UP projections ([IM,H], standard orientation), w2 is the
                //   DOWN ([H,IM]). So: engine w1(gate) ← w1, w2(up) ← w3,
                //   w3(down) ← w2. No transposes.
                load2("attention.wo.weight", lw.wo, NH * HD, H);
                load2("feed_forward.w1.weight", lw.w1, H, IM);
                load2("feed_forward.w3.weight", lw.w2, H, IM);
                load2("feed_forward.w2.weight", lw.w3, IM, H);
                load2("attention_norm.weight", lw.rms_attn, H, 1);
                load2("ffn_norm.weight", lw.rms_ffn, H, 1);
            } else if (cfg.arch == RCPP_ARCH_LLAMA && cfg.architecture == "exaone") {
                // EXAONE-3.5: llama-layout with GPT-2-style names. Sequential
                // attn→ln_2→FFN; RMSNorm (no bias); silu GLU with SEPARATE
                // gate/up tensors (c_fc_0 = gate, c_fc_1 = up), c_proj down.
                auto load2 = [&](const char* tname, size_t& idx, int rows, int cols) {
                    std::vector<float> w;
                    snprintf(buf, sizeof(buf), "transformer.h.%d.%s", l, tname);
                    if (!r.get_tensor_f32(buf, w)) return;
                    if ((int)w.size() == rows * cols) idx = push(std::move(w));
                    else fprintf(stderr, "Generic: safetensors %s: %d elems, want %d\n", buf, (int)w.size(), rows * cols);
                };
                load2("attn.attention.q_proj.weight", lw.wq, H, NH * HD);
                load2("attn.attention.k_proj.weight", lw.wk, H, NKV * HD);
                load2("attn.attention.v_proj.weight", lw.wv, H, NKV * HD);
                load2("attn.attention.out_proj.weight", lw.wo, NH * HD, H);
                load2("mlp.c_fc_0.weight", lw.w1, H, IM);
                load2("mlp.c_fc_1.weight", lw.w2, H, IM);
                load2("mlp.c_proj.weight", lw.w3, IM, H);
                load2("ln_1.weight", lw.rms_attn, H, 1);
                load2("ln_2.weight", lw.rms_ffn, H, 1);
            } else if (cfg.arch == RCPP_ARCH_GPTNEOX) {
                // GPT-NeoX/Pythia: gpt_neox.layers.N.* names, fused
                // query_key_value [3H, H] (no GQA — q/k/v each H rows),
                // biases everywhere, nn.LayerNorm weight+bias, PARALLEL
                // attn+FFN (single input_layernorm), non-gated erf-gelu FFN.
                // lw.rms_ffn/w2 stay SIZE_MAX (parallel + no gate).
                std::vector<float> qkv;
                snprintf(buf, sizeof(buf), "gpt_neox.layers.%d.attention.query_key_value.weight", l);
                if (!r.get_tensor_f32(buf, qkv) || (int)qkv.size() != 3 * NH * HD * H) {
                    fprintf(stderr, "Generic: safetensors %s: missing/misized qkv\n", buf);
                    return false;
                }
                // GPT-NeoX qkv is HEAD-INTERLEAVED [q_h, k_h, v_h] per head
                // (llama.cpp conversion reshapes (n_head, 3, hd, embed) then
                // cats — the raw safetensors is NOT [q|k|v] grouped).
                size_t hd2 = (size_t)HD;
                std::vector<float> q(NH * hd2 * H), k(NH * hd2 * H), v(NH * hd2 * H);
                for (size_t h = 0; h < (size_t)NH; h++) {
                    memcpy(q.data() + (h * hd2) * H, qkv.data() + ((h * 3 + 0) * hd2) * H, (size_t)hd2 * H * sizeof(float));
                    memcpy(k.data() + (h * hd2) * H, qkv.data() + ((h * 3 + 1) * hd2) * H, (size_t)hd2 * H * sizeof(float));
                    memcpy(v.data() + (h * hd2) * H, qkv.data() + ((h * 3 + 2) * hd2) * H, (size_t)hd2 * H * sizeof(float));
                }
                lw.wq = push(std::move(q)); lw.wk = push(std::move(k)); lw.wv = push(std::move(v));
                auto load2 = [&](const char* tname, size_t& idx, int rows, int cols) {
                    std::vector<float> w;
                    snprintf(buf, sizeof(buf), "gpt_neox.layers.%d.%s", l, tname);
                    if (!r.get_tensor_f32(buf, w)) return;
                    if ((int)w.size() == rows * cols) idx = push(std::move(w));
                    else fprintf(stderr, "Generic: safetensors %s: %d elems, want %d\n", buf, (int)w.size(), rows * cols);
                };
                load2("attention.dense.weight", lw.wo, NH * HD, H);
                load2("mlp.dense_h_to_4h.weight", lw.w1, H, IM);
                load2("mlp.dense_4h_to_h.weight", lw.w3, IM, H);
                load2("input_layernorm.weight", lw.rms_attn, H, 1);
                load2("input_layernorm.bias", lw.rms_attn_b, H, 1);
                // qkv bias [3H] head-interleaved → bq/bk/bv head groups.
                std::vector<float> qkvb;
                snprintf(buf, sizeof(buf), "gpt_neox.layers.%d.attention.query_key_value.bias", l);
                if (r.get_tensor_f32(buf, qkvb) && (int)qkvb.size() == 3 * NH * HD) {
                    std::vector<float> bq((size_t)NH * HD), bk((size_t)NH * HD), bv((size_t)NH * HD);
                    for (size_t h = 0; h < (size_t)NH; h++) {
                        memcpy(bq.data() + h * HD, qkvb.data() + (h * 3 + 0) * HD, HD * sizeof(float));
                        memcpy(bk.data() + h * HD, qkvb.data() + (h * 3 + 1) * HD, HD * sizeof(float));
                        memcpy(bv.data() + h * HD, qkvb.data() + (h * 3 + 2) * HD, HD * sizeof(float));
                    }
                    lw.bq = push(std::move(bq)); lw.bk = push(std::move(bk)); lw.bv = push(std::move(bv));
                }
                load2("attention.dense.bias", lw.bo, NH * HD, 1);
                load2("mlp.dense_h_to_4h.bias", lw.w1_b, IM, 1);
                load2("mlp.dense_4h_to_h.bias", lw.w3_b, H, 1);
            } else if (cfg.arch == RCPP_ARCH_OPT) {
                // OPT: model.decoder.layers.N.* names, sequential structure,
                // nn.LayerNorm weight+bias (self_attn_layer_norm /
                // final_layer_norm), projection biases everywhere, NON-GATED
                // RELU FFN (fc1/fc2), no rotary. lw.w2 stays SIZE_MAX.
                auto load2 = [&](const char* tname, size_t& idx, int rows, int cols) {
                    std::vector<float> w;
                    snprintf(buf, sizeof(buf), "model.decoder.layers.%d.%s", l, tname);
                    if (!r.get_tensor_f32(buf, w)) return;
                    if ((int)w.size() == rows * cols) idx = push(std::move(w));
                    else fprintf(stderr, "Generic: safetensors %s: %d elems, want %d\n", buf, (int)w.size(), rows * cols);
                };
                load2("self_attn.q_proj.weight", lw.wq, H, NH * HD);
                load2("self_attn.k_proj.weight", lw.wk, H, NKV * HD);
                load2("self_attn.v_proj.weight", lw.wv, H, NKV * HD);
                load2("self_attn.out_proj.weight", lw.wo, NH * HD, H);
                load2("self_attn.q_proj.bias", lw.bq, NH * HD, 1);
                load2("self_attn.k_proj.bias", lw.bk, NKV * HD, 1);
                load2("self_attn.v_proj.bias", lw.bv, NKV * HD, 1);
                load2("self_attn.out_proj.bias", lw.bo, H, 1);
                load2("fc1.weight", lw.w1, H, IM);
                load2("fc1.bias", lw.w1_b, IM, 1);
                load2("fc2.weight", lw.w3, IM, H);
                load2("fc2.bias", lw.w3_b, H, 1);
                load2("self_attn_layer_norm.weight", lw.rms_attn, H, 1);
                load2("self_attn_layer_norm.bias", lw.rms_attn_b, H, 1);
                load2("final_layer_norm.weight", lw.rms_ffn, H, 1);
                load2("final_layer_norm.bias", lw.rms_ffn_b, H, 1);
            } else {
            load("self_attn.q_proj.weight", lw.wq, H, NH * HD);
            load("self_attn.k_proj.weight", lw.wk, H, NKV * HD);
            load("self_attn.v_proj.weight", lw.wv, H, NKV * HD);
            load("self_attn.o_proj.weight", lw.wo, NH * HD, H);
            if (cfg.n_experts > 0) {
                // MoE layer — two HF conventions:
                //  Mixtral/Qwen3: router mlp.gate.weight + per-expert files.
                //  Granite: fused block_sparse_moe 3D tensors.
                snprintf(buf, sizeof(buf), "model.layers.%d.mlp.gate.weight", l);
                const bool mixtral_style = r.has(buf);
                if (mixtral_style) {
                    std::vector<float> router;
                    if (!r.get_tensor_f32(buf, router) || (int)router.size() != cfg.n_experts * H) {
                        fprintf(stderr, "Generic: safetensors %s: missing/misized router\n", buf);
                        return false;
                    }
                    lw.moe_gate_inp = push(std::move(router));
                    bool experts_ok = true;
                    auto load_experts = [&](const char* hf_suffix, size_t& idx2, int rows, int cols) {
                        if (!experts_ok) return;
                        std::vector<float> stacked;
                        stacked.reserve((size_t)cfg.n_experts * rows * cols);
                        for (int e = 0; e < cfg.n_experts; e++) {
                            std::vector<float> w;
                            snprintf(buf, sizeof(buf), "model.layers.%d.mlp.experts.%d.%s", l, e, hf_suffix);
                            if (!r.get_tensor_f32(buf, w) || (int)w.size() != rows * cols) {
                                fprintf(stderr, "Generic: safetensors %s: %d elems, want %d\n",
                                        buf, (int)w.size(), rows * cols);
                                experts_ok = false;
                                return;
                            }
                            stacked.insert(stacked.end(), w.begin(), w.end());
                        }
                        idx2 = push(std::move(stacked));
                    };
                    load_experts("gate_proj.weight", lw.moe_gate_exps, IM, H);
                    load_experts("up_proj.weight", lw.moe_up_exps, IM, H);
                    load_experts("down_proj.weight", lw.moe_down_exps, H, IM);
                    snprintf(buf, sizeof(buf), "model.layers.%d.mlp.shared_expert.gate_proj.weight", l);
                    if (r.has(buf))
                        fprintf(stderr, "Generic: safetensors layer %d: shared expert present but unsupported (engine MoE path) — IGNORING\n", l);
                    if (!experts_ok) {
                        fprintf(stderr, "Generic: safetensors layer %d: MoE tensor missing — ABORTING LOAD\n", l);
                        return false;
                    }
                } else {
                    // Granite-style fused MoE.
                    snprintf(buf, sizeof(buf), "model.layers.%d.block_sparse_moe.router.layer.weight", l);
                    std::vector<float> router;
                    if (!r.get_tensor_f32(buf, router) || (int)router.size() != cfg.n_experts * H) {
                        fprintf(stderr, "Generic: safetensors %s: missing/misized router\n", buf);
                        return false;
                    }
                    lw.moe_gate_inp = push(std::move(router));
                    std::vector<float> input_linear, output_linear;
                    snprintf(buf, sizeof(buf), "model.layers.%d.block_sparse_moe.input_linear.weight", l);
                    if (!r.get_tensor_f32(buf, input_linear) ||
                        (int)input_linear.size() != cfg.n_experts * 2 * IM * H) {
                        fprintf(stderr, "Generic: safetensors %s: missing/misized input_linear\n", buf);
                        return false;
                    }
                    snprintf(buf, sizeof(buf), "model.layers.%d.block_sparse_moe.output_linear.weight", l);
                    if (!r.get_tensor_f32(buf, output_linear) ||
                        (int)output_linear.size() != cfg.n_experts * H * IM) {
                        fprintf(stderr, "Generic: safetensors %s: missing/misized output_linear\n", buf);
                        return false;
                    }
                    std::vector<float> gate, up;
                    gate.reserve((size_t)cfg.n_experts * IM * H);
                    up.reserve((size_t)cfg.n_experts * IM * H);
                    size_t per = (size_t)2 * IM * H, half = (size_t)IM * H;
                    for (int e = 0; e < cfg.n_experts; e++) {
                        gate.insert(gate.end(), input_linear.begin() + e * per,
                                     input_linear.begin() + e * per + half);
                        up.insert(up.end(), input_linear.begin() + e * per + half,
                                  input_linear.begin() + (e + 1) * per);
                    }
                    lw.moe_gate_exps = push(std::move(gate));
                    lw.moe_up_exps = push(std::move(up));
                    lw.moe_down_exps = push(std::move(output_linear));
                }
            } else {
                load("mlp.gate_proj.weight", lw.w1, H, IM);
                load("mlp.up_proj.weight", lw.w2, H, IM);
                load("mlp.down_proj.weight", lw.w3, IM, H);
            }
            }
            // Norms: gemma2/3/4 need the +1 and post-norm naming.
            auto load_norm = [&](const char* hf_name, size_t& idx, int n) {
                std::vector<float> w;
                snprintf(buf, sizeof(buf), "model.layers.%d.%s", l, hf_name);
                if (!r.get_tensor_f32(buf, w)) return;
                if ((int)w.size() != n) return;
                if (gemma_post_norms)
                    for (auto& v : w) v += 1.0f;
                idx = push(std::move(w));
            };
            load_norm("input_layernorm.weight", lw.rms_attn, H);
            if (gemma_post_norms) {
                load_norm("pre_feedforward_layernorm.weight", lw.rms_ffn, H);
                load_norm("post_attention_layernorm.weight", lw.post_attn_norm, H);
                load_norm("post_feedforward_layernorm.weight", lw.post_ffn_norm, H);
            } else {
                load_norm("post_attention_layernorm.weight", lw.rms_ffn, H);
            }
            load_norm("self_attn.q_norm.weight", lw.q_norm, HD);
            load_norm("self_attn.k_norm.weight", lw.k_norm, HD);
            // Optional QKV bias (Qwen2 family).
            auto load_bias = [&](const char* hf_name, size_t& idx, int n) {
                std::vector<float> w;
                snprintf(buf, sizeof(buf), "model.layers.%d.%s", l, hf_name);
                if (!r.get_tensor_f32(buf, w)) return;
                if ((int)w.size() == n) idx = push(std::move(w));
            };
            load_bias("self_attn.q_proj.bias", lw.bq, NH * HD);
            load_bias("self_attn.k_proj.bias", lw.bk, NKV * HD);
            load_bias("self_attn.v_proj.bias", lw.bv, NKV * HD);

            if (cfg.n_experts > 0) {
                if (lw.moe_gate_inp == SIZE_MAX || lw.moe_gate_exps == SIZE_MAX ||
                    lw.moe_up_exps == SIZE_MAX || lw.moe_down_exps == SIZE_MAX) {
                    fprintf(stderr, "Generic: safetensors layer %d: MoE tensors incomplete — ABORTING LOAD\n", l);
                    return false;
                }
            } else if (lw.wq == SIZE_MAX || lw.wk == SIZE_MAX || lw.wv == SIZE_MAX ||
                       lw.wo == SIZE_MAX ||
                       (!cfg.norm_is_layernorm &&
                        (lw.rms_attn == SIZE_MAX || lw.rms_ffn == SIZE_MAX)) ||
                       // GPT-2/Falcon/GPT-NeoX/OPT: non-gated FFN — w2 (gate) legitimately absent
                       (!((cfg.arch == RCPP_ARCH_GPT2 || cfg.arch == RCPP_ARCH_FALCON ||
                           cfg.arch == RCPP_ARCH_GPTNEOX || cfg.arch == RCPP_ARCH_OPT) && lw.w2 == SIZE_MAX) &&
                        (lw.w1 == SIZE_MAX || lw.w2 == SIZE_MAX || lw.w3 == SIZE_MAX))) {
                fprintf(stderr, "Generic: safetensors layer %d: missing required tensor — ABORTING LOAD\n", l);
                return false;
            }
        }
        printf("Generic: safetensors loaded — %d layers, %.1fM params, arch=%s\n",
               cfg.n_layers, (double)embed.size() / 1e6, cfg.architecture.c_str());
        return true;
    }

    bool load_gguf(const std::string& path) {
        ModelConfig hdr_cfg;
        if (!read_gguf_header(path, hdr_cfg)) return false;

        // Architecture guard: refuse architectures with tensor layouts this
        // backend doesn't understand (issue #947). ZAYA MoE uses a non-standard
        // GGUF tensor layout (e.g. ffn_gate_inp with shape [NE*NE*H] instead
        // of [NE*H]) that would produce garbage. Zamba/Mamba2 are handled by
        // their own dedicated backends.
        if (hdr_cfg.arch == RCPP_ARCH_ZAYA || hdr_cfg.arch == RCPP_ARCH_ZAMBA2 ||
            hdr_cfg.arch == RCPP_ARCH_ZAMBA || hdr_cfg.arch == RCPP_ARCH_MAMBA ||
            hdr_cfg.arch == RCPP_ARCH_QWEN35) {
            const char* hint = "";
            if (hdr_cfg.arch == RCPP_ARCH_QWEN35)
                hint = " — Qwen3.5 Gate-Delta requires NPU (FLM/XRT) or HIP backend";
            fprintf(stderr, "  [generic] Refusing to load %s (arch=%d)%s\n",
                    path.c_str(), (int)hdr_cfg.arch, hint);
            return false;
        }
        fprintf(stderr, "load_gguf: %s, %d layers, %d hidden\n", hdr_cfg.model_name.c_str(), hdr_cfg.n_layers, hdr_cfg.hidden);
        
        int H = hdr_cfg.hidden_size, L = hdr_cfg.n_layers, NH = hdr_cfg.n_heads;
        int NKV = hdr_cfg.n_kv_heads, HD = hdr_cfg.head_dim;
        int FF = hdr_cfg.intermediate_size;
        [[maybe_unused]] int V = hdr_cfg.vocab_size;
        
        auto load = [&](const std::string& name, std::vector<float>& dst, size_t expected) -> bool {
            std::vector<float> buf;
            size_t n = 0;
            if (!read_gguf_tensor(path, name, buf, &n)) return false;
            if (n != expected) { fprintf(stderr, "  %s: expected %zu, got %zu\n", name.c_str(), expected, n); return false; }
            dst = std::move(buf);
            return true;
        };
        
        // Embedding
        int real_vocab = read_gguf_vocab(path);
        if (real_vocab > 0) cfg.vocab = cfg.vocab_size = real_vocab;
        load("token_embd.weight", embed, (size_t)real_vocab * H);
        
        // Final norm
        load("output_norm.weight", final_norm, H);

        // LM head — optional. Many GGUF exports omit it entirely when the
        // source model ties embeddings (tie_word_embeddings: true); when
        // present, it's a genuinely different matrix from token_embd.weight
        // and using the tied embedding instead silently produces wrong
        // logits (issue #319). output_weight stays empty when absent, and
        // forward() falls back to the tied embedding in that case.
        load("output.weight", output_weight, (size_t)real_vocab * H);

        
        // Per-layer weights
        layers.resize(L);
        flat_weights.clear();
        
        auto load_tensor = [&](const std::string& name, size_t expected) -> size_t {
            std::vector<float> buf;
            size_t n = 0;
            if (!read_gguf_tensor(path, name, buf, &n)) return SIZE_MAX;
            if (n != expected) {
                fprintf(stderr, "  %s: expected %zu, got %zu — ABORTING LOAD\n", name.c_str(), expected, n);
                return SIZE_MAX;
            }
            size_t idx = flat_weights.size();
            flat_weights.insert(flat_weights.end(), buf.begin(), buf.end());
            return idx;
        };
        // Like load_tensor, but returns SIZE_MAX (not 0) when the tensor
        // simply isn't present — for genuinely optional tensors (QKV bias),
        // where "absent" and "present at index 0" must stay distinguishable.
        auto load_tensor_optional = [&](const std::string& name, size_t expected) -> size_t {
            std::vector<float> buf;
            size_t n = 0;
            if (!read_gguf_tensor(path, name, buf, &n)) return SIZE_MAX;
            if (n != expected) { fprintf(stderr, "  %s: expected %zu, got %zu\n", name.c_str(), expected, n); return SIZE_MAX; }
            size_t idx = flat_weights.size();
            flat_weights.insert(flat_weights.end(), buf.begin(), buf.end());
            return idx;
        };

        // MoE detection must use the FILE's expert count, not the member cfg:
        // the member defaults to 16 (Zaya .bin convention) and callers may not
        // have synced it — a dense GGUF would hit the MoE branch and abort.
        int NE = hdr_cfg.n_experts;
        for (int i = 0; i < L; i++) {
            std::string p = "blk." + std::to_string(i) + ".";
            LayerW lw = {};  // zero-initialize all indices to SIZE_MAX
            lw.wq = SIZE_MAX; lw.wk = SIZE_MAX; lw.wv = SIZE_MAX; lw.wo = SIZE_MAX;
            lw.rms_attn = SIZE_MAX; lw.rms_ffn = SIZE_MAX;
            lw.w1 = SIZE_MAX; lw.w2 = SIZE_MAX; lw.w3 = SIZE_MAX;
            lw.moe_gate_inp = SIZE_MAX; lw.moe_gate_exps = SIZE_MAX;
            lw.moe_up_exps = SIZE_MAX; lw.moe_down_exps = SIZE_MAX;
            lw.q_norm = SIZE_MAX; lw.k_norm = SIZE_MAX;

            lw.rms_attn = load_tensor(p + "attn_norm.weight", H);
            lw.rms_ffn  = load_tensor(p + "ffn_norm.weight", H);
            lw.wq = load_tensor(p + "attn_q.weight", (size_t)NH*HD*H);
            lw.wk = load_tensor(p + "attn_k.weight", (size_t)NKV*HD*H);
            lw.wv = load_tensor(p + "attn_v.weight", (size_t)NKV*HD*H);
            lw.wo = load_tensor(p + "attn_output.weight", (size_t)H*NH*HD);
            if (lw.wq != SIZE_MAX) unrotate_rope_rows(flat_weights, lw.wq, NH, HD, H);
            if (lw.wk != SIZE_MAX) unrotate_rope_rows(flat_weights, lw.wk, NKV, HD, H);

            // Check that all required tensors loaded correctly. If any shape
            // mismatch occurred, load_tensor returns SIZE_MAX and the model
            // will produce garbage — abort early (issue #947).
            // OLMo: rms_attn/rms_ffn are legitimately absent (no-affine
            // LayerNorm — llama.cpp writes no norm tensors for OLMo).
            bool layer_ok = (lw.wq != SIZE_MAX) && (lw.wk != SIZE_MAX)
                         && (lw.wv != SIZE_MAX) && (lw.wo != SIZE_MAX)
                         && (hdr_cfg.arch == RCPP_ARCH_OLMO ||
                             (lw.rms_attn != SIZE_MAX && lw.rms_ffn != SIZE_MAX));
            if (!layer_ok) {
                fprintf(stderr, "  [generic] Layer %d: required tensor shape mismatch — ABORTING LOAD\n", i);
                return false;
            }

            if (NE > 0) {
                // MoE layer: no dense ffn_gate/up/down — route through the
                // stacked per-expert "_exps" tensors instead.
                lw.moe_gate_inp  = load_tensor(p + "ffn_gate_inp.weight", (size_t)NE*H);
                lw.moe_gate_exps = load_tensor(p + "ffn_gate_exps.weight", (size_t)NE*FF*H);
                lw.moe_up_exps   = load_tensor(p + "ffn_up_exps.weight", (size_t)NE*FF*H);
                lw.moe_down_exps = load_tensor(p + "ffn_down_exps.weight", (size_t)NE*H*FF);

                // Also check MoE tensor shapes
                if (lw.moe_gate_inp == SIZE_MAX || lw.moe_gate_exps == SIZE_MAX ||
                    lw.moe_up_exps == SIZE_MAX || lw.moe_down_exps == SIZE_MAX) {
                    fprintf(stderr, "  [generic] Layer %d: MoE tensor shape mismatch — ABORTING LOAD\n", i);
                    return false;
                }
            } else {
                lw.w1 = load_tensor(p + "ffn_gate.weight", (size_t)FF*H);
                lw.w2 = load_tensor(p + "ffn_up.weight", (size_t)FF*H);
                lw.w3 = load_tensor(p + "ffn_down.weight", (size_t)H*FF);

                if (lw.w1 == SIZE_MAX || lw.w2 == SIZE_MAX || lw.w3 == SIZE_MAX) {
                    fprintf(stderr, "  [generic] Layer %d: FFN tensor shape mismatch — ABORTING LOAD\n", i);
                    return false;
                }
            }
            lw.bq = load_tensor_optional(p + "attn_q.bias", (size_t)NH*HD);
            lw.bk = load_tensor_optional(p + "attn_k.bias", (size_t)NKV*HD);
            lw.bv = load_tensor_optional(p + "attn_v.bias", (size_t)NKV*HD);
            lw.q_norm = load_tensor_optional(p + "attn_q_norm.weight", HD);
            lw.k_norm = load_tensor_optional(p + "attn_k_norm.weight", HD);
            // Gemma-2/3 post-norms: RMSNorm applied to the attention and FFN
            // outputs before the residual adds. Without them the FFN output
            // (RMS ~40 on Gemma-3) blasts the residual stream -> garbage
            // logits (caught by the issue #1243 per-vocab ppl gate).
            lw.post_attn_norm = load_tensor_optional(p + "post_attention_norm.weight", H);
            lw.post_ffn_norm = load_tensor_optional(p + "post_ffw_norm.weight", H);
            layers[i] = lw;
        }
        {
            int with_bias = 0, with_qknorm = 0;
            for (auto& lw : layers) {
                if (lw.bq != SIZE_MAX) with_bias++;
                if (lw.q_norm != SIZE_MAX) with_qknorm++;
            }
            if (with_bias > 0) printf("Generic: %d/%d layers have biased QKV projections\n", with_bias, L);
            if (with_qknorm > 0) printf("Generic: %d/%d layers have Q/K-norm\n", with_qknorm, L);
            if (NE > 0) printf("Generic: MoE model — %d experts, %d used per token\n", NE, cfg.num_experts_top);
        }
        
        printf("Generic: loaded %zu layers, embed=%zu, final_norm=%zu, lm_head=%s\n",
               layers.size(), embed.size(), final_norm.size(),
               output_weight.empty() ? "tied" : "untied");

        // The GGUF header is authoritative for dims — the member cfg may still
        // hold caller defaults (e.g. ppl_generic passes a default-constructed
        // cfg). Without this sync, forward() runs with cfg.n_layers=40 on a
        // 28-layer model → OOB on layers[] (found via the Bonsai ppl gate).
        cfg.set_hidden(hdr_cfg.hidden_size);
        cfg.set_heads(hdr_cfg.num_heads);
        cfg.set_kv_heads(hdr_cfg.num_kv_heads ? hdr_cfg.num_kv_heads : hdr_cfg.num_heads);
        cfg.set_layers(hdr_cfg.num_layers);
        cfg.head_dim = hdr_cfg.head_dim ? hdr_cfg.head_dim : hdr_cfg.hidden_size / hdr_cfg.num_heads;
        cfg.set_ff(hdr_cfg.intermediate_size);
        cfg.set_experts(hdr_cfg.num_experts);
        cfg.num_experts_top = hdr_cfg.num_experts_top;
        cfg.rms_norm_eps = hdr_cfg.rms_norm_eps;
        cfg.rope_theta = hdr_cfg.rope_theta;
        if (hdr_cfg.max_seq_len > 0) cfg.max_seq_len = hdr_cfg.max_seq_len;
        // OLMo: LayerNorm (no affine params, eps 1e-5) + QKV clip 8.0. The
        // GGUF carries NO norm tensors (llama.cpp skips no-affine norms —
        // see the layer_ok relaxation above). Must run AFTER the hdr->cfg
        // copies above (they'd clobber eps with the header default 1e-6).
        if (hdr_cfg.arch == RCPP_ARCH_OLMO) {
            cfg.norm_is_layernorm = true;
            cfg.clip_qkv = 8.0f;
            cfg.rms_norm_eps = 1e-5f;
        }
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

    // OLMo LayerNorm: no learnable weight/bias (OlmoLayerNorm in
    // modeling_olmo.py — F.layer_norm(x, shape, None, None, eps=1e-5), biased
    // variance). Different from rmsnorm: centered, no affine params.
    static void layernorm(float* o, const float* x, int n, float eps) {
        double mean = 0; for (int i = 0; i < n; i++) mean += x[i];
        mean /= n;
        double var = 0; for (int i = 0; i < n; i++) { double d = x[i] - mean; var += d * d; }
        var /= n;
        float r = 1.0f / sqrtf((float)var + eps);
        for (int i = 0; i < n; i++) o[i] = (float)((x[i] - mean) * r);
    }

    // LayerNorm with affine weight (+optional bias) — GPT-2 ln_1/ln_2/ln_f.
    static void layernorm_affine(float* o, const float* x, const float* w,
                                 const float* b, int n, float eps) {
        layernorm(o, x, n, eps);
        for (int i = 0; i < n; i++) o[i] = o[i] * w[i] + (b ? b[i] : 0.0f);
    }

    // NeoX-style (half-split) RoPE — the convention GGUF/llama.cpp-family
    // models (Llama, Qwen, Mistral, ...) actually use: pairs element i with
    // i+rot_dim/2, not adjacent elements (i, i+1). Cross-checked against
    // ZINC's shaders (src/shaders/rope_fused.comp and siblings, all
    // independently confirm half_rot = rope_dim/2 pairing) since ZINC is
    // independently verified to produce coherent output on these same
    // models — this file's previous adjacent-pair version was the GPT-J
    // convention, wrong for this model family, and produced incoherent
    // (real-vocabulary but semantically scrambled) output as a result.
    static void rope(float* q, float* k, int pos, int n_heads, int n_kv, int hd, int rot_dim, float theta, const float* freqs) {
        int half = rot_dim / 2;
        for (int h = 0; h < n_heads; h++) {
            for (int i = 0; i < half; i++) {
                float t = pos * freqs[i];
                float cosv = cosf(t), sinv = sinf(t);
                int i0 = h * hd + i, i1 = h * hd + i + half;
                float q0 = q[i0], q1 = q[i1];
                q[i0] = q0 * cosv - q1 * sinv;
                q[i1] = q0 * sinv + q1 * cosv;
            }
        }
        for (int h = 0; h < n_kv; h++) {
            for (int i = 0; i < half; i++) {
                float t = pos * freqs[i];
                float cosv = cosf(t), sinv = sinf(t);
                int i0 = h * hd + i, i1 = h * hd + i + half;
                float k0 = k[i0], k1 = k[i1];
                k[i0] = k0 * cosv - k1 * sinv;
                k[i1] = k0 * sinv + k1 * cosv;
            }
        }
    }

    // AVX-512 GEMV row kernel (always compiled, runtime-dispatched — #1346).
    // Accumulates in F64: gemma-family models reach activations O(1e4) with
    // final-layer norm gammas ~500, so f32 accumulation noise (1e-5 relative)
    // is amplified to visible logit errors. F64 accumulation is exact to
    // f32 inputs and removes that noise source.
    __attribute__((target("avx512f,avx512dq")))
    static float gemv_row_avx512(const float* in, const float* wr, int K) {
        double s = 0;
        int j = 0;
        __m512d acc = _mm512_setzero_pd();
        for (; j + 15 < K; j += 16) {
            __m256 a0 = _mm256_loadu_ps(in + j);
            __m256 a1 = _mm256_loadu_ps(in + j + 8);
            __m256 b0 = _mm256_loadu_ps(wr + j);
            __m256 b1 = _mm256_loadu_ps(wr + j + 8);
            acc = _mm512_fmadd_pd(_mm512_cvtps_pd(a0), _mm512_cvtps_pd(b0), acc);
            acc = _mm512_fmadd_pd(_mm512_cvtps_pd(a1), _mm512_cvtps_pd(b1), acc);
        }
        s = _mm512_reduce_add_pd(acc);
        for (; j < K; j++) s += (double)in[j] * wr[j];
        return (float)s;
    }

    static void matmul(float* out, const float* in, const float* w, int M, int K) {
        // Vectorized GEMV with OpenMP. Each thread processes rows, inner dot
        // product uses 16-wide FMA on AVX-512 hosts (runtime-gated, #1346),
        // 8-wide FMA on AVX2 builds, scalar tail otherwise.
        const bool avx512 = cpu_has_avx512();
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < M; i++) {
            const float* wr = w + (size_t)i * K;
            if (avx512) { out[i] = gemv_row_avx512(in, wr, K); continue; }
            double s = 0;
            int j = 0;
#if defined(__AVX2__)
            // FMA-8 accumulator for the main body; converted to double for the
            // tail so long dot products don't accumulate f32 rounding.
            __m256 acc = _mm256_setzero_ps();
            for (; j + 7 < K; j += 8) {
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(in + j), _mm256_loadu_ps(wr + j), acc);
            }
            __m128 lo = _mm256_castps256_ps128(acc);
            __m128 hi = _mm256_extractf128_ps(acc, 1);
            lo = _mm_add_ps(lo, hi);
            lo = _mm_hadd_ps(lo, lo);
            lo = _mm_hadd_ps(lo, lo);
            s = _mm_cvtss_f32(lo);
#endif
            for (; j < K; j++) s += (double)in[j] * wr[j];  // scalar tail in double
            out[i] = (float)s;
        }
    }

    // Packed TQ2 GEMV (WS-04): multiplication-free, pext masks + maskz add/sub,
    // per-(row,32-group) BF16 scales. TQ2 code mapping: 0=-s, 1=0, 2=+s, 3=0.
    // The AVX-512/BMI2 body lives in a target-attributed free function — the
    // caller only reaches it after cpu_has_packed_isa() (issue #1240). (A
    // target attribute on the member itself doesn't survive OpenMP outlining.)
    __attribute__((target("avx512f,bmi2")))
    static float gemv_packed_row(const uint8_t* base, const float* x, int row,
                                 int ntc, int tc, int gs, int tr) {
        float acc_row = 0;
        int trr = row / tr, rr = row % tr;
        int groups = tc / gs;
        size_t tb = (size_t)tr * groups * 2 + (size_t)tr * tc / 4;
        for (int tcc = 0; tcc < ntc; tcc++) {
            const uint8_t* tile = base + ((size_t)trr * ntc + tcc) * tb;
            const uint16_t* sc = (const uint16_t*)tile;
            const uint8_t* qd = tile + (size_t)tr * groups * 2;
            int c0 = tcc * tc;
            for (int g = 0; g < groups; g++) {
                float s = bf16_to_f32(sc[rr * groups + g]);
                const uint8_t* q = qd + (size_t)(rr * tc + g * gs) / 4;
                __m512 acc = _mm512_setzero_ps();
                for (int hh = 0; hh < gs / 16; hh++) {
                    uint32_t v;
                    memcpy(&v, q + 4 * hh, 4);
                    uint32_t lo = _pext_u32(v, 0x55555555u);
                    uint32_t hi = _pext_u32(v, 0xAAAAAAAAu);
                    uint32_t pos = hi & ~lo;
                    uint32_t neg = ~(hi | lo);
                    __m512 acts = _mm512_loadu_ps(x + c0 + g * gs + 16 * hh);
                    acc = _mm512_add_ps(acc, _mm512_maskz_mov_ps((__mmask16)pos, acts));
                    acc = _mm512_sub_ps(acc, _mm512_maskz_mov_ps((__mmask16)neg, acts));
                }
                acc_row += _mm512_reduce_add_ps(acc) * s;
            }
        }
        return acc_row;
    }

    void gemv_packed(const PackedW& pw, const float* x, float* y) {
        int N = pw.N, K = pw.K;
        int ntc = (K + tc_ - 1) / tc_;
        int groups = tc_ / gs_;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; i++)
            y[i] = gemv_packed_row(pw.base, x, i, ntc, tc_, gs_, tr_);
    }

    // Weight-GEMV dispatcher: packed TQ2 path when a packed slot exists, else fp32.
    void mm(float* out, const float* in, const float* W, int N, int K, int pk) {
        if (packed_ && pk >= 0) gemv_packed(packed_w_[pk], in, out);
        else matmul(out, in, W, N, K);
    }

    // Exact SiLU: x * sigmoid(x) * up. The vectorized Padé [7/6] sigmoid
    // kernels (#1350) were removed: they are numerically wrong for |x| > ~4
    // (measured 9x error at x=-7.8; saturates at 0.99086 for x>=10 instead of
    // 1.0). That corrupts SwiGLU gates on models whose gates run large
    // (Qwen3-8B: real gates reach 7-12 — llama.cpp PPL 17.5 on identical
    // weights) and collapses the hidden state. Caught by the issue #1243
    // re-conversion gate: our ppl 58k vs llama.cpp's 17.5 on the same model.
    // Note: callers must NOT compile this with -ffast-math — it makes the
    // expf loop produce NaN for large |g| (reassociation/vectorization).
    static void silu(float* out, const float* gate, const float* up, int n) {
        for (int i = 0; i < n; i++) {
            float g = gate[i];
            out[i] = (g / (1.0f + expf(-g))) * up[i];
        }
    }

    // GELU activation (tanh approximation): 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
    static void gelu(float* out, const float* x, int n) {
        const float c = 0.7978845608f; // sqrt(2/pi)
        for (int i = 0; i < n; i++) {
            float v = x[i];
            float x3 = v * v * v;
            float inner = c * (v + 0.044715f * x3);
            out[i] = 0.5f * v * (1.0f + tanhf(inner));
        }
    }

    // GeGLU: gelu(gate) * up — used by Gemma
    static void geglu(float* out, const float* gate, const float* up, int n) {
        gelu(out, gate, n);
        for (int i = 0; i < n; i++) out[i] *= up[i];
    }

    // Standard erf-based GELU (nn.GELU default) — Falcon's MLP activation.
    // Distinct from gelu() (gelu_new / tanh approx) used elsewhere.
    static void gelu_erf(float* o, const float* x, int n) {
        for (int i = 0; i < n; i++) o[i] = 0.5f * x[i] * (1.0f + erff(x[i] * 0.70710678118f));
    }

    // Squared ReLU GLU: relu(gate)^2 * up — used by Phi
    static void squared_relu_glu(float* out, const float* gate, const float* up, int n) {
        for (int i = 0; i < n; i++) {
            float g = gate[i];
            float r = g > 0.0f ? g : 0.0f;
            out[i] = r * r * up[i];
        }
    }

    // Architecture-specific FFN activation dispatch
    static void ffn_activate(float* out, const float* gate, const float* up, int n,
                             rcpp_arch_t arch, const char* arch_str) {
        switch (arch) {
            case RCPP_ARCH_GEMMA:
                // GeGLU: gelu(gate) * up — true gemma families.
                // Granite shares the GEMMA enum but is SwiGLU.
                if (arch_str && (strcmp(arch_str, "granite") == 0 || strcmp(arch_str, "granitemoe") == 0))
                    silu(out, gate, up, n);
                else
                    geglu(out, gate, up, n);
                break;
            case RCPP_ARCH_PHI:
                // Phi-3/4 (and phi-2's GELU is handled at load-time only for
                // the fused gate_up split — the activation is SwiGLU for
                // phi-3-mini/4; squared-relu GLU was speculative and wrong).
                silu(out, gate, up, n);
                break;
            default:
                // SwiGLU: silu(gate) * up — Llama, Mistral, Qwen2, Qwen3, BitNet, fallback
                silu(out, gate, up, n);
                break;
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

    const float* last_logits() override {
        return initialized ? logits_buf.data() : nullptr;
    }

    bool forward(int token_id, float* hidden_out) override {
        int tok = forward(token_id);
        if (hidden_out) *hidden_out = 0.0f;
        return tok >= 0;
    }

    bool lm_head(const float* hidden, float* logits, int* argmax) override {
        // LM head: logits = lm_weight @ hidden
        // Uses untied output.weight when the model has one, else tied embedding
        // (issue #958 — was always returning false, breaking cascade/adaptive
        //  strategy routing which needs per-token logprobs).
        const float* lm_w = output_weight.empty() ? embed.data() : output_weight.data();
        if (!lm_w) return false;
        mm(logits, hidden, lm_w, cfg.vocab, cfg.hidden, pk_lm_);
        if (argmax) {
            *argmax = 0;
            float max_val = logits[0];
            for (int i = 1; i < cfg.vocab; ++i) {
                if (logits[i] > max_val) {
                    max_val = logits[i];
                    *argmax = i;
                }
            }
        }
        return true;
    }

    int forward(int token) {
        if (token < 0 || token >= (int)cfg.vocab) {
            fprintf(stderr, "[generic] token_id=%d out of range [0,%d)\n", token, (int)cfg.vocab);
            return -1;
        }
        std::vector<float> x0(cfg.hidden);
        for (int i = 0; i < cfg.hidden; i++) x0[i] = embed[token * (size_t)cfg.hidden + i];
        // GPT-2/OPT: learned position embeddings — add the wpe/embed_positions
        // row for the CURRENT position (pos + pos_offset; OPT pads 2 slots).
        if (cfg.use_learned_pos)
            for (int i = 0; i < cfg.hidden; i++) x0[i] += pos_embed[((size_t)pos + cfg.pos_offset) * cfg.hidden + i];
        // Granite: embeddings pre-scaled by embedding_multiplier (12.0).
        if (cfg.embedding_multiplier != 1.0f)
            for (int i = 0; i < cfg.hidden; i++) x0[i] *= cfg.embedding_multiplier;
        // Gemma-family: scale the token embedding by sqrt(hidden) — the model
        // was trained with this scaling (gemma_pytorch; llama.cpp gemma2/3
        // graphs). Only for real token rows; forward_embed (vision splice)
        // callers pass already-scaled embeddings and are untouched.
        if (gemma_emb_scale_) {
            float s = sqrtf((float)cfg.hidden);
            for (int i = 0; i < cfg.hidden; i++) x0[i] *= s;
        }
        return forward_embed(x0.data());
    }

    // Same transformer body as forward(int), but takes a precomputed
    // embedding vector directly instead of doing a token_embd lookup —
    // the splice point for injecting vision embeddings (mm.2 output) at
    // image-placeholder positions instead of a text token's row.
     int forward_embed(const float* x_in) override {
        // Bounds-check KV cache position before writing (fixes OOB/overflow)
        if (pos >= cfg.max_seq_len) {
            fprintf(stderr, "[generic] KV cache overflow: pos=%d >= max_seq_len=%d\n", pos, cfg.max_seq_len);
            return -1;
        }
        int H = cfg.hidden, NH = cfg.n_heads, NKV = cfg.n_kv_heads, HD = cfg.head_dim;
        int GQA = NH / NKV, FF = cfg.intermediate_size, V = cfg.vocab;
        float eps = cfg.rms_norm_eps, theta = cfg.rope_theta;
        int rot_dim = cfg.rope_dim > 0 ? cfg.rope_dim : cfg.head_dim;  // full RoPE by default

        // scores indexed by past position (t <= pos) — must hold max_seq_len,
        // not HD (issue #1263: heap OOB for prompts > head_dim tokens).
        // All buffers are pre-allocated class members — no heap allocs per token.
        float* x = scratch_x.data();
        float* x2 = scratch_x2.data();
        float* x3 = scratch_x3.data();  // falcon parallel: saved normed input
        float* q = scratch_q.data();
        float* k = scratch_k.data();
        float* v = scratch_v.data();
        float* scores = scratch_scores.data();
        float* att = scratch_att.data();
        float* gate_up = scratch_gate_up.data();
        float* silu_buf = scratch_silu_buf.data();
        int debug_ops = cached_debug_ops_;  // cached getenv — checked once at init
        (void)scores; // used below
        (void)gate_up; (void)silu_buf; // used in FFN path

        for (int i = 0; i < H; i++) x[i] = x_in[i];
        int _n_layers = cached_num_layers_;
        for (int il = 0; il < _n_layers; il++) {
            auto& l = layers[il];
            int kv_begin = pos * NKV * HD;

            // RMSNorm → QKV
            if (cfg.norm_is_layernorm) {
                if (l.rms_attn != SIZE_MAX)
                    layernorm_affine(x2, x, w(l.rms_attn),
                                     l.rms_attn_b != SIZE_MAX ? w(l.rms_attn_b) : nullptr, H, eps);
                else layernorm(x2, x, H, eps);
            }
            else rmsnorm(x2, x, w(l.rms_attn), H, eps);
            if (il == 0 && debug_ops) {
                double ss = 0; for (int i = 0; i < H; i++) ss += (double)x[i] * x[i];
                fprintf(stderr, "[cpu] pos=%d in=[%g %g %g] normed=[%g %g %g] mean2=%g eps=%g r=%g\n",
                        pos, x[0], x[1], x[2], x2[0], x2[1], x2[2],
                        ss / H, eps, 1.0f / sqrtf((float)(ss / H) + eps));
            }
            mm(q, x2, w(l.wq), NH*HD, H, l.pk_q);
            mm(k, x2, w(l.wk), NKV*HD, H, l.pk_k);
            mm(v, x2, w(l.wv), NKV*HD, H, l.pk_v);

            // Optional QKV bias (Qwen2 and others use biased attention
            // projections; absent for architectures like Llama).
            if (l.bq != SIZE_MAX) { float* b = w(l.bq); for (int i = 0; i < NH*HD; i++) q[i] += b[i]; }
            if (l.bk != SIZE_MAX) { float* b = w(l.bk); for (int i = 0; i < NKV*HD; i++) k[i] += b[i]; }
            if (l.bv != SIZE_MAX) { float* b = w(l.bv); for (int i = 0; i < NKV*HD; i++) v[i] += b[i]; }

            // Falcon parallel: save the normed input for the FFN (attn and
            // FFN both consume input_layernorm output — no second norm).
            if (cfg.parallel_attn_ffn)
                memcpy(x3, x2, (size_t)H * sizeof(float));


            // OLMo: clip QKV values to [-clip_qkv, clip_qkv] before RoPE
            // (config clip_qkv: 8.0, applied in OlmoAttention.forward).
            if (cfg.clip_qkv > 0.0f) {
                for (int i = 0; i < NH*HD; i++)
                    q[i] = q[i] > cfg.clip_qkv ? cfg.clip_qkv : (q[i] < -cfg.clip_qkv ? -cfg.clip_qkv : q[i]);
                for (int i = 0; i < NKV*HD; i++) {
                    k[i] = k[i] > cfg.clip_qkv ? cfg.clip_qkv : (k[i] < -cfg.clip_qkv ? -cfg.clip_qkv : k[i]);
                    v[i] = v[i] > cfg.clip_qkv ? cfg.clip_qkv : (v[i] < -cfg.clip_qkv ? -cfg.clip_qkv : v[i]);
                }
            }

            // Optional per-head QK-norm (Qwen3 and others): RMSNorm applied
            // independently to each head's head_dim-sized slice with a
            // shared [head_dim] weight. Must happen before RoPE.
            if (l.q_norm != SIZE_MAX) {
                float* qn = w(l.q_norm);
                for (int h = 0; h < NH; h++) rmsnorm(&q[h*HD], &q[h*HD], qn, HD, eps);
            }
            if (l.k_norm != SIZE_MAX) {
                float* kn = w(l.k_norm);
                for (int h = 0; h < NKV; h++) rmsnorm(&k[h*HD], &k[h*HD], kn, HD, eps);
            }

            bool _dbg_ops = (il == 0 && debug_ops);
            if (_dbg_ops) fprintf(stderr,
                "[cpu] L0 q_pre=[%g %g %g] k_pre=[%g %g %g] v=[%g %g %g]\n",
                q[0], q[1], q[2], k[0], k[1], k[2], v[0], v[1], v[2]);

            // RoPE — gemma3 hybrid: layer il is FULL when il%pattern==pattern-1
            // (uses rope_theta), otherwise LOCAL (rope_local_base_freq).
            const float* freqs = rope_freqs.data();
            if (!rope_freqs_local.empty() && il % cfg.sliding_window_pattern != cfg.sliding_window_pattern - 1)
                freqs = rope_freqs_local.data();
            if (!cfg.no_rope) rope(q, k, pos, NH, NKV, HD, rot_dim, theta, freqs);
            if (_dbg_ops) fprintf(stderr, "[cpu] L0 q_post=[%g %g %g] k_post=[%g %g %g]\n",
                q[0], q[1], q[2], k[0], k[1], k[2]);

            // KV cache
            memcpy(&k_cache[il][kv_begin], k, (size_t)NKV * HD * sizeof(float));
            memcpy(&v_cache[il][kv_begin], v, (size_t)NKV * HD * sizeof(float));

            // Attention: GQA
            memset(att, 0, (size_t)NH * HD * sizeof(float));
            float attn_scale = cfg.attention_multiplier > 0.0f
                ? cfg.attention_multiplier   // granite: explicit per-model scale
                : inv_sqrt_hd_;              // default: 1/sqrt(HD)
            int kvs = NKV * HD;  // stride per position in KV cache
            for (int h = 0; h < NH; h++) {
                int kv_h = h / GQA;
                float* Q = &q[h * HD];
                int kv_base = kv_h * HD;  // precompute base offset for this KV head
                // Score over all past positions
                for (int t = 0; t <= pos; t++) {
                    float* K = &k_cache[il][t * kvs + kv_base];
                    float s = 0;
                    for (int d = 0; d < HD; d++) s += Q[d] * K[d];
                    scores[t] = s * attn_scale;  // multiply instead of divide
                }
                // Gemma-2/3 attention-logit soft-cap (config attn_logit_softcapping,
                // gemma2=50.0; gemma3-1b has NONE). qk-norm + query_pre_attn_scalar
                // produce large scores that must be tanh-capped before softmax.
                if (attn_cap_ > 0.0f)
                    for (int t = 0; t <= pos; t++) scores[t] = attn_cap_ * tanhf(scores[t] / attn_cap_);
                softmax(scores, pos + 1);
                // Weighted sum of V
                for (int d = 0; d < HD; d++) {
                    float sum = 0;
                    for (int t = 0; t <= pos; t++) {
                        float* V = &v_cache[il][t * kvs + kv_base];
                        sum += scores[t] * V[d];
                    }
                    att[h * HD + d] = sum;
                }
            }
            if (_dbg_ops) fprintf(stderr, "[cpu] L0 att=[%g %g %g] att128=[%g %g %g]\n",
                att[0], att[1], att[2], att[128], att[129], att[130]);

            // O proj
            mm(x2, att, w(l.wo), H, NH*HD, l.pk_o);
            if (l.bo != SIZE_MAX) { float* b = w(l.bo); for (int i = 0; i < H; i++) x2[i] += b[i]; }
            if (l.post_attn_norm != SIZE_MAX) rmsnorm(x2, x2, w(l.post_attn_norm), H, eps);
            // Residual (granite: block output scaled by residual_multiplier=0.22)
            if (cfg.residual_multiplier != 1.0f)
                for (int i = 0; i < H; i++) x[i] += x2[i] * cfg.residual_multiplier;
            else
                for (int i = 0; i < H; i++) x[i] += x2[i];
            if (debug_ops && il < 3) {
                double s = 0; for (int i = 0; i < H; i++) s += fabs(x[i]);
                fprintf(stderr, "[cpu] L%d after attn: mean|.|=%g\n", il, s / H);
            }

            // FFN: RMSNorm → gate/up → activation (arch-specific) → down → residual (dense), or
            // RMSNorm → router top-k → per-expert gate/up/activation/down,
            // weighted sum → residual (MoE).
            if (cfg.parallel_attn_ffn) {
                // Falcon: FFN input = the SAME normed output as attention.
                memcpy(x2, x3, (size_t)H * sizeof(float));
            } else if (cfg.norm_is_layernorm) {
                if (l.rms_ffn != SIZE_MAX)
                    layernorm_affine(x2, x, w(l.rms_ffn),
                                     l.rms_ffn_b != SIZE_MAX ? w(l.rms_ffn_b) : nullptr, H, eps);
                else layernorm(x2, x, H, eps);
            }
            else rmsnorm(x2, x, w(l.rms_ffn), H, eps);
            if (l.moe_gate_inp != SIZE_MAX) {
                int NE = cfg.n_experts, NEU = cfg.num_experts_top;
                float* router_probs = scratch_moe_router_probs.data();
                mm(router_probs, x2, w(l.moe_gate_inp), NE, H, -1);

                // Top-k expert selection (same indices either way — softmax
                // is monotonic, so top-k on raw logits == top-k on probs).
                int* idx = scratch_moe_idx.data();
                for (int e = 0; e < NE; e++) idx[e] = e;
                std::partial_sort(idx, idx + NEU, idx + NE,
                                   [&](int a, int b) { return router_probs[a] > router_probs[b]; });

                // Gating convention: Mixtral softmax-all-renorm vs
                // granite/qwen3 top-k-logits-then-softmax (normalizations
                // differ).
                const bool topk_softmax_gating =
                    (cfg.architecture == "granite" || cfg.architecture == "granitemoe" ||
                     cfg.arch == RCPP_ARCH_QWEN3 || cfg.arch == RCPP_ARCH_QWEN35);
                float wsum = 0.0f;
                if (topk_softmax_gating) {
                    float mx = router_probs[idx[0]];
                    for (int t = 1; t < NEU; t++)
                        if (router_probs[idx[t]] > mx) mx = router_probs[idx[t]];
                    for (int t = 0; t < NEU; t++) {
                        router_probs[idx[t]] = expf(router_probs[idx[t]] - mx);
                        wsum += router_probs[idx[t]];
                    }
                } else {
                    softmax(router_probs, NE);
                    for (int t = 0; t < NEU; t++) wsum += router_probs[idx[t]];
                }
                if (wsum < 6.103515625e-5f) wsum = 6.103515625e-5f;

                float* ffn_acc = scratch_moe_ffn_acc.data();
                memset(ffn_acc, 0, H * sizeof(float));
                float* gate_buf = scratch_moe_gate.data();
                float* up_buf = scratch_moe_up.data();
                float* moe_silu = scratch_moe_silu.data();  // #1342: FF-sized, distinct from down output
                float* down_buf = scratch_moe_down.data();  // H-sized
                for (int t = 0; t < NEU; t++) {
                    int e = idx[t];
                    float we = router_probs[e] / wsum;
                    float* wg = w(l.moe_gate_exps) + (size_t)e * FF * H;
                    float* wu = w(l.moe_up_exps)   + (size_t)e * FF * H;
                    float* wd = w(l.moe_down_exps) + (size_t)e * H * FF;
                    mm(gate_buf, x2, wg, FF, H, -1);
                    mm(up_buf, x2, wu, FF, H, -1);
                    ffn_activate(moe_silu, gate_buf, up_buf, FF, cfg.arch, cfg.architecture.c_str());
                    mm(down_buf, moe_silu, wd, H, FF, -1);
                    for (int i = 0; i < H; i++) ffn_acc[i] += we * down_buf[i];
                }
                if (cfg.residual_multiplier != 1.0f)
                    for (int i = 0; i < H; i++) x[i] += ffn_acc[i] * cfg.residual_multiplier;
                else
                    for (int i = 0; i < H; i++) x[i] += ffn_acc[i];
            } else if (l.w2 == SIZE_MAX) {
                // Non-gated FFN (GPT-2: gelu_new; Falcon: erf-gelu):
                // act(w1 x + b1) then w3 + b3.
                mm(gate_up, x2, w(l.w1), FF, H, l.pk_w1);
                if (l.w1_b != SIZE_MAX) { float* b = w(l.w1_b); for (int i = 0; i < FF; i++) gate_up[i] += b[i]; }
                if (cfg.arch == RCPP_ARCH_FALCON || cfg.arch == RCPP_ARCH_GPTNEOX) gelu_erf(gate_up, gate_up, FF);
                else if (cfg.arch == RCPP_ARCH_OPT) { for (int i = 0; i < FF; i++) gate_up[i] = gate_up[i] > 0 ? gate_up[i] : 0.0f; }  // ReLU
                else gelu(gate_up, gate_up, FF);
                mm(x2, gate_up, w(l.w3), H, FF, l.pk_w3);
                if (l.w3_b != SIZE_MAX) { float* b = w(l.w3_b); for (int i = 0; i < H; i++) x2[i] += b[i]; }
                if (l.post_ffn_norm != SIZE_MAX) rmsnorm(x2, x2, w(l.post_ffn_norm), H, eps);
                // Residual — scale_depth (MiniCPM) / residual_multiplier (granite)
                // applies to BOTH residual adds; the gated dense path was missing
                // it (only attn + MoE paths scaled). Fixed 2026-08-14 (minicpm).
                if (cfg.residual_multiplier != 1.0f)
                    for (int i = 0; i < H; i++) x[i] += x2[i] * cfg.residual_multiplier;
                else
                    for (int i = 0; i < H; i++) x[i] += x2[i];
            } else {
                mm(gate_up, x2, w(l.w1), FF, H, l.pk_w1);
                mm(&gate_up[FF], x2, w(l.w2), FF, H, l.pk_w2);
                ffn_activate(silu_buf, gate_up, &gate_up[FF], FF, cfg.arch, cfg.architecture.c_str());
                mm(x2, silu_buf, w(l.w3), H, FF, l.pk_w3);
                if (l.post_ffn_norm != SIZE_MAX) rmsnorm(x2, x2, w(l.post_ffn_norm), H, eps);
                if (debug_ops && il < 3) {
                    double sg = 0, sd = 0;
                    for (int i = 0; i < FF; i++) sg += fabs(gate_up[i]);
                    for (int i = 0; i < H; i++) sd += fabs(x2[i]);
                    fprintf(stderr, "[cpu] L%d ffn gate mean|.|=%g down mean|.|=%g\n", il, sg / FF, sd / H);
                }
                // Residual — same scale_depth/residual_multiplier fix as the
                // no-gate path (gated dense FFN, 2026-08-14).
                if (cfg.residual_multiplier != 1.0f)
                    for (int i = 0; i < H; i++) x[i] += x2[i] * cfg.residual_multiplier;
                else
                    for (int i = 0; i < H; i++) x[i] += x2[i];
            }
        }

        if (debug_ops) {
            double s = 0, mx = 0;
            for (int i = 0; i < H; i++) { s += fabs(x[i]); if (fabs(x[i]) > mx) mx = fabs(x[i]); }
            fprintf(stderr, "[cpu] final hidden mean|.|=%g max|.|=%g\n", s / H, mx);
        }
        // Final norm — RMSNorm (weighted), OLMo LayerNorm (no affine), or
        // GPT-2 LayerNorm with weight+bias.
        if (cfg.norm_is_layernorm) {
            if (!final_norm.empty())
                layernorm_affine(x2, x, final_norm.data(),
                                 final_norm_bias.empty() ? nullptr : final_norm_bias.data(), H, eps);
            else layernorm(x2, x, H, eps);
        }
        else rmsnorm(x2, x, final_norm.data(), H, eps);

        // LM head — untied output.weight when the model has one, else tied embedding.
        const float* lm_head = output_weight.empty() ? embed.data() : output_weight.data();
        mm(logits_buf.data(), x2, lm_head, V, H, pk_lm_);
        // Granite (6th quirk): LM-head logits divided by logits_scaling (6.0)
        // BEFORE any soft-cap. Argmax-invariant — invisible to every argmax
        // test; matters for sampling and perplexity.
        if (cfg.logits_scaling != 1.0f)
            for (int i = 0; i < V; i++) logits_buf[i] /= cfg.logits_scaling;

        // Gemma-2/3 final-logit soft-cap (config final_logit_softcapping,
        // gemma2=30.0; gemma3-1b has NONE — the arch-string default wrongly
        // compressed gemma3 logits). Uncapped LM-head logits run ±100+ for
        // gemma2 (ppl gate #1243), but gemma3-1b logits are small and must
        // pass through untouched.
        if (final_cap_ > 0.0f)
            for (int i = 0; i < V; i++) logits_buf[i] = final_cap_ * tanhf(logits_buf[i] / final_cap_);

        pos++;

        // Argmax
        int best = 0; float bestv = logits_buf[0];
        for (int i = 1; i < V; i++) {
            if (logits_buf[i] > bestv) { bestv = logits_buf[i]; best = i; }
        }
        if (getenv("CPU_DEBUG")) fprintf(stderr, "[cpu] argmax idx=%d maxlogit=%g\n", best, bestv);  // one-time check, not in hot inner loop
        return best;
    }

    // Perplexity over per-sample token-id sequences (WS-05/WS-00 harness).
    // logits_buf holds the previous token's logits after each forward().
    double compute_ppl(const std::vector<std::vector<int>>& samples) {
        if (!initialized) return 0.0;
        double total_nll = 0.0;
        long long n = 0;
        int V = cfg.vocab;
        for (auto& ids : samples) {
            reset();
            for (size_t i = 0; i + 1 < ids.size(); i++) {
                if (forward(ids[i]) < 0) continue;
                // logits_buf = logits for ids[i]; NLL of ids[i+1]
                int nxt = ids[i + 1];
                if (nxt >= 0 && nxt < V) {
                    float mx = logits_buf[0];
                    for (int j = 1; j < V; j++) if (logits_buf[j] > mx) mx = logits_buf[j];
                    double s = 0;
                    for (int j = 0; j < V; j++) s += expf(logits_buf[j] - mx);
                    double lse = mx + log(s);
                    total_nll += lse - logits_buf[nxt];
                    n++;
                    if (n <= 5 && getenv("CPU_DEBUG_PPL")) {
                        int am = 0;
                        for (int j = 1; j < V; j++) if (logits_buf[j] > logits_buf[am]) am = j;
                        fprintf(stderr, "[ppl] tok=%d argmax=%d next=%d logit_next=%g lse=%g nll=%g\n",
                                ids[i], am, nxt, logits_buf[nxt], lse, lse - logits_buf[nxt]);
                    }
                }
            }
        }
        return n ? exp(total_nll / (double)n) : 0.0;
    }

    // WS-05 P1 (issue #1245): apply TQ2 residual correction planes to the fp32
    // weight pool. File format "PNL1" (little-endian): magic[4] = "PNL1",
    // u32 n_entries, then per entry: u32 layer, u32 kind (0=q 1=k 2=v 3=o
    // 4=w1 5=w2 6=w3), u32 rows, u32 cols, u32 k, i8 B[rows*k], i8 C[k*cols],
    // f32 d[k]. Each entry adds W += B·diag(d)·C (row-major [rows,cols]).
    // Forcing packed_ off keeps the decode on the corrected fp32 pool (the
    // packed path reads raw tiles and would ignore the correction).
    bool apply_plane_corrections(const char* path) {
        FILE* f = fopen(path, "rb");
        if (!f) { fprintf(stderr, "planes: cannot open %s\n", path); return false; }
        char magic[4];
        if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "PNL1", 4) != 0) {
            fprintf(stderr, "planes: bad magic in %s\n", path); fclose(f); return false;
        }
        uint32_t n_entries = 0;
        if (fread(&n_entries, 4, 1, f) != 1) { fclose(f); return false; }
        int applied = 0;
        for (uint32_t e = 0; e < n_entries; e++) {
            uint32_t lyr, kind, rows, cols, k;
            if (fread(&lyr, 4, 1, f) != 1 || fread(&kind, 4, 1, f) != 1 ||
                fread(&rows, 4, 1, f) != 1 || fread(&cols, 4, 1, f) != 1 ||
                fread(&k, 4, 1, f) != 1) { fclose(f); return false; }
            if (lyr >= (uint32_t)layers.size()) { fclose(f); return false; }
            size_t idx = SIZE_MAX;
            const char* names[7] = {"q", "k", "v", "o", "w1", "w2", "w3"};
            size_t* slots[7] = {&layers[lyr].wq, &layers[lyr].wk, &layers[lyr].wv,
                                &layers[lyr].wo, &layers[lyr].w1, &layers[lyr].w2,
                                &layers[lyr].w3};
            if (kind < 7) idx = *slots[kind];
            size_t nbytes = (size_t)rows * k + (size_t)k * cols;
            std::vector<int8_t> bc(nbytes);
            std::vector<float> d(k);
            if (fread(bc.data(), 1, nbytes, f) != nbytes ||
                fread(d.data(), 4, k, f) != k) { fclose(f); return false; }
            if (idx == SIZE_MAX || idx + (size_t)rows * cols > flat_weights.size()) {
                fprintf(stderr, "planes: layer %u kind=%s mismatch (rows=%u cols=%u, idx=%zu total=%zu) — skipping\n",
                        lyr, kind < 7 ? names[kind] : "?", rows, cols,
                        idx, flat_weights.size());
                continue;
            }
            float* w = flat_weights.data() + idx;
            const int8_t* B = bc.data();
            const int8_t* C = bc.data() + (size_t)rows * k;
            for (uint32_t i = 0; i < rows; i++)
                for (uint32_t j = 0; j < k; j++) {
                    float bij = (float)B[(size_t)i * k + j];
                    if (bij == 0.0f) continue;
                    const int8_t* cj = C + (size_t)j * cols;
                    float dj = d[j];
                    float* wr = w + (size_t)i * cols;
                    for (uint32_t c = 0; c < cols; c++) wr[c] += bij * dj * (float)cj[c];
                }
            applied++;
        }
        fclose(f);
        packed_ = false;   // decode must use the corrected fp32 pool
        printf("planes: applied %d/%u corrections — decode switched to fp32 pool\n", applied, n_entries);
        return applied > 0;
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
