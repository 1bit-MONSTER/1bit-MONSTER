// deepseek_v4.cpp — DeepSeek V4 Flash / Pro — CPU forward (rewrite 2026-08-16)
//
// Real-architecture implementation. See deepseek_v4.h for the arch summary.
// Math follows HF modeling_deepseek_v4.py 5.14 (the gate oracle) exactly:
//   - mHC: fn/base/scale -> pre/post/comb + Sinkhorn-Knopp (hc_sinkhorn_iters)
//   - Shared-KV MQA: q_a->q_a_norm->q_b->q_b_norm(UNWEIGHTED RMSNorm),
//     kv_proj->kv_norm, partial rope on last qk_rope_head_dim, per-head sinks
//   - Grouped output: o_a (GroupedLinear, o_groups) -> o_b
//   - MoE: sqrtsoftplus scoring; first num_hash_layers use tid2eid[input_ids];
//     fused gate_up experts; shared SwiGLU experts; swiglu_limit clamps
//   - final hc_head -> RMSNorm -> lm_head
//
// Compressor branches (CSA/HCA) and the Lightning Indexer are NOT implemented:
// the mini gate uses sliding_attention layers only (compress_ratios all 0),
// where the compressor contributes nothing. Add them when gating a real
// checkpoint.

#include "deepseek_v4.h"
#include "safetensors_reader.h"
using safetensors_detail::json_find_int;
using safetensors_detail::json_find_float;
using safetensors_detail::json_find_bool;
#include <fstream>
#include <sstream>

// ─── math helpers ─────────────────────────────────────────────────────────────
namespace ds4math {

static inline float silu(float x) { return x / (1.0f + std::exp(-x)); }
// sqrtsoftplus = sqrt(log(1 + exp(x)))
static inline float sqrtsoftplus(float x) {
    if (x > 30.0f) return std::sqrt(x);        // log1p(exp) saturates to x
    return std::sqrt(std::log1p(std::exp(x)));
}

// y = x @ W  (x [K], W [N, K] row-major -> y [N])
static inline void matmul(float* y, const float* x, const float* W, int N, int K) {
    for (int i = 0; i < N; i++) {
        float s = 0;
        const float* w = W + (size_t)i * K;
        for (int j = 0; j < K; j++) s += x[j] * w[j];
        y[i] = s;
    }
}

// RMSNorm with weight (or null for unweighted)
static inline void rmsnorm(float* y, const float* x, const float* w, int n, float eps) {
    float ss = 0;
    for (int i = 0; i < n; i++) ss += x[i] * x[i];
    float inv = 1.0f / std::sqrt(ss / n + eps);
    for (int i = 0; i < n; i++) y[i] = x[i] * inv * (w ? w[i] : 1.0f);
}

// Interleaved partial RoPE on the LAST rd channels of each head in x
// ([nope | rope] layout, rope = interleaved pairs).
// theta: rope base; pos: token position.
static inline void rope_partial(float* x, int head_dim, int rd, int pos, float theta) {
    if (rd <= 0 || rd > head_dim) return;
    const int n_pairs = rd / 2;
    float* rope = x + (head_dim - rd);
    for (int p = 0; p < n_pairs; p++) {
        float freq = 1.0f / std::pow(theta, (float)(2 * p) / rd);
        float angle = pos * freq;
        float c = std::cos(angle), s = std::sin(angle);
        float x0 = rope[2 * p], x1 = rope[2 * p + 1];
        rope[2 * p]     = x0 * c - x1 * s;
        rope[2 * p + 1] = x0 * s + x1 * c;
    }
}

// ─── mHC module ───────────────────────────────────────────────────────────────
// hidden_streams [hc, H]. Computes pre (collapse weights), post, comb
// (Sinkhorn-projected), and returns the collapsed input [H] = sum_k pre[k]*s[k].
static void mhc_forward(const std::vector<std::vector<float>>& streams, int H,
                        const float* fn, const float* base, const float* scale,
                        int hc, float eps, int sinkhorn_iters, float rms_eps,
                        std::vector<float>& collapsed,
                        std::vector<float>& post_out,
                        std::vector<std::vector<float>>& comb_out) {
    const int hc_dim = hc * H;
    const int hc_mix = (2 + hc) * hc;
    std::vector<float> flat(hc_dim), normed(hc_dim), mixes(hc_mix);
    for (int k = 0; k < hc; k++)
        std::copy(streams[k].begin(), streams[k].end(), flat.begin() + (size_t)k * H);
    rmsnorm(normed.data(), flat.data(), nullptr, hc_dim, rms_eps);
    // mixes = fn @ normed   (fn [hc_mix, hc_dim])
    matmul(mixes.data(), normed.data(), fn, hc_mix, hc_dim);

    // pre = sigmoid(mixes[0:hc] * scale[0] + base[0:hc]) + eps
    // post = 2*sigmoid(mixes[hc:2hc] * scale[1] + base[hc:2hc])
    std::vector<float> pre(hc), post(hc);
    for (int k = 0; k < hc; k++) {
        pre[k] = 1.0f / (1.0f + std::exp(-(mixes[k] * scale[0] + base[k]))) + eps;
        post[k] = 2.0f / (1.0f + std::exp(-(mixes[hc + k] * scale[1] + base[hc + k])));
    }
    // comb: mixes[2hc:] * scale[2] + base[2hc:]  -> [hc, hc]
    std::vector<float> comb((size_t)hc * hc);
    for (int i = 0; i < hc * hc; i++)
        comb[i] = mixes[2 * hc + i] * scale[2] + base[2 * hc + i];
    // softmax over last dim (per row)
    for (int r = 0; r < hc; r++) {
        float mx = comb[(size_t)r * hc];
        for (int c = 1; c < hc; c++) mx = std::max(mx, comb[r * hc + c]);
        float ssum = 0;
        for (int c = 0; c < hc; c++) { comb[r * hc + c] = std::exp(comb[r * hc + c] - mx); ssum += comb[r * hc + c]; }
        for (int c = 0; c < hc; c++) comb[r * hc + c] /= ssum;
    }
    // Sinkhorn: +eps, col-norm, then iters-1 x {row-norm, col-norm}
    // (HF: softmax(-1)+eps; /sum(-2); for _ in range(iters-1): /sum(-1); /sum(-2))
    for (int i = 0; i < hc * hc; i++) comb[i] += eps;
    auto norm_cols = [&]() {  // divide each COLUMN by its sum (sum over rows, dim -2)
        for (int c = 0; c < hc; c++) {
            float ssum = eps;
            for (int r = 0; r < hc; r++) ssum += comb[r * hc + c];
            for (int r = 0; r < hc; r++) comb[r * hc + c] /= ssum;
        }
    };
    auto norm_rows = [&]() {  // divide each ROW by its sum (sum over cols, dim -1)
        for (int r = 0; r < hc; r++) {
            float ssum = eps;
            for (int c = 0; c < hc; c++) ssum += comb[r * hc + c];
            for (int c = 0; c < hc; c++) comb[r * hc + c] /= ssum;
        }
    };
    norm_cols();
    for (int it = 1; it < sinkhorn_iters; it++) { norm_rows(); norm_cols(); }

    // collapsed = sum_k pre[k] * streams[k]
    collapsed.assign(H, 0.0f);
    for (int k = 0; k < hc; k++)
        for (int d = 0; d < H; d++) collapsed[d] += pre[k] * streams[k][d];

    post_out = std::move(post);
    comb_out.assign(hc, std::vector<float>(hc));
    for (int r = 0; r < hc; r++)
        std::copy(comb.begin() + (size_t)r * hc, comb.begin() + (size_t)(r + 1) * hc, comb_out[r].begin());
}

} // namespace ds4math

// ─── loader ───────────────────────────────────────────────────────────────────
bool DeepSeekV4Model::load_from_safetensors(const std::string& dir, const DeepSeekV4Config* override_cfg) {
    // config.json scan
    {
        std::ifstream cf(dir + "/config.json");
        std::string txt((std::istreambuf_iterator<char>(cf)), std::istreambuf_iterator<char>());
        int iv = 0; float fv = 0;
        if (json_find_int(txt, "hidden_size", iv)) cfg.hidden_size = iv;
        if (json_find_int(txt, "num_hidden_layers", iv)) cfg.num_layers = iv;
        if (json_find_int(txt, "num_attention_heads", iv)) cfg.num_heads = iv;
        if (json_find_int(txt, "num_key_value_heads", iv)) cfg.num_kv_heads = iv;
        if (json_find_int(txt, "head_dim", iv)) cfg.head_dim = iv;
        if (json_find_int(txt, "q_lora_rank", iv)) cfg.q_lora_rank = iv;
        if (json_find_int(txt, "o_lora_rank", iv)) cfg.o_lora_rank = iv;
        if (json_find_int(txt, "o_groups", iv)) cfg.o_groups = iv;
        if (json_find_int(txt, "vocab_size", iv)) cfg.vocab_size = iv;
        if (json_find_int(txt, "moe_intermediate_size", iv)) cfg.moe_intermediate = iv;
        if (json_find_int(txt, "n_routed_experts", iv)) cfg.n_routed_experts = iv;
        if (json_find_int(txt, "n_shared_experts", iv)) cfg.n_shared_experts = iv;
        if (json_find_int(txt, "num_experts_per_tok", iv)) cfg.top_k = iv;
        if (json_find_int(txt, "num_hash_layers", iv)) cfg.num_hash_layers = iv;
        // mlp_layer_types list: count "hash_moe" entries (V4 configs ship the
        // list, not num_hash_layers).
        {
            size_t p = txt.find("\"mlp_layer_types\"");
            if (p != std::string::npos) {
                int n_hash = 0;
                size_t q = p;
                while ((q = txt.find("hash_moe", q)) != std::string::npos) { n_hash++; q += 8; }
                if (n_hash > 0) cfg.num_hash_layers = n_hash;
            }
        }
        if (json_find_int(txt, "sliding_window", iv)) cfg.sliding_window = iv;
        if (json_find_int(txt, "hc_mult", iv)) cfg.hc_mult = iv;
        if (json_find_int(txt, "hc_sinkhorn_iters", iv)) cfg.hc_sinkhorn_iters = iv;
        if (json_find_float(txt, "routed_scaling_factor", fv)) cfg.routed_scale = fv;
        if (json_find_float(txt, "swiglu_limit", fv)) cfg.swiglu_limit = fv;
        if (json_find_float(txt, "hc_eps", fv)) cfg.hc_eps = fv;
        if (json_find_float(txt, "rms_norm_eps", fv)) cfg.rms_norm_eps = fv;
        if (json_find_float(txt, "rope_theta", fv)) cfg.rope_theta = fv;
        // qk_rope_head_dim via partial_rotary_factor
        float prf = 0;
        if (json_find_float(txt, "partial_rotary_factor", prf) && prf > 0)
            cfg.qk_rope_head_dim = (int)(cfg.head_dim * prf);
        else if (json_find_int(txt, "qk_rope_head_dim", iv) && iv > 0)
            cfg.qk_rope_head_dim = iv;
        bool bv = false;
        if (json_find_bool(txt, "tie_word_embeddings", bv)) cfg.tie_embeddings = bv;
    }
    if (override_cfg) cfg = *override_cfg;

    const int H = cfg.hidden_size;
    const int hc = cfg.hc_mult;
    const int hc_dim = hc * H;
    const int hc_mix = (2 + hc) * hc;

    SafetensorsWeightReader r;
    // single-file checkpoint: no index.json -> open() the file directly
    std::ifstream idx(dir + "/model.safetensors.index.json");
    bool has_index = idx.good();
    bool r_ok = has_index ? r.open_dir(dir) : r.open(dir + "/model.safetensors");
    if (!r_ok) { fprintf(stderr, "[deepseek_v4] FAIL: open %s (%s)\n", dir.c_str(), r.error().c_str()); return false; }
    auto get = [&](const char* name, std::vector<float>& dst, size_t expect, bool required = true) -> bool {
        if (!r.get_tensor_f32(name, dst)) {
            if (required) fprintf(stderr, "  [deepseek_v4] missing tensor: %s\n", name);
            return false;
        }
        if (expect && dst.size() != expect) {
            fprintf(stderr, "  [deepseek_v4] %s: %zu elems, want %zu\n", name, dst.size(), expect);
            return false;
        }
        return true;
    };

    if (!get("model.embed_tokens.weight", embed, (size_t)cfg.vocab_size * H)) return false;
    if (!get("model.norm.weight", final_norm_w, H)) return false;
    if (!cfg.tie_embeddings) {
        if (!get("lm_head.weight", lm_head, (size_t)cfg.vocab_size * H)) return false;
    } else lm_head = embed;
    if (!get("model.hc_head.hc_fn", hc_head_fn, (size_t)hc * hc_dim)) return false;
    if (!get("model.hc_head.hc_base", hc_head_base, hc)) return false;
    if (!get("model.hc_head.hc_scale", hc_head_scale, 1)) return false;

    layers.resize(cfg.num_layers);
    const int in_per_group = (cfg.num_heads * cfg.head_dim) / cfg.o_groups;  // o_a input width per group

    for (int il = 0; il < cfg.num_layers; il++) {
        auto& l = layers[il];
        char buf[256];
        auto name = [&](const char* p) {
            snprintf(buf, sizeof(buf), "model.layers.%d.%s", il, p);
            return (const char*)buf;
        };
        bool ok = true;
        ok &= get(name("input_layernorm.weight"), l.rms_attn_w, H);
        ok &= get(name("post_attention_layernorm.weight"), l.rms_ffn_w, H);
        ok &= get(name("self_attn.sinks"), l.sinks, cfg.num_heads);
        ok &= get(name("self_attn.q_a_proj.weight"), l.q_a, (size_t)cfg.q_lora_rank * H);
        ok &= get(name("self_attn.q_a_norm.weight"), l.q_a_norm, cfg.q_lora_rank);
        ok &= get(name("self_attn.q_b_proj.weight"), l.q_b, (size_t)cfg.num_heads * cfg.head_dim * cfg.q_lora_rank);
        ok &= get(name("self_attn.kv_proj.weight"), l.kv_w, (size_t)cfg.head_dim * H);
        ok &= get(name("self_attn.kv_norm.weight"), l.kv_norm, cfg.head_dim);
        ok &= get(name("self_attn.o_a_proj.weight"), l.o_a, (size_t)cfg.o_groups * cfg.o_lora_rank * in_per_group);
        ok &= get(name("self_attn.o_b_proj.weight"), l.o_b, (size_t)H * cfg.o_groups * cfg.o_lora_rank);
        ok &= get(name("attn_hc.fn"), l.hc_attn_fn, (size_t)hc_mix * hc_dim);
        ok &= get(name("attn_hc.base"), l.hc_attn_base, hc_mix);
        ok &= get(name("attn_hc.scale"), l.hc_attn_scale, 3);
        ok &= get(name("ffn_hc.fn"), l.hc_ffn_fn, (size_t)hc_mix * hc_dim);
        ok &= get(name("ffn_hc.base"), l.hc_ffn_base, hc_mix);
        ok &= get(name("ffn_hc.scale"), l.hc_ffn_scale, 3);
        ok &= get(name("mlp.gate.weight"), l.gate, (size_t)cfg.n_routed_experts * H);
        ok &= get(name("mlp.experts.gate_up_proj"), l.exp_gate_up,
                  (size_t)cfg.n_routed_experts * 2 * cfg.moe_intermediate * H);
        ok &= get(name("mlp.experts.down_proj"), l.exp_down,
                  (size_t)cfg.n_routed_experts * cfg.moe_intermediate * H);
        ok &= get(name("mlp.shared_experts.gate_proj.weight"), l.sh_gate, (size_t)cfg.moe_intermediate * H);
        ok &= get(name("mlp.shared_experts.up_proj.weight"), l.sh_up, (size_t)cfg.moe_intermediate * H);
        ok &= get(name("mlp.shared_experts.down_proj.weight"), l.sh_down, (size_t)H * cfg.moe_intermediate);

        if (il < cfg.num_hash_layers) {
            // hash routing: tid2eid [vocab, top_k]
            ok &= get(name("mlp.gate.tid2eid"), l.tid2eid, (size_t)cfg.vocab_size * cfg.top_k);
        } else {
            // topk routing: e_score_correction_bias
            ok &= get(name("mlp.gate.e_score_correction_bias"), l.exp_probs_b, cfg.n_routed_experts);
        }
        if (!ok) {
            fprintf(stderr, "  [deepseek_v4] layer %d: incomplete weights — ABORT\n", il);
            return false;
        }
    }
    fprintf(stderr,
        "[deepseek_v4] loaded: %d layers, H=%d, heads=%d, rope_dim=%d, q_lora=%d, o_lora=%d, "
        "o_groups=%d, experts=%d+%d, top_k=%d, moe_int=%d, hc_mult=%d, hash_layers=%d\n",
        cfg.num_layers, H, cfg.num_heads, cfg.qk_rope_head_dim, cfg.q_lora_rank, cfg.o_lora_rank,
        cfg.o_groups, cfg.n_routed_experts, cfg.n_shared_experts, cfg.top_k,
        cfg.moe_intermediate, hc, cfg.num_hash_layers);
    return true;
}

void DeepSeekV4Model::clear() {
    embed.clear(); final_norm_w.clear(); lm_head.clear();
    hc_head_fn.clear(); hc_head_base.clear(); hc_head_scale.clear();
    layers.clear();
}

// ─── forward ──────────────────────────────────────────────────────────────────
std::vector<float> deepseek_v4_forward(DeepSeekV4Model& model, int token_id,
                                       DeepSeekV4KVCache& kv_cache,
                                       DeepSeekV4mHCState& mhc, int& pos) {
    using namespace ds4math;
    const auto& cfg = model.cfg;
    const int H = cfg.hidden_size;
    const int hc = cfg.hc_mult;
    const int hc_dim = hc * H;
    // grouped output projection: heads split into o_groups; each group feeds a
    // per-group block of o_a (GroupedLinear), then o_b mixes the concat.
    const int in_per_group = (cfg.num_heads * cfg.head_dim) / cfg.o_groups;

    std::vector<float> embed(H);
    if (token_id >= 0 && token_id < cfg.vocab_size)
        std::copy(&model.embed[(size_t)token_id * H], &model.embed[(size_t)token_id * H + H], embed.begin());
    mhc.set_embed(embed.data());
    if (kv_cache.head_dim == 0)
        kv_cache.init(cfg.num_layers, 4096, cfg.head_dim);

    std::vector<float> norm(H), collapsed(H);
    std::vector<float> post(hc);
    std::vector<std::vector<float>> comb(hc, std::vector<float>(hc));

    // per-layer buffers
    std::vector<float> q_a(cfg.q_lora_rank), q_full((size_t)cfg.num_heads * cfg.head_dim);
    std::vector<float> kv((size_t)cfg.head_dim);
    std::vector<float> attn_out((size_t)cfg.num_heads * cfg.head_dim);
    std::vector<float> oa((size_t)cfg.o_groups * cfg.o_lora_rank);
    std::vector<float> attn_proj(H);
    std::vector<float> moe_in(H), shared_out(H), moe_out(H);
    std::vector<float> exp_gate_up(2 * cfg.moe_intermediate), exp_act(cfg.moe_intermediate);
    std::vector<float> sh_gate(cfg.moe_intermediate), sh_up(cfg.moe_intermediate);
    std::vector<float> router_scores(cfg.n_routed_experts);
    std::vector<int>   expert_ids(cfg.top_k);
    std::vector<float> expert_wts(cfg.top_k);
    std::vector<float> scores_buf(4096 + 1), probs_buf(4096 + 1);

    for (int il = 0; il < cfg.num_layers; il++) {
        const auto& l = model.layers[il];
        const float* x = mhc.streams[0].data(); // for norms below we use collapsed

        // Debug: dump stream-0 of the CURRENT token after each sublayer

        // ── attn site mHC pre ──
        mhc_forward(mhc.streams, H, l.hc_attn_fn.data(), l.hc_attn_base.data(), l.hc_attn_scale.data(),
                    hc, cfg.hc_eps, cfg.hc_sinkhorn_iters, cfg.rms_norm_eps, collapsed, post, comb);
        rmsnorm(norm.data(), collapsed.data(), l.rms_attn_w.data(), H, cfg.rms_norm_eps);

        // ── Q compression: q_a = norm @ W_q_a ; q_a_norm ; q_b = q_a @ W_q_b ──
        matmul(q_a.data(), norm.data(), l.q_a.data(), cfg.q_lora_rank, H);
        rmsnorm(q_a.data(), q_a.data(), l.q_a_norm.data(), cfg.q_lora_rank, cfg.rms_norm_eps);
        matmul(q_full.data(), q_a.data(), l.q_b.data(), cfg.num_heads * cfg.head_dim, cfg.q_lora_rank);
        // q_b_norm: UNWEIGHTED RMSNorm per head
        for (int h = 0; h < cfg.num_heads; h++)
            rmsnorm(&q_full[(size_t)h * cfg.head_dim], &q_full[(size_t)h * cfg.head_dim],
                    nullptr, cfg.head_dim, cfg.rms_norm_eps);
        // partial rope on last qk_rope_head_dim of each head
        for (int h = 0; h < cfg.num_heads; h++)
            rope_partial(&q_full[(size_t)h * cfg.head_dim], cfg.head_dim, cfg.qk_rope_head_dim, pos, cfg.rope_theta);

        // ── KV: kv = norm @ W_kv ; kv_norm ; rope; store ──
        matmul(kv.data(), norm.data(), l.kv_w.data(), cfg.head_dim, H);
        rmsnorm(kv.data(), kv.data(), l.kv_norm.data(), cfg.head_dim, cfg.rms_norm_eps);
        rope_partial(kv.data(), cfg.head_dim, cfg.qk_rope_head_dim, pos, cfg.rope_theta);
        float* cache_row = kv_cache.kv[il].data() + (size_t)pos * cfg.head_dim;
        std::copy(kv.begin(), kv.end(), cache_row);

        // ── attention (shared KV head, K=V, per-head sinks) ──
        const int seq_len = pos + 1;
        const float scale = 1.0f / std::sqrt((float)cfg.head_dim);
        int win_start = std::max(0, seq_len - cfg.sliding_window);
        std::fill(attn_out.begin(), attn_out.end(), 0.0f);
        for (int h = 0; h < cfg.num_heads; h++) {
            const float* qh = &q_full[(size_t)h * cfg.head_dim];
            float sink = l.sinks[h];
            // scores over window + sink (S+1 entries)
            int n = 0;
            for (int s = win_start; s < seq_len; s++) {
                const float* krow = kv_cache.kv[il].data() + (size_t)s * cfg.head_dim;
                float acc = 0;
                for (int d = 0; d < cfg.head_dim; d++) acc += qh[d] * krow[d];
                scores_buf[n++] = acc * scale;
            }
            scores_buf[n] = sink; n++;
            // softmax
            float mx = scores_buf[0];
            for (int i = 1; i < n; i++) mx = std::max(mx, scores_buf[i]);
            float ssum = 0;
            for (int i = 0; i < n; i++) { probs_buf[i] = std::exp(scores_buf[i] - mx); ssum += probs_buf[i]; }
            // weighted sum of V (drop the sink entry)
            float* outh = &attn_out[(size_t)h * cfg.head_dim];
            for (int i = 0; i < n - 1; i++) {
                float w = probs_buf[i] / ssum;
                const float* vrow = kv_cache.kv[il].data() + (size_t)(win_start + i) * cfg.head_dim;
                for (int d = 0; d < cfg.head_dim; d++) outh[d] += w * vrow[d];
            }
        }

        // ── derope output rope slice (apply -sin at query position) ──
        // HF: apply_rotary_pos_emb(attn_output, cos, -sin) — rotate the LAST
        // rope dims by the NEGATIVE query angle (K=V picked up rope; undo it).
        {
            const int rd = cfg.qk_rope_head_dim;
            const int n_pairs = rd / 2;
            for (int h = 0; h < cfg.num_heads; h++) {
                float* outh = &attn_out[(size_t)h * cfg.head_dim];
                float* rope = outh + (cfg.head_dim - rd);
                for (int p = 0; p < n_pairs; p++) {
                    float freq = 1.0f / std::pow(cfg.rope_theta, (float)(2 * p) / rd);
                    float angle = pos * freq;
                    float c = std::cos(angle), s = std::sin(angle);
                    float x0 = rope[2 * p], x1 = rope[2 * p + 1];
                    rope[2 * p]     = x0 * c + x1 * s;   // -sin => +sin
                    rope[2 * p + 1] = -x0 * s + x1 * c;
                }
            }
        }

        // ── grouped output projection ──
        // o_a: GroupedLinear — weight [o_groups*o_lora, in_per_group]; per group g:
        //   oa_g = x_g @ W_g^T where x_g = out[g*in_per:(g+1)*in_per], W_g = weight[g*o_lora:(g+1)*o_lora]
        for (int g = 0; g < cfg.o_groups; g++) {
            const float* xg = &attn_out[(size_t)g * in_per_group];
            for (int r = 0; r < cfg.o_lora_rank; r++) {
                float s = 0;
                const float* wrow = &l.o_a[(size_t)(g * cfg.o_lora_rank + r) * in_per_group];
                for (int c = 0; c < in_per_group; c++) s += xg[c] * wrow[c];
                oa[(size_t)g * cfg.o_lora_rank + r] = s;
            }
        }
        matmul(attn_proj.data(), oa.data(), l.o_b.data(), H, cfg.o_groups * cfg.o_lora_rank);

        // ── attn site mHC post: streams[k] = post[k]*attn_proj + sum_j comb[j][k]*streams[j] ──
        // (compute into a temp buffer first: comb mixes the OLD streams, and
        // writing in place would corrupt later k — HF uses a fresh matmul)
        {
            std::vector<std::vector<float>> new_streams(hc, std::vector<float>(H));
            for (int k = 0; k < hc; k++)
                for (int d = 0; d < H; d++) {
                    float v = post[k] * attn_proj[d];
                    for (int j = 0; j < hc; j++) v += comb[j][k] * mhc.streams[j][d];
                    new_streams[k][d] = v;
                }
            mhc.streams = std::move(new_streams);
        }

        // ── FFN site mHC pre ──
        mhc_forward(mhc.streams, H, l.hc_ffn_fn.data(), l.hc_ffn_base.data(), l.hc_ffn_scale.data(),
                    hc, cfg.hc_eps, cfg.hc_sinkhorn_iters, cfg.rms_norm_eps, collapsed, post, comb);
        rmsnorm(norm.data(), collapsed.data(), l.rms_ffn_w.data(), H, cfg.rms_norm_eps);

        // ── MoE ──
        // router scores: sqrtsoftplus(gate @ norm)
        matmul(router_scores.data(), norm.data(), l.gate.data(), cfg.n_routed_experts, H);
        for (int i = 0; i < cfg.n_routed_experts; i++) router_scores[i] = sqrtsoftplus(router_scores[i]);

        if (il < cfg.num_hash_layers && !l.tid2eid.empty()) {
            // hash routing: tid2eid[token_id][k]
            for (int k = 0; k < cfg.top_k; k++)
                expert_ids[k] = (int)l.tid2eid[(size_t)token_id * cfg.top_k + k];
            for (int k = 0; k < cfg.top_k; k++) expert_wts[k] = router_scores[expert_ids[k]];
        } else {
            // topk routing: indices = topk(scores + e_score_correction_bias)
            std::vector<float> sc(router_scores);
            if (!l.exp_probs_b.empty())
                for (int i = 0; i < cfg.n_routed_experts; i++) sc[i] += l.exp_probs_b[i];
            for (int k = 0; k < cfg.top_k; k++) {
                int best = 0;
                for (int i = 1; i < cfg.n_routed_experts; i++) if (sc[i] > sc[best]) best = i;
                expert_ids[k] = best;
                expert_wts[k] = router_scores[best];
                sc[best] = -1e30f;
            }
        }
        // normalize (norm_topk_prob) + routed scaling
        float wt_sum = 0;
        for (int k = 0; k < cfg.top_k; k++) wt_sum += expert_wts[k];
        if (wt_sum > 1e-20f) for (int k = 0; k < cfg.top_k; k++) expert_wts[k] /= wt_sum;
        for (int k = 0; k < cfg.top_k; k++) expert_wts[k] *= cfg.routed_scale;

        // routed experts (fused gate_up)
        std::fill(moe_out.begin(), moe_out.end(), 0.0f);
        for (int k = 0; k < cfg.top_k; k++) {
            int eid = expert_ids[k];
            float wt = expert_wts[k];
            const float* wgu = &l.exp_gate_up[(size_t)eid * 2 * cfg.moe_intermediate * H];
            const float* wdn = &l.exp_down[(size_t)eid * cfg.moe_intermediate * H];
            for (int i = 0; i < 2 * cfg.moe_intermediate; i++) {
                const float* wrow = wgu + (size_t)i * H;
                float s = 0;
                for (int j = 0; j < H; j++) s += norm[j] * wrow[j];
                exp_gate_up[i] = s;
            }
            // gate = clamp(max=limit); up = clamp(min=-limit, max=limit); act = silu(gate)*up
            for (int i = 0; i < cfg.moe_intermediate; i++) {
                float g = exp_gate_up[i], u = exp_gate_up[cfg.moe_intermediate + i];
                g = std::min(g, cfg.swiglu_limit);
                u = std::max(-cfg.swiglu_limit, std::min(u, cfg.swiglu_limit));
                exp_act[i] = silu(g) * u;
            }
            for (int d = 0; d < H; d++) {
                const float* wrow = wdn + (size_t)d * cfg.moe_intermediate;
                float s = 0;
                for (int j = 0; j < cfg.moe_intermediate; j++) s += exp_act[j] * wrow[j];
                moe_out[d] += wt * s;
            }
        }

        // shared experts (SwiGLU with clamps)
        matmul(sh_gate.data(), norm.data(), l.sh_gate.data(), cfg.moe_intermediate, H);
        matmul(sh_up.data(),   norm.data(), l.sh_up.data(),   cfg.moe_intermediate, H);
        std::fill(shared_out.begin(), shared_out.end(), 0.0f);
        {
            std::vector<float> act(cfg.moe_intermediate);
            for (int i = 0; i < cfg.moe_intermediate; i++) {
                float g = std::min(sh_gate[i], cfg.swiglu_limit);
                float u = std::max(-cfg.swiglu_limit, std::min(sh_up[i], cfg.swiglu_limit));
                act[i] = silu(g) * u;
            }
            matmul(shared_out.data(), act.data(), l.sh_down.data(), H, cfg.moe_intermediate);
        }
        for (int d = 0; d < H; d++) moe_out[d] += shared_out[d];

        // ── FFN site mHC post ──
        {
            std::vector<std::vector<float>> new_streams(hc, std::vector<float>(H));
            for (int k = 0; k < hc; k++)
                for (int d = 0; d < H; d++) {
                    float v = post[k] * moe_out[d];
                    for (int j = 0; j < hc; j++) v += comb[j][k] * mhc.streams[j][d];
                    new_streams[k][d] = v;
                }
            mhc.streams = std::move(new_streams);
        }
    }

    // ── hc_head: collapse streams ──
    {
        std::vector<float> flat(hc_dim), normed(hc_dim), mixes(hc);
        for (int k = 0; k < hc; k++)
            std::copy(mhc.streams[k].begin(), mhc.streams[k].end(), flat.begin() + (size_t)k * H);
        rmsnorm(normed.data(), flat.data(), nullptr, hc_dim, cfg.rms_norm_eps);
        matmul(mixes.data(), normed.data(), model.hc_head_fn.data(), hc, hc_dim);
        std::vector<float> pre(hc);
        for (int k = 0; k < hc; k++)
            pre[k] = 1.0f / (1.0f + std::exp(-(mixes[k] * model.hc_head_scale[0] + model.hc_head_base[k]))) + cfg.hc_eps;
        collapsed.assign(H, 0.0f);
        for (int k = 0; k < hc; k++)
            for (int d = 0; d < H; d++) collapsed[d] += pre[k] * mhc.streams[k][d];
    }

    // ── final norm + lm_head ──
    rmsnorm(norm.data(), collapsed.data(), model.final_norm_w.data(), H, cfg.rms_norm_eps);
    std::vector<float> logits(cfg.vocab_size);
    const float* lm = model.lm_head.empty() ? model.embed.data() : model.lm_head.data();
    for (int i = 0; i < cfg.vocab_size; i++) {
        const float* wrow = lm + (size_t)i * H;
        float s = 0;
        for (int j = 0; j < H; j++) s += norm[j] * wrow[j];
        logits[i] = s;
    }

    pos++;
    return logits;
}
