// deepseek_v4.h — DeepSeek V4 Flash / Pro — CPU inference
//
// Rewritten 2026-08-16 against the REAL architecture (HF modeling_deepseek_v4.py
// 5.14 + our llama.cpp fork src/models/deepseek4.cpp). The previous version was
// written against a fictional design (MLA + kv_lora_rank + a 4x4 mHC mix matrix)
// that does not exist in V4. The real V4:
//
//  1. Shared-KV Multi-Query Attention (num_kv_heads=1, K=V):
//       q_a_proj (H -> q_lora) -> q_a_norm (RMSNorm)
//       q_b_proj (q_lora -> n_heads*head_dim) -> q_b_norm (UNWEIGHTED RMSNorm)
//       kv_proj (H -> head_dim) -> kv_norm (RMSNorm)
//       partial RoPE on the LAST qk_rope_head_dim channels of each head
//       ([nope | rope] layout); per-head learnable attention sinks
//       (gpt-OSS style: cat to scores pre-softmax, drop after)
//       grouped output projection: o_a_proj (GroupedLinear, o_groups) + o_b_proj
//  2. mHC (Manifold-Constrained Hyper-Connections): hc_mult=4 parallel streams,
//       per-sublayer fn/base/scale modules + Sinkhorn-Knopp projection of the
//       comb matrix onto doubly-stochastic (20 iters), plus a final hc_head.
//  3. MoE: sqrtsoftplus router scoring, e_score_correction_bias, first
//       num_hash_layers layers use frozen tid2eid[input_ids] hash routing;
//       FUSED experts.gate_up_proj 3D tensors; shared_experts SwiGLU with
//       swiglu_limit clamps.
//  4. Per-layer compressors (CSA ratio 4 / HCA ratio 128) — NOT implemented
//       here (mini gate has sliding_attention layers only); a real-checkpoint
//       gate needs them.
//
// Loads HF safetensors names (model.layers.N.self_attn.*, mlp.*, attn_hc.*).
// GGUF blk.* aliases are NOT handled by this loader yet.

#ifndef DEEPSEEK_V4_H
#define DEEPSEEK_V4_H

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <vector>
#include <string>
#include <cstring>
#include <algorithm>
#include <numeric>

// ─── DeepSeek V4 Config ───────────────────────────────────────────────────────
struct DeepSeekV4Config {
    int hidden_size      = 4096;
    int num_layers       = 43;
    int num_heads        = 64;
    int num_kv_heads     = 1;     // Shared-KV MQA: 1 KV head, K=V
    int head_dim         = 512;
    int qk_rope_head_dim = 64;    // partial rotary (last 64 of 512 per head)
    int q_lora_rank      = 1024;
    int o_lora_rank      = 1024;
    int o_groups         = 8;     // grouped output projection
    int vocab_size       = 129280;
    int max_seq_len      = 1048576;

    // MoE FFN
    int n_routed_experts = 256;
    int n_shared_experts = 1;
    int top_k            = 6;
    int moe_intermediate = 2048;
    float routed_scale   = 1.5f;
    float swiglu_limit   = 10.0f;
    int num_hash_layers  = 3;     // first N layers route via tid2eid lookup
    bool norm_topk_prob  = true;

    // Attention / context
    int sliding_window   = 128;
    float rope_theta     = 10000.0f;
    float compress_rope_theta = 160000.0f;
    int rope_orig_ctx    = 65536;
    float rope_yarn_factor = 16.0f;

    // mHC
    int hc_mult          = 4;
    int hc_sinkhorn_iters = 20;
    float hc_eps         = 1e-6f;

    float rms_norm_eps   = 1e-6f;
    bool tie_embeddings  = false;
    int pad_token_id     = -1;

    // Per-layer attention compress ratio (0 = sliding). Vector indexed by layer.
    std::vector<int> compress_ratios;
    std::vector<int> layer_attn_type;  // 0=sliding, 1=CSA(4), 2=HCA(128)
    // Window widths per flavour. Parsed from `compress_rates` (transformers /
    // fixture naming) or implied by `compress_ratios` (checkpoint naming).
    int csa_rate = 4;
    int hca_rate = 128;
    // Indexer dims (Lightning Indexer; CSA only).
    int index_n_heads = 64;
    int index_head_dim = 128;
    int index_topk = 512;
};

// ─── Weights ──────────────────────────────────────────────────────────────────
struct DeepSeekV4Layer {
    // norms (RMSNorm weights)
    std::vector<float> rms_attn_w, rms_ffn_w;
    // attention
    std::vector<float> sinks;          // per-head learnable sink [n_heads]
    std::vector<float> q_a, q_a_norm;  // q_a_proj [q_lora, H], q_a_norm [q_lora]
    std::vector<float> q_b;            // q_b_proj [n_heads*head_dim, q_lora]
    std::vector<float> kv_w, kv_norm;  // kv_proj [head_dim, H], kv_norm [head_dim]
    std::vector<float> o_a, o_b;       // o_a [o_groups*o_lora, n_heads*head_dim/o_groups], o_b [H, o_groups*o_lora]
    // mHC
    std::vector<float> hc_attn_fn, hc_attn_base, hc_attn_scale;  // fn [hc_mix_dim, hc*hidden]
    std::vector<float> hc_ffn_fn,  hc_ffn_base,  hc_ffn_scale;
    // MoE router
    std::vector<float> gate;           // mlp.gate.weight [n_routed, H]
    std::vector<float> exp_probs_b;    // e_score_correction_bias [n_routed] (moe layers)
    std::vector<float> tid2eid;        // hash layers: [vocab*top_k] ints
    // experts (fused gate_up 3D)
    std::vector<float> exp_gate_up;    // [n_routed, 2*moe_int, H]
    std::vector<float> exp_down;       // [n_routed, H, moe_int]
    // shared expert (SwiGLU)
    std::vector<float> sh_gate, sh_up, sh_down;  // [moe_int, H], [moe_int, H], [H, moe_int]
    // ── compressor (CSA/HCA); empty on sliding layers ─────────────────────────
    // The compressor pools every `cp_rate` tokens: softmax(gate + position_bias)
    // over the 2*rate Ca/Cb slots (two series, stride `rate`), RMSNorm, then the
    // "compress" RoPE at positions w*rate. See the reference
    // DeepseekV4CSACompressor / DeepseekV4HCACompressor (transformers 5.16.1).
    int cp_rate = 0;                 // 0 = this layer has no compressor
    int cp_series = 1;               // 1 = HCA (single series), 2 = CSA (Ca/Cb overlap)
    std::vector<float> cp_kv;        // kv_proj.weight    [cp_series*head_dim, H]
    std::vector<float> cp_gate;      // gate_proj.weight  [cp_series*head_dim, H]
    std::vector<float> cp_pos_bias;  // position_bias     [cp_rate, cp_series*head_dim]
    std::vector<float> cp_norm;      // kv_norm.weight    [head_dim]
    // ── Lightning Indexer (CSA layers only) ───────────────────────────────────
    // Its own compressor at index_head_dim (2 series), a query projection off
    // q_residual, and a ReLU-weighted head sum that ranks compressed entries.
    int ix_heads = 0;                // index_n_heads (0 = no indexer)
    int ix_hd = 0;                   // index_head_dim
    int ix_topk = 0;                 // index_topk
    std::vector<float> ix_kv, ix_gate;  // [2*ix_hd, H]
    std::vector<float> ix_pos_bias;     // [cp_rate, 2*ix_hd]
    std::vector<float> ix_norm;         // [ix_hd]
    std::vector<float> ix_qb;           // [ix_heads*ix_hd, q_lora_rank]
    std::vector<float> ix_wproj;        // [ix_heads, H]
};

struct DeepSeekV4Model {
    DeepSeekV4Config cfg;
    std::vector<float> embed;        // [vocab, H]
    std::vector<float> final_norm_w; // [H]
    std::vector<float> lm_head;      // [vocab, H] (untied)
    std::vector<float> hc_head_fn;   // [hc, hc*H]
    std::vector<float> hc_head_base; // [hc]
    std::vector<float> hc_head_scale;// [1]
    std::vector<DeepSeekV4Layer> layers;

    // Loader (HF safetensors names). override_cfg optional.
    bool load_from_safetensors(const std::string& dir, const DeepSeekV4Config* override_cfg = nullptr);
    void clear();
};

// Compressor state/params are needed by the KV cache below, which holds one
// state per layer for the layer's own compressor and (CSA) the indexer's.
struct DeepSeekV4CompState {
    bool inited = false;
    int n_buf = 0;              // tokens buffered toward the current window
    int n_entries = 0;          // entries emitted so far
    std::vector<float> buf_kv;  // [rate * series*hd] pending rows
    std::vector<float> buf_gate;
    std::vector<float> prev_ca, prev_ca_gate;  // previous window's Ca (CSA only)
    std::vector<float> entries;  // [n_entries * hd], emitted in order
    void init(int rate, int series, int hd) {
        const size_t rows = (size_t)(rate > 0 ? rate : 1) * series * hd;
        buf_kv.assign(rows, 0.0f);
        buf_gate.assign(rows, 0.0f);
        prev_ca.assign((size_t)(rate > 0 ? rate : 1) * hd, 0.0f);
        prev_ca_gate.assign((size_t)(rate > 0 ? rate : 1) * hd, -INFINITY);
        entries.clear();
        n_buf = 0;
        n_entries = 0;
        inited = true;
    }
};

// The compressor's weights, so the SAME state machine serves both the outer
// compressor (head_dim) and the indexer's own compression (index_head_dim) —
// they differ only in weights, width and series count.
struct DeepSeekV4CompParams {
    const float* kv = nullptr;        // [series*hd, H]
    const float* gate = nullptr;      // [series*hd, H]
    const float* pos_bias = nullptr;  // [rate, series*hd]
    const float* norm = nullptr;      // [hd]
    int series = 1;
    int hd = 0;
};


// ─── KV cache ─────────────────────────────────────────────────────────────────
struct DeepSeekV4KVCache {
    // per layer: rolling buffer [max_slots, head_dim]; single KV head, K=V
    std::vector<std::vector<float>> kv;
    int head_dim = 0;
    int max_slots = 0;
    // per-layer compressor state (the layer's own, and — CSA only — the indexer's)
    std::vector<DeepSeekV4CompState> cp, ix;
    void init(int n_layers, int max_seq, int hd) {
        head_dim = hd; max_slots = max_seq;
        kv.assign(n_layers, std::vector<float>((size_t)max_seq * hd, 0.0f));
        cp.assign(n_layers, DeepSeekV4CompState());
        ix.assign(n_layers, DeepSeekV4CompState());
    }
    void clear() {
        for (auto& l : kv) std::fill(l.begin(), l.end(), 0.0f);
        for (auto& s : cp) { s.entries.clear(); s.n_buf = 0; s.n_entries = 0; s.inited = false; }
        for (auto& s : ix) { s.entries.clear(); s.n_buf = 0; s.n_entries = 0; s.inited = false; }
    }
};

// ─── mHC state (hc_mult parallel streams) ─────────────────────────────────────
struct DeepSeekV4mHCState {
    int hc = 4, H = 0;
    std::vector<std::vector<float>> streams;  // [hc][H]
    void init(int hc_mult, int hidden) { hc = hc_mult; H = hidden; streams.assign(hc, std::vector<float>(hidden, 0.0f)); }
    void set_embed(const float* e) { for (int k = 0; k < hc; k++) std::copy(e, e + H, streams[k].begin()); }
    const float* current() const { return streams[0].data(); }
};

// ─── Compressor (CSA/HCA) ─────────────────────────────────────────────────────
// NOTE ON THE TWO ENTRY POINTS: `deepseek_v4_compressor_forward` below is the
// window-batched form used by the P1.1 stage-1 gate (it matches the reference's
// single full-sequence forward). Decoding needs the *incremental* form:
// `deepseek_v4_compressor_step` buffers one token at a time and emits an entry
// the moment the window fills, exactly as the reference's cache does
// (`store_compression_weights` -> usable prefix -> one entry, Ca carried over in
// `overlap_kv`). Both must produce the same entries; the incremental one is what
// the attention path will consume (gated by cmp_deepseek_v4_compressor_incremental).
// Feed one token through a parameterised compressor state.
bool deepseek_v4_compressor_step_p(const DeepSeekV4CompParams& p, const DeepSeekV4Config& cfg,
                                   int rate, const float* x_token, DeepSeekV4CompState& st);

// Compressor params for a layer's Lightning Indexer (CSA layers only).
DeepSeekV4CompParams deepseek_v4_indexer_comp_params(const DeepSeekV4Layer& l, const DeepSeekV4Config& cfg);

// Feed one token (its collapsed attention-site hidden vector `x_token` [H]).
// Returns true when this token completed a window and appended an entry.
bool deepseek_v4_compressor_step(const DeepSeekV4Layer& l, const DeepSeekV4Config& cfg, int rate,
                                 const float* x_token, DeepSeekV4CompState& st);

std::vector<float> deepseek_v4_compressor_forward(const DeepSeekV4Layer& l, const DeepSeekV4Config& cfg,
                                                  int rate, const std::vector<float>& x, int T);

// Indexer top-k for one CSA layer (batched prefill, positions 0..T-1): the
// indexer compresses with its own weights at `index_head_dim`, scores each
// query against those entries as `sum_h relu(q_h . K) * w_h` (both scaled),
// masks entries past the query's causal threshold, and keeps `index_topk` of
// them. Returns [T][k] entry indices; -1 marks a pick the query may not use
// (fewer than k entries are ready) — the reference's sentinel, which the
// attention path must treat as "not selected".
// The gate-able half: the score table [T][n_win], exactly as the reference's
// scorer returns it (before the causal mask / sentinel).
std::vector<float> deepseek_v4_indexer_scores(const DeepSeekV4Layer& l, const DeepSeekV4Config& cfg,
                                             int rate, const std::vector<float>& x, int T);

// The selection half: the causal mask + top-k with the reference's `-1` sentinel.
// Ties make the *order* implementation-defined (see the .cpp note), so index
// equality is not a valid gate — the score table above is.
std::vector<int> deepseek_v4_indexer_topk(const DeepSeekV4Layer& l, const DeepSeekV4Config& cfg,
                                          int rate, const std::vector<float>& x, int T);

// ─── Forward ──────────────────────────────────────────────────────────────────
// One token. Returns logits [vocab]. kv_cache + mhc updated in place.
//
// `layer_states`, when non-null, receives the residual streams at every layer
// boundary for this token: `num_layers` states of `hc_mult * hidden_size`
// floats each ([hc][H] row-major). State i is the input to layer i — the same
// tensor HF exposes as `hidden_states[i]` (its last entry, the collapsed and
// normed output, has no stream equivalent and is covered by the logits gate).
std::vector<float> deepseek_v4_forward(DeepSeekV4Model& model, int token_id,
                                       DeepSeekV4KVCache& kv_cache,
                                       DeepSeekV4mHCState& mhc, int& pos,
                                       std::vector<float>* layer_states = nullptr);

#endif
