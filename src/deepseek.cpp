// deepseek.cpp — DeepSeek-V2-Lite MLA implementation (rewritten 2026-08-15)
//
// Layout matches the real llama.cpp deepseek2 GGUF (verified against
// mradermacher/DeepSeek-V2-Lite.Q8_0, the manifest's e2e oracle):
//   - uncompressed Q: attn_q [H, n_heads*(qk_nope+qk_rope)], per head [nope|rope]
//   - MLA KV: attn_kv_a_mqa [H, kv_lora+qk_rope] (latent + per-position rope),
//     attn_kv_a_norm (RMSNorm on the latent), attn_kv_b [kv_lora, n_heads*(nope+v)]
//   - MoE: ffn_gate_inp router, ffn_{gate,up,down}_exps [H|moe_int, moe_int|H, experts]
//     (experts INNERMOST), ffn_{gate,up}_shexp + ffn_down_shexp with the
//     n_shared experts fused along the intermediate dim
//   - first_k_dense_replace=1 -> layer 0 is a DENSE FFN (ffn_gate/up/down)
// Memory: routed experts are dequantized once at load and kept as f16
// (transposed to [expert, H, moe_int]) — Q8->f32 would be ~63GB, f16 is ~29GB.

#include "deepseek.h"
#include "gguf_reader.h"
#include "../engine/npu/src/onebp_loader.cpp"
#include <cstring>

static inline float f16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16, exp = (h & 0x7C00u) >> 10;
    uint32_t man = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) { bits = sign | (man << 13); }
    else if (exp == 0x1F) { bits = sign | 0x7F800000u | (man << 13); }
    else { bits = sign | ((exp + 112) << 23) | (man << 13); }
    float f; memcpy(&f, &bits, 4); return f;
}

static inline uint16_t f32_to_f16(float f) {
    uint32_t bits; memcpy(&bits, &f, 4);
    uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = (int32_t)((bits >> 23) & 0xFF) - 127 + 15;
    uint32_t man = (bits >> 13) & 0x3FFu;
    if (((bits >> 23) & 0xFF) == 0xFF) return (uint16_t)(sign | 0x7C00u | man);  // inf/nan
    if (exp >= 0x1F) return (uint16_t)(sign | 0x7C00u);
    if (exp <= 0) { if (exp < -10) return (uint16_t)sign; man |= 0x400; int shift = 14 - exp; man >>= shift; return (uint16_t)(sign | man); }
    return (uint16_t)(sign | ((uint32_t)exp << 10) | man);
}

bool DeepSeekModel::load_from_gguf(const std::string& path, const DeepSeekConfig* override_cfg) {
    GgufReader r;
    if (!r.open(path)) {
        fprintf(stderr, "[deepseek] FAIL: could not open %s\n", path.c_str());
        return false;
    }
    auto gu32 = [&](const std::string& key, int def) -> int {
        uint32_t v; if (r.get_u32(key, v)) return (int)v;
        std::string arch = r.architecture();
        if (!arch.empty() && r.get_u32(arch + "." + key, v)) return (int)v;
        return def;
    };
    if (override_cfg) cfg = *override_cfg;
    else {
        cfg.hidden_size       = gu32("embedding_length", 2048);
        cfg.num_layers        = gu32("block_count", 27);
        cfg.num_heads         = gu32("attention.head_count", 16);
        cfg.num_kv_heads      = gu32("attention.head_count_kv", 16);
        cfg.head_dim          = gu32("attention.key_length", 128);
        cfg.vocab_size        = gu32("vocab_size", 102400);
        cfg.max_seq_len       = gu32("context_length", 4096);
        cfg.qk_nope_dim       = gu32("attention.qk_nope_head_dim", 0);
        cfg.qk_rope_dim       = gu32("attention.qk_rope_head_dim", 0);
        cfg.v_dim             = gu32("attention.v_head_dim", 0);
        // Fallback (llama.cpp deepseek2 convention): derive from the mla keys.
        // qk_rope = rope.dimension_count; qk_nope = key_length_mla - qk_rope;
        // v_dim = value_length_mla.
        if (!cfg.qk_nope_dim || !cfg.qk_rope_dim || !cfg.v_dim) {
            int kl_mla = gu32("attention.key_length_mla", 0);
            int vl_mla = gu32("attention.value_length_mla", 0);
            int n_rot  = gu32("rope.dimension_count", 0);
            if (!cfg.qk_rope_dim) cfg.qk_rope_dim = n_rot;
            if (!cfg.qk_nope_dim) cfg.qk_nope_dim = kl_mla - cfg.qk_rope_dim;
            if (!cfg.v_dim)       cfg.v_dim       = vl_mla;
        }
        if (!cfg.qk_nope_dim || !cfg.qk_rope_dim || !cfg.v_dim) {
            // last resort: qk_nope/qk_rope split like DeepSeek-V2 defaults
            if (!cfg.qk_rope_dim) cfg.qk_rope_dim = cfg.head_dim / 2;
            if (!cfg.qk_nope_dim) cfg.qk_nope_dim = cfg.head_dim - cfg.qk_rope_dim;
            if (!cfg.v_dim)       cfg.v_dim       = cfg.head_dim;
        }
        cfg.kv_lora_rank      = gu32("attention.kv_lora_rank", 512);
        cfg.q_lora_rank       = gu32("attention.q_lora_rank", 0);  // 0 = uncompressed (V2-Lite)
        cfg.n_routed_experts  = gu32("expert_count", 64);
        cfg.n_shared_experts  = gu32("expert_shared_count", 2);
        cfg.top_k             = gu32("expert_used_count", 6);
        cfg.moe_intermediate  = gu32("expert_feed_forward_length", 1408);
        cfg.first_k_dense     = gu32("attention.first_k_dense_replace", 1);
        cfg.dense_intermediate = 0;  // set from the actual tensor shape below
        cfg.rms_norm_eps      = 1e-6f;
        // Instella trained-in bits (deepseek2.* KVs, emitted by the fork's converter)
        cfg.gated_attention   = gu32("attention.gated_attention", 0);
        cfg.score_func        = gu32("expert_gating_func", 0);  // 0=none 1=softmax 2=sigmoid (llama enum)
        cfg.norm_topk_prob    = gu32("expert_weights_norm", 0);
        float rs_f = 0;
        if (r.get_f32("deepseek2.expert_weights_scale", rs_f) || r.get_f32("expert_weights_scale", rs_f))
            cfg.routed_scaling = rs_f;
        if (cfg.routed_scaling <= 0.0f) cfg.routed_scaling = 1.0f;
        if (r.get_u32("deepseek2.instella_farskip", (uint32_t&)cfg.farskip)) {
            cfg.farskip_start = gu32("deepseek2.instella_farskip_start", 0);
            cfg.farskip_end   = gu32("deepseek2.instella_farskip_end", 100000);
        }
    }
    int H = cfg.hidden_size, NH = cfg.num_heads;
    int QKD = cfg.qk_nope_dim + cfg.qk_rope_dim;          // per-head q dim
    int KVB = cfg.qk_nope_dim + cfg.v_dim;                // per-head kv_b output dim

    auto get = [&](const std::string& name, std::vector<float>& dst, size_t expect) -> bool {
        size_t n = 0;
        if (!r.get_tensor_f32(name, dst, &n)) { fprintf(stderr, "  [deepseek] missing: %s\n", name.c_str()); return false; }
        if (expect > 0 && n != expect) { fprintf(stderr, "  [deepseek] %s: expected %zu, got %zu\n", name.c_str(), expect, n); return false; }
        return true;
    };

    if (!get("token_embd.weight", token_emb, (size_t)cfg.vocab_size * H)) return false;
    if (!get("output_norm.weight", final_norm_w, (size_t)H)) get("final_norm.weight", final_norm_w, (size_t)H);
    if (final_norm_w.empty()) final_norm_w.resize(H, 1.0f);
    get("output.weight", output_w, (size_t)cfg.vocab_size * H);

    layers.resize(cfg.num_layers);
    for (int il = 0; il < cfg.num_layers; il++) {
        auto& l = layers[il];
        std::string p = "blk." + std::to_string(il) + ".";
        bool ok = true;
        ok &= get(p + "attn_norm.weight", l.rms_attn_w, (size_t)H);
        ok &= get(p + "ffn_norm.weight",  l.rms_ffn_w,  (size_t)H);
        ok &= get(p + "attn_q.weight", l.w_q, (size_t)H * NH * QKD);
        ok &= get(p + "attn_kv_a_mqa.weight", l.w_kv_a, (size_t)H * (cfg.kv_lora_rank + cfg.qk_rope_dim));
        ok &= get(p + "attn_kv_a_norm.weight", l.w_kv_a_norm, (size_t)cfg.kv_lora_rank);
        // kv_b comes in two on-disk forms:
        //   legacy 2D  [kv_lora_rank, NH*(qk_nope+v_dim)] fused (attn_kv_b.weight)
        //   current   split 3D (llama.cpp post-refactor; Instella's converter
        //             emits this): attn_k_b [qk_nope, kv_lora, NH] +
        //             attn_v_b [kv_lora, v_dim, NH]. GGUF ne[] has ne0
        //             contiguous (fastest), so element (a,b,c) sits at
        //             a + ne0*b + ne0*ne1*c. Normalize to the fused 2D layout
        //             [kv_lora, NH*(nope+v)] so the forward is shared.
        {
            size_t kvb_n = (size_t)cfg.kv_lora_rank * NH * KVB;
            if (!get(p + "attn_kv_b.weight", l.w_kv_b, kvb_n)) {
                std::vector<float> kb, vb;
                bool okb = true;
                okb &= get(p + "attn_k_b.weight", kb, (size_t)cfg.qk_nope_dim * cfg.kv_lora_rank * NH);
                okb &= get(p + "attn_v_b.weight", vb, (size_t)cfg.v_dim * cfg.kv_lora_rank * NH);
                if (okb) {
                    l.w_kv_b.assign(kvb_n, 0.0f);
                    // attn_k_b [qk_nope, kv_lora, NH]: element (d,j,h) at
                    // d + qk_nope*j + qk_nope*kv_lora*h. k_nope[h][d] =
                    // sum_j latent[j] * k_b[d,j,h].
                    for (int h = 0; h < NH; h++)
                        for (int d = 0; d < cfg.qk_nope_dim; d++) {
                            float* dst = l.w_kv_b.data() + ((size_t)h * KVB + d) * cfg.kv_lora_rank;
                            for (int j = 0; j < cfg.kv_lora_rank; j++)
                                dst[j] = kb[(size_t)d + (size_t)cfg.qk_nope_dim * j +
                                            (size_t)cfg.qk_nope_dim * cfg.kv_lora_rank * h];
                        }
                    // attn_v_b [kv_lora, v_dim, NH]: element (j,d,h) at
                    // j + kv_lora*d + kv_lora*v_dim*h. v[h][d] =
                    // sum_j latent[j] * v_b[j,d,h].
                    for (int h = 0; h < NH; h++)
                        for (int d = 0; d < cfg.v_dim; d++) {
                            float* dst = l.w_kv_b.data() + ((size_t)h * KVB + cfg.qk_nope_dim + d) * cfg.kv_lora_rank;
                            for (int j = 0; j < cfg.kv_lora_rank; j++)
                                dst[j] = vb[(size_t)j + (size_t)cfg.kv_lora_rank * d +
                                            (size_t)cfg.kv_lora_rank * cfg.v_dim * h];
                        }
                } else {
                    ok = false;
                }
            }
        }
        ok &= get(p + "attn_output.weight", l.w_o, (size_t)H * NH * cfg.v_dim);
        // Instella gated MLA: gate_proj [NH*v_dim, H] applied to the PRE-NORM
        // attention input (sigmoid), before o_proj. The GGUF has no
        // gated_attention KV — presence of the attn_gate tensor is the signal.
        // Enable per-layer only where the tensor exists (TENSOR_NOT_REQUIRED
        // semantics in the fork).
        if (r.tensor_info(p + "attn_gate.weight")) {
            cfg.gated_attention = 1;
            ok &= get(p + "attn_gate.weight", l.w_attn_gate, (size_t)NH * cfg.v_dim * H);
        }

        if (il < cfg.first_k_dense) {
            // Dense FFN (layer 0): intermediate from the actual tensor.
            auto* dti = r.tensor_info(p + "ffn_gate.weight");
            int DI = (dti && dti->shape.size() == 2) ? (int)dti->shape[1] : 10944;
            cfg.dense_intermediate = DI;
            ok &= get(p + "ffn_gate.weight", l.d_gate, (size_t)H * DI);
            ok &= get(p + "ffn_up.weight",   l.d_up,   (size_t)H * DI);
            ok &= get(p + "ffn_down.weight", l.d_down, (size_t)DI * H);
        } else {
            // MoE layer
            ok &= get(p + "ffn_gate_inp.weight", l.w_gate, (size_t)H * cfg.n_routed_experts);
            get(p + "exp_probs_b.bias", l.w_exp_probs_b, (size_t)cfg.n_routed_experts);  // e_score_correction_bias (optional)
            ok &= get(p + "ffn_gate_shexp.weight", l.w_shared_gate, (size_t)H * cfg.n_shared_experts * cfg.moe_intermediate);
            ok &= get(p + "ffn_up_shexp.weight",   l.w_shared_up,   (size_t)H * cfg.n_shared_experts * cfg.moe_intermediate);
            ok &= get(p + "ffn_down_shexp.weight", l.w_shared_down, (size_t)cfg.n_shared_experts * cfg.moe_intermediate * H);

            // Routed experts: GGUF ne=[{A},{B},{NE}] with {A} INNERMOST (blocks
            // of 32 along A), expert OUTERMOST (llama.cpp deepseek2 convention
            // {n_embd, n_ff, n_expert}). Dequant -> f16 [experts, A, B].
            auto load_exps = [&](const char* tname, std::vector<uint16_t>& dst,
                                 int A, int B, int NE) -> bool {
                std::string full = p + tname;
                auto* ti = r.tensor_info(full);
                if (!ti) { fprintf(stderr, "  [deepseek] missing: %s\n", full.c_str()); return false; }
                GgufBlockInfo bi = gguf_block_info(ti->dtype);
                if (bi.block_size <= 0 || bi.block_bytes <= 0) { fprintf(stderr, "  [deepseek] %s: unsupported dtype\n", full.c_str()); return false; }
                std::vector<uint8_t> raw;
                uint64_t numel = 0;
                if (!r.get_tensor_raw(full, bi.block_size, bi.block_bytes, raw, &numel)) { fprintf(stderr, "  [deepseek] raw read fail: %s\n", full.c_str()); return false; }
                if ((uint64_t)A * B * NE != numel) { fprintf(stderr, "  [deepseek] %s: %llu elems, want %d\n", full.c_str(), (unsigned long long)numel, A * B * NE); return false; }
                dst.resize((size_t)A * B * NE);
                int nblocks = (A + bi.block_size - 1) / bi.block_size;  // blocks along A
                std::vector<float> dq(bi.block_size);
                const uint8_t* rawp = raw.data();
                for (int e = 0; e < NE; e++) {
                    for (int b = 0; b < B; b++) {
                        const uint8_t* row = rawp + ((size_t)(e * B + b) * nblocks) * bi.block_bytes;
                        for (int nb = 0; nb < nblocks; nb++) {
                            int cnt = std::min(bi.block_size, A - nb * bi.block_size);
                            if (!gguf_dequant(ti->dtype, row + (size_t)nb * bi.block_bytes, dq.data(), cnt)) return false;
                            int base = nb * bi.block_size;
                            for (int j = 0; j < cnt; j++)
                                dst[(size_t)e * A * B + (size_t)b * A + base + j] = f32_to_f16(dq[j]);
                        }
                    }
                }
                return true;
            };
            ok &= load_exps("ffn_gate_exps.weight", l.exp_gate, H, cfg.moe_intermediate, cfg.n_routed_experts);
            ok &= load_exps("ffn_up_exps.weight",   l.exp_up,   H, cfg.moe_intermediate, cfg.n_routed_experts);
            ok &= load_exps("ffn_down_exps.weight", l.exp_down, cfg.moe_intermediate, H, cfg.n_routed_experts);
        }
        if (!ok) { fprintf(stderr, "  [deepseek] layer %d incomplete\n", il); return false; }
    }
    fprintf(stderr, "[deepseek] loaded: %s (%d layers, H=%d, MLA kv_lora=%d rope=%d, experts=%d top_k=%d shared=%d dense_first=%d)\n",
            r.architecture().c_str(), cfg.num_layers, H, cfg.kv_lora_rank, cfg.qk_rope_dim,
            cfg.n_routed_experts, cfg.top_k, cfg.n_shared_experts, cfg.first_k_dense);
    return true;
}

bool DeepSeekModel::load_from_1bp(const std::string& path, const DeepSeekConfig* override_cfg) {
    // Load the same tensor set as load_from_gguf, but from a .1bp ternary file.
    // The 1bp loader (NpuOnebpModel) dequantizes tiles to f32; ndim==3 MLA
    // tensors (attn_k_b/attn_v_b) are stored as per-head "expert" stacks in
    // the converter (ne = shape[0]) — read them back and renormalize to the
    // fused 2D layout exactly like the GGUF split path.
    NpuOnebpModel m;
    if (!m.open(path.c_str())) {
        fprintf(stderr, "[deepseek] FAIL: could not open %s\n", path.c_str());
        return false;
    }
    const auto& h = m.header();
    if (h.arch != ONEBP_DEEPSEEK2) {
        fprintf(stderr, "[deepseek] %s: arch=%u (want ONEBP_DEEPSEEK2=20)\n", path.c_str(), (unsigned)h.arch);
        return false;
    }
    if (override_cfg) cfg = *override_cfg;
    else {
        cfg.hidden_size       = h.hidden_size;
        cfg.num_layers        = h.num_layers;
        cfg.num_heads         = h.num_attention_heads;
        cfg.num_kv_heads      = h.num_kv_heads ? h.num_kv_heads : h.num_attention_heads;
        cfg.head_dim          = h.head_dim;
        cfg.vocab_size        = h.vocab_size;
        cfg.max_seq_len       = h.max_seq_len ? h.max_seq_len : 4096;
        cfg.qk_nope_dim       = h.mla_qk_nope_dim;
        cfg.qk_rope_dim       = h.mla_qk_rope_dim;
        cfg.v_dim             = h.mla_v_dim;
        cfg.kv_lora_rank      = h.mla_kv_lora_rank;
        cfg.q_lora_rank       = 0;  // Instella/DeepSeek-V2-Lite: uncompressed Q
        cfg.n_routed_experts  = h.num_experts;
        cfg.n_shared_experts  = 1;  // refined from the fused shexp tensor below
        cfg.top_k             = h.n_expert_used;
        cfg.moe_intermediate  = h.n_ff_exp;   // per-expert FFN width (not intermediate_size)
        cfg.first_k_dense     = h.n_layer_dense_lead;
        cfg.rms_norm_eps      = 1e-6f;
        cfg.gated_attention   = h.mla_gated_attn;
        cfg.farskip           = h.mla_farskip;
        cfg.farskip_start     = h.mla_farskip_start;
        cfg.farskip_end       = h.mla_farskip_end;
        cfg.score_func        = h.expert_gating_func;  // llama enum: 2=sigmoid
        cfg.norm_topk_prob    = h.expert_weights_norm;
        cfg.routed_scaling    = h.expert_weights_scale();
        // shared experts: n_ff_shexp is the FUSED intermediate (n_shared * moe_int)
        if (h.n_ff_shexp > 0 && cfg.moe_intermediate > 0)
            cfg.n_shared_experts = std::max(1, (int)h.n_ff_shexp / cfg.moe_intermediate);
    }
    int H = cfg.hidden_size, NH = cfg.num_heads;
    int QKD = cfg.qk_nope_dim + cfg.qk_rope_dim;
    int KVB = cfg.qk_nope_dim + cfg.v_dim;

    auto get = [&](const char* name, std::vector<float>& dst, size_t expect) -> bool {
        if (!m.get_tensor_f32(name, dst)) { fprintf(stderr, "  [deepseek] missing: %s\n", name); return false; }
        if (expect > 0 && dst.size() != expect) { fprintf(stderr, "  [deepseek] %s: expected %zu, got %zu\n", name, expect, dst.size()); return false; }
        return true;
    };

    if (!get("token_embd.weight", token_emb, (size_t)cfg.vocab_size * H)) return false;
    get("output_norm.weight", final_norm_w, (size_t)H);
    if (final_norm_w.empty()) final_norm_w.resize(H, 1.0f);
    get("output.weight", output_w, (size_t)cfg.vocab_size * H);

    layers.resize(cfg.num_layers);
    for (int il = 0; il < cfg.num_layers; il++) {
        auto& l = layers[il];
        std::string p = "blk." + std::to_string(il) + ".";
        bool ok = true;
        ok &= get((p + "attn_norm.weight").c_str(), l.rms_attn_w, (size_t)H);
        ok &= get((p + "ffn_norm.weight").c_str(),  l.rms_ffn_w,  (size_t)H);
        ok &= get((p + "attn_q.weight").c_str(), l.w_q, (size_t)H * NH * QKD);
        ok &= get((p + "attn_kv_a_mqa.weight").c_str(), l.w_kv_a, (size_t)H * (cfg.kv_lora_rank + cfg.qk_rope_dim));
        ok &= get((p + "attn_kv_a_norm.weight").c_str(), l.w_kv_a_norm, (size_t)cfg.kv_lora_rank);
        // kv_b: stored as 2D flat [shape[0], shape[1]*shape[2]] (ne0-contiguous,
        // same order as the GGUF reader — verified 2026-08-16). Normalize to
        // the fused [kv_lora, NH*(nope+v)] layout.
        {
            size_t kvb_n = (size_t)cfg.kv_lora_rank * NH * KVB;
            if (!get((p + "attn_kv_b.weight").c_str(), l.w_kv_b, kvb_n)) {
                std::vector<float> kb, vb;
                bool okb = true;
                okb &= get((p + "attn_k_b.weight").c_str(), kb, (size_t)cfg.qk_nope_dim * cfg.kv_lora_rank * NH);
                okb &= get((p + "attn_v_b.weight").c_str(), vb, (size_t)cfg.v_dim * cfg.kv_lora_rank * NH);
                if (okb) {
                    l.w_kv_b.assign(kvb_n, 0.0f);
                    // kb flat = ne0-contiguous [qk_nope, kv_lora, NH]: element
                    // (d, j, h) at d + qk_nope*j + qk_nope*kv_lora*h.
                    for (int h = 0; h < NH; h++)
                        for (int d = 0; d < cfg.qk_nope_dim; d++) {
                            float* dst = l.w_kv_b.data() + ((size_t)h * KVB + d) * cfg.kv_lora_rank;
                            for (int j = 0; j < cfg.kv_lora_rank; j++)
                                dst[j] = kb[(size_t)d + (size_t)cfg.qk_nope_dim * j +
                                            (size_t)cfg.qk_nope_dim * cfg.kv_lora_rank * h];
                        }
                    // vb flat = ne0-contiguous [kv_lora, v_dim, NH]: element
                    // (j, d, h) at j + kv_lora*d + kv_lora*v_dim*h.
                    for (int h = 0; h < NH; h++)
                        for (int d = 0; d < cfg.v_dim; d++) {
                            float* dst = l.w_kv_b.data() + ((size_t)h * KVB + cfg.qk_nope_dim + d) * cfg.kv_lora_rank;
                            for (int j = 0; j < cfg.kv_lora_rank; j++)
                                dst[j] = vb[(size_t)j + (size_t)cfg.kv_lora_rank * d +
                                            (size_t)cfg.kv_lora_rank * cfg.v_dim * h];
                        }
                } else {
                    ok = false;
                }
            }
        }
        ok &= get((p + "attn_output.weight").c_str(), l.w_o, (size_t)H * NH * cfg.v_dim);
        if (cfg.gated_attention)
            ok &= get((p + "attn_gate.weight").c_str(), l.w_attn_gate, (size_t)NH * cfg.v_dim * H);

        if (il < cfg.first_k_dense) {
            if (auto* te = m.find_tensor((p + "ffn_gate.weight").c_str())) cfg.dense_intermediate = te->rows;
            ok &= get((p + "ffn_gate.weight").c_str(), l.d_gate, (size_t)H * cfg.dense_intermediate);
            ok &= get((p + "ffn_up.weight").c_str(),   l.d_up,   (size_t)H * cfg.dense_intermediate);
            ok &= get((p + "ffn_down.weight").c_str(), l.d_down, (size_t)cfg.dense_intermediate * H);
        } else {
            ok &= get((p + "ffn_gate_inp.weight").c_str(), l.w_gate, (size_t)H * cfg.n_routed_experts);
            get((p + "exp_probs_b.bias").c_str(), l.w_exp_probs_b, (size_t)cfg.n_routed_experts);
            ok &= get((p + "ffn_gate_shexp.weight").c_str(), l.w_shared_gate, (size_t)H * cfg.n_shared_experts * cfg.moe_intermediate);
            ok &= get((p + "ffn_up_shexp.weight").c_str(),   l.w_shared_up,   (size_t)H * cfg.n_shared_experts * cfg.moe_intermediate);
            ok &= get((p + "ffn_down_shexp.weight").c_str(), l.w_shared_down, (size_t)cfg.n_shared_experts * cfg.moe_intermediate * H);
            auto load_exps = [&](const char* tname, std::vector<uint16_t>& dst, int A, int B, int NE) -> bool {
                std::string full = p + tname;
                auto* te = m.find_tensor(full.c_str());
                if (!te) { fprintf(stderr, "  [deepseek] missing: %s\n", full.c_str()); return false; }
                dst.resize((size_t)A * B * NE);
                for (int e = 0; e < NE; e++) {
                    std::vector<float> w;
                    if (!m.get_tensor_f32_expert(full.c_str(), e, w)) { fprintf(stderr, "  [deepseek] expert fail: %s[%d]\n", full.c_str(), e); return false; }
                    if (w.size() != (size_t)A * B) { fprintf(stderr, "  [deepseek] %s[%d]: %zu elems, want %d\n", full.c_str(), e, w.size(), A * B); return false; }
                    for (size_t i = 0; i < w.size(); i++) dst[(size_t)e * A * B + i] = f32_to_f16(w[i]);
                }
                return true;
            };
            ok &= load_exps("ffn_gate_exps.weight", l.exp_gate, H, cfg.moe_intermediate, cfg.n_routed_experts);
            ok &= load_exps("ffn_up_exps.weight",   l.exp_up,   H, cfg.moe_intermediate, cfg.n_routed_experts);
            ok &= load_exps("ffn_down_exps.weight", l.exp_down, cfg.moe_intermediate, H, cfg.n_routed_experts);
        }
        if (!ok) { fprintf(stderr, "  [deepseek] layer %d incomplete\n", il); return false; }
    }
    fprintf(stderr, "[deepseek] loaded 1bp: %s (%d layers, H=%d, MLA kv_lora=%d rope=%d, experts=%d top_k=%d shared=%d dense_first=%d, gated=%d farskip=%d)\n",
            path.c_str(), cfg.num_layers, H, cfg.kv_lora_rank, cfg.qk_rope_dim,
            cfg.n_routed_experts, cfg.top_k, cfg.n_shared_experts, cfg.first_k_dense,
            cfg.gated_attention, cfg.farskip);
    return true;
}

void DeepSeekModel::clear() {
    token_emb.clear(); final_norm_w.clear(); output_w.clear();
    for (auto& l : layers) {
        l.rms_attn_w.clear(); l.rms_ffn_w.clear();
        l.w_q.clear(); l.w_kv_a.clear(); l.w_kv_a_norm.clear(); l.w_kv_b.clear(); l.w_o.clear();
        l.w_attn_gate.clear(); l.w_exp_probs_b.clear();
        l.d_gate.clear(); l.d_up.clear(); l.d_down.clear();
        l.w_gate.clear(); l.w_shared_gate.clear(); l.w_shared_up.clear(); l.w_shared_down.clear();
        l.exp_gate.clear(); l.exp_up.clear(); l.exp_down.clear();
    }
    layers.clear();
}

using namespace deepseek_math;

std::vector<float> deepseek_forward(
    const DeepSeekModel& model, int token_id,
    std::vector<std::vector<float>>& mla_kv_cache,
    int& pos)
{
    const auto& cfg = model.cfg;
    int H = cfg.hidden_size, NH = cfg.num_heads;
    int NOPE = cfg.qk_nope_dim, ROPE = cfg.qk_rope_dim, VD = cfg.v_dim;
    int QKD = NOPE + ROPE, KVB = NOPE + VD;
    int MOE = cfg.moe_intermediate, NE = cfg.n_routed_experts, TOPK = cfg.top_k;
    float scale = 1.0f / sqrtf((float)QKD);

    std::vector<float> x(H);
    if (token_id >= 0 && token_id < cfg.vocab_size)
        for (int i = 0; i < H; i++) x[i] = model.token_emb[(size_t)token_id * H + i];

    std::vector<float> norm(H), q((size_t)NH * QKD), kv(cfg.kv_lora_rank + ROPE);
    std::vector<float> latent(cfg.kv_lora_rank), k_rope(ROPE), q_rope(ROPE);
    std::vector<float> k_nope((size_t)NH * NOPE), v((size_t)NH * VD), attn_out((size_t)NH * VD);
    std::vector<float> scores(cfg.max_seq_len);
    std::vector<float> shared_gate(MOE), shared_up(MOE), shared_down(H);
    // dense_gate covers BOTH the dense-layer FFN (dense_intermediate) and the
    // shared-expert FFN (n_shared * moe_intermediate) — they share the scratch.
    int scratch_di = std::max(cfg.dense_intermediate > 0 ? cfg.dense_intermediate : 10944,
                              cfg.n_shared_experts * MOE);
    std::vector<float> dense_gate(scratch_di);
    std::vector<float> expert_gate(MOE), expert_up(MOE), expert_down(H);
    std::vector<float> router_probs(NE), expert_wts(TOPK);
    int expert_ids[64];

    // Instella FarSkip dual-residual streams (verified against the llama.cpp
    // fork patch a897ea2f1 + HF modeling_instella_moe.py):
    //   main_x = main stream  (residual + routed, feeds MLP norm)
    //   alt_x  = alt stream   (routed-free, feeds ATTENTION)
    // fork: ffn_inp = attn_out + inpL(main); farskip FFN norm reads inpL
    // (main WITHOUT this layer's attn); alt' = ffn_inp + shared;
    // main' = alt' + routed.
    std::vector<float> main_x(H), alt_x(H);
    const bool fs = cfg.farskip;
    auto in_fs = [&](int il) { return il >= cfg.farskip_start && il <= cfg.farskip_end; };

    // attention/FFN scratch shared across layers
    std::vector<float> attn_proj(H), gate_proj_out(H);

    for (int il = 0; il < cfg.num_layers; il++) {
        auto& l = model.layers[il];
        const bool layer_fs = fs && in_fs(il);

        // ── FarSkip stream select ──
        // x is the MAIN stream entering this layer. Attention reads alt
        // (routed-free); on the first farskip layer alt is undefined so
        // attention reads x itself (fork: inpL_alt == nullptr -> inpSA = inpL).
        std::vector<float> attn_inp;
        if (layer_fs) {
            if (il == cfg.farskip_start || (il > 0 && !in_fs(il - 1))) {
                attn_inp = x;       // first farskip layer: single stream
            } else {
                attn_inp = alt_x;   // later farskip layers: routed-free stream
            }
            main_x = x;             // main stream for this layer = incoming x
        } else {
            attn_inp = x;
        }

        // ── MLA attention (on attn_inp) ──
        rmsnorm(norm.data(), attn_inp.data(), l.rms_attn_w.data(), H, cfg.rms_norm_eps);
        matmul(q.data(), norm.data(), l.w_q.data(), NH * QKD, H);
        matmul(kv.data(), norm.data(), l.w_kv_a.data(), cfg.kv_lora_rank + ROPE, H);
        memcpy(latent.data(), kv.data(), cfg.kv_lora_rank * sizeof(float));
        rmsnorm(latent.data(), latent.data(), l.w_kv_a_norm.data(), cfg.kv_lora_rank, cfg.rms_norm_eps);
        memcpy(k_rope.data(), kv.data() + cfg.kv_lora_rank, ROPE * sizeof(float));
        rope(k_rope.data(), ROPE, pos);

        // Decompress K_nope / V from the latent. kv_b is [out=NH*KVB][in=kv_lora]
        // (element (h*KVB + d, j) at (h*KVB + d)*kv_lora + j).
        for (int h = 0; h < NH; h++) {
            for (int d = 0; d < NOPE; d++) {
                float s = 0;
                const float* row = l.w_kv_b.data() + (size_t)(h * KVB + d) * cfg.kv_lora_rank;
                for (int j = 0; j < cfg.kv_lora_rank; j++) s += latent[j] * row[j];
                k_nope[(size_t)h * NOPE + d] = s;
            }
            for (int d = 0; d < VD; d++) {
                float s = 0;
                const float* row = l.w_kv_b.data() + (size_t)(h * KVB + NOPE + d) * cfg.kv_lora_rank;
                for (int j = 0; j < cfg.kv_lora_rank; j++) s += latent[j] * row[j];
                v[(size_t)h * VD + d] = s;
            }
        }

        // KV cache (MLA: store decompressed per-head nope+v + shared k_rope).
        if ((int)mla_kv_cache.size() <= il) mla_kv_cache.resize(cfg.num_layers);
        auto& cache = mla_kv_cache[il];
        int per_pos = NH * NOPE + NH * VD + ROPE;
        if ((int)cache.size() < (pos + 1) * per_pos) cache.resize((size_t)(pos + 1) * per_pos);
        float* cp = cache.data() + (size_t)pos * per_pos;
        memcpy(cp, k_nope.data(), (size_t)NH * NOPE * sizeof(float));
        memcpy(cp + NH * NOPE, k_rope.data(), ROPE * sizeof(float));
        memcpy(cp + NH * NOPE + ROPE, v.data(), (size_t)NH * VD * sizeof(float));

        // Q rope part per head.
        for (int h = 0; h < NH; h++) {
            memcpy(q_rope.data(), q.data() + (size_t)h * QKD + NOPE, ROPE * sizeof(float));
            rope(q_rope.data(), ROPE, pos);
            memcpy(q.data() + (size_t)h * QKD + NOPE, q_rope.data(), ROPE * sizeof(float));
        }

        // Attention over cached positions.
        for (int h = 0; h < NH; h++) {
            const float* qn = q.data() + (size_t)h * QKD;
            const float* qr = qn + NOPE;
            for (int s = 0; s <= pos; s++) {
                const float* cs = cache.data() + (size_t)s * per_pos;
                const float* kn = cs;
                const float* kr = cs + NH * NOPE;
                float acc = 0;
                for (int d = 0; d < NOPE; d++) acc += qn[d] * kn[(size_t)h * NOPE + d];
                for (int d = 0; d < ROPE; d++) acc += qr[d] * kr[d];
                scores[s] = acc * scale;
            }
            softmax_inplace(scores.data(), pos + 1);
            const float* cv = cache.data() + (size_t)0 * per_pos + NH * NOPE + ROPE;
            for (int d = 0; d < VD; d++) {
                float acc = 0;
                for (int s = 0; s <= pos; s++)
                    acc += scores[s] * cv[(size_t)s * per_pos + (size_t)h * VD + d];
                attn_out[(size_t)h * VD + d] = acc;
            }
        }

        // Instella gated MLA: attn_out * sigmoid(gate_proj(pre_norm_input))
        // before o_proj. gate input = the PRE-NORM attention input (HF:
        // self_attn(gate_proj(hidden_states)) where hidden_states is the
        // normed input to attention).
        if (cfg.gated_attention) {
            matmul(gate_proj_out.data(), norm.data(), l.w_attn_gate.data(), NH * VD, H);
            for (int i = 0; i < NH * VD; i++)
                attn_out[i] *= 1.0f / (1.0f + expf(-gate_proj_out[i]));
        }

        // o_proj then residual: attn output lands on the MAIN stream.
        matmul(attn_proj.data(), attn_out.data(), l.w_o.data(), H, NH * VD);
        // o_proj then residual. ffn_inp = main + attn (the residual base);
        // for farskip the FFN NORM reads main WITHOUT attn (fork build_norm(inpL)),
        // for non-farskip it reads main+attn.
        matmul(attn_proj.data(), attn_out.data(), l.w_o.data(), H, NH * VD);
        std::vector<float> ffn_inp;
        if (layer_fs) {
            ffn_inp.resize(H);
            for (int i = 0; i < H; i++) ffn_inp[i] = main_x[i] + attn_proj[i];  // main + attn
            main_x = ffn_inp;   // main' = main + attn (attn lands on main)
        } else {
            ffn_inp = x;
            for (int i = 0; i < H; i++) { x[i] += attn_proj[i]; ffn_inp[i] = x[i]; }
        }

        // ── FFN ──
        // FarSkip: MLP consumes the outdated MAIN stream WITHOUT this layer's
        // attention output (fork: build_norm(inpL)); non-farskip reads ffn_inp.
        std::vector<float> ffn_norm_inp;
        if (layer_fs) {
            ffn_norm_inp = main_x;
            for (int i = 0; i < H; i++) ffn_norm_inp[i] -= attn_proj[i];  // main w/o attn
        } else {
            ffn_norm_inp = ffn_inp;
        }
        rmsnorm(norm.data(), ffn_norm_inp.data(), l.rms_ffn_w.data(), H, cfg.rms_norm_eps);
        if (il < cfg.first_k_dense) {
            // Dense FFN (layer 0). d_gate/d_up are [out=DI][in=H], d_down [out=H][in=DI].
            for (int i = 0; i < cfg.dense_intermediate; i++) {
                float g = 0, u = 0;
                const float* gr = l.d_gate.data() + (size_t)i * H;
                const float* ur = l.d_up.data()   + (size_t)i * H;
                for (int j = 0; j < H; j++) { g += norm[j] * gr[j]; u += norm[j] * ur[j]; }
                dense_gate[i] = silu(g) * u;
            }
            for (int i = 0; i < H; i++) {
                float s = 0;
                const float* dr = l.d_down.data() + (size_t)i * cfg.dense_intermediate;
                for (int j = 0; j < cfg.dense_intermediate; j++) s += dense_gate[j] * dr[j];
                if (layer_fs) main_x[i] += s;
                else x[i] += s;
            }
            if (layer_fs) alt_x = main_x;
        } else {
            // MoE
            matmul(router_probs.data(), norm.data(), l.w_gate.data(), NE, H);
            if (!l.w_exp_probs_b.empty())
                for (int i = 0; i < NE; i++) router_probs[i] += l.w_exp_probs_b[i];
            if (cfg.score_func == 2) {
                // Instella: sigmoid router (llama enum: 2=SIGMOID; DeepSeek-V3
                // scoring_func=sigmoid)
                for (int i = 0; i < NE; i++) router_probs[i] = 1.0f / (1.0f + expf(-router_probs[i]));
                for (int k = 0; k < TOPK; k++) {
                    int best = 0; float bv = -1e30f;
                    for (int i = 0; i < NE; i++) if (router_probs[i] > bv) { bv = router_probs[i]; best = i; }
                    expert_ids[k] = best; expert_wts[k] = bv; router_probs[best] = -1e30f;
                }
                // norm_topk_prob: sum-normalize the top-k weights
                if (cfg.norm_topk_prob) {
                    float ssum = 0; for (int k = 0; k < TOPK; k++) ssum += expert_wts[k];
                    if (ssum > 1e-9f) for (int k = 0; k < TOPK; k++) expert_wts[k] /= ssum;
                }
                if (cfg.routed_scaling != 1.0f)
                    for (int k = 0; k < TOPK; k++) expert_wts[k] *= cfg.routed_scaling;
            } else {
                softmax_inplace(router_probs.data(), NE);
                for (int k = 0; k < TOPK; k++) {
                    int best = 0; float bv = -1e30f;
                    for (int i = 0; i < NE; i++) if (router_probs[i] > bv) { bv = router_probs[i]; best = i; }
                    expert_ids[k] = best; expert_wts[k] = bv; router_probs[best] = -1e30f;
                }
            }
            std::fill(shared_down.begin(), shared_down.end(), 0.0f);
            // Shared experts: one fused [out=n_shared*MOE][in=H] MLP, down [out=H][in=n_shared*MOE].
            {
                const float* sg = l.w_shared_gate.data();
                const float* su = l.w_shared_up.data();
                const float* sd = l.w_shared_down.data();
                int SM = cfg.n_shared_experts * MOE;
                for (int i = 0; i < SM; i++) {
                    float g = 0, u = 0;
                    const float* gr = sg + (size_t)i * H;
                    const float* ur = su + (size_t)i * H;
                    for (int j = 0; j < H; j++) { g += norm[j] * gr[j]; u += norm[j] * ur[j]; }
                    dense_gate[i] = silu(g) * u;
                }
                for (int i = 0; i < H; i++) {
                    float s = 0;
                    const float* dr = sd + (size_t)i * SM;
                    for (int j = 0; j < SM; j++) s += dense_gate[j] * dr[j];
                    shared_down[i] = s;
                }
            }
            std::fill(expert_down.begin(), expert_down.end(), 0.0f);
            for (int k = 0; k < TOPK; k++) {
                int e = expert_ids[k];
                const uint16_t* eg = l.exp_gate.data() + (size_t)e * MOE * H;
                const uint16_t* eu = l.exp_up.data()   + (size_t)e * MOE * H;
                const uint16_t* ed = l.exp_down.data() + (size_t)e * H * MOE;
                for (int i = 0; i < MOE; i++) {
                    float g = 0, u = 0;
                    for (int j = 0; j < H; j++) {
                        g += norm[j] * f16_to_f32(eg[(size_t)i * H + j]);
                        u += norm[j] * f16_to_f32(eu[(size_t)i * H + j]);
                    }
                    expert_gate[i] = silu(g) * u;
                }
                for (int i = 0; i < H; i++) {
                    float s = 0;
                    for (int j = 0; j < MOE; j++) s += expert_gate[j] * f16_to_f32(ed[(size_t)i * MOE + j]);
                    expert_down[i] += s * expert_wts[k];
                }
            }
            if (layer_fs) {
                // FarSkip MoE: alt' = main + attn + shared ; main' = alt' + routed
                // main_x already = old main + attn (attn_proj landed above);
                // ffn_inp = the pre-FFN main stream (main WITHOUT this attn).
                // HF: residual = main + attn; residual_no_routed = residual +
                // shared; main = residual + routed.
                for (int i = 0; i < H; i++) {
                    float alt = ffn_inp[i] + shared_down[i];  // main+attn+shared (routed-free)
                    alt_x[i] = alt;
                    main_x[i] = alt + expert_down[i];         // main' = alt' + routed
                }
            } else {
                for (int i = 0; i < H; i++) x[i] += shared_down[i] + expert_down[i];
            }
        }

        if (getenv("DS_DUMP_ALL")) {
            FILE* df = fopen("/tmp/ds_all.txt", "a");
            if (df) { fprintf(df, "%s L%d main[0..3]: %.6f %.6f %.6f %.6f  alt[0..3]: %.6f %.6f %.6f %.6f\n",
                getenv("DS_TAG") ? getenv("DS_TAG") : "?",
                il, main_x[0],main_x[1],main_x[2],main_x[3], alt_x[0],alt_x[1],alt_x[2],alt_x[3]); fclose(df); }
        }
        // Hand the stream forward: after a farskip layer the running x for the
        // NEXT layer's non-farskip path is the main stream.
        if (layer_fs) x = main_x;
    }

    rmsnorm(norm.data(), x.data(), model.final_norm_w.data(), H, cfg.rms_norm_eps);
    std::vector<float> logits(cfg.vocab_size);
    if (!model.output_w.empty()) matmul(logits.data(), norm.data(), model.output_w.data(), cfg.vocab_size, H);
    else for (int i = 0; i < cfg.vocab_size; i++) {
        float s = 0; for (int j = 0; j < H; j++) s += norm[j] * model.token_emb[(size_t)i * H + j];
        logits[i] = s;
    }
    pos++;
    return logits;
}
