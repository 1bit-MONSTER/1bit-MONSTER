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
        cfg.qk_nope_dim       = gu32("attention.qk_nope_head_dim", 128);
        cfg.qk_rope_dim       = gu32("attention.qk_rope_head_dim", 64);
        cfg.v_dim             = gu32("attention.v_head_dim", 128);
        cfg.kv_lora_rank      = gu32("attention.kv_lora_rank", 512);
        cfg.q_lora_rank       = gu32("attention.q_lora_rank", 0);  // 0 = uncompressed (V2-Lite)
        cfg.n_routed_experts  = gu32("attention.expert_count", 64);
        cfg.n_shared_experts  = gu32("feed_forward.moe.shared_expert_count", 2);
        cfg.top_k             = gu32("attention.expert_used_count", 6);
        cfg.moe_intermediate  = gu32("feed_forward.moe.intermediate_size", 1408);
        cfg.first_k_dense     = gu32("attention.first_k_dense_replace", 1);
        cfg.dense_intermediate = 0;  // set from the actual tensor shape below
        cfg.rms_norm_eps      = 1e-6f;
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
        ok &= get(p + "attn_kv_b.weight", l.w_kv_b, (size_t)cfg.kv_lora_rank * NH * KVB);
        ok &= get(p + "attn_output.weight", l.w_o, (size_t)H * NH * cfg.v_dim);

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

void DeepSeekModel::clear() {
    token_emb.clear(); final_norm_w.clear(); output_w.clear();
    for (auto& l : layers) {
        l.rms_attn_w.clear(); l.rms_ffn_w.clear();
        l.w_q.clear(); l.w_kv_a.clear(); l.w_kv_a_norm.clear(); l.w_kv_b.clear(); l.w_o.clear();
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
    std::vector<float> dense_gate(cfg.dense_intermediate > 0 ? cfg.dense_intermediate : 10944);
    std::vector<float> expert_gate(MOE), expert_up(MOE), expert_down(H);
    std::vector<float> router_probs(NE), expert_wts(TOPK);
    int expert_ids[64];

    for (int il = 0; il < cfg.num_layers; il++) {
        auto& l = model.layers[il];

        // ── MLA attention ──
        rmsnorm(norm.data(), x.data(), l.rms_attn_w.data(), H, cfg.rms_norm_eps);
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
        // Residual: attn output projected and added to the pre-attention x.
        {
            std::vector<float> attn_proj(H);
            matmul(attn_proj.data(), attn_out.data(), l.w_o.data(), H, NH * VD);
            for (int i = 0; i < H; i++) x[i] += attn_proj[i];
        }

        // ── FFN ──
        rmsnorm(norm.data(), x.data(), l.rms_ffn_w.data(), H, cfg.rms_norm_eps);
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
                x[i] += s;
            }
        } else {
            // MoE
            matmul(router_probs.data(), norm.data(), l.w_gate.data(), NE, H);
            softmax_inplace(router_probs.data(), NE);
            for (int k = 0; k < TOPK; k++) {
                int best = 0; float bv = -1e30f;
                for (int i = 0; i < NE; i++) if (router_probs[i] > bv) { bv = router_probs[i]; best = i; }
                expert_ids[k] = best; expert_wts[k] = bv; router_probs[best] = -1e30f;
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
            for (int i = 0; i < H; i++) x[i] += shared_down[i] + expert_down[i];
        }

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
