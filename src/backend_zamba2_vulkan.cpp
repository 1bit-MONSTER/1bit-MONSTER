// backend_zamba2_vulkan.cpp — Zamba2 on Vulkan via the ZINC C++ compute path.
//
// P1 scope: decode-only correctness. Mirrors the CPU reference
// (zamba2_engine.cpp / mamba2_kernels.cpp) op-for-op; every op reuses an
// existing zinc_cpp shader except four new ones (mamba2_scan, mamba_conv1d,
// group_rms_norm, mamba_rope) that implement the Mamba2 math exactly as the
// CPU code computes it. Validated token-for-token against backend_zamba2
// (CPU) by tools/validate_zamba2_vulkan.cpp.
//
// Gate: ZAMBA2_VK=1. Enabled per-model: the router lists zamba2_vulkan first
// for the zamba2 arch, and init() declines unless the env var is set, so
// existing behavior is unchanged without it.
//
// Layouts (all fp32, matching the CPU reference):
//   in_proj scratch  [ z(d_inner) | xBC(conv_dim) | dt(n_head) ] — conv1d and
//                     scan operate in-place on the xBC region.
//   params buffer    [ per layer: A_log(n_head) | D(n_head) | dt_bias(n_head)
//                      | ssm_state(layers*n_head*head_dim*d_state) ]
//   conv buffer      [ per layer: conv1d_w(d_conv*conv_dim) | conv1d_b(conv_dim)
//                      | conv_state(layers*(d_conv-1)*conv_dim) ]
//   kv cache         [ 2 | n_hybrid | max_seq | n_kv*hd ] — the exact layout
//                     zinc_cpp flash_attn.comp expects (h=n_hybrid).
#include "backend.h"
#include "vulkan_wrapper.h"
#include "compute_engine.h"
#include "zamba2_engine.h"
#include "gguf_zamba2_loader.cpp"  // house style: included directly (backend_zamba2.cpp)
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <memory>
#include <cstdlib>
#include <cmath>
#include <sys/stat.h>

// ── helpers ──────────────────────────────────────────────────────
static float fbits(uint32_t v) { float f; memcpy(&f, &v, 4); return f; }

static std::string z2v_resolve_shader_dir() {
    auto has_shaders = [](const std::string& d) -> bool {
        if (d.empty()) return false;
        struct stat st;
        return stat((d + "/embed.spv").c_str(), &st) == 0;
    };
    if (const char* env = std::getenv("ZINC_SHADER_DIR"); env && has_shaders(env))
        return env;
#ifdef ZINC_SHADER_DIR
    if (has_shaders(ZINC_SHADER_DIR)) return ZINC_SHADER_DIR;
#endif
    static const char* candidates[] = {
        "shaders",
        "build_cmake/zinc_cpp_build/shaders",
        "build/zinc_cpp_build/shaders",
        "engine/gpu/zinc_cpp/build/shaders",
        "engine/gpu/shaders",
        "/usr/share/1bit-systems/shaders",
        "/usr/local/share/1bit-systems/shaders",
    };
    for (const char* c : candidates)
        if (has_shaders(c)) return c;
    return "";
}

static const char* kZ2vRequiredShaders[] = {
    "embed", "gemv_f32", "rms_norm_mul", "copy_buffer", "vadd", "argmax",
    "flash_attn", "swiglu", "silu_mul", "gelu_mul",
    "mamba2_scan", "mamba_conv1d", "group_rms_norm", "mamba_rope",
};

// Upload fp32 host data to a device buffer via a staging copy.
static void upload_float_data(VkDevice dev, VkQueue queue, CommandPool& pool,
                              GpuBuffer& dst, const float* src, size_t count) {
    if (count == 0 || !dst) return;
    size_t bytes = count * sizeof(float);
    GpuBuffer staging(dev, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    float* mapped = (float*)staging.map();
    memcpy(mapped, src, bytes);
    staging.unmap();
    VkCommandBuffer cmd = pool.begin_once();
    VkBufferCopy copy = {0, 0, bytes};
    vkCmdCopyBuffer(cmd, staging.buffer(), dst.buffer(), 1, &copy);
    pool.submit_and_wait(cmd, queue);
}

// Zero a device buffer range (state / kv reset). 'off' is a byte offset —
// state regions live mid-buffer, after the per-layer params/weights.
static void fill_zero(VkDevice dev, VkQueue queue, CommandPool& pool,
                      VkBuffer buf, size_t bytes, VkDeviceSize off = 0) {
    VkCommandBuffer cmd = pool.begin_once();
    vkCmdFillBuffer(cmd, buf, off, bytes, 0u);
    pool.submit_and_wait(cmd, queue);
}

// ── per-layer GPU weights ────────────────────────────────────────
struct Z2vLayerGPU {
    GpuBuffer in_proj;    // [d_in_proj, d_model]
    GpuBuffer out_proj;   // [d_model, d_inner]
    GpuBuffer norm_w;     // [d_inner] group norm (layout [group_size, n_group])
    GpuBuffer input_norm; // [d_model]
    bool hybrid = false;
    // hybrid extras:
    GpuBuffer linear;     // [d_model, d_model]
    GpuBuffer qw, kw, vw, ow;      // attn projections
    GpuBuffer gate_w, up_w, down_w;  // FFN
    GpuBuffer pre_ff_norm, ffn_norm; // [d_model]
    GpuBuffer mamba_input_norm;      // [d_model] — hybrid mamba decoder input norm
};

// ── backend ──────────────────────────────────────────────────────
struct Zamba2VulkanBackend : Backend {
    std::unique_ptr<ZincEngine> engine_;
    std::unique_ptr<ComputeEngine> compute_;
    Zamba2Model model_;
    bool vulkan_ok_ = false;

    // global GPU buffers
    GpuBuffer embed;         // [vocab, d_model]
    GpuBuffer final_norm;    // [d_model]
    GpuBuffer params;        // A|D|dt_bias per layer + ssm states
    GpuBuffer conv_pack;     // conv1d_w|b per layer + conv states
    GpuBuffer kv_cache;      // [2, n_hybrid, max_seq, n_kv*hd]
    std::vector<Z2vLayerGPU> layers_;

    // scratch
    GpuBuffer hidden_, residual_, tmp_, projected_, in_proj_, y_inner_;
    GpuBuffer qkv_, attn_out_, gate_up_, act_, logits_;
    GpuBuffer embedding_, concat_;  // hybrid: embedding copy + [hidden|embed] concat

    // host-side conv circular offsets (one per layer, mod d_conv-1)
    std::vector<uint32_t> conv_off_;
    int pos_ = 0;

    Zamba2VulkanBackend() {
        type = BackendType::ZINC_GPU;
        name = "Zamba2 (Vulkan, ZINC C++)";
    }
    ~Zamba2VulkanBackend() override { destroy(); }

    bool init(const ModelConfig& model_cfg, const std::string& weights_dir) override {
        (void)weights_dir;
        cfg = model_cfg;
        destroy();

        if (!getenv("ZAMBA2_VK")) {
            fprintf(stderr, "Zamba2VK: disabled via unset ZAMBA2_VK — using HIP/CPU.\n");
            return false;
        }
        if (cfg.model_path.empty()) {
            fprintf(stderr, "Zamba2VK: no GGUF model path available\n");
            return false;
        }

        std::string shader_dir = z2v_resolve_shader_dir();
        if (shader_dir.empty()) {
            fprintf(stderr, "Zamba2VK: compiled Vulkan shaders not found — disabling.\n");
            return false;
        }
        {
            std::string missing;
            for (const char* s : kZ2vRequiredShaders) {
                struct stat st;
                if (stat((shader_dir + "/" + s + ".spv").c_str(), &st) != 0) {
                    if (!missing.empty()) missing += ", ";
                    missing += s;
                }
            }
            if (!missing.empty()) {
                fprintf(stderr, "Zamba2VK: missing shaders (%s) — disabling.\n",
                        missing.c_str());
                return false;
            }
        }

        // 1. Load weights to host (fp32) with the existing Zamba2 GGUF loader.
        std::string model_path = !cfg.model_path.empty() ? cfg.model_path : weights_dir;
        if (!load_zamba2_from_gguf(model_path, model_)) {
            fprintf(stderr, "Zamba2VK: failed to load model %s\n", model_path.c_str());
            return false;
        }
        Zamba2Config& c = model_.cfg;

        try {
            engine_ = std::make_unique<ZincEngine>();
            engine_->init(shader_dir, -1);
            compute_ = std::make_unique<ComputeEngine>(
                engine_->device(), engine_->queue(),
                engine_->queue_family(), *engine_->cmd_pool(),
                *engine_->pipeline_cache());
            VkDevice dev = engine_->device();
            VkQueue q = engine_->queue();
            CommandPool& pool = *engine_->cmd_pool();

            const size_t d_model = c.d_model, d_inner = c.d_inner;
            const size_t conv_dim = d_inner + 2 * (size_t)c.n_group * c.d_state;
            const size_t d_in_proj = d_inner + conv_dim + c.n_head;

            VkBufferUsageFlags rw = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                    VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            auto alloc = [&](GpuBuffer& b, size_t bytes, const char* tag) {
                b = GpuBuffer(dev, bytes, rw, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
                printf("  Z2V %-12s %6.1f MB\n", tag, bytes / (1024.0 * 1024.0));
            };

            // 2. Global weights + states.
            alloc(embed, (size_t)c.vocab_size * d_model * 4, "embed");
            alloc(final_norm, d_model * 4, "final_norm");
            alloc(params, ((size_t)3 * c.n_head * c.n_layers +
                           (size_t)c.n_layers * c.n_head * c.head_dim * c.d_state) * 4, "params+ssm");
            alloc(conv_pack, ((size_t)c.n_layers * (c.d_conv * conv_dim + conv_dim) +
                              (size_t)c.n_layers * (c.d_conv - 1) * conv_dim) * 4, "conv");
            alloc(kv_cache, (size_t)model_.num_hybrid_layers() * 2 * c.max_seq_len *
                            c.n_kv_heads * c.attn_head_dim * 4, "kv_cache");

            upload_float_data(dev, q, pool, embed, model_.embed_w.data(), model_.embed_w.size());
            upload_float_data(dev, q, pool, final_norm, model_.final_norm_w.data(), model_.final_norm_w.size());

            // params: per layer A_log | D | dt_bias, then zero ssm state.
            // Per-(head, head_dim) state slices (true Mamba2, #zamba2-validation).
            std::vector<float> params_host((size_t)3 * c.n_head * c.n_layers, 0.f);
            size_t st_off = params_host.size();
            params_host.resize(st_off + (size_t)c.n_layers * c.n_head * c.head_dim * c.d_state, 0.f);
            // conv_pack: per layer conv1d_w | conv1d_b, then zero conv state.
            size_t w_cnt = (size_t)c.d_conv * conv_dim;
            std::vector<float> conv_host((size_t)c.n_layers * (w_cnt + conv_dim), 0.f);
            size_t cs_off = conv_host.size();
            conv_host.resize(cs_off + (size_t)c.n_layers * (c.d_conv - 1) * conv_dim, 0.f);

            auto upload_layer = [&](int layer, const Mamba2LayerWeights& ml) {
                Z2vLayerGPU g;
                alloc(g.in_proj, d_in_proj * d_model * 4, "in_proj");
                alloc(g.out_proj, d_model * d_inner * 4, "out_proj");
                alloc(g.norm_w, (size_t)c.d_inner * 4, "norm_w");
                alloc(g.input_norm, d_model * 4, "input_norm");
                upload_float_data(dev, q, pool, g.in_proj, ml.in_proj_w.data(), ml.in_proj_w.size());
                upload_float_data(dev, q, pool, g.out_proj, ml.out_proj_w.data(), ml.out_proj_w.size());
                upload_float_data(dev, q, pool, g.norm_w, ml.norm_w.data(), ml.norm_w.size());
                upload_float_data(dev, q, pool, g.input_norm, ml.input_norm_w.data(), ml.input_norm_w.size());
                // packed conv: kernel + bias into the shared buffer
                memcpy(conv_host.data() + (size_t)layer * w_cnt, ml.conv1d_w.data(),
                       ml.conv1d_w.size() * 4);
                memcpy(conv_host.data() + (size_t)c.n_layers * w_cnt + (size_t)layer * conv_dim,
                       ml.conv1d_b.data(), ml.conv1d_b.size() * 4);
                // params: A | D | dt_bias
                size_t pl = (size_t)layer * 3 * c.n_head;
                memcpy(params_host.data() + pl, ml.A_log.data(), ml.A_log.size() * 4);
                memcpy(params_host.data() + pl + c.n_head, ml.D.data(), ml.D.size() * 4);
                memcpy(params_host.data() + pl + 2 * c.n_head, ml.dt_bias.data(), ml.dt_bias.size() * 4);
                return g;
            };

            layers_.resize(c.n_layers);
            for (int layer = 0; layer < c.n_layers; ++layer) {
                auto it = model_.hybrid_layers.find(layer);
                if (it != model_.hybrid_layers.end()) {
                    const HybridLayerWeights& hl = it->second;
                    Z2vLayerGPU g = upload_layer(layer, hl.mamba);
                    g.hybrid = true;
                    size_t n_at = (size_t)c.n_attn_heads * c.attn_head_dim;
                    size_t d_ff = hl.shared_transformer_up.size() / d_model;
                    // Attention input is 2*d_model (concat(hidden, embedding)) —
                    // the q/k/v weights are [n_at, 2*d_model]. The old d_model
                    // sizing silently overflowed the buffers (GPU fault).
                    size_t attn_in = 2 * d_model;
                    alloc(g.linear, d_model * d_model * 4, "linear");
                    alloc(g.qw, n_at * attn_in * 4, "q_w");
                    alloc(g.kw, n_at * attn_in * 4, "k_w");
                    alloc(g.vw, n_at * attn_in * 4, "v_w");
                    alloc(g.ow, n_at * d_model * 4, "o_w");
                    alloc(g.gate_w, d_ff * d_model * 4, "gate_w");
                    alloc(g.up_w, d_ff * d_model * 4, "up_w");
                    alloc(g.down_w, d_model * d_ff * 4, "down_w");
                    // post_attention_norm is the CONCAT norm: [2*d_model]
                    // (was alloc'd d_model → 4096-float upload overflow)
                    alloc(g.pre_ff_norm, attn_in * 4, "pre_ff");
                    alloc(g.ffn_norm, d_model * 4, "ffn_norm");
                    alloc(g.mamba_input_norm, d_model * 4, "mb_in_norm");
                    upload_float_data(dev, q, pool, g.linear, hl.linear_w.data(), hl.linear_w.size());
                    upload_float_data(dev, q, pool, g.qw, hl.shared_transformer_q.data(), hl.shared_transformer_q.size());
                    upload_float_data(dev, q, pool, g.kw, hl.shared_transformer_k.data(), hl.shared_transformer_k.size());
                    upload_float_data(dev, q, pool, g.vw, hl.shared_transformer_v.data(), hl.shared_transformer_v.size());
                    upload_float_data(dev, q, pool, g.ow, hl.shared_transformer_o.data(), hl.shared_transformer_o.size());
                    upload_float_data(dev, q, pool, g.gate_w, hl.shared_transformer_gate.data(), hl.shared_transformer_gate.size());
                    upload_float_data(dev, q, pool, g.up_w, hl.shared_transformer_up.data(), hl.shared_transformer_up.size());
                    upload_float_data(dev, q, pool, g.down_w, hl.shared_transformer_down.data(), hl.shared_transformer_down.size());
                    upload_float_data(dev, q, pool, g.pre_ff_norm, hl.shared_transformer_pre_ff_norm.data(), hl.shared_transformer_pre_ff_norm.size());
                    upload_float_data(dev, q, pool, g.ffn_norm, hl.shared_transformer_ffn_norm.data(), hl.shared_transformer_ffn_norm.size());
                    upload_float_data(dev, q, pool, g.mamba_input_norm, hl.mamba_input_norm_w.data(), hl.mamba_input_norm_w.size());
                    layers_[layer] = std::move(g);
                } else {
                    auto ml_it = model_.mamba_layers.find(layer);
                    if (ml_it == model_.mamba_layers.end()) {
                        fprintf(stderr, "Zamba2VK: missing weights for layer %d\n", layer);
                        return false;
                    }
                    layers_[layer] = upload_layer(layer, ml_it->second);
                }
            }
            upload_float_data(dev, q, pool, params, params_host.data(), params_host.size());
            upload_float_data(dev, q, pool, conv_pack, conv_host.data(), conv_host.size());
            // states start zeroed (resize zero-fills; buffers uploaded once).

            // 3. Scratch buffers.
            alloc(hidden_, d_model * 4, "hidden");
            alloc(residual_, d_model * 4, "residual");
            alloc(tmp_, d_model * 4, "tmp");
            alloc(projected_, d_model * 4, "projected");
            alloc(in_proj_, d_in_proj * 4, "in_proj");
            alloc(y_inner_, d_inner * 4, "y_inner");
            size_t n_qkv = ((size_t)c.n_attn_heads + 2 * c.n_kv_heads) * c.attn_head_dim;
            alloc(qkv_, n_qkv * 4, "qkv");
            alloc(attn_out_, (size_t)c.n_attn_heads * c.attn_head_dim * 4, "attn_out");
            // FFN intermediate: gate/up rows each = up_w rows (2*d_model on Zamba2)
            size_t d_ff = 0;
            for (auto& kvp : model_.hybrid_layers)
                d_ff = std::max(d_ff, kvp.second.shared_transformer_up.size() / d_model);
            if (d_ff == 0) d_ff = 2 * d_model;
            alloc(gate_up_, 2 * d_ff * 4, "gate_up");
            alloc(act_, d_ff * 4, "act");
            alloc(logits_, (size_t)c.vocab_size * 4, "logits");
            alloc(embedding_, d_model * 4, "embedding");
            alloc(concat_, 2 * d_model * 4, "concat");

            conv_off_.assign(c.n_layers, 0);
            pos_ = 0;
            vulkan_ok_ = true;
            initialized = true;
            printf("Zamba2VK: ready — %d layers (%zu hybrid), d=%zu\n",
                   c.n_layers, model_.hybrid_layers.size(), d_model);
            return true;
        } catch (const std::exception& e) {
            fprintf(stderr, "Zamba2VK: init failed: %s\n", e.what());
            destroy();
            return false;
        }
    }

    bool reset() override {
        if (!initialized) return false;
        const Zamba2Config& c = model_.cfg;
        const size_t conv_dim = c.d_inner + 2 * (size_t)c.n_group * c.d_state;
        // Zero only the STATE regions — the buffers are [params | states] and
        // [conv_w | conv_b | conv_state]. The old offset-0 fills clobbered
        // A/D/dt_bias and the conv weights (#1462 follow-up).
        // State is per-(head, head_dim) slices: layers*n_head*head_dim*d_state.
        fill_zero(engine_->device(), engine_->queue(), *engine_->cmd_pool(),
                  params.buffer(),
                  (size_t)c.n_layers * c.n_head * c.head_dim * c.d_state * 4,
                  (VkDeviceSize)3 * c.n_head * c.n_layers * 4);
        fill_zero(engine_->device(), engine_->queue(), *engine_->cmd_pool(),
                  conv_pack.buffer(),
                  (size_t)c.n_layers * (c.d_conv - 1) * conv_dim * 4,
                  (VkDeviceSize)c.n_layers * (c.d_conv * conv_dim + conv_dim) * 4);
        fill_zero(engine_->device(), engine_->queue(), *engine_->cmd_pool(),
                  kv_cache.buffer(), kv_cache.size());
        std::fill(conv_off_.begin(), conv_off_.end(), 0u);
        pos_ = 0;
        return true;
    }

    bool forward(int, float*) override { return false; }
    bool lm_head(const float*, float*, int*) override { return false; }

    // ── one Mamba2 block (shared by pure and hybrid layers) ──
    // in/out: tmp_ (in-place mamba output on x)
    void mamba_block(int layer, int n_layers) {
        const Zamba2Config& c = model_.cfg;
        const size_t d_model = c.d_model, d_inner = c.d_inner;
        const size_t conv_dim = d_inner + 2 * (size_t)c.n_group * c.d_state;
        const size_t d_in_proj = d_inner + conv_dim + c.n_head;
        Z2vLayerGPU& g = layers_[layer];

        // in_proj: [d_in_proj, d_model] @ tmp
        compute_->gemv_f32(in_proj_.buffer(), tmp_.buffer(), g.in_proj.buffer(),
                           (int)d_in_proj, 1, (int)d_model);
        // conv1d in-place on the xBC region (descriptor offset d_inner).
        PushConstants pc_c{};
        pc_c.M = (uint32_t)conv_dim; pc_c.N = (uint32_t)c.d_conv;
        pc_c.stride = conv_off_[layer]; pc_c.layer = layer; pc_c.head = n_layers;
        compute_->dispatch_off("mamba_conv1d", pc_c,
                               in_proj_.buffer(), (VkDeviceSize)d_inner * 4,
                               in_proj_.buffer(), (VkDeviceSize)d_inner * 4,
                               conv_pack.buffer(), 0,
                               (uint32_t)((conv_dim + 255) / 256));
        conv_off_[layer] = (conv_off_[layer] + 1) % (uint32_t)(c.d_conv - 1);
        // scan: x/B/C/dt in in_proj_, y in y_inner_, params+state in params_.
        PushConstants pc_s{};
        pc_s.M = (uint32_t)d_inner; pc_s.N = (uint32_t)conv_dim;
        pc_s.K = (uint32_t)c.n_group; pc_s.stride = (uint32_t)c.d_state;
        pc_s.token = c.head_dim; pc_s.layer = layer; pc_s.head = c.n_head;
        pc_s.pos = c.n_head / c.n_group;
        pc_s.pad0 = fbits((uint32_t)n_layers);
        compute_->dispatch("mamba2_scan", pc_s,
                           in_proj_.buffer(), y_inner_.buffer(), params.buffer(),
                           (uint32_t)c.n_head);
        // z-gate: y_inner *= silu(z) — BEFORE the group norm (HF
        // Zamba2RMSNormGated: hidden * silu(gate) first, then RMSNorm;
        // matches mamba2_cpu_forward step 5b). z at in_proj_[0..d_inner).
        PushConstants pc_z{}; pc_z.M = (uint32_t)d_inner;
        compute_->dispatch("silu_mul", pc_z,
                           in_proj_.buffer(), y_inner_.buffer(), y_inner_.buffer(),
                           (uint32_t)((d_inner + 255) / 256));
        // group RMSNorm (eps 1e-6 hardcoded in the CPU reference).
        PushConstants pc_n{};
        pc_n.M = (uint32_t)d_inner; pc_n.N = (uint32_t)c.n_group;
        pc_n.K = (uint32_t)(d_inner / c.n_group); pc_n.eps = 1e-6f;
        compute_->dispatch("group_rms_norm", pc_n,
                           y_inner_.buffer(), VK_NULL_HANDLE, g.norm_w.buffer(),
                           (uint32_t)c.n_group);
        // out_proj → tmp_
        compute_->gemv_f32(tmp_.buffer(), y_inner_.buffer(), g.out_proj.buffer(),
                           (int)d_model, 1, (int)d_inner);
    }

    int generate(int token_id) override {
        if (!initialized || !vulkan_ok_) return -1;
        const Zamba2Config& c = model_.cfg;
        const size_t d_model = c.d_model;

        PushConstants pc{};
        compute_->embed_lookup(hidden_.buffer(), embed.buffer(), token_id, (int)d_model);
        // keep the original embedding for the hybrid concat (reference: Zamba2
        // shared transformer input = concat(hidden, embedding))
        pc.M = (uint32_t)d_model;
        compute_->dispatch_batch("copy_buffer", pc,
                          hidden_.buffer(), embedding_.buffer(), VK_NULL_HANDLE, 1);

        for (int layer = 0; layer < c.n_layers; ++layer) {
            Z2vLayerGPU& g = layers_[layer];
            if (!g.hybrid) {
                // ── pure Mamba2 layer ──
                // residual = input; tmp = rms(input); mamba on tmp; + residual
                pc.M = (uint32_t)d_model;
                compute_->dispatch_batch("copy_buffer", pc,
                                  hidden_.buffer(), residual_.buffer(), VK_NULL_HANDLE, 1);
                compute_->dispatch_batch("copy_buffer", pc,
                                  hidden_.buffer(), tmp_.buffer(), VK_NULL_HANDLE, 1);
                compute_->rms_norm(tmp_.buffer(), g.input_norm.buffer(),
                                   (int)d_model, c.rms_norm_eps);
                mamba_block(layer, c.n_layers);
                // tmp_ = mamba_out; hidden = mamba_out + residual
                compute_->dispatch_batch("add_residual", pc,
                                  tmp_.buffer(), residual_.buffer(), VK_NULL_HANDLE, 1);
                compute_->dispatch_batch("copy_buffer", pc,
                                  tmp_.buffer(), hidden_.buffer(), VK_NULL_HANDLE, 1);
            } else {
                // ── hybrid layer — reference structure (modeling_zamba2.py) ──
                //   th = shared_transformer(concat(hidden, embedding))
                //   hidden = hidden + ssm_mix(th)
                //   hidden = mamba_decoder(hidden)
                // (verified op-for-op against the CPU reference in PR #1462)
                const size_t n_at = (size_t)c.n_attn_heads * c.attn_head_dim;
                const size_t attn_in = 2 * d_model;

                // 1. concat_ = [hidden_ | embedding_]; RMSNorm (post_attention_norm, 2d)
                pc.M = (uint32_t)d_model;
                compute_->dispatch_batch("copy_buffer", pc,
                                  hidden_.buffer(), concat_.buffer(), VK_NULL_HANDLE, 1);
                compute_->dispatch_off("copy_buffer", pc,
                                  embedding_.buffer(), 0,
                                  concat_.buffer(), (VkDeviceSize)d_model * 4,
                                  VK_NULL_HANDLE, 0, 1);
                compute_->rms_norm(concat_.buffer(), g.pre_ff_norm.buffer(),
                                   (int)attn_in, c.rms_norm_eps);

                // 2. QKV projections from the 2*d_model concat (weights [n_at, 2*d_model])
                compute_->gemv_f32(qkv_.buffer(), concat_.buffer(), g.qw.buffer(),
                                   (int)n_at, 1, (int)attn_in);
                compute_->gemv_f32_off(qkv_.buffer(), (VkDeviceSize)n_at * 4,
                                       concat_.buffer(), g.kw.buffer(),
                                       (int)n_at, 1, (int)attn_in);
                compute_->gemv_f32_off(qkv_.buffer(), (VkDeviceSize)2 * n_at * 4,
                                       concat_.buffer(), g.vw.buffer(),
                                       (int)n_at, 1, (int)attn_in);

                // 3. RoPE (1.2B/7B use it; 2.7B does not — TODO config flag)
                PushConstants pc_r{};
                pc_r.M = (uint32_t)c.n_attn_heads; pc_r.N = (uint32_t)c.n_kv_heads;
                pc_r.K = (uint32_t)c.attn_head_dim;
                pc_r.rope_theta = c.rope_theta; pc_r.pos = pos_;
                compute_->dispatch("mamba_rope", pc_r,
                                   qkv_.buffer(), VK_NULL_HANDLE, VK_NULL_HANDLE,
                                   (uint32_t)(c.n_attn_heads + c.n_kv_heads));

                // 4. fused attention + KV write, scale sqrt(2/hd) (concat compensation)
                int hyb_idx = 0;
                for (int ll = 0; ll <= layer; ++ll)
                    if (model_.hybrid_layers.count(ll)) hyb_idx++;
                hyb_idx--;
                compute_->flash_attn(qkv_.buffer(), kv_cache.buffer(), kv_cache.buffer(),
                                     attn_out_.buffer(), pos_ + 1,
                                     c.n_attn_heads, c.n_kv_heads, c.attn_head_dim,
                                     c.n_attn_heads / c.n_kv_heads,
                                     hyb_idx, c.max_seq_len, model_.num_hybrid_layers(),
                                     (float)std::sqrt(2.0 / c.attn_head_dim));

                // 5. o_proj [d_model, n_at] → tmp_
                compute_->gemv_f32(tmp_.buffer(), attn_out_.buffer(), g.ow.buffer(),
                                   (int)d_model, 1, (int)n_at);

                // 6. pre-FFN RMSNorm (ffn_norm) + SiLU FFN → projected_ = th
                compute_->rms_norm(tmp_.buffer(), g.ffn_norm.buffer(),
                                   (int)d_model, c.rms_norm_eps);
                // GpuBuffer::size() returns BYTES — /4 for element count.
                size_t d_ff = g.up_w.size() / (4 * d_model);
                compute_->gemv_f32(gate_up_.buffer(), tmp_.buffer(), g.gate_w.buffer(),
                                   (int)d_ff, 1, (int)d_model);
                compute_->gemv_f32_off(gate_up_.buffer(), (VkDeviceSize)d_ff * 4,
                                       tmp_.buffer(), g.up_w.buffer(),
                                       (int)d_ff, 1, (int)d_model);
                PushConstants pc_g{}; pc_g.M = (uint32_t)d_ff;
                // Zamba2 hybrid FFN is GELU-gated, not SiLU (hidden_act=gelu,
                // #zamba2-validation — matches gelu_mul_kernel + CPU reference).
                compute_->dispatch("gelu_mul", pc_g,
                                   gate_up_.buffer(), act_.buffer(), VK_NULL_HANDLE,
                                   (uint32_t)((d_ff + 255) / 256));
                compute_->gemv_f32(projected_.buffer(), act_.buffer(), g.down_w.buffer(),
                                   (int)d_model, 1, (int)d_ff);

                // 7. hidden = hidden + ssm_mix(th). The mamba residual must be
                // the layer INPUT, not input+th — HF Zamba2MambaDecoderLayer:
                // out = input + mamba(norm(input + th)); th is consumed inside
                // the norm. Saving input+th as residual double-counts ssm_mix
                // (#zamba2-validation, matches forward_hybrid_layer).
                compute_->gemv_f32(tmp_.buffer(), projected_.buffer(), g.linear.buffer(),
                                   (int)d_model, 1, (int)d_model);
                pc.M = (uint32_t)d_model;
                compute_->dispatch_batch("copy_buffer", pc,
                                  hidden_.buffer(), residual_.buffer(), VK_NULL_HANDLE, 1);
                compute_->dispatch_batch("add_residual", pc,
                                  hidden_.buffer(), tmp_.buffer(), VK_NULL_HANDLE, 1);

                // 8. mamba decoder: rms(hidden) → mamba → + hidden (residual)
                compute_->dispatch_batch("copy_buffer", pc,
                                  hidden_.buffer(), tmp_.buffer(), VK_NULL_HANDLE, 1);
                compute_->rms_norm(tmp_.buffer(), g.mamba_input_norm.buffer(),
                                   (int)d_model, c.rms_norm_eps);
                mamba_block(layer, c.n_layers);   // tmp_ = mamba_out
                compute_->dispatch_batch("add_residual", pc,
                                  tmp_.buffer(), residual_.buffer(), VK_NULL_HANDLE, 1);
                compute_->dispatch_batch("copy_buffer", pc,
                                  tmp_.buffer(), hidden_.buffer(), VK_NULL_HANDLE, 1);
            }
        }

        // final norm + tied lm_head
        compute_->rms_norm(hidden_.buffer(), final_norm.buffer(), (int)d_model, c.rms_norm_eps);
        compute_->gemv_f32(logits_.buffer(), hidden_.buffer(), embed.buffer(),
                           c.vocab_size, 1, (int)d_model);
        pos_++;
        return compute_->argmax(logits_.buffer(), c.vocab_size);
    }

    void destroy() override {
        layers_.clear();
        conv_off_.clear();
        logits_ = GpuBuffer(); act_ = GpuBuffer(); gate_up_ = GpuBuffer();
        attn_out_ = GpuBuffer(); qkv_ = GpuBuffer(); y_inner_ = GpuBuffer();
        in_proj_ = GpuBuffer(); projected_ = GpuBuffer(); tmp_ = GpuBuffer();
        residual_ = GpuBuffer(); hidden_ = GpuBuffer();
        kv_cache = GpuBuffer(); conv_pack = GpuBuffer(); params = GpuBuffer();
        final_norm = GpuBuffer(); embed = GpuBuffer();
        embedding_ = GpuBuffer(); concat_ = GpuBuffer();  // must outlive engine_ (device)
        model_ = Zamba2Model();
        compute_.reset();
        engine_.reset();
        vulkan_ok_ = false;
        initialized = false;
    }

    float benchmark(int tokens) override {
        if (!initialized) return 0.0f;
        reset();
        auto t0 = std::chrono::high_resolution_clock::now();
        int tok = 1;
        for (int i = 0; i < tokens; i++) {
            tok = generate(tok);
            if (tok < 0) break;
        }
        float ms = std::chrono::duration<float, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
        return tokens > 0 ? ms / tokens : 0.0f;
    }

    bool can_infer() const override { return initialized && vulkan_ok_; }
};

Backend* create_zamba2_vulkan_backend() { return new Zamba2VulkanBackend(); }
