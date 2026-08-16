// backend_metal.mm — Apple Metal GPU backend
//
// Uses Metal Performance Shaders (MPS) for matrix operations and
// custom Metal compute shaders for LLM inference primitives.
// macOS 14.0+ / iOS 17.0+ required for MPS graph API.

#include "backend.h"

// Portable half — the HIP headers that normally provide `half` don't exist on
// macOS; __fp16 is the native Apple equivalent (same storage, same conversions).
#ifndef __HIP__
using half = __fp16;
#endif

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

// MPSNDArray enforces a minimum buffer footprint (observed floor = 2×hidden,
// scaling with dims). Round every device buffer up to 16 KiB so GEMV/MPS
// operands never trip it — real 2K-hidden models need 8 KiB already.
static NSUInteger mps_round(NSUInteger n) { return ((n + 16383) / 16384) * 16384; }

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <fstream>
#include <chrono>
#include <algorithm>
#include <cmath>

// ── Metal helpers ──
static const char* metal_kernel_src = R"(
#include <metal_stdlib>
using namespace metal;

// RMS normalization kernel
kernel void rmsnorm(
    device half *x [[buffer(0)]],
    const device half *w [[buffer(1)]],
    constant int &n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    // Simplified single-threaded RMS norm for Metal
    // For production, use MPS matrix ops or tile shaders
    if (gid >= n) return;
    
    // Compute mean square (parallel reduction would be better)
    // For now, use a simple CPU-fallback-compatible approach
}

// SiLU * mul elementwise
kernel void silu_mul(
    device half *out [[buffer(0)]],
    const device half *g [[buffer(1)]],
    const device half *u [[buffer(2)]],
    constant int &n [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    float v = float(g[gid]);
    out[gid] = half(v / (1.0 + exp(-v)) * float(u[gid]));
}

// Argmax
kernel void argmax(
    const device float *logits [[buffer(0)]],
    device int *idx [[buffer(1)]],
    device float *val [[buffer(2)]],
    constant int &n [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    // Simplified: each thread checks its position
    // Full implementation uses parallel reduction
    if (gid == 0) {
        int best_i = 0;
        float best = logits[0];
        for (int i = 1; i < n; i++) {
            if (logits[i] > best) { best = logits[i]; best_i = i; }
        }
        *idx = best_i;
        *val = best;
    }
}

// Element-wise copy
kernel void copy(
    device half *dst [[buffer(0)]],
    const device half *src [[buffer(1)]],
    constant int &n [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    dst[gid] = src[gid];
}

// Embedding lookup
kernel void embed_lookup(
    device half *out [[buffer(0)]],
    const device half *embed [[buffer(1)]],
    const device half *ibias [[buffer(2)]],
    const device half *iscale [[buffer(3)]],
    constant int &token_id [[buffer(4)]],
    constant int &h [[buffer(5)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= h) return;
    float raw = float(embed[(size_t)token_id * h + gid]);
    out[gid] = half((raw + float(ibias[gid])) * float(iscale[gid]));
}
)";

// ── Metal Backend implementation ──
struct MetalBackend : Backend {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> cmd_queue = nil;
    id<MTLLibrary> library = nil;
    id<MTLComputePipelineState> rmsnorm_pso = nil;
    id<MTLComputePipelineState> silu_mul_pso = nil;
    id<MTLComputePipelineState> argmax_pso = nil;
    id<MTLComputePipelineState> copy_pso = nil;
    id<MTLComputePipelineState> embed_pso = nil;
    
    MPSMatrixMultiplication* matmul = nil;
    MPSMatrix* mps_a = nil;
    MPSMatrix* mps_b = nil;
    MPSMatrix* mps_c = nil;
    
    // GPU buffers
    id<MTLBuffer> d_hs = nil;
    id<MTLBuffer> d_ao = nil;
    id<MTLBuffer> d_tmp = nil;
    id<MTLBuffer> d_gate = nil;
    id<MTLBuffer> d_ffn = nil;
    id<MTLBuffer> d_fnw = nil;
    id<MTLBuffer> d_embed = nil;
    id<MTLBuffer> d_kcache = nil;
    id<MTLBuffer> d_vcache = nil;
    id<MTLBuffer> d_qout = nil;
    id<MTLBuffer> d_kout = nil;
    id<MTLBuffer> d_vout = nil;
    id<MTLBuffer> d_lm_vocab = nil;
    id<MTLBuffer> d_ibias = nil;
    id<MTLBuffer> d_iscale = nil;
    id<MTLBuffer> d_argmax_idx = nil;
    id<MTLBuffer> d_argmax_val = nil;
    
    // Weight buffers (per-layer)
    std::vector<id<MTLBuffer>> layer_wq, layer_wk, layer_wv, layer_wo;
    std::vector<id<MTLBuffer>> layer_w1, layer_w2, layer_w3;
    std::vector<id<MTLBuffer>> layer_rms_a, layer_rms_f;
    
    std::vector<float> embed, iscale, ibias;
    float* logits_buf = nullptr;
    
    int hidden = 2048;
    int n_layers = 40;
    int n_heads = 8;
    int n_kv_heads = 2;
    int head_dim = 128;
    int vocab = 262272;
    int n_ff = 2048;
    int pos = 0;
    int max_seq = 4096;

    MetalBackend() {
        type = BackendType::METAL_GPU;
        name = "Metal GPU (Apple)";
    }

    ~MetalBackend() override { destroy(); }

    bool init_metal_device() {
        device = MTLCreateSystemDefaultDevice();
        if (!device) {
            fprintf(stderr, "Metal: No Metal-capable GPU found\n");
            return false;
        }
        
        // Check for Apple GPU (Apple Silicon) or AMD GPU
        if (![device supportsFamily:MTLGPUFamilyApple7] &&
            ![device supportsFamily:MTLGPUFamilyMac2]) {
            fprintf(stderr, "Metal: GPU may not support required features\n");
        }
        
        printf("Metal: Using %s\n", [[device name] UTF8String]);
        return true;
    }

    bool compile_kernels() {
        NSError* error = nil;
        NSString* src = [NSString stringWithUTF8String:metal_kernel_src];
        library = [device newLibraryWithSource:src options:nil error:&error];
        if (error || !library) {
            fprintf(stderr, "Metal: Failed to compile kernels: %s\n",
                    [[error localizedDescription] UTF8String]);
            return false;
        }
        
        auto make_pso = [&](const char* name) -> id<MTLComputePipelineState> {
            NSString* ns_name = [NSString stringWithUTF8String:name];
            id<MTLFunction> fn = [library newFunctionWithName:ns_name];
            if (!fn) return nil;
            return [device newComputePipelineStateWithFunction:fn error:&error];
        };
        
        rmsnorm_pso = make_pso("rmsnorm");
        silu_mul_pso = make_pso("silu_mul");
        argmax_pso = make_pso("argmax");
        copy_pso = make_pso("copy");
        embed_pso = make_pso("embed_lookup");
        
        if (!rmsnorm_pso || !silu_mul_pso || !argmax_pso || !copy_pso || !embed_pso) {
            fprintf(stderr, "Metal: Failed to create pipeline states\n");
            return false;
        }
        
        cmd_queue = [device newCommandQueue];
        return cmd_queue != nil;
    }

    // Matrix multiply using MPS (Metal Performance Shaders)
    // C[M,N] = A[M,K] @ B[K,N]
    void metal_matmul(id<MTLCommandBuffer> cb,
                      id<MTLBuffer> C, const id<MTLBuffer> A, const id<MTLBuffer> B,
                      int M, int N, int K) {
        // Use MPSMatrixMultiplication for fp16 matmul
        MPSMatrixDescriptor* desc_a = [MPSMatrixDescriptor
            matrixDescriptorWithRows:M columns:K
            rowBytes:K * sizeof(half) dataType:MPSDataTypeFloat16];
        MPSMatrixDescriptor* desc_b = [MPSMatrixDescriptor
            matrixDescriptorWithRows:K columns:N
            rowBytes:N * sizeof(half) dataType:MPSDataTypeFloat16];
        MPSMatrixDescriptor* desc_c = [MPSMatrixDescriptor
            matrixDescriptorWithRows:M columns:N
            rowBytes:N * sizeof(half) dataType:MPSDataTypeFloat16];
        
        MPSMatrix* mat_a = [[MPSMatrix alloc] initWithBuffer:A descriptor:desc_a];
        MPSMatrix* mat_b = [[MPSMatrix alloc] initWithBuffer:B descriptor:desc_b];
        MPSMatrix* mat_c = [[MPSMatrix alloc] initWithBuffer:C descriptor:desc_c];
        
        MPSMatrixMultiplication* mm = [[MPSMatrixMultiplication alloc]
            initWithDevice:device resultRows:M resultColumns:N interiorColumns:K];
        
        // CPU-side ops between gemvs read these buffers — the MPS kernel must
        // complete before they do. Each gemv gets its own command buffer,
        // committed + waited here (synchronous, correct-first).
        [mm encodeToCommandBuffer:cb leftMatrix:mat_a rightMatrix:mat_b resultMatrix:mat_c];
        [cb commit];
        [cb waitUntilCompleted];
    }

    // GEMV: out[M] = in[K] @ wt[K, M] using MPS as GeMM with N=1
    void metal_gemv(id<MTLBuffer> out, const id<MTLBuffer> in_vec,
                    const id<MTLBuffer> wt, int M, int K) {
        // Fresh command buffer per gemv — a committed buffer cannot be reused.
        id<MTLCommandBuffer> cb = [cmd_queue commandBuffer];
        metal_matmul(cb, out, in_vec, wt, 1, M, K);
    }

    bool init(const ModelConfig& cfg, const std::string& weights_dir) override {
        this->cfg = cfg;
        
        hidden = cfg.hidden_size > 0 ? cfg.hidden_size : cfg.hidden;        n_layers = cfg.num_layers > 0 ? cfg.num_layers : cfg.n_layers;
        n_heads = cfg.num_heads > 0 ? cfg.num_heads : cfg.n_heads;
        n_kv_heads = cfg.num_kv_heads > 0 ? cfg.num_kv_heads : cfg.n_kv_heads;
        head_dim = cfg.head_dim;
        vocab = cfg.vocab_size > 0 ? cfg.vocab_size : cfg.vocab;
        n_ff = cfg.intermediate_size > 0 ? cfg.intermediate_size : hidden;
        max_seq = cfg.max_seq_len > 0 ? cfg.max_seq_len : 4096;

        printf("Metal: Initializing (H=%d, L=%d, NH=%d, NKV=%d, V=%d)...\n",
               hidden, n_layers, n_heads, n_kv_heads, vocab);

        if (!init_metal_device()) return false;
        if (!compile_kernels()) return false;

        std::string wd = weights_dir;
        if (!wd.empty() && wd.back() != '/') wd += '/';

        // Load weights
        auto load = [&](const std::string& name) -> std::vector<float> {
            std::string path = wd + name;
            std::ifstream f(path, std::ios::binary | std::ios::ate);
            if (!f) return {};
            size_t n = f.tellg() / sizeof(float); f.seekg(0);
            std::vector<float> d(n); f.read((char*)d.data(), n * sizeof(float));
            return d;
        };
        
        // Weights come from torch as [out, in] row-major; the MPS gemv treats
        // the buffer as [K, N] = [in, out] row-major. Transpose at load so the
        // matmul is mathematically correct: B[k][n] = w[n][k].
        auto to_half_buffer = [&](const std::vector<float>& src, int rows, int cols) -> id<MTLBuffer> {
            if (src.empty()) return nil;
            std::vector<half> h((size_t)rows * cols);
            for (int r = 0; r < rows; r++)
                for (int c = 0; c < cols; c++)
                    h[(size_t)c * rows + r] = half(src[(size_t)r * cols + c]);
            return [device newBufferWithBytes:h.data()
                                       length:h.size() * sizeof(half)
                                      options:MTLResourceStorageModeShared];
        };

        embed = load("model_embed_tokens_weight.bin");
        auto fnorm = load("model_norm_weight.bin");
        iscale = load("model_input_hidden_states_scale.bin");
        ibias = load("model_input_hidden_states_bias.bin");

        if (embed.empty() || fnorm.empty() || iscale.empty() || ibias.empty()) {
            fprintf(stderr, "Metal: failed to load initial weight files\n");
            return false;
        }

        // Allocate buffers
        int buf_size = hidden * sizeof(half);
        d_hs = [device newBufferWithLength:mps_round(buf_size) options:MTLResourceStorageModeShared];
        d_ao = [device newBufferWithLength:mps_round(buf_size) options:MTLResourceStorageModeShared];
        d_tmp = [device newBufferWithLength:mps_round(std::max(hidden, 2 * n_ff) * sizeof(half)) options:MTLResourceStorageModeShared];
        d_gate = [device newBufferWithLength:mps_round(n_ff * sizeof(half)) options:MTLResourceStorageModeShared];
        d_ffn = [device newBufferWithLength:mps_round(hidden * sizeof(half)) options:MTLResourceStorageModeShared];
        d_fnw = to_half_buffer(fnorm, hidden, 1);
        d_embed = to_half_buffer(embed, vocab, hidden);  // [V,H] → transposed [H,V] for lm_head B
        d_ibias = to_half_buffer(ibias, hidden, 1);
        d_iscale = to_half_buffer(iscale, hidden, 1);
        
        int kv_elem = n_layers * max_seq * n_kv_heads * head_dim;
        d_kcache = [device newBufferWithLength:kv_elem * sizeof(half) options:MTLResourceStorageModeShared];
        d_vcache = [device newBufferWithLength:kv_elem * sizeof(half) options:MTLResourceStorageModeShared];
        
        d_qout = [device newBufferWithLength:mps_round((n_heads * head_dim) * sizeof(half)) options:MTLResourceStorageModeShared];
        d_kout = [device newBufferWithLength:mps_round((n_kv_heads * head_dim) * sizeof(half)) options:MTLResourceStorageModeShared];
        d_vout = [device newBufferWithLength:mps_round((n_kv_heads * head_dim) * sizeof(half)) options:MTLResourceStorageModeShared];
        d_lm_vocab = [device newBufferWithLength:mps_round(vocab * sizeof(float)) options:MTLResourceStorageModeShared];
        d_argmax_idx = [device newBufferWithLength:sizeof(int) options:MTLResourceStorageModeShared];
        d_argmax_val = [device newBufferWithLength:sizeof(float) options:MTLResourceStorageModeShared];

        size_t qd = n_heads * head_dim;
        size_t kd = n_kv_heads * head_dim;

        // Load per-layer weights
        for (int il = 0; il < n_layers; il++) {
            std::string p = "model_layers_" + std::to_string(il) + "_";
            auto wq = load(p + "self_attn_q_proj.weight.bin");
            auto wk = load(p + "self_attn_k_proj.weight.bin");
            auto wv = load(p + "self_attn_v_proj.weight.bin");
            auto wo = load(p + "self_attn_o_proj.weight.bin");
            auto w1 = load(p + "mlp_gate_proj.weight.bin");
            auto w2 = load(p + "mlp_down_proj.weight.bin");
            auto w3 = load(p + "mlp_up_proj.weight.bin");
            auto rms_a = load(p + "input_layernorm.weight.bin");
            auto rms_f = load(p + "post_attention_layernorm.weight.bin");

            layer_wq.push_back(to_half_buffer(wq, qd, hidden));
            layer_wk.push_back(to_half_buffer(wk, kd, hidden));
            layer_wv.push_back(to_half_buffer(wv, kd, hidden));
            layer_wo.push_back(to_half_buffer(wo, hidden, qd));
            layer_w1.push_back(to_half_buffer(w1, n_ff, hidden));
            layer_w2.push_back(to_half_buffer(w2, hidden, n_ff));
            layer_w3.push_back(to_half_buffer(w3, n_ff, hidden));
            layer_rms_a.push_back(to_half_buffer(rms_a, hidden, 1));
            layer_rms_f.push_back(to_half_buffer(rms_f, hidden, 1));
        }

        try {
            logits_buf = new float[vocab];
        } catch (std::bad_alloc&) {
            fprintf(stderr, "Metal: failed to allocate logits buffer\n");
            return false;
        }

        initialized = true;
        printf("Metal: Engine ready\n");
        return true;
    }

    bool reset() override {
        pos = 0;
        return true;
    }

    bool forward(int token_id, float* hidden_out) override {
        (void)token_id; (void)hidden_out;
        static bool warned = false;
        if (!warned) {
            fprintf(stderr, "Metal Backend: forward() not implemented (generate() works)\n");
            warned = true;
        }
        return false;
    }

    bool lm_head(const float* hidden, float* logits, int* argmax) override {
        (void)hidden; (void)logits; (void)argmax;
        static bool warned = false;
        if (!warned) {
            fprintf(stderr, "Metal Backend: lm_head() not implemented (generate() works)\n");
            warned = true;
        }
        return false;
    }

    int generate(int token_id) override {
        if (!initialized) return -1;
        
        @autoreleasepool {
            id<MTLCommandBuffer> cb = [cmd_queue commandBuffer];
            
            // 1. Embedding lookup
            // Simplified for now: use host-side embedding lookup
            for (int i = 0; i < hidden; i++) {
                float raw = embed[(size_t)token_id * hidden + i];
                half val = half((raw + ibias[i]) * iscale[i]);
                ((half*)[d_hs contents])[i] = val;
            }
            {double m=0;half* dbg=(half*)[d_hs contents];for(int i=0;i<hidden;i++)m+=(double)dbg[i]*(double)dbg[i];
             fprintf(stderr,"  embed mag=%.4f (embed vec size=%zu)\n",sqrt(m/hidden),embed.size());
             fprintf(stderr,"  wq0 size=%zu wq1 size=%zu\n", layer_wq.empty()?0:(size_t)layer_wq[0].length, layer_wq.size()>1?(size_t)layer_wq[1].length:0);}
            
            size_t qd = n_heads * head_dim;
            size_t kd = n_kv_heads * head_dim;
            
            // 2. Process layers
            for (int il = 0; il < n_layers; il++) {
                half* hs = (half*)[d_hs contents];
                half* tmp = (half*)[d_tmp contents];
                half* ao = (half*)[d_ao contents];
                
                // RMS norm (CPU-side for now, Metal kernel for production)
                float ss = 0;
                for (int i = 0; i < hidden; i++) ss += (float)hs[i] * (float)hs[i];
                float rms = sqrtf(ss / hidden + 1e-5f);
                float inv_rms = 1.0f / rms;
                half* rms_w = (half*)[layer_rms_a[il] contents];
                for (int i = 0; i < hidden; i++) tmp[i] = half((float)hs[i] * inv_rms * (float)rms_w[i]);
                
                // QKV projections via MPS
                if (layer_wq[il]) {
                    metal_gemv(d_qout, d_tmp, layer_wq[il], qd, hidden);
                }
                if (layer_wk[il]) {
                    metal_gemv(d_kout, d_tmp, layer_wk[il], kd, hidden);
                }
                if (layer_wv[il]) {
                    metal_gemv(d_vout, d_tmp, layer_wv[il], kd, hidden);
                }
                
                // Store KV in cache (host-side for simplicity)
                half* k_dst = (half*)[d_kcache contents] + (size_t)il * max_seq * kd + (size_t)pos * kd;
                half* v_dst = (half*)[d_vcache contents] + (size_t)il * max_seq * kd + (size_t)pos * kd;
                memcpy(k_dst, [d_kout contents], kd * sizeof(half));
                memcpy(v_dst, [d_vout contents], kd * sizeof(half));
                
                {
                    double mq=0,mk=0,mv=0;
                    half* dq=(half*)[d_qout contents]; half* dk=(half*)[d_kout contents]; half* dv=(half*)[d_vout contents];
                    for(int i=0;i<qd;i++)mq+=(double)dq[i]*(double)dq[i];
                    for(int i=0;i<kd;i++)mk+=(double)dk[i]*(double)dk[i];
                    for(int i=0;i<kd;i++)mv+=(double)dv[i]*(double)dv[i];
                    fprintf(stderr,"  L0 qkv mag: q=%.4f k=%.4f v=%.4f\n",sqrt(mq/qd),sqrt(mk/kd),sqrt(mv/kd));
                }
                // Attention (CPU-side accumulation; Metal kernel for production)
                int seq_len = pos + 1;
                float scale = 1.0f / sqrtf((float)head_dim);
                half* ao_out = (half*)[d_ao contents];
                int heads_per_kv = n_kv_heads > 0 ? n_heads / n_kv_heads : 1;
                for (int h = 0; h < n_heads; h++) {
                    int kh = h / heads_per_kv;  // GQA: query head h shares KV head kh
                    // scores[t] = q·k_t * scale
                    std::vector<float> scores(seq_len);
                    for (int t = 0; t < seq_len; t++) {
                        float s = 0;
                        for (int d = 0; d < head_dim; d++) {
                            s += (float)((half*)[d_qout contents])[h * head_dim + d] *
                                 (float)((half*)[d_kcache contents])[(size_t)il * max_seq * kd + (size_t)t * kd + kh * head_dim + d];
                        }
                        scores[t] = s * scale;
                    }
                    // softmax over scores
                    float mx = *std::max_element(scores.begin(), scores.end());
                    float esum = 0;
                    for (int t = 0; t < seq_len; t++) esum += expf(scores[t] - mx);
                    // weighted V sum: out[d] = Σ_t softmax(t) * V[t][h][d]
                    for (int d = 0; d < head_dim; d++) {
                        float acc = 0;
                        for (int t = 0; t < seq_len; t++) {
                            float w = expf(scores[t] - mx) / esum;
                            acc += w * (float)((half*)[d_vcache contents])[(size_t)il * max_seq * kd + (size_t)t * kd + kh * head_dim + d];
                        }
                        ao_out[h * head_dim + d] = half(acc);
                    }
                }
                if (il==0){double m=0;for(int i=0;i<qd;i++)m+=(double)ao_out[i]*(double)ao_out[i];fprintf(stderr,"  L0 attn-out mag=%.4f\n",sqrt(m/qd));}
                
                // Output projection via MPS → d_ffn, then residual: hs += Wo·attn
                if (layer_wo[il]) {
                    metal_gemv(d_ffn, d_ao, layer_wo[il], hidden, qd);
                    if (il==0){double m=0;half* dbg=(half*)[d_ffn contents];for(int i=0;i<hidden;i++)m+=(double)dbg[i]*(double)dbg[i];fprintf(stderr,"  L0 Wo mag=%.4f\n",sqrt(m/hidden));}
                    half* hs_in = (half*)[d_hs contents];
                    half* attn_out = (half*)[d_ffn contents];
                    for (int i = 0; i < hidden; i++) hs_in[i] = half((float)hs_in[i] + (float)attn_out[i]);
                    if (il==0){double m=0;for(int i=0;i<hidden;i++)m+=(double)hs_in[i]*(double)hs_in[i];fprintf(stderr,"  L0 after attn-resid mag=%.4f\n",sqrt(m/hidden));}
                }
                
                // FFN
                // RMS norm of the residual hs → tmp
                ss = 0;
                for (int i = 0; i < hidden; i++) ss += (float)hs[i] * (float)hs[i];
                rms = sqrtf(ss / hidden + 1e-5f);
                inv_rms = 1.0f / rms;
                rms_w = (half*)[layer_rms_f[il] contents];
                for (int i = 0; i < hidden; i++) tmp[i] = half((float)hs[i] * inv_rms * (float)rms_w[i]);
                
                // gate = W1·norm → d_gate ; up = W3·norm → d_ao (separate buffers!)
                if (layer_w1[il]) {
                    metal_gemv(d_gate, d_tmp, layer_w1[il], n_ff, hidden);
                }
                if (layer_w3[il]) {
                    metal_gemv(d_ao, d_tmp, layer_w3[il], n_ff, hidden);
                }
                
                // tmp = silu(gate) * up  (reuse d_tmp as the product buffer)
                half* gate = (half*)[d_gate contents];
                half* up = (half*)[d_ao contents];
                for (int i = 0; i < n_ff; i++) {
                    float v = (float)gate[i];
                    tmp[i] = half((v / (1.0f + expf(-v))) * (float)up[i]);
                }
                
                // Down projection → d_ffn, then residual: hs += W2·product
                if (layer_w2[il]) {
                    metal_gemv(d_ffn, d_tmp, layer_w2[il], hidden, n_ff);
                    half* hs_in = (half*)[d_hs contents];
                    half* ffn_out = (half*)[d_ffn contents];
                    for (int i = 0; i < hidden; i++) hs_in[i] = half((float)hs_in[i] + (float)ffn_out[i]);
                }
            }
            
            // 3. Final RMS norm
            half* hs = (half*)[d_hs contents];
            half* fnw = (half*)[d_fnw contents];
            float ss = 0;
            for (int i = 0; i < hidden; i++) ss += (float)hs[i] * (float)hs[i];
            float rms = sqrtf(ss / hidden + 1e-5f);
            float inv_rms = 1.0f / rms;
            for (int i = 0; i < hidden; i++) hs[i] = half((float)hs[i] * inv_rms * (float)fnw[i]);
            
            // 4. LM head via MPS
            metal_gemv(d_lm_vocab, d_hs, d_embed, vocab, hidden);
            
            [cb commit];
            [cb waitUntilCompleted];
            
            // 5. Argmax on CPU — MPS wrote fp16 into d_lm_vocab (desc is
            // MPSDataTypeFloat16), so read it as half, not float.
            half* logits_h = (half*)[d_lm_vocab contents];
            int next_token = 0;
            float best_val = (float)logits_h[0];
            for (int i = 1; i < vocab; i++) {
                float v = (float)logits_h[i];
                if (v > best_val) { best_val = v; next_token = i; }
            }
            
            pos++;
            return next_token;
        }
    }

    float benchmark(int tokens = 10) override {
        if (!initialized) return 0;
        reset();
        auto t0 = std::chrono::high_resolution_clock::now();
        int tok = 100;
        for (int i = 0; i < tokens; i++) {
            tok = generate(tok);
        }
        float ms = std::chrono::duration<float, std::milli>(
            std::chrono::high_resolution_clock::now() - t0).count();
        return ms / tokens;
    }

    void destroy() override {
        #define METAL_RELEASE(p) do { if (p) { p = nil; } } while(0)
        METAL_RELEASE(device);
        METAL_RELEASE(cmd_queue);
        METAL_RELEASE(library);
        METAL_RELEASE(rmsnorm_pso);
        METAL_RELEASE(silu_mul_pso);
        METAL_RELEASE(argmax_pso);
        METAL_RELEASE(copy_pso);
        METAL_RELEASE(embed_pso);
        METAL_RELEASE(d_hs);
        METAL_RELEASE(d_ao);
        METAL_RELEASE(d_tmp);
        METAL_RELEASE(d_gate);
        METAL_RELEASE(d_ffn);
        METAL_RELEASE(d_fnw);
        METAL_RELEASE(d_embed);
        METAL_RELEASE(d_kcache);
        METAL_RELEASE(d_vcache);
        METAL_RELEASE(d_qout);
        METAL_RELEASE(d_kout);
        METAL_RELEASE(d_vout);
        METAL_RELEASE(d_lm_vocab);
        METAL_RELEASE(d_ibias);
        METAL_RELEASE(d_iscale);
        METAL_RELEASE(d_argmax_idx);
        METAL_RELEASE(d_argmax_val);
        #undef METAL_RELEASE
        layer_wq.clear();
        layer_wk.clear();
        layer_wv.clear();
        layer_wo.clear();
        layer_w1.clear();
        layer_w2.clear();
        layer_w3.clear();
        layer_rms_a.clear();
        layer_rms_f.clear();
        delete[] logits_buf;
        initialized = false;
    }
};

extern "C" Backend* create_metal_backend() { return new MetalBackend(); }
