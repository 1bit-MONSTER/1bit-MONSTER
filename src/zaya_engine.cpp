// zaya_engine.cpp — Zaya inference engine as a C-callable library.
// Compiles into libzaya_engine.a, linked into token_router.
// No main(), no networking — just the model.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>
#include <fstream>
#include <algorithm>
#include <new>
#include "hip_check.h"
#include "zaya_engine.h"

#include "hip_check.h"

// OneBP model loader (NpuOnebpModel::open / get_tensor_f32 / get_tensor_f32_expert
// for ndim=1/2/3 dequant). The class lives entirely in the .cpp with no header,
// so it is raw-included exactly like backend_hip_1bp.cpp does.
#include "../engine/npu/src/onebp_loader.cpp"


// ── Runtime config (set by zaya_init from model header) ──
static constexpr float RMD_EPS=1e-5f;
static constexpr int BLK=256;
static constexpr size_t ZAYA_B_MAX = 8;  // batch cap for zaya_forward_batch (also sizes d_hs/d_lm_vocab)
static thread_local ZayaConfig eng;  // populated by zaya_init from ZayaConfig parameter

// Page gather kernel: copies page-allocated KV slots into a contiguous
// scratch buffer for the flash-attention kernel (which expects sequential
// positions [0, seq_len) in a single buffer).
// Each block handles one KV head for one position.
__global__ void gather_kv_pages_k(
    __half* __restrict__ dst,        // [seq_len, nkv, hd] contiguous output
    const __half* __restrict__ pool, // [pool_pages, page_size, nkv, hd] paged input
    const int* __restrict__ page_ids, // [n_pages] page_id -> pool_page, -1 = zero
    int nkv, int hd, int page_size, int seq_len)
{
    int pos = blockIdx.x;
    int kh = blockIdx.y;
    if (pos >= seq_len || kh >= nkv) return;
    int page_id = pos / page_size;
    int page_off = pos % page_size;
    int pool_page = page_ids[page_id];
    int tx = threadIdx.x;
    for (int i = tx; i < hd; i += blockDim.x) {
        if (pool_page >= 0) {
            dst[(size_t)pos * nkv * hd + (size_t)kh * hd + i] =
                pool[((size_t)pool_page * page_size + page_off) * nkv * hd + (size_t)kh * hd + i];
        } else {
            dst[(size_t)pos * nkv * hd + (size_t)kh * hd + i] = __float2half(0.0f);
        }
    }
}

// ── Embedding lookup kernel (#5): avoids per-token H2D copy ──
// Looks up token_id in the GPU-resident embedding table, applies scale/bias.
__global__ void embed_lookup_k(__half* out, const __half* embed, const __half* ibias, const __half* iscale, const int* d_token_id, int h){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= h) return;
    int token_id = *d_token_id;
    float raw = (float)embed[(size_t)token_id * h + i];
    out[i] = __float2half((raw + (float)ibias[i]) * (float)iscale[i]);
}

// ── Helper kernels ──
__global__ void nan_clean_k(__half*buf,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;float v=__half2float(buf[i]);if(isnan(v)||isinf(v))buf[i]=__float2half(0.0f);}
__global__ void rmsnorm_k(__half*x,const __half*w,int n){__shared__ float r[32];int tx=threadIdx.x,wid=tx/32,l=tx%32;float ss=0;for(int i=tx;i<n;i+=blockDim.x)ss+=(float)x[i]*(float)x[i];for(int o=16;o>0;o>>=1)ss+=__shfl_xor_sync(0xffffffffULL, ss,o);if(l==0)r[wid]=ss;__syncthreads();if(wid==0){ss=(l<(256/32))?r[l]:0;for(int o=16;o>0;o>>=1)ss+=__shfl_xor_sync(0xffffffffULL, ss,o);if(l==0)r[0]=ss;}__syncthreads();float iv=1.0f/sqrtf(r[0]/n+1e-5f);for(int i=tx;i<n;i+=blockDim.x)x[i]=__float2half((float)x[i]*iv*(float)w[i]);}
__global__ void copy_k(__half*d,const __half*s,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;d[i]=s[i];}
__global__ void mm_k(__half*out,const __half*in,const __half*wt,int M,int K){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=M)return;float s=0;for(int k=0;k<K;k++)s+=(float)in[k]*(float)wt[k*(size_t)M+i];if(s>65504.0f)s=65504.0f;else if(s<-65504.0f)s=-65504.0f;out[i]=__float2half(s);}
__global__ void silu_mul_k(__half*out,const __half*g,const __half*u,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;float v=(float)g[i];out[i]=__float2half((v/(1.0f+expf(-v)))*(float)u[i]);}
__global__ void residual_scale_k(__half*out,const __half*res,const float*hs_s,const float*hs_b,const float*res_s,const float*res_b,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;out[i]=__float2half((float)out[i]*hs_s[i]+hs_b[i]+(float)res[i]*res_s[i]+res_b[i]);}

// ── Complex kernels from .hip files ──
#include "zaya_cca_attn.hip"
#include "zaya_gpu_router.hip"
#include "zaya_router_moe.hip"
#include "zaya_moe_tiled_gemv.hip"
#include "zaya_fused_qkv.hip"
#include "zaya_moe_expert_ffn.hip"
#include "zaya_moe_batch_union.hip"
#include "argmax_kernel.hip"
#include "zaya_cca_custom.hip"
#include "v_interleave_kernel.hip"

// ── Persistent thread blocks for MoE expert FFN (single grid, all layers) ──
#include "zaya_persistent_moe.hip"

// ── Reference-faithful CCA Q/K/V prep. ──
#include "zaya_cca_prep.hip"

// ── Post-router skip-expert fixup. ──
#include "zaya_skip_fixup.hip"

// ── GQA V-broadcast for batch path. ──
#include "zaya_batch_v_attn.hip"

// ── Flash-decoding KV-cache attention (ensure kv_cache_attn_fd.hip is linked). ──
extern "C" int rcpp_kv_cache_attn_decode_fd(const void* Q,const void* K,const void* V,void* out,
                                            int num_q_heads,int num_kv_heads,int head_dim,
                                            int seq_len,float scale,void* stream);
extern "C" int rcpp_kv_cache_attn_decode_fd_prealloc(const void* Q,const void* K,const void* V,void* out,float* partials,int num_q_heads,int num_kv_heads,int head_dim,int seq_len,float scale,void* stream);

// ── rocWMMA batched GEMV (requires rocWMMA header library) ──
// Guarded: the CMakeLists.txt sets ROCWMMA_FOUND and the include path.
// If rocWMMA is not available, this kernel is skipped and the engine
// falls back to the scalar-tiled batched GEMV.
#if __has_include(<rocwmma/rocwmma.hpp>) && !defined(WMMA_WAVE32_DISABLED)
#include "zaya_moe_wmma_batched.hip"
#endif

// ── Weight loading (structs LayerW, ZayaState defined in zaya_engine.h) ──

static std::vector<float> load_bin(const std::string& p){
    std::ifstream f(p,std::ios::binary|std::ios::ate);
    if(!f){fprintf(stderr,"Missing: %s\n",p.c_str());return {};}
    size_t n=f.tellg()/sizeof(float);f.seekg(0);
    std::vector<float> d(n);f.read((char*)d.data(),n*sizeof(float));return d;
}
// Load FP16 weight file with correct half→float conversion and overflow guard
static float f16_to_f32(uint16_t h){
    uint32_t sign=(h&0x8000u)<<16;
    int exp=(h>>10)&31;
    uint32_t mant=(h&0x3FFu)<<13;
    if(exp==0){
        if(mant==0){uint32_t r=sign;float f;memcpy(&f,&r,4);return f;}
        while(!(mant&0x400000u)){mant<<=1;exp--;}
        mant&=0x3FFFFFu;
    }else if(exp==31){exp=255;
    }else{exp=exp-15+127;}
    uint32_t r=sign|((uint32_t)exp<<23)|mant;
    float f;memcpy(&f,&r,4);return f;
}
static std::vector<float> load_bin_f16(const std::string& p){
    std::ifstream f(p,std::ios::binary|std::ios::ate);
    if(!f){fprintf(stderr,"Missing: %s\n",p.c_str());return {};}
    size_t sz=f.tellg();f.seekg(0);
    size_t n=sz/2;
    std::vector<uint16_t> tmp(n);
    std::vector<float> d(n);
    f.read((char*)tmp.data(),n*2);
    const float wclip=32.0f;
    for(size_t i=0;i<n;i++){
        float v=f16_to_f32(tmp[i]);
        if(v>wclip)v=wclip;else if(v<-wclip)v=-wclip;
        d[i]=v;
    }
    return d;
}
static std::string L(int i){return std::to_string(i);}
static std::string ensure_trailing_slash(const std::string& s) {
    if (s.empty()) return s;
    return s.back() == '/' ? s : s + "/";
}
#define W(N) load_bin(g_weights_dir+N)
#define WF(N) load_bin_f16(g_weights_dir+N)  // FP16 weight matrices
static void upf16(const std::vector<float>& s,__half*d,int n,hipStream_t h=0){
    if((size_t)n>s.size()){fprintf(stderr,"upf16: expected %d floats, got %zu — aborting upload\n",n,s.size());return;}
    std::vector<__half>b(n);for(int i=0;i<n;i++)b[i]=__float2half(s[i]);
    HIP_OK_V(hipMemcpyAsync(d,b.data(),n*2,hipMemcpyHostToDevice,h));
}
static void upf32(const std::vector<float>& s,float*d,int n,hipStream_t h=0){
    if((size_t)n>s.size()){fprintf(stderr,"upf32: expected %d floats, got %zu — aborting upload\n",n,s.size());return;}
    HIP_OK_V(hipMemcpyAsync(d,s.data(),n*4,hipMemcpyHostToDevice,h));
}

// ── GPU buffer allocation (shared by zaya_init and zaya_init_onebp) ──
// Extracted from zaya_init() so both init paths allocate an identical buffer
// set. Returns false on OOM; the caller destroys the state. Sized off the
// runtime config `eng` (thread_local).
static bool zaya_alloc_buffers(ZayaState* s) {
    auto alloc_f16 = [&](auto& p, size_t n) -> bool {
        hipError_t _e = hipMalloc(&p, n*2);
        if (_e != hipSuccess) { fprintf(stderr,"HIP OOM at %s:%d — %s\n",__FILE__,__LINE__,hipGetErrorString(_e)); return false; }
        return true;
    };
    auto alloc_f32 = [&](auto& p, size_t n) -> bool {
        hipError_t _e = hipMalloc(&p, n*4);
        if (_e != hipSuccess) { fprintf(stderr,"HIP OOM at %s:%d — %s\n",__FILE__,__LINE__,hipGetErrorString(_e)); return false; }
        return true;
    };
    #define ALLOC_OR_FAIL(s, alloc_fn, ptr, n) do { if (!alloc_fn(ptr, n)) { return false; } } while(0)
    // Batch path (zaya_forward_batch) writes per-token slices d_hs + b*eng.h
    // and d_lm_vocab + b*eng.vocab for up to B_MAX tokens — size for B_MAX,
    // not one (issue #1264: OOB GPU writes for any B >= 2).
    constexpr size_t ZAYA_B_MAX = 8;
    ALLOC_OR_FAIL(s, alloc_f16, s->d_hs, eng.h * ZAYA_B_MAX);
    ALLOC_OR_FAIL(s, alloc_f16, s->d_ao, eng.h * ZAYA_B_MAX);
    ALLOC_OR_FAIL(s, alloc_f16, s->d_tmp, std::max(eng.h, 2*eng.n_ff));
    ALLOC_OR_FAIL(s, alloc_f16, s->d_fnw, eng.h);
    ALLOC_OR_FAIL(s, alloc_f16, s->d_lm_out, 4096);
    ALLOC_OR_FAIL(s, alloc_f16, s->d_lm_vocab, eng.vocab * ZAYA_B_MAX);
    // d_argmax_idx, d_sorted_ids, d_expert_counts, d_expert_offsets are
    // declared as int* (and used as int by kernels) but allocated via
    // alloc_f32 since hipMalloc works in bytes and both int/float are 4 B.
    ALLOC_OR_FAIL(s, alloc_f32, s->d_argmax_idx, 1);
    ALLOC_OR_FAIL(s, alloc_f32, s->d_argmax_val, 1);
    ALLOC_OR_FAIL(s, alloc_f32, s->d_sorted_ids, 8);
    ALLOC_OR_FAIL(s, alloc_f32, s->d_expert_counts, 17);
    ALLOC_OR_FAIL(s, alloc_f32, s->d_expert_offsets, 17);
    ALLOC_OR_FAIL(s, alloc_f16, s->d_embed, eng.vocab * eng.h);
    ALLOC_OR_FAIL(s, alloc_f16, s->d_ibias, eng.h);
    ALLOC_OR_FAIL(s, alloc_f16, s->d_iscale, eng.h);
    ALLOC_OR_FAIL(s, alloc_f32, s->d_token_id, 1);
    ALLOC_OR_FAIL(s, alloc_f16, s->d_conv, eng.n_layers * 2 * eng.qkv);
    ALLOC_OR_FAIL(s, alloc_f16, s->d_phs, eng.n_layers * eng.h);
    // KV cache: linear contiguous by default (#3), paged pool as fallback of KV_PAGE_SIZE token pages instead of
    // the full max_seq_len contiguous buffer. Saves ~75% memory at 64K context.
    s->max_seq = eng.max_seq_len > 0 ? eng.max_seq_len : 4096;
    if (s->use_linear_kv) {
        // Linear KV cache (#3): contiguous [n_layers, max_seq, NKV, HD] — zero gather overhead
        size_t kv_elems = (size_t)eng.n_layers * s->max_seq * eng.nkv * eng.hd;
        ALLOC_OR_FAIL(s, alloc_f16, s->d_kcache, kv_elems);
        ALLOC_OR_FAIL(s, alloc_f16, s->d_vcache, kv_elems);
        fprintf(stderr, "  KV cache: linear contiguous %d tok x %d layers = %.1f MB\n",
                s->max_seq, eng.n_layers, (double)kv_elems * 2 / (1024*1024));
    } else {
        // Paged KV cache fallback
        s->page_size = KV_PAGE_SIZE;
        s->n_kv_pages = (s->max_seq + s->page_size - 1) / s->page_size;
        s->kv_pool_pages = eng.kv_pool_pages > 0 ?
            std::min(eng.kv_pool_pages, s->n_kv_pages) :
            std::min(s->n_kv_pages, KV_DEFAULT_PAGES);
        if (s->kv_pool_pages < 1) s->kv_pool_pages = 1;
        size_t kv_pool_elems = (size_t)eng.n_layers * s->kv_pool_pages * s->page_size * eng.nkv * eng.hd;
        ALLOC_OR_FAIL(s, alloc_f16, s->d_kcache, kv_pool_elems);
        ALLOC_OR_FAIL(s, alloc_f16, s->d_vcache, kv_pool_elems);
        fprintf(stderr, "  KV cache: %d pages (%d tok/page, %d pool, %d max_seq) = %.1f MB\n",
                s->n_kv_pages, s->page_size, s->kv_pool_pages, s->max_seq,
                (double)kv_pool_elems * 2 / (1024*1024));
    }
    // Page table / gather buffers only needed for paged KV cache (#3)
    if (!s->use_linear_kv) {
        s->page_alloc.resize(eng.n_layers);
        s->page_map.resize(eng.n_layers);
        s->page_lru.resize(eng.n_layers);
        s->page_next_evict.resize(eng.n_layers);
        for (int il = 0; il < eng.n_layers; il++) {
            s->page_alloc[il].assign(s->n_kv_pages, false);
            s->page_map[il].assign(s->n_kv_pages, -1);
            s->page_lru[il].assign(s->kv_pool_pages, -1);
            s->page_next_evict[il] = 0;
        }
        ALLOC_OR_FAIL(s, alloc_f16, s->d_k_gather, s->max_seq * eng.nkv * eng.hd);
        ALLOC_OR_FAIL(s, alloc_f16, s->d_v_gather, s->max_seq * eng.nkv * eng.hd);
        ALLOC_OR_FAIL(s, alloc_f32, s->d_page_map, s->n_kv_pages);
    }
    s->gather_seq_len = 0;
    // Pre-allocated flash-decoding partials buffer for graph capture (#2)
    {
        int max_tiles = (s->max_seq + 128 - 1) / 128;
        int partials_elems = eng.nq * max_tiles * (eng.hd + 2);
        ALLOC_OR_FAIL(s, alloc_f32, s->d_partials, partials_elems);
    }
    ALLOC_OR_FAIL(s, alloc_f16, s->d_vrec, eng.n_layers * (eng.kd / 2));
    ALLOC_OR_FAIL(s, alloc_f16, s->d_qout, eng.qd);
    ALLOC_OR_FAIL(s, alloc_f16, s->d_kout, eng.kd);
    ALLOC_OR_FAIL(s, alloc_f16, s->d_vout, eng.kd);
    ALLOC_OR_FAIL(s, alloc_f32, s->d_skip_flag, 1);
    ALLOC_OR_FAIL(s, alloc_f32, s->d_prev_rs, (size_t)eng.n_layers * eng.rtr_h);
    ALLOC_OR_FAIL(s, alloc_f32, s->d_expert_idx, 1);
    ALLOC_OR_FAIL(s, alloc_f32, s->d_expert_wt, 1);
    #undef ALLOC_OR_FAIL
    return true;
}

// ── OneBP per-layer weight loader (plan: tensor-name mapping table) ──
// Maps OneBP tensor names (GGUF `blk.N.*` preserved verbatim by
// gguf_to_onebp) onto the engine's LayerW GPU buffers.
//
// Zaya layers ALTERNATE: even layers are CCA-attention only, odd layers are
// MoE-only (llama.cpp zaya.cpp: "ZAYA1-8B: 40 sub-blocks (20 CCA + 20 MoE)").
// A sublayer is present iff its tensors exist in the file, so:
//   * missing weight  → pointer stays null → zaya_forward's guards skip that
//     sublayer (same pattern as the reference tests/zaya_gpu_decode.cpp;
//     there is NO dense-FFN fallback in this architecture — a layer with no
//     MoE tensors simply has no MLP, and the old comment claiming a "dense
//     fallback" was fiction, issue #1527).
//   * present but wrong-sized, PARTIAL groups (some of a sublayer's tensors
//     only), or a layer with neither sublayer → ABORT init loudly. Zero-
//     filling attention/FFN weights used to "load successfully" and emit
//     garbage logits.
//   * optional BIAS tensors (conv/router/residual-scale biases) missing →
//     zero-fill (bias=0 is a correct default; the kernels dereference them
//     unconditionally — llama.cpp marks them TENSOR_NOT_REQUIRED).
// Returns false on any fatal condition (alloc failure, size mismatch,
// inconsistent layer structure).

// Issue #1527 diagnostics: list every tensor of one layer so a mapping
// mismatch (wrong GGUF names, e.g. ssm_conv1d vs cca_conv_grp) is visible
// in the abort message instead of degrading to zero weights.
static void dump_layer_tensors(const NpuOnebpModel& model, const std::string& p) {
    for (int i = 0; i < model.tensor_count(); i++) {
        const auto* t = model.tensor(i);
        if (t && t->name.rfind(p, 0) == 0) {
            fprintf(stderr, "    %s [ndim=%d %dx%dx%d quant=%u]\n",
                    t->name.c_str(), t->ndim, t->num_experts, t->rows, t->cols,
                    (unsigned)t->quant);
        }
    }
}

static bool load_layer_onebp(NpuOnebpModel& model, int il, LayerW& l,
                             const ZayaConfig& eng, ZayaState* s, hipStream_t st) {
    const std::string p = "blk." + std::to_string(il) + ".";
    bool ok = true;

    // Issue #1521: an allocation failure mid-load used to be silent (the layer
    // just "failed to load" with no reason). The 74B's fp16 expert buffers
    // (~145 GB) exceed the 62 GB Strix Halo pool — without this print that OOM
    // looked like a loader mapping bug.
// Issue #1521: an allocation failure mid-load used to be silent (the layer
// just "failed to load" with no reason). The 74B's fp16 expert buffers
// (~145 GB) exceed the 62 GB Strix Halo pool — without this print that OOM
// looked like a loader mapping bug.
#define LOADER_ALLOC_OR_FAIL(expr, what)                                          \
    do {                                                                          \
        if ((expr) != hipSuccess) {                                               \
            fprintf(stderr, "  onebp: %s allocation failed (%s) — aborting\n",   \
                    (what), hipGetErrorString(hipGetLastError()));                \
            ok = false;                                                           \
        }                                                                         \
    } while (0)

    auto alloc_fail = [&](const char* what) {
        fprintf(stderr, "  onebp: %s allocation failed (%s) — aborting\n",
                what, hipGetErrorString(hipGetLastError()));
        ok = false;
    };

    // FP16 upload. optional=true (bias terms): missing → zero-fill (bias=0
    // default). optional=false (weights): missing → gpu stays null (sublayer
    // absent, valid on the alternating layers). Any present-but-wrong-sized
    // tensor is fatal (issue #1527).
    auto f16 = [&](const char* n, __half*& gpu, int count, bool optional) {
        gpu = nullptr;
        std::vector<float> v;
        if (!model.get_tensor_f32((p + n).c_str(), v)) {
            if (optional) {
                fprintf(stderr, "  onebp: '%s%s' absent — zero-filled (optional bias)\n", p.c_str(), n);
                if (hipMalloc(&gpu, (size_t)count * 2) != hipSuccess) { alloc_fail("f16"); return; }
                (void)hipMemsetAsync(gpu, 0, (size_t)count * 2, st);
            }
            return;
        }
        if ((int)v.size() != count) {
            fprintf(stderr, "  onebp: '%s%s' size %zu != %d — aborting (a weight path must never be zero-filled)\n",
                    p.c_str(), n, v.size(), count);
            ok = false; return;
        }
        if (hipMalloc(&gpu, (size_t)count * 2) != hipSuccess) { alloc_fail("f16"); return; }
        upf16(v, gpu, count, st);
    };
    // FP32 upload — same semantics as f16.
    auto f32 = [&](const char* n, float*& gpu, int count, bool optional) {
        gpu = nullptr;
        std::vector<float> v;
        if (!model.get_tensor_f32((p + n).c_str(), v)) {
            if (optional) {
                fprintf(stderr, "  onebp: '%s%s' absent — zero-filled (optional bias)\n", p.c_str(), n);
                if (hipMalloc(&gpu, (size_t)count * 4) != hipSuccess) { alloc_fail("f32"); return; }
                (void)hipMemsetAsync(gpu, 0, (size_t)count * 4, st);
            }
            return;
        }
        if ((int)v.size() != count) {
            fprintf(stderr, "  onebp: '%s%s' size %zu != %d — aborting (a weight path must never be zero-filled)\n",
                    p.c_str(), n, v.size(), count);
            ok = false; return;
        }
        if (hipMalloc(&gpu, (size_t)count * 4) != hipSuccess) { alloc_fail("f32"); return; }
        upf32(v, gpu, count, st);
    };

    // ── Attention group (even layers) ──
    // attn_norm: required on EVERY layer — it is the pre-norm that runs
    // before either sublayer (llama.cpp builds it for all layers).
    f16("attn_norm.weight", l.nw, eng.h, false);
    if (!ok) return false;
    if (!l.nw) {
        fprintf(stderr, "  onebp: '%sattn_norm.weight' missing — required on every layer — aborting\n", p.c_str());
        return false;
    }
    // QKV projections (issue #1521): GGUF stores [in=H, out=d] (llama.cpp
    // mul_mat convention); the engine's moe_tiled_gemv reads [out=d, in=H].
    // Accept both orientations — transpose the GGUF one, take 8B-era dumps
    // ([d, h]) as-is. The old loader took the file as-is → garbage for any
    // GGUF-derived .1bp.
    auto f16_mm = [&](const char* n, __half*& gpu, int d) {
        gpu = nullptr;
        std::string nm = p + n;
        auto* te = model.find_tensor(nm.c_str());
        int rows = te ? te->rows : 0, cols = te ? te->cols : 0;
        std::vector<float> v;
        if (!model.get_tensor_f32(nm.c_str(), v)) return;  // sublayer absent
        if (rows == d && cols == eng.h && (int)v.size() >= (size_t)d * eng.h) {
            if (hipMalloc(&gpu, (size_t)d * eng.h * 2) != hipSuccess) { alloc_fail("qkv proj"); return; }
            upf16(v, gpu, d * eng.h, st);
        } else if (rows == eng.h && cols == d && (int)v.size() >= (size_t)d * eng.h) {
            std::vector<float> tr((size_t)d * eng.h);
            for (int i = 0; i < d; i++)
                for (int j = 0; j < eng.h; j++)
                    tr[(size_t)i * eng.h + j] = v[(size_t)j * d + i];
            if (hipMalloc(&gpu, (size_t)d * eng.h * 2) != hipSuccess) { alloc_fail("qkv proj"); return; }
            upf16(tr, gpu, d * eng.h, st);
        } else {
            fprintf(stderr, "  onebp: '%s' dims [%d,%d] size %zu don't match [d=%d,H=%d] either way — aborting\n",
                    nm.c_str(), rows, cols, v.size(), d, eng.h);
            ok = false;
        }
    };
    f16_mm("attn_q.weight", l.wq, eng.qd);
    f16_mm("attn_k.weight", l.wk, eng.kd);
    f16_mm("cca_val_proj1.weight", l.wv1, eng.kd / 2);
    f16_mm("cca_val_proj2.weight", l.wv2, eng.kd / 2);
    f16("attn_norm_2.weight", l.pan, eng.h, false);  // .bin-era only; absent in GGUF → null → skipped at use
    f32("cca_k_scale.weight", l.ks, eng.nkv, false);
    f32("res_scale_hs.weight", l.pahss, eng.h, false);
    f32("res_scale_res.weight", l.parss, eng.h, true);  // TENSOR_NOT_REQUIRED in llama.cpp — absent on attention layers in the real 74B; zero = residual term dropped
    if (!ok) return false;

    // CCA conv weights (issue #1521). GGUF names: ssm_conv1d (depthwise,
    // [t=2, qkv]) and cca_conv_grp (grouped, 3D [t, qkv/n_groups, qkv]). The
    // cca_prep kernel reads cdw[c*2+t] and cgw[oc*2*gc + j*2 + t]. gguf_to_onebp
    // stores 2D tensors transposed (rows=shape[1], cols=shape[0]), so the
    // converted ssm_conv1d is already [c, t] = engine-native; accept both
    // orientations. The old names (cca_conv_grp.weight [qkv,2],
    // cca_conv_grp_grouped.*) do not exist in zaya GGUFs.
    {
        std::string nm = p + "ssm_conv1d.weight";
        auto* te = model.find_tensor(nm.c_str());
        int rows = te ? te->rows : 0, cols = te ? te->cols : 0;
        std::vector<float> v;
        l.cdw = nullptr;
        if (!te) { /* sublayer absent */ }
        else if (model.get_tensor_f32(nm.c_str(), v) && rows == eng.qkv && cols == 2 && (int)v.size() >= eng.qkv * 2) {
            // [c, t] — engine-native (converter's transposed 2D layout)
            if (hipMalloc(&l.cdw, (size_t)eng.qkv * 2 * 4) != hipSuccess) alloc_fail("ssm_conv1d");
            else upf32(v, l.cdw, eng.qkv * 2, st);
        } else if (model.get_tensor_f32(nm.c_str(), v) && rows == 2 && cols == eng.qkv && (int)v.size() >= eng.qkv * 2) {
            std::vector<float> tr((size_t)eng.qkv * 2);
            for (int c = 0; c < eng.qkv; c++)
                for (int t = 0; t < 2; t++)
                    tr[(size_t)c * 2 + t] = v[(size_t)t * eng.qkv + c];
            if (hipMalloc(&l.cdw, (size_t)eng.qkv * 2 * 4) != hipSuccess) alloc_fail("ssm_conv1d");
            else upf32(tr, l.cdw, eng.qkv * 2, st);
        } else {
            fprintf(stderr, "  onebp: '%s' dims [%d,%d] size %zu don't match [qkv=%d,2] either way — aborting\n",
                    nm.c_str(), rows, cols, v.size(), eng.qkv);
            ok = false;
        }
    }
    if (!ok) return false;
    {
        std::string nm = p + "cca_conv_grp.weight";
        auto* te = model.find_tensor(nm.c_str());
        const int gc = eng.qkv / (eng.nq + eng.nkv);  // group width (128 for 8B/74B)
        l.cgw = nullptr;
        if (!te) { /* sublayer absent */ }
        else if (te->ndim != 3 || te->num_experts != 2 || te->rows != gc || te->cols != eng.qkv) {
            fprintf(stderr, "  onebp: '%s' is [%d x %d x %d], want 2 x %d x %d — aborting\n",
                    nm.c_str(), te->num_experts, te->rows, te->cols, gc, eng.qkv);
            ok = false;
        } else {
            std::vector<float> tr((size_t)eng.qkv * gc * 2);
            for (int t = 0; t < 2; t++) {
                std::vector<float> v;
                if (!model.get_tensor_f32_expert(nm.c_str(), t, v) || (int)v.size() < (size_t)gc * eng.qkv) {
                    fprintf(stderr, "  onebp: '%s' expert %d/2 missing/short — aborting\n", nm.c_str(), t);
                    ok = false;
                    break;
                }
                for (int j = 0; j < gc; j++)
                    for (int oc = 0; oc < eng.qkv; oc++)
                        tr[(size_t)oc * gc * 2 + j * 2 + t] = v[(size_t)j * eng.qkv + oc];
            }
            if (ok) {
                if (hipMalloc(&l.cgw, (size_t)eng.qkv * gc * 2 * 4) != hipSuccess) alloc_fail("cca_conv_grp");
                if (ok) upf32(tr, l.cgw, eng.qkv * gc * 2, st);
            }
        }
    }
    if (!ok) return false;
    // Conv / residual-scale BIASES are optional (bias=0 default).
    f32("ssm_conv1d.bias", l.cdb, eng.qkv, true);
    f32("cca_conv_grp.bias", l.cgb, eng.qkv, true);
    f32("res_scale_hs.bias", l.pahsb, eng.h, true);
    f32("res_scale_res.bias", l.parsb, eng.h, true);
    if (!ok) return false;

    // wo (attn_output.weight) — kernel wants [H, QD] (moe_tiled_gemv M=H,K=QD).
    // OneBP may store [H, QD] or the transpose [QD, H]: verify against the
    // recorded dims and transpose only when needed (plan step 6).
    {
        std::string nm = p + "attn_output.weight";
        std::vector<float> v;
        auto* te = model.find_tensor(nm.c_str());
        int rows = te ? te->rows : 0, cols = te ? te->cols : 0;
        bool found = model.get_tensor_f32(nm.c_str(), v);
        l.wo = nullptr;
        if (found && rows == eng.h && cols == eng.qd && (int)v.size() >= eng.h * eng.qd) {
            if (hipMalloc(&l.wo, (size_t)eng.h * eng.qd * 2) != hipSuccess) alloc_fail("attn_output");
            else upf16(v, l.wo, eng.h * eng.qd, st);
        } else if (found && rows == eng.qd && cols == eng.h && (int)v.size() >= eng.h * eng.qd) {
            std::vector<float> tr((size_t)eng.h * eng.qd);
            for (int i = 0; i < eng.h; i++)
                for (int k = 0; k < eng.qd; k++)
                    tr[(size_t)i * eng.qd + k] = v[(size_t)k * eng.h + i];
            if (hipMalloc(&l.wo, (size_t)eng.h * eng.qd * 2) != hipSuccess) alloc_fail("attn_output");
            else upf16(tr, l.wo, eng.h * eng.qd, st);
        } else if (found) {
            fprintf(stderr, "  onebp: '%s' dims [%d,%d] size %zu don't match [H=%d,QD=%d] — aborting\n",
                    nm.c_str(), rows, cols, v.size(), eng.h, eng.qd);
            ok = false;
        }
        // not found → l.wo stays null (attention sublayer absent on odd layers)
    }
    if (!ok) return false;

    // ── Router group (odd layers) — GGUF topology (issue #1521):
    //   down(ffn_gate_inp) → norm(ffn_norm) → mlp0(ffn_gate) → gelu →
    //   mlp2(zaya_router_mlp2) → gelu → mlp4(zaya_router_mlp4) → softmax
    // maps 1:1 onto the engine kernel (gdw → rfn → rf1 → gelu → rf2 → gelu
    // → rout). The old 8B-era names (zaya_router_mlp2→rf1, mlp4-pad→rf2,
    // out_proj→rout) do not exist in zaya GGUFs and were silently
    // zero-filled/garbage before the strict loader.
    // gdw (ffn_gate_inp.weight) — engine reads [H, rtr_h] stride-1
    // (gdw[j*rtr_h+i]); GGUF stores [in=H, out=rtr_h] — direct, with the
    // transpose branch for [rtr_h, H] dumps.
    {
        std::string nm = p + "ffn_gate_inp.weight";
        std::vector<float> v;
        auto* te = model.find_tensor(nm.c_str());
        int rows = te ? te->rows : 0, cols = te ? te->cols : 0;
        bool found = model.get_tensor_f32(nm.c_str(), v);
        l.gdw = nullptr;
        if (found && rows == eng.h && cols == eng.rtr_h && (int)v.size() >= eng.h * eng.rtr_h) {
            if (hipMalloc(&l.gdw, (size_t)eng.h * eng.rtr_h * 4) != hipSuccess) alloc_fail("ffn_gate_inp");
            else upf32(v, l.gdw, eng.h * eng.rtr_h, st);
        } else if (found && rows == eng.rtr_h && cols == eng.h && (int)v.size() >= eng.h * eng.rtr_h) {
            std::vector<float> tr((size_t)eng.h * eng.rtr_h);
            for (int i = 0; i < eng.rtr_h; i++)
                for (int j = 0; j < eng.h; j++)
                    tr[(size_t)j * eng.rtr_h + i] = v[(size_t)i * eng.h + j];
            if (hipMalloc(&l.gdw, (size_t)eng.h * eng.rtr_h * 4) != hipSuccess) alloc_fail("ffn_gate_inp");
            else upf32(tr, l.gdw, eng.h * eng.rtr_h, st);
        } else if (found) {
            fprintf(stderr, "  onebp: '%s' dims [%d,%d] size %zu don't match [H=%d,rtr_h=%d] — aborting\n",
                    nm.c_str(), rows, cols, v.size(), eng.h, eng.rtr_h);
            ok = false;
        }
        // not found → l.gdw stays null (MoE sublayer absent on even layers)
    }
    if (!ok) return false;
    // Router biases/norms (biases optional, bias=0 default).
    f32("ffn_gate_inp.bias", l.gdb, eng.rtr_h, true);
    f32("ffn_norm.weight", l.rfn, eng.rtr_h, false);
    f32("ffn_gate.weight", l.rf1, eng.rtr_h * eng.rtr_h, false);        // mlp0
    f32("ffn_gate.bias", l.rf1b, eng.rtr_h, true);
    f32("zaya_router_mlp2.weight", l.rf2, eng.rtr_h * eng.rtr_h, false); // mlp2
    f32("zaya_router_mlp2.bias", l.rf2b, eng.rtr_h, true);
    f32("zaya_router_biases.weight", l.bb, eng.n_exp_t, true);
    if (!ok) return false;

    // rout (zaya_router_mlp4.weight) — the GGUF's final projection is
    // [rtr_h, n_exp_t] (mlp4, llama.cpp mul_mat [in, out]); the engine's
    // router kernel reads [out=n_exp_t, in=rtr_h] — transpose. Accept
    // [n_exp_t, rtr_h] dumps as-is. (The old zero-pad-to-square quirk was
    // based on a stale plan and produced garbage for the real 74B.)
    {
        std::string nm = p + "zaya_router_mlp4.weight";
        std::vector<float> v;
        auto* te = model.find_tensor(nm.c_str());
        int rows = te ? te->rows : 0, cols = te ? te->cols : 0;
        bool found = model.get_tensor_f32(nm.c_str(), v);
        l.rout = nullptr;
        if (!found) { /* sublayer absent */ }
        else if (found && rows == eng.rtr_h && cols == eng.n_exp_t && (int)v.size() >= (size_t)eng.rtr_h * eng.n_exp_t) {
            std::vector<float> tr((size_t)eng.n_exp_t * eng.rtr_h);
            for (int i = 0; i < eng.n_exp_t; i++)
                for (int j = 0; j < eng.rtr_h; j++)
                    tr[(size_t)i * eng.rtr_h + j] = v[(size_t)j * eng.n_exp_t + i];
            if (hipMalloc(&l.rout, (size_t)eng.n_exp_t * eng.rtr_h * 4) != hipSuccess) alloc_fail("zaya_router_mlp4");
            else upf32(tr, l.rout, eng.n_exp_t * eng.rtr_h, st);
        } else if (found && rows == eng.n_exp_t && cols == eng.rtr_h && (int)v.size() >= (size_t)eng.n_exp_t * eng.rtr_h) {
            if (hipMalloc(&l.rout, (size_t)eng.n_exp_t * eng.rtr_h * 4) != hipSuccess) alloc_fail("zaya_router_mlp4");
            else upf32(v, l.rout, eng.n_exp_t * eng.rtr_h, st);
        } else {
            fprintf(stderr, "  onebp: '%s' dims [%d,%d] size %zu don't match [rtr_h=%d,n_exp_t=%d] either way — aborting\n",
                    nm.c_str(), rows, cols, v.size(), eng.rtr_h, eng.n_exp_t);
            ok = false;
        }
    }
    if (!ok) return false;

    // EDA scale (zaya_router_eda.weight [rtr_h] → mean scalar; same handling
    // as the .bin loader's router_states_scale). Optional.
    {
        std::string nm = p + "zaya_router_eda.weight";
        std::vector<float> v;
        s->has_eda[il] = false;
        s->eda_scale[il] = 0.0f;
        if (model.get_tensor_f32(nm.c_str(), v) && (int)v.size() >= eng.rtr_h) {
            float sum = 0; for (float x : v) sum += x;
            s->eda_scale[il] = sum / (float)eng.rtr_h;
            s->has_eda[il] = true;
        }
    }

    // ── MoE experts (plan step 5): ndim=3 [NE, R, C], dequant each expert
    // via get_tensor_f32_expert and concatenate. gu = [NE, 2*n_ff, H] with
    // gate rows [0,n_ff) then up rows [n_ff,2*n_ff) per expert (the layout
    // zaya_moe_expert_ffn.hip reads); dn = [NE, H, n_ff]. Missing → null
    // (MoE sublayer absent on even layers); expert-count mismatch or a
    // short expert → abort (issue #1527 — never "skip MoE" silently).
    //
    // Streamed per-expert (issue #1529): dequant one expert to fp32, convert
    // to fp16, and async-copy into the preallocated half buffer. The old
    // all-experts fp32 concat peaked at ~19 GB transient host allocation for
    // the 74B (24 experts × 2*n_ff × H × 4 B) and churned ~2.3 TB of host
    // traffic per model load; per-expert the peak is rows*cols*(4+2) B and
    // the next expert's dequant overlaps with the previous upload.
    {
        auto experts = [&](const char* n, __half*& gpu, int rows, int cols) {
            std::string nm = p + n;
            auto* te = model.find_tensor(nm.c_str());
            gpu = nullptr;
            if (!te) return;  // sublayer absent
            if (te->ndim != 3 || te->num_experts != eng.n_exp) {
                fprintf(stderr, "  onebp: '%s' is [%d x %d x %d], want %d-expert ndim=3 — aborting\n",
                        nm.c_str(), te->num_experts, te->rows, te->cols, eng.n_exp);
                ok = false;
                return;
            }
            if (hipMalloc(&gpu, (size_t)eng.n_exp * rows * cols * 2) != hipSuccess) { alloc_fail("experts"); return; }
            std::vector<__half> hbuf((size_t)rows * cols);
            for (int e = 0; e < eng.n_exp; e++) {
                std::vector<float> v;
                if (!model.get_tensor_f32_expert(nm.c_str(), e, v) || (int)v.size() < rows * cols) {
                    fprintf(stderr, "  onebp: '%s' expert %d/%d missing/short — aborting\n",
                            nm.c_str(), e, eng.n_exp);
                    ok = false;
                    return;
                }
                for (size_t i = 0; i < (size_t)rows * cols; i++) hbuf[i] = __float2half(v[i]);
                (void)hipMemcpyAsync(gpu + (size_t)e * rows * cols, hbuf.data(),
                                     (size_t)rows * cols * 2, hipMemcpyHostToDevice, st);
            }
        };
        experts("ffn_gate_up_exps.weight", l.gu, 2 * eng.n_ff, eng.h);
        experts("ffn_down_exps.weight", l.dn, eng.h, eng.n_ff);
    }
    if (!ok) return false;

    // ── Post-MLP residual scales. The reference llama.cpp loader falls back
    // to the attention res_scale when res_scale_hs.mlp.* is absent, so a
    // missing .mlp group is fine (and it is never used on attention-only
    // layers). A PARTIAL .mlp group is a broken file. The scales are
    // weights: residual_scale_k dereferences them unconditionally, so a MoE
    // layer with no scales at all is fatal too.
    f32("res_scale_hs.mlp.weight", l.pmhss, eng.h, false);
    f32("res_scale_res.mlp.weight", l.pmrss, eng.h, false);
    if (!ok) return false;
    f32("res_scale_hs.mlp.bias", l.pmhsb, eng.h, true);
    f32("res_scale_res.mlp.bias", l.pmrsb, eng.h, true);
    if (!ok) return false;
    if ((l.pmhss != nullptr) != (l.pmrss != nullptr)) {
        fprintf(stderr, "  onebp: layer %d has a PARTIAL res_scale_hs.mlp group — aborting\n", il);
        dump_layer_tensors(model, p);
        return false;
    }
    if (!l.pmhss && l.pahss && l.parss) {
        // llama.cpp fallback: res_scale_hs_mlp ? res_scale_hs_mlp : res_scale_hs
        l.pmhss = l.pahss; l.pmhsb = l.pahsb; l.pmrss = l.parss; l.pmrsb = l.parsb;
    }

    // ── Sublayer consistency (issue #1527) ──
    // A valid layer has ALL of a sublayer's weight tensors or NONE of them
    // (even layers are attention-only, odd layers are MoE-only). Anything
    // else means a broken or mismatched file — abort rather than silently
    // run garbage through zeroed weights. res_scale_hs/res are excluded:
    // they exist on EVERY layer in the reference format.
    const bool attn_any = l.wq || l.wk || l.wv1 || l.wv2 || l.wo || l.cdw || l.cgw || l.ks;
    const bool attn_all = l.wq && l.wk && l.wv1 && l.wv2 && l.wo && l.cdw && l.cgw && l.ks;
    const bool moe_any  = l.gdw || l.rfn || l.rf1 || l.rf2 || l.rout || l.gu || l.dn;
    const bool moe_all  = l.gdw && l.rfn && l.rf1 && l.rf2 && l.rout && l.gu && l.dn;
    if (attn_any != attn_all) {
        fprintf(stderr, "  onebp: layer %d has PARTIAL attention tensors (some but not all) — broken or mismatched file, aborting\n", il);
        dump_layer_tensors(model, p);
        return false;
    }
    if (moe_any != moe_all) {
        fprintf(stderr, "  onebp: layer %d has PARTIAL MoE tensors (router without experts or vice versa) — aborting\n", il);
        dump_layer_tensors(model, p);
        return false;
    }
    if (!attn_any && !moe_any) {
        fprintf(stderr, "  onebp: layer %d has NEITHER attention nor MoE weights — aborting\n", il);
        dump_layer_tensors(model, p);
        return false;
    }
    if (attn_all && !l.pahss) {
        fprintf(stderr, "  onebp: layer %d has attention but no res_scale_hs — aborting\n", il);
        dump_layer_tensors(model, p);
        return false;
    }
    if (moe_all && !l.pmhss) {
        fprintf(stderr, "  onebp: layer %d has MoE but no residual scales (res_scale_hs.mlp.* or res_scale_hs.*) — aborting\n", il);
        dump_layer_tensors(model, p);
        return false;
    }

    return ok;
}

extern "C" {

// ── WMMA defines (redefined after zaya_moe_wmma_batched.hip undefs them) ──
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 64
#define WMMA_THREADS 128

// ── Init: load weights, allocate GPU memory ──
// ── Resolve weights directory ──
// thread_local so multiple engines in different threads can load different models.
// Priority: 1. weights_dir parameter  2. ZAYA_WEIGHTS_DIR env  3. XDG_DATA_HOME  4. HOME  5. /tmp
static thread_local std::string g_weights_dir;
static void resolve_weights_dir(const char* weights_dir) {
    if (weights_dir && weights_dir[0]) {
        // Reject path traversal attempts (fixes #1328)
        std::string wd(weights_dir);
        if (wd.find("..") != std::string::npos) {
            fprintf(stderr, "ERROR: weights directory contains '..' — rejected for security\n");
            g_weights_dir.clear();
            return;
        }
        g_weights_dir = ensure_trailing_slash(weights_dir);
        return;
    }
    const char* env = getenv("ZAYA_WEIGHTS_DIR");
    if (env && env[0]) { g_weights_dir = ensure_trailing_slash(env); return; }
    const char* xdg = getenv("XDG_DATA_HOME");
    if (xdg && xdg[0]) { g_weights_dir = std::string(xdg) + "/1bit-systems/weights/"; return; }
    const char* home = getenv("HOME");
    if (home && home[0]) { g_weights_dir = std::string(home) + "/.local/share/1bit-systems/weights/"; return; }
    fprintf(stderr, "ERROR: No weights directory found — set ZAYA_WEIGHTS_DIR or ensure HOME is set\n");
    g_weights_dir.clear();  // fixes #1332: never default to /tmp
}

ZayaState* zaya_init(const char* weights_dir, const ZayaConfig* cfg) {
    // Use the passed-in weights_dir (or env fallback)
    resolve_weights_dir(weights_dir);
    // Populate runtime config from the provided ZayaConfig (or default Zaya1-8B)
    if (cfg) {
        eng = *cfg;
    } else {
        eng = ZayaConfig::zaya1_8b();
    }

    ZayaState* s = new (std::nothrow) ZayaState();
    if (!s) {
        fprintf(stderr, "zaya_init: failed to allocate ZayaState (OOM)\n");
        return nullptr;
    }
    HIP_OK_R(hipStreamCreate(&s->st), nullptr);
    
    s->embed = W("model_embed_tokens_weight.bin");
    auto fnorm = W("model_norm_weight.bin");
    s->iscale = W("model_input_hidden_states_scale.bin");
    s->ibias = W("model_input_hidden_states_bias.bin");

    // If any of the four initial weight files is missing, abort init gracefully
    // instead of crashing downstream (fixes #61).
    if (s->embed.empty() || fnorm.empty() || s->iscale.empty() || s->ibias.empty()) {
        fprintf(stderr, "zaya_init: failed to load one or more initial weight files — aborting init\n");
        fprintf(stderr, "  Missing: %s%s%s%s\n",
                s->embed.empty() ? "model_embed_tokens_weight.bin " : "",
                fnorm.empty() ? "model_norm_weight.bin " : "",
                s->iscale.empty() ? "model_input_hidden_states_scale.bin " : "",
                s->ibias.empty() ? "model_input_hidden_states_bias.bin " : "");
        fprintf(stderr, "  This model is not in Zaya .bin format — use ZINC/Vulkan or CPU backend, or convert to Zaya.\n");
        zaya_destroy(s);
        return nullptr;
    }

    // Verify a GPU is available before attempting allocations
    int ndev = 0;
    HIP_OK_R(hipGetDeviceCount(&ndev), nullptr);
    if (ndev < 1) {
        fprintf(stderr, "zaya_init: No HIP-capable GPU found (device count=%d).\n", ndev);
        zaya_destroy(s);
        return nullptr;
    }

    // Dimension validation: check loaded weights match expected config dimensions.
    size_t expected_embed = (size_t)eng.vocab * eng.h;
    if (s->embed.size() != expected_embed) {
        fprintf(stderr, "zaya_init: model embed size %zu != expected %zu (cfg H=%d, vocab=%d)\n",
                s->embed.size(), expected_embed, eng.h, eng.vocab);
        fprintf(stderr, "  Engine configured for H=%d, NQ=%d, NKV=%d, L=%d, V=%d.\n",
                eng.h, eng.nq, eng.nkv, eng.n_layers, eng.vocab);
        fprintf(stderr, "  Refusing to load — would produce silent garbage.\n");
        zaya_destroy(s);
        return nullptr;
    }

    // Allocate GPU buffers (shared with zaya_init_onebp via zaya_alloc_buffers)
    if (!zaya_alloc_buffers(s)) { zaya_destroy(s); return nullptr; }
    
    upf16(s->embed,s->d_embed,eng.vocab*eng.h,s->st);
    // Upload ibias/iscale to GPU for device-side embedding lookup (#5)
    std::vector<__half> h_ibias(eng.h), h_iscale(eng.h);
    for(int i=0;i<eng.h;i++){h_ibias[i]=__float2half(s->ibias[i]);h_iscale[i]=__float2half(s->iscale[i]);}
    HIP_OK_R(hipMemcpy(s->d_ibias,h_ibias.data(),eng.h*2,hipMemcpyHostToDevice), nullptr);
    HIP_OK_R(hipMemcpy(s->d_iscale,h_iscale.data(),eng.h*2,hipMemcpyHostToDevice), nullptr);
    // Final norm
    std::vector<__half>hf(eng.h);for(int i=0;i<eng.h;i++)hf[i]=__float2half(fnorm[i]);
    HIP_OK_R(hipMemcpy(s->d_fnw,hf.data(),eng.h*2,hipMemcpyHostToDevice), nullptr);
    
    auto A=[&](auto&p,int n)->bool{hipError_t _e=hipMalloc(&p,n*2);if(_e!=hipSuccess){fprintf(stderr,"HIP Error %d at %s:%d — %s\n",_e,__FILE__,__LINE__,hipGetErrorString(_e));return false;}return true;};
    auto B=[&](auto&p,int n)->bool{hipError_t _e=hipMalloc(&p,n*4);if(_e!=hipSuccess){fprintf(stderr,"HIP Error %d at %s:%d — %s\n",_e,__FILE__,__LINE__,hipGetErrorString(_e));return false;}return true;};
    
    s->lw.resize(eng.n_layers);
    s->has_eda.resize(eng.n_layers);
    s->eda_scale.resize(eng.n_layers);
    for(int il=0;il<eng.n_layers;il++){
        auto& l=s->lw[il];
        if(!A(l.nw,eng.h)){zaya_destroy(s);return nullptr;}upf16(W("model_layers_"+L(il)+"_input_layernorm_weight.bin"),l.nw,eng.h,s->st);
        if(!A(l.wq,eng.qd*eng.h)){zaya_destroy(s);return nullptr;}upf16(WF("model_layers_"+L(il)+"_self_attn_qkv_proj_q_proj_weight.bin"),l.wq,eng.qd*eng.h,s->st);
        if(!A(l.wk,eng.kd*eng.h)){zaya_destroy(s);return nullptr;}upf16(WF("model_layers_"+L(il)+"_self_attn_qkv_proj_k_proj_weight.bin"),l.wk,eng.kd*eng.h,s->st);
        if(!A(l.wv1,(eng.kd/2)*eng.h)){zaya_destroy(s);return nullptr;}upf16(WF("model_layers_"+L(il)+"_self_attn_qkv_proj_v_proj_current_weight.bin"),l.wv1,(eng.kd/2)*eng.h,s->st);
        if(!A(l.wv2,(eng.kd/2)*eng.h)){zaya_destroy(s);return nullptr;}upf16(WF("model_layers_"+L(il)+"_self_attn_qkv_proj_v_proj_delayed_weight.bin"),l.wv2,(eng.kd/2)*eng.h,s->st);
        if(!A(l.wo,eng.h*eng.qd)){zaya_destroy(s);return nullptr;}upf16(WF("model_layers_"+L(il)+"_self_attn_o_proj_weight.bin"),l.wo,eng.h*eng.qd,s->st);
        if(!B(l.cdw,eng.qkv*2)){zaya_destroy(s);return nullptr;}upf32(WF("model_layers_"+L(il)+"_self_attn_qkv_proj_conv_qk_depthwise_weight.bin"),l.cdw,eng.qkv*2,s->st);
        if(!B(l.cdb,eng.qkv)){zaya_destroy(s);return nullptr;}upf32(WF("model_layers_"+L(il)+"_self_attn_qkv_proj_conv_qk_depthwise_bias.bin"),l.cdb,eng.qkv,s->st);
        if(!B(l.cgw,eng.qkv*128*2)){zaya_destroy(s);return nullptr;}upf32(WF("model_layers_"+L(il)+"_self_attn_qkv_proj_conv_qk_grouped_weight.bin"),l.cgw,eng.qkv*128*2,s->st);
        if(!B(l.cgb,eng.qkv)){zaya_destroy(s);return nullptr;}upf32(WF("model_layers_"+L(il)+"_self_attn_qkv_proj_conv_qk_grouped_bias.bin"),l.cgb,eng.qkv,s->st);
        if(!B(l.ks,eng.nkv)){zaya_destroy(s);return nullptr;}upf32(W("model_layers_"+L(il)+"_self_attn_qk_norm_temp.bin"),l.ks,eng.nkv,s->st);
        if(!B(l.pahss,eng.h)){zaya_destroy(s);return nullptr;}upf32(W("model_layers_"+L(il)+"_post_attention_residual_scale_hidden_states_scale.bin"),l.pahss,eng.h,s->st);
        if(!B(l.pahsb,eng.h)){zaya_destroy(s);return nullptr;}upf32(W("model_layers_"+L(il)+"_post_attention_residual_scale_hidden_states_bias.bin"),l.pahsb,eng.h,s->st);
        if(!B(l.parss,eng.h)){zaya_destroy(s);return nullptr;}upf32(W("model_layers_"+L(il)+"_post_attention_residual_scale_residual_scale.bin"),l.parss,eng.h,s->st);
        if(!B(l.parsb,eng.h)){zaya_destroy(s);return nullptr;}upf32(W("model_layers_"+L(il)+"_post_attention_residual_scale_residual_bias.bin"),l.parsb,eng.h,s->st);
        if(!A(l.pan,eng.h)){zaya_destroy(s);return nullptr;}upf16(W("model_layers_"+L(il)+"_post_attention_layernorm_weight.bin"),l.pan,eng.h,s->st);
        // gdw stored transposed [eng.h, eng.rtr_h] for cache-friendly GPU access
        // (each thread reads stride-1 floats in the inner loop instead of stride eng.h).
        // Upstream PyTorch saves it as [eng.rtr_h, eng.h] — we transpose on load.
        if(!B(l.gdw,eng.h*eng.rtr_h)){zaya_destroy(s);return nullptr;}
        {
            auto raw=WF("model_layers_"+L(il)+"_mlp_gate_down_proj_weight.bin");
            std::vector<float> tr(eng.h*eng.rtr_h);
            for(int i=0;i<eng.rtr_h;i++)for(int j=0;j<eng.h;j++)tr[j*eng.rtr_h+i]=raw[i*eng.h+j];
            upf32(tr,l.gdw,eng.h*eng.rtr_h,s->st);
        }
        if(!B(l.gdb,eng.rtr_h)){zaya_destroy(s);return nullptr;}upf32(W("model_layers_"+L(il)+"_mlp_gate_down_proj_bias.bin"),l.gdb,eng.rtr_h,s->st);
        if(!B(l.rfn,eng.rtr_h)){zaya_destroy(s);return nullptr;}upf32(W("model_layers_"+L(il)+"_mlp_gate_router_mlp_norm_weight.bin"),l.rfn,eng.rtr_h,s->st);
        if(!B(l.rf1,eng.rtr_h*eng.rtr_h)){zaya_destroy(s);return nullptr;}upf32(W("model_layers_"+L(il)+"_mlp_gate_router_mlp_fc1_weight.bin"),l.rf1,eng.rtr_h*eng.rtr_h,s->st);
        if(!B(l.rf1b,eng.rtr_h)){zaya_destroy(s);return nullptr;}upf32(W("model_layers_"+L(il)+"_mlp_gate_router_mlp_fc1_bias.bin"),l.rf1b,eng.rtr_h,s->st);
        if(!B(l.rf2,eng.rtr_h*eng.rtr_h)){zaya_destroy(s);return nullptr;}upf32(W("model_layers_"+L(il)+"_mlp_gate_router_mlp_fc2_weight.bin"),l.rf2,eng.rtr_h*eng.rtr_h,s->st);
        if(!B(l.rf2b,eng.rtr_h)){zaya_destroy(s);return nullptr;}upf32(W("model_layers_"+L(il)+"_mlp_gate_router_mlp_fc2_bias.bin"),l.rf2b,eng.rtr_h,s->st);
        if(!B(l.rout,eng.n_exp_t*eng.rtr_h)){zaya_destroy(s);return nullptr;}upf32(W("model_layers_"+L(il)+"_mlp_gate_router_mlp_out_proj_weight.bin"),l.rout,eng.n_exp_t*eng.rtr_h,s->st);
        if(!B(l.bb,eng.n_exp_t)){zaya_destroy(s);return nullptr;}upf32(W("model_layers_"+L(il)+"_mlp_gate_balancing_biases.bin"),l.bb,eng.n_exp_t,s->st);
        auto sz_gu=eng.n_exp*2*eng.n_ff*eng.h;auto sz_dn=eng.n_exp*eng.h*eng.n_ff;
        hipError_t e1=hipMalloc(&l.gu,sz_gu*2), e2=hipMalloc(&l.dn,sz_dn*2);
        if(e1!=hipSuccess||e2!=hipSuccess){
            // Free whichever allocations succeeded before bailing,
            // then null them out so zaya_destroy doesn't double-free
            // (fixes GPU memory leak when only one MoE expert tensor
            //  alloc succeeds and the other fails).
            if(e1==hipSuccess && l.gu){ (void)hipFree(l.gu); l.gu=nullptr; }
            if(e2==hipSuccess && l.dn){ (void)hipFree(l.dn); l.dn=nullptr; }
            // Non-fatal: MoE expert tensors are optional (some models use
            // dense FFN only). Let the engine continue without them.
            // The null check in zaya_forward guards against null d_gu/d_dn.
            l.gu=nullptr; l.dn=nullptr;
        } else {
            upf16(W("model_layers_"+L(il)+"_mlp_experts_gate_up_proj.bin"),l.gu,sz_gu,s->st);
            upf16(W("model_layers_"+L(il)+"_mlp_experts_down_proj.bin"),l.dn,sz_dn,s->st);
        }
        if(!B(l.pmhss,eng.h)){zaya_destroy(s);return nullptr;}upf32(W("model_layers_"+L(il)+"_post_mlp_residual_scale_hidden_states_scale.bin"),l.pmhss,eng.h,s->st);
        if(!B(l.pmhsb,eng.h)){zaya_destroy(s);return nullptr;}upf32(W("model_layers_"+L(il)+"_post_mlp_residual_scale_hidden_states_bias.bin"),l.pmhsb,eng.h,s->st);
        if(!B(l.pmrss,eng.h)){zaya_destroy(s);return nullptr;}upf32(W("model_layers_"+L(il)+"_post_mlp_residual_scale_residual_scale.bin"),l.pmrss,eng.h,s->st);
        if(!B(l.pmrsb,eng.h)){zaya_destroy(s);return nullptr;}upf32(W("model_layers_"+L(il)+"_post_mlp_residual_scale_residual_bias.bin"),l.pmrsb,eng.h,s->st);
        
        std::string ep=g_weights_dir+"model_layers_"+L(il)+"_mlp_gate_router_states_scale.bin";
        std::ifstream ff(ep,std::ios::binary|std::ios::ate);
        if(ff){
            size_t fsize=ff.tellg();ff.seekg(0);
            if(fsize>=(size_t)eng.rtr_h*4){
                std::vector<float> buf(eng.rtr_h);
                ff.read((char*)buf.data(),eng.rtr_h*4);
                float sum=0; for(auto v:buf) sum+=v;
                s->eda_scale[il]=sum/(float)eng.rtr_h;
            }else{
                ff.read((char*)&s->eda_scale[il],4);
            }
            s->has_eda[il]=true;
        }else{
            s->eda_scale[il]=0;
            s->has_eda[il]=false;
        }
    }
    HIP_OK_R(hipStreamSynchronize(s->st), nullptr);
    return s;
}

// ── Init from a .1bp file (OneBP/Q4NX format) ──
// Opens the model with NpuOnebpModel::open(), validates dimensions against the
// config (same embed-size gate as zaya_init), allocates the identical GPU
// buffer set via zaya_alloc_buffers, and loads every weight through the
// tensor-name mapping in load_layer_onebp (get_tensor_f32 for ndim=1/2,
// get_tensor_f32_expert per-expert for ndim=3 MoE stacks, upf16/upf32
// uploads). Returns the engine state, or nullptr on failure.
ZayaState* zaya_init_onebp(const char* onebp_path, const ZayaConfig* cfg) {
    // Populate runtime config from the provided ZayaConfig (or default Zaya1-8B)
    if (cfg) {
        eng = *cfg;
    } else {
        eng = ZayaConfig::zaya1_8b();
    }

    ZayaState* s = new (std::nothrow) ZayaState();
    if (!s) {
        fprintf(stderr, "zaya_init_onebp: failed to allocate ZayaState (OOM)\n");
        return nullptr;
    }
    HIP_OK_R(hipStreamCreate(&s->st), nullptr);

    NpuOnebpModel model;
    if (!model.open(onebp_path)) {
        fprintf(stderr, "zaya_init_onebp: failed to open %s\n", onebp_path);
        zaya_destroy(s);
        return nullptr;
    }
    const auto& hdr = model.header();
    fprintf(stderr, "zaya_init_onebp: %s — H=%d L=%d NH=%d NKV=%d HD=%d IM=%d V=%d (%d tensors, quant=%u)\n",
            onebp_path, hdr.hidden_size, hdr.num_layers, hdr.num_attention_heads,
            hdr.num_kv_heads, hdr.head_dim, hdr.intermediate_size, hdr.vocab_size,
            model.tensor_count(), (unsigned)hdr.quant);

    // Validate the model header against the runtime config before allocating or
    // loading weights — a silent dimension mismatch would zero-fill every tensor
    // and produce a model that loads without error but outputs garbage.
    // Extended in #1527: attention geometry and the expert count were previously
    // unchecked, so a 24-expert 74B could load under a 16-expert config and
    // silently zero-fill every layer. intermediate_size is deliberately NOT
    // checked: gguf_to_onebp infers it from the first MoE tensor's shape
    // (2*n_ff rows for gate_up), which is not the engine's n_ff — those
    // mismatches surface as wrong-sized expert tensors in load_layer_onebp.
    if (eng.h      != (int)hdr.hidden_size    ||
        eng.n_layers != (int)hdr.num_layers    ||
        eng.vocab  != (int)hdr.vocab_size       ||
        (eng.nq  > 0 && hdr.num_attention_heads > 0 && eng.nq  != (int)hdr.num_attention_heads) ||
        (eng.nkv > 0 && hdr.num_kv_heads       > 0 && eng.nkv != (int)hdr.num_kv_heads)       ||
        (eng.hd  > 0 && hdr.head_dim           > 0 && eng.hd  != (int)hdr.head_dim)           ||
        (hdr.num_experts > 0 && hdr.num_experts != (uint32_t)eng.n_exp)) {
        fprintf(stderr, "zaya_init_onebp: config/header dimension mismatch — aborting init "
                "(cfg H=%d L=%d V=%d NH=%d NKV=%d HD=%d NE=%d vs hdr H=%d L=%d V=%d NH=%d NKV=%d HD=%d NE=%u)\n",
                eng.h, eng.n_layers, eng.vocab, eng.nq, eng.nkv, eng.hd, eng.n_exp,
                (int)hdr.hidden_size, (int)hdr.num_layers, (int)hdr.vocab_size,
                (int)hdr.num_attention_heads, (int)hdr.num_kv_heads, (int)hdr.head_dim,
                (unsigned)hdr.num_experts);
        zaya_destroy(s);
        return nullptr;
    }
    // Verify a GPU is available before attempting allocations (same as zaya_init)
    int ndev = 0;
    HIP_OK_R(hipGetDeviceCount(&ndev), nullptr);
    if (ndev < 1) {
        fprintf(stderr, "zaya_init_onebp: No HIP-capable GPU found (device count=%d).\n", ndev);
        zaya_destroy(s);
        return nullptr;
    }

    // Globals: token_embd + final norm — required (same gate as zaya_init).
    // output_norm.weight with model.norm.weight as fallback alias (plan risk #1).
    if (!model.get_tensor_f32("token_embd.weight", s->embed)) {
        fprintf(stderr, "zaya_init_onebp: missing token_embd.weight — aborting init\n");
        zaya_destroy(s);
        return nullptr;
    }
    std::vector<float> fnorm;
    if (!model.get_tensor_f32("output_norm.weight", fnorm))
        model.get_tensor_f32("model.norm.weight", fnorm);
    if ((int)fnorm.size() != eng.h) {
        fprintf(stderr, "zaya_init_onebp: missing output_norm.weight/model.norm.weight (%zu, want %d) — aborting init\n",
                fnorm.size(), eng.h);
        zaya_destroy(s);
        return nullptr;
    }

    // Dimension validation — same embed-size gate as zaya_init (fixes #61):
    // refuse to load a model whose embedding table doesn't match the config.
    size_t expected_embed = (size_t)eng.vocab * eng.h;
    if (s->embed.size() != expected_embed) {
        fprintf(stderr, "zaya_init_onebp: model embed size %zu != expected %zu (cfg H=%d, vocab=%d)\n",
                s->embed.size(), expected_embed, eng.h, eng.vocab);
        fprintf(stderr, "  Refusing to load — would produce silent garbage.\n");
        zaya_destroy(s);
        return nullptr;
    }

    // input_hidden_states_scale/bias are optional (plan risk #5): default
    // scale=1.0, bias=0.0 (pattern from tests/zaya_gpu_decode.cpp).
    s->iscale.assign(eng.h, 1.0f);
    s->ibias.assign(eng.h, 0.0f);
    {
        std::vector<float> v;
        if (model.get_tensor_f32("input_hidden_states_scale.weight", v) && (int)v.size() == eng.h) {
            for (int i = 0; i < eng.h; i++) s->iscale[i] = v[i];
        } else {
            fprintf(stderr, "  zaya_init_onebp: input_hidden_states_scale.weight absent — using 1.0\n");
        }
        if (model.get_tensor_f32("input_hidden_states_scale.bias", v) && (int)v.size() == eng.h) {
            for (int i = 0; i < eng.h; i++) s->ibias[i] = v[i];
        } else {
            fprintf(stderr, "  zaya_init_onebp: input_hidden_states_scale.bias absent — using 0.0\n");
        }
    }

    // Allocate GPU buffers — identical set to zaya_init's
    if (!zaya_alloc_buffers(s)) {
        zaya_destroy(s);
        return nullptr;
    }

    // Upload globals (host copies kept for the batch path / backend adapter)
    upf16(s->embed, s->d_embed, eng.vocab * eng.h, s->st);
    upf16(fnorm, s->d_fnw, eng.h, s->st);
    upf16(s->ibias, s->d_ibias, eng.h, s->st);
    upf16(s->iscale, s->d_iscale, eng.h, s->st);

    // Per-layer weights via the tensor-name mapping table.
    // value-init LayerW (not resize) so all raw pointers are nullptr — zaya_destroy
    // frees s->lw via safe() and must not see indeterminate pointers on partial failure.
    s->lw.assign(eng.n_layers, LayerW{});
    s->has_eda.assign(eng.n_layers, false);
    s->eda_scale.assign(eng.n_layers, 0.0f);
    for (int il = 0; il < eng.n_layers; il++) {
        if (!load_layer_onebp(model, il, s->lw[il], eng, s, s->st)) {
            fprintf(stderr, "zaya_init_onebp: layer %d failed to load (see messages above) — aborting init\n", il);
            zaya_destroy(s);
            return nullptr;
        }
    }
    HIP_OK_R(hipStreamSynchronize(s->st), nullptr);
    fprintf(stderr, "zaya_init_onebp: engine ready (%d layers)\n", eng.n_layers);
    return s;
}

// ── Page-aware KV cache write ──
// Allocate a page for the given (layer, pos) if not already allocated.
// Returns the GPU offset within the layer's page pool.
static size_t zaya_kv_page_write(ZayaState* s, int il, int pos) {
    int page_id = pos / s->page_size;
    int page_off = pos % s->page_size;
    if (!s->page_alloc[il][page_id]) {
        // Evict from pool if needed
        int pool_page = s->page_next_evict[il];
        s->page_map[il][page_id] = pool_page;
        s->page_alloc[il][page_id] = true;
        s->page_next_evict[il] = (pool_page + 1) % s->kv_pool_pages;
        // Update device-side page map for gather kernel
        std::vector<int> host_map(s->n_kv_pages);
        for (int p = 0; p < s->n_kv_pages; p++) {
            host_map[p] = s->page_map[il][p];
        }
        hipError_t _s_ = hipMemcpy(s->d_page_map, host_map.data(), s->n_kv_pages * sizeof(int), hipMemcpyHostToDevice);
        if (_s_ != hipSuccess) {
            fprintf(stderr, "HIP Error %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(_s_));
            return 0;
        }
    }
    int pool_page = s->page_map[il][page_id];
    return (size_t)pool_page * s->page_size * eng.nkv * eng.hd + (size_t)page_off * eng.nkv * eng.hd;
}

// ── Page-aware KV cache gather ──
// Gathers all pages for positions [0, seq_len) into the scratch buffers.
static void zaya_kv_gather(ZayaState* s, int il, int seq_len) {
    dim3 grid(seq_len, eng.nkv);
    gather_kv_pages_k<<<grid, 128, 0, s->st>>>(
        s->d_k_gather,
        s->d_kcache + (size_t)il * s->kv_pool_pages * s->page_size * eng.nkv * eng.hd,
        s->d_page_map, eng.nkv, eng.hd, s->page_size, seq_len);
    HIP_CHECK(hipGetLastError());
    gather_kv_pages_k<<<grid, 128, 0, s->st>>>(
        s->d_v_gather,
        s->d_vcache + (size_t)il * s->kv_pool_pages * s->page_size * eng.nkv * eng.hd,
        s->d_page_map, eng.nkv, eng.hd, s->page_size, seq_len);
    HIP_CHECK(hipGetLastError());
}

// ── Forward: token in, logits out ──
// Zaya layers alternate CCA-attention / MoE (issue #1527): each sublayer
// runs only when its weights were loaded (null when absent in the file —
// same pattern as the reference tests/zaya_gpu_decode.cpp). The per-layer
// pre-norm (attn_norm) runs on every layer, matching llama.cpp's graph.
static inline bool layer_has_attn(const LayerW& l) {
    return l.wq && l.wk && l.wv1 && l.wv2 && l.wo && l.cdw && l.cgw && l.ks;
}

void zaya_forward(ZayaState* s, int token_id, float* logits_out) {
    if (token_id < 0 || token_id >= eng.vocab) { if (logits_out) memset(logits_out, 0, eng.vocab * sizeof(float)); return; }
    int g1 = (eng.h+BLK-1)/BLK;
    // Device-side embedding lookup (#5): no H2D copy
    HIP_OK_V(hipMemcpyAsync(s->d_token_id, &token_id, 4, hipMemcpyHostToDevice, s->st));
    embed_lookup_k<<<g1,BLK,0,s->st>>>(s->d_hs, s->d_embed, s->d_ibias, s->d_iscale, s->d_token_id, eng.h);
    HIP_CHECK(hipGetLastError());

    for(int il=0;il<eng.n_layers;il++){
        auto& l=s->lw[il];
        // ── CCA attention: q/k/v proj → cca_prep → KV-cache stash → flash-decode → o_proj ──
        // RMSNorm (separate launch to avoid cross-block data race, issue #870)
        rmsnorm_fused_kernel<<<1, 256, 0, s->st>>>(s->d_hs, l.nw, eng.h);
        HIP_CHECK(hipGetLastError());
        if (layer_has_attn(l)) {
        // Separate Q/K/V1/V2 using mm_k (no warp shuffle, AMD HIP compatible)
        { int hv2=eng.kd/2;
          mm_k<<<(eng.qd+BLK-1)/BLK,BLK,0,s->st>>>(s->d_tmp,s->d_hs,l.wq,eng.qd,eng.h);
          HIP_CHECK(hipGetLastError());
          mm_k<<<(eng.kd+BLK-1)/BLK,BLK,0,s->st>>>(s->d_tmp+eng.qd,s->d_hs,l.wk,eng.kd,eng.h);
          HIP_CHECK(hipGetLastError());
          mm_k<<<(hv2+BLK-1)/BLK,BLK,0,s->st>>>(s->d_tmp+eng.qd+eng.kd,s->d_hs,l.wv1,hv2,eng.h);
          HIP_CHECK(hipGetLastError());
          mm_k<<<(hv2+BLK-1)/BLK,BLK,0,s->st>>>(s->d_tmp+eng.qd+eng.kd+hv2,s->d_hs,l.wv2,hv2,eng.h);
          HIP_CHECK(hipGetLastError()); }
          nan_clean_k<<<(eng.qkv+BLK-1)/BLK,BLK,0,s->st>>>(s->d_tmp,eng.qkv);
        cca_prep_kernel<<<1,256,cca_prep_smem_bytes(eng.nq,eng.nkv,eng.hd,eng.hd/2),s->st>>>(s->d_tmp,s->d_tmp+eng.qd,s->d_tmp+eng.qd+eng.kd,s->d_tmp+eng.qd+eng.kd+eng.kd/2,
            s->d_conv+(size_t)il*2*eng.qkv, s->d_vrec+(size_t)il*(eng.kd/2),
            l.cdw,l.cdb,l.cgw,l.cgb,l.ks,
            s->d_qout,s->d_kout,s->d_vout, s->pos, eng.nq,eng.nkv,eng.hd,eng.hd/2,5000000.0f,eng.qkv/(eng.nq+eng.nkv));
        HIP_CHECK(hipGetLastError());
        // KV cache: linear contiguous write (#3) — no gather overhead
        {
            __half* layer_k = s->d_kcache + ((size_t)il * s->max_seq + s->pos) * eng.nkv * eng.hd;
            __half* layer_v = s->d_vcache + ((size_t)il * s->max_seq + s->pos) * eng.nkv * eng.hd;
            copy_k<<<(eng.kd+BLK-1)/BLK,BLK,0,s->st>>>(layer_k, s->d_kout, eng.kd);
            HIP_CHECK(hipGetLastError());
            copy_k<<<(eng.kd+BLK-1)/BLK,BLK,0,s->st>>>(layer_v, s->d_vout, eng.kd);
            HIP_CHECK(hipGetLastError());
            rcpp_kv_cache_attn_decode_fd_prealloc(s->d_qout,
                s->d_kcache + (size_t)il * s->max_seq * eng.nkv * eng.hd,
                s->d_vcache + (size_t)il * s->max_seq * eng.nkv * eng.hd,
                s->d_ao, s->d_partials, eng.nq, eng.nkv, eng.hd, s->pos+1, 1.0f/sqrtf((float)eng.hd), (void*)s->st);
        }
        moe_tiled_gemv<<<eng.h/WMMA_M,WMMA_THREADS,0,s->st>>>(s->d_tmp,s->d_ao,l.wo,eng.h,eng.qd);  // o_proj -> d_tmp (avoid in-place race)
        HIP_CHECK(hipGetLastError());
        copy_k<<<(eng.h+BLK-1)/BLK,BLK,0,s->st>>>(s->d_ao,s->d_tmp,eng.h);
        residual_scale_k<<<g1,BLK,0,s->st>>>(s->d_ao,s->d_hs,l.pahss,l.pahsb,l.parss,l.parsb,eng.h);
        HIP_CHECK(hipGetLastError());
        copy_k<<<g1,BLK,0,s->st>>>(s->d_hs,s->d_ao,eng.h);
        HIP_CHECK(hipGetLastError());
        if (l.pan) {  // .bin-era second norm; absent in GGUF-derived models
        rmsnorm_k<<<1,BLK,0,s->st>>>(s->d_hs,l.pan,eng.h);
        HIP_CHECK(hipGetLastError());
        }
        }
        if(l.gu&&l.dn){
            eda_router_gpu_kernel<<<1,eng.rtr_h,eda_router_smem_bytes(eng.rtr_h,2),s->st>>>(s->d_hs,s->d_prev_rs+(size_t)il*eng.rtr_h,s->has_eda[il]?1:0,s->eda_scale[il],l.gdw,l.gdb,l.rfn,l.rf1,l.rf1b,l.rf2,l.rf2b,l.rout,l.bb,s->d_prev_rs+(size_t)il*eng.rtr_h,s->d_expert_idx,s->d_expert_wt,eng.n_exp,eng.h,eng.rtr_h,2);
            HIP_CHECK(hipGetLastError());
            encode_expert_cache_kernel<<<1,32,0,s->st>>>(s->d_prev_rs+(size_t)il*eng.rtr_h,s->d_expert_idx,eng.rtr_h);
            HIP_CHECK(hipGetLastError());
            fixup_skip_expert_kernel<<<1,256,0,s->st>>>(s->d_expert_idx,s->d_skip_flag,eng.n_exp,eng.n_exp_t);
            HIP_CHECK(hipGetLastError());
            // #1: No host sync — skip_flag read by gateup/down on GPU
            const int gb=(2*eng.n_ff+WMMA_M-1)/WMMA_M;
            const int db=(eng.h+WMMA_M-1)/WMMA_M;
            const int sb=(eng.n_ff+BLK-1)/BLK;
            wmma_gateup_kernel<<<gb,WMMA_THREADS,0,s->st>>>(s->d_tmp,s->d_hs,l.gu,s->d_expert_idx,s->d_skip_flag,eng.h,eng.n_ff,eng.n_exp);
            HIP_CHECK(hipGetLastError());
            silu_mul_k<<<sb,BLK,0,s->st>>>(s->d_ao,s->d_tmp,s->d_tmp+eng.n_ff,eng.n_ff);
            HIP_CHECK(hipGetLastError());
            wmma_down_kernel<<<db,WMMA_THREADS,0,s->st>>>(s->d_tmp,s->d_ao,l.dn,s->d_expert_idx,s->d_skip_flag,eng.h,eng.n_ff,eng.n_exp);
            HIP_CHECK(hipGetLastError());
            residual_scale_k<<<g1,BLK,0,s->st>>>(s->d_tmp,s->d_hs,l.pmhss,l.pmhsb,l.pmrss,l.pmrsb,eng.h);
            HIP_CHECK(hipGetLastError());
            copy_k<<<g1,BLK,0,s->st>>>(s->d_hs,s->d_tmp,eng.h);
        HIP_CHECK(hipGetLastError());
        }else{
            copy_k<<<g1,BLK,0,s->st>>>(s->d_tmp,s->d_hs,eng.h);
        HIP_CHECK(hipGetLastError());
        }
    }
    rmsnorm_k<<<1,BLK,0,s->st>>>(s->d_hs,s->d_fnw,eng.h);
    HIP_CHECK(hipGetLastError());

    // lm_head — tiled GEMV in a single launch; buffer allocated in zaya_init (fixes #59)
    // No sync needed before the lm_head: both the RMSNorm and the lm_head GEMV are on
    // the same stream, so the GEMV waits for the RMSNorm automatically (fixes perf).
    moe_tiled_gemv<<<(eng.vocab+WMMA_M-1)/WMMA_M,WMMA_THREADS,0,s->st>>>(s->d_lm_vocab,s->d_hs,s->d_embed,eng.vocab,eng.h);
    HIP_CHECK(hipGetLastError());
    HIP_OK_V(hipStreamSynchronize(s->st));
    std::vector<__half> lh(eng.vocab);
    HIP_OK_V(hipMemcpy(lh.data(),s->d_lm_vocab,(size_t)eng.vocab*2,hipMemcpyDeviceToHost));
    for(int v=0;v<eng.vocab;v++)logits_out[v]=__half2float(lh[v]);
    if(s->pos < s->max_seq-1) s->pos++;
}

// ── Forward greedy: same as forward but only returns argmax (much faster) ──
int zaya_forward_greedy(ZayaState* s, int token_id) {
    if (token_id < 0 || token_id >= eng.vocab) return -1;
    int g1 = (eng.h+BLK-1)/BLK;
    // Device-side embedding lookup (#5): no H2D copy
    HIP_OK_R(hipMemcpyAsync(s->d_token_id, &token_id, 4, hipMemcpyHostToDevice, s->st), -1);
    embed_lookup_k<<<g1,BLK,0,s->st>>>(s->d_hs, s->d_embed, s->d_ibias, s->d_iscale, s->d_token_id, eng.h);
    HIP_CHECK(hipGetLastError());

    for(int il=0;il<eng.n_layers;il++){
        auto& l=s->lw[il];
        // ── CCA attention: q/k/v proj → cca_prep → KV-cache stash → flash-decode → o_proj ──
        // RMSNorm (separate launch to avoid cross-block data race, issue #870)
        rmsnorm_fused_kernel<<<1, 256, 0, s->st>>>(s->d_hs, l.nw, eng.h);
        HIP_CHECK(hipGetLastError());
        if (layer_has_attn(l)) {
        // Separate Q/K/V1/V2 using mm_k (no warp shuffle, AMD HIP compatible)
        { int hv2=eng.kd/2;
          mm_k<<<(eng.qd+BLK-1)/BLK,BLK,0,s->st>>>(s->d_tmp,s->d_hs,l.wq,eng.qd,eng.h);
          HIP_CHECK(hipGetLastError());
          mm_k<<<(eng.kd+BLK-1)/BLK,BLK,0,s->st>>>(s->d_tmp+eng.qd,s->d_hs,l.wk,eng.kd,eng.h);
          HIP_CHECK(hipGetLastError());
          mm_k<<<(hv2+BLK-1)/BLK,BLK,0,s->st>>>(s->d_tmp+eng.qd+eng.kd,s->d_hs,l.wv1,hv2,eng.h);
          HIP_CHECK(hipGetLastError());
          mm_k<<<(hv2+BLK-1)/BLK,BLK,0,s->st>>>(s->d_tmp+eng.qd+eng.kd+hv2,s->d_hs,l.wv2,hv2,eng.h);
          HIP_CHECK(hipGetLastError()); }
          nan_clean_k<<<(eng.qkv+BLK-1)/BLK,BLK,0,s->st>>>(s->d_tmp,eng.qkv);
        cca_prep_kernel<<<1,256,cca_prep_smem_bytes(eng.nq,eng.nkv,eng.hd,eng.hd/2),s->st>>>(s->d_tmp,s->d_tmp+eng.qd,s->d_tmp+eng.qd+eng.kd,s->d_tmp+eng.qd+eng.kd+eng.kd/2,
            s->d_conv+(size_t)il*2*eng.qkv, s->d_vrec+(size_t)il*(eng.kd/2),
            l.cdw,l.cdb,l.cgw,l.cgb,l.ks,
            s->d_qout,s->d_kout,s->d_vout, s->pos, eng.nq,eng.nkv,eng.hd,eng.hd/2,5000000.0f,eng.qkv/(eng.nq+eng.nkv));
        HIP_CHECK(hipGetLastError());
        // KV cache: linear contiguous write (#3) — no gather overhead
        {
            __half* layer_k = s->d_kcache + ((size_t)il * s->max_seq + s->pos) * eng.nkv * eng.hd;
            __half* layer_v = s->d_vcache + ((size_t)il * s->max_seq + s->pos) * eng.nkv * eng.hd;
            copy_k<<<(eng.kd+BLK-1)/BLK,BLK,0,s->st>>>(layer_k, s->d_kout, eng.kd);
            HIP_CHECK(hipGetLastError());
            copy_k<<<(eng.kd+BLK-1)/BLK,BLK,0,s->st>>>(layer_v, s->d_vout, eng.kd);
            HIP_CHECK(hipGetLastError());
            rcpp_kv_cache_attn_decode_fd_prealloc(s->d_qout,
                s->d_kcache + (size_t)il * s->max_seq * eng.nkv * eng.hd,
                s->d_vcache + (size_t)il * s->max_seq * eng.nkv * eng.hd,
                s->d_ao, s->d_partials, eng.nq, eng.nkv, eng.hd, s->pos+1, 1.0f/sqrtf((float)eng.hd), (void*)s->st);
        }
        moe_tiled_gemv<<<eng.h/WMMA_M,WMMA_THREADS,0,s->st>>>(s->d_tmp,s->d_ao,l.wo,eng.h,eng.qd);  // o_proj -> d_tmp (avoid in-place race)
        HIP_CHECK(hipGetLastError());
        copy_k<<<(eng.h+BLK-1)/BLK,BLK,0,s->st>>>(s->d_ao,s->d_tmp,eng.h);
        residual_scale_k<<<g1,BLK,0,s->st>>>(s->d_ao,s->d_hs,l.pahss,l.pahsb,l.parss,l.parsb,eng.h);
        HIP_CHECK(hipGetLastError());
        copy_k<<<g1,BLK,0,s->st>>>(s->d_hs,s->d_ao,eng.h);
        HIP_CHECK(hipGetLastError());
        nan_clean_k<<<g1,BLK,0,s->st>>>(s->d_hs,eng.h);
        if (l.pan) {  // .bin-era second norm; absent in GGUF-derived models
        rmsnorm_k<<<1,BLK,0,s->st>>>(s->d_hs,l.pan,eng.h);
        HIP_CHECK(hipGetLastError());
        }
        }
        if(l.gu&&l.dn){
            eda_router_gpu_kernel<<<1,eng.rtr_h,eda_router_smem_bytes(eng.rtr_h,2),s->st>>>(s->d_hs,s->d_prev_rs+(size_t)il*eng.rtr_h,s->has_eda[il]?1:0,s->eda_scale[il],l.gdw,l.gdb,l.rfn,l.rf1,l.rf1b,l.rf2,l.rf2b,l.rout,l.bb,s->d_prev_rs+(size_t)il*eng.rtr_h,s->d_expert_idx,s->d_expert_wt,eng.n_exp,eng.h,eng.rtr_h,2);
            HIP_CHECK(hipGetLastError());
            encode_expert_cache_kernel<<<1,32,0,s->st>>>(s->d_prev_rs+(size_t)il*eng.rtr_h,s->d_expert_idx,eng.rtr_h);
            HIP_CHECK(hipGetLastError());
            fixup_skip_expert_kernel<<<1,256,0,s->st>>>(s->d_expert_idx,s->d_skip_flag,eng.n_exp,eng.n_exp_t);
            HIP_CHECK(hipGetLastError());
            // #1: No host sync — skip_flag read by gateup/down on GPU
            const int gb=(2*eng.n_ff+WMMA_M-1)/WMMA_M;
            const int db=(eng.h+WMMA_M-1)/WMMA_M;
            const int sb=(eng.n_ff+BLK-1)/BLK;
            wmma_gateup_kernel<<<gb,WMMA_THREADS,0,s->st>>>(s->d_tmp,s->d_hs,l.gu,s->d_expert_idx,s->d_skip_flag,eng.h,eng.n_ff,eng.n_exp);
            HIP_CHECK(hipGetLastError());
            silu_mul_k<<<sb,BLK,0,s->st>>>(s->d_ao,s->d_tmp,s->d_tmp+eng.n_ff,eng.n_ff);
            HIP_CHECK(hipGetLastError());
            wmma_down_kernel<<<db,WMMA_THREADS,0,s->st>>>(s->d_tmp,s->d_ao,l.dn,s->d_expert_idx,s->d_skip_flag,eng.h,eng.n_ff,eng.n_exp);
            HIP_CHECK(hipGetLastError());
            residual_scale_k<<<g1,BLK,0,s->st>>>(s->d_tmp,s->d_hs,l.pmhss,l.pmhsb,l.pmrss,l.pmrsb,eng.h);
            HIP_CHECK(hipGetLastError());
            copy_k<<<g1,BLK,0,s->st>>>(s->d_hs,s->d_tmp,eng.h);
        HIP_CHECK(hipGetLastError());
        }else{copy_k<<<g1,BLK,0,s->st>>>(s->d_tmp,s->d_hs,eng.h);}
    HIP_CHECK(hipGetLastError());
    }
    rmsnorm_k<<<1,BLK,0,s->st>>>(s->d_hs,s->d_fnw,eng.h);
    HIP_CHECK(hipGetLastError());

    // lm_head + GPU argmax (no full logit copy); buffers allocated in zaya_init (fixes #59)
    moe_tiled_gemv<<<(eng.vocab+WMMA_M-1)/WMMA_M,WMMA_THREADS,0,s->st>>>(s->d_lm_vocab,s->d_hs,s->d_embed,eng.vocab,eng.h);
    HIP_CHECK(hipGetLastError());
    argmax_kernel<<<1,256,0,s->st>>>(s->d_lm_vocab,eng.vocab,s->d_argmax_idx,s->d_argmax_val);
    HIP_CHECK(hipGetLastError());
    HIP_OK_R(hipStreamSynchronize(s->st), -1);
    int best;
    HIP_OK_R(hipMemcpy(&best,s->d_argmax_idx,4,hipMemcpyDeviceToHost), -1);
    if(s->pos < s->max_seq-1) s->pos++;

    // HIP graph capture disabled: the stream is shared with non-captured
    // MoE NPU pipeline operations, and capturing the greedy path interferes
    // with subsequent kernel launches on the same stream.
    // Always use per-call kernel launch (consistent with zaya_forward path).
    if (!s->graph_captured) {
        s->graph_captured = true;  // mark as done to skip this block forever
        fprintf(stderr, "  HIP graph capture disabled — using per-call launch\n");
    }
    return best;
}

// ── Forward batch: process B tokens through all layers ──
// Uses batched router + batch-union MoE for expert dedup.
// B <= 8 recommended (constrained by shared memory in union kernel).
void zaya_forward_batch(ZayaState* s, const int* token_ids, float* logits_out, int B) {
    int g1 = (eng.h+BLK-1)/BLK;
    if (B > 8) {
        fprintf(stderr, "zaya_forward_batch: B=%d > 8, truncating to 8 (tokens %d-%d will NOT be processed)\n", B, 8, B-1);
        B = 8;
    }

    // ── Embedding lookup for all B tokens ──
    if (B < 1 || B > (int)ZAYA_B_MAX) {
        fprintf(stderr, "[zaya] batch B=%d out of range [1,%zu]\n", B, ZAYA_B_MAX);
        return;
    }
    std::vector<__half> hh(B * eng.h);
    for (int b = 0; b < B; b++) {
        int tid = token_ids[b];
        if (tid < 0 || tid >= eng.vocab) {
            fprintf(stderr, "[zaya] token id %d out of range [0,%d)\n", tid, eng.vocab);
            return;
        }
        for (int i = 0; i < eng.h; i++) {
            float raw = s->embed[tid * (size_t)eng.h + i];
            hh[b * (size_t)eng.h + i] = __float2half((raw + s->ibias[i]) * s->iscale[i]);
        }
    }
    HIP_OK_V(hipMemcpyAsync(s->d_hs, hh.data(), B * eng.h * 2, hipMemcpyHostToDevice, s->st));

    for (int il = 0; il < eng.n_layers; il++) {
        auto& l = s->lw[il];

        // Per-layer pre-norm (attn_norm) runs on every layer, matching
        // llama.cpp's graph. The sublayers are guarded by their weights
        // (issue #1527 — Zaya layers alternate attention-only / MoE-only).
        for (int b = 0; b < B; b++) {
            __half* hs_b = s->d_hs + (size_t)b * eng.h;
            rmsnorm_k<<<1, BLK, 0, s->st>>>(hs_b, l.nw, eng.h);
            HIP_CHECK(hipGetLastError());
        }
        if (layer_has_attn(l)) {
        // CCA attention: per-token Q/K/V proj → V interleave → GQA broadcast → o_proj
        for (int b = 0; b < B; b++) {
            __half* hs_b = s->d_hs + (size_t)b * eng.h;
            __half* ao_b = s->d_ao + (size_t)b * eng.h;
            moe_tiled_gemv<<<eng.qd/WMMA_M,WMMA_THREADS,0,s->st>>>(s->d_tmp, hs_b, l.wq, eng.qd, eng.h);
            HIP_CHECK(hipGetLastError());
            moe_tiled_gemv<<<eng.kd/WMMA_M,WMMA_THREADS,0,s->st>>>(s->d_tmp+eng.qd, hs_b, l.wk, eng.kd, eng.h);
            HIP_CHECK(hipGetLastError());
            moe_tiled_gemv<<<eng.kd/2/WMMA_M,WMMA_THREADS,0,s->st>>>(s->d_tmp+eng.qd+eng.kd, hs_b, l.wv1, eng.kd/2, eng.h);
            HIP_CHECK(hipGetLastError());
            moe_tiled_gemv<<<eng.kd/2/WMMA_M,WMMA_THREADS,0,s->st>>>(s->d_tmp+eng.qd+eng.kd+eng.kd/2, hs_b, l.wv2, eng.kd/2, eng.h);
            HIP_CHECK(hipGetLastError());
            v_interleave_kernel<<<(eng.kd/2+BLK-1)/BLK,BLK,0,s->st>>>(s->d_tmp+eng.qd, s->d_tmp+eng.qd+eng.kd, s->d_tmp+eng.qd+eng.kd+eng.kd/2, eng.kd/2);
            HIP_CHECK(hipGetLastError());
            gqa_broadcast_k<<<(eng.qd+BLK-1)/BLK,BLK,0,s->st>>>(s->d_tmp+eng.qd, s->d_tmp, eng.nq, eng.nkv, eng.hd);
            HIP_CHECK(hipGetLastError());
            moe_tiled_gemv<<<eng.h/WMMA_M,WMMA_THREADS,0,s->st>>>(ao_b, s->d_tmp, l.wo, eng.h, eng.qd);
        HIP_CHECK(hipGetLastError());
        }

        // Post-attention residual + RMSNorm (per token)
        for (int b = 0; b < B; b++) {
            __half* hs_b  = s->d_hs + (size_t)b * eng.h;
            __half* ao_b  = s->d_ao + (size_t)b * eng.h;
            residual_scale_k<<<g1, BLK, 0, s->st>>>(ao_b, hs_b,
                l.pahss, l.pahsb, l.parss, l.parsb, eng.h);
            HIP_CHECK(hipGetLastError());
            copy_k<<<g1, BLK, 0, s->st>>>(hs_b, ao_b, eng.h);
            HIP_CHECK(hipGetLastError());
            if (l.pan) {  // .bin-era second norm; absent in GGUF-derived models
            rmsnorm_k<<<1, BLK, 0, s->st>>>(hs_b, l.pan, eng.h);
            HIP_CHECK(hipGetLastError());
            }
        }
        }

        // Batched router + batch-union MoE
        // For B >= 4, use sorted dispatch to group tokens by expert
        // (loads each expert's weights ONCE, avoids L2 thrashing).
        // For B < 4, the fused per-token kernel is simpler and fast enough.
        if (l.gu && l.dn) {
            if (B >= 4) {
                // Phase 1: Route all B tokens (GPU-resident)
                batched_moe_router_kernel<<<B, 256, batched_router_smem_bytes(eng.rtr_h,eng.n_exp_t), s->st>>>(
                    s->d_hs, s->d_prev_rs + (size_t)il * eng.rtr_h,
                    s->has_eda[il] ? 1 : 0, s->eda_scale[il],
                    l.gdw, l.gdb, l.rfn, l.rf1, l.rf1b,
                    l.rf2, l.rf2b, l.rout, l.bb,
                    s->d_prev_rs + (size_t)il * eng.rtr_h,
                    s->d_expert_idx, s->d_expert_wt,
                    B, eng.h, eng.rtr_h, eng.n_exp_t);
                HIP_CHECK(hipGetLastError());

                // Phase 2: Sort token IDs by expert (histogram + prefix sum + scatter);
                // buffers allocated in zaya_init (fixes #63).
                moe_sort_histogram_kernel<<<1, 32, sort_histogram_smem_bytes(eng.n_exp_t), s->st>>>(
                    s->d_expert_idx, s->d_expert_counts, s->d_expert_offsets,
                    s->d_sorted_ids, B, eng.n_exp_t);
                HIP_CHECK(hipGetLastError());

                // Phase 3: Expert FFN (sorted, one block per expert with count>0)
                moe_sorted_expert_kernel<<<eng.n_exp, 256, sorted_expert_smem_bytes(eng.n_ff), s->st>>>(
                    s->d_hs, s->d_sorted_ids, s->d_expert_counts, s->d_expert_offsets,
                    l.gu, l.dn, s->d_tmp, B, eng.h, eng.n_exp, eng.n_ff);
                HIP_CHECK(hipGetLastError());

                // Phase 4: Handle MOD skip tokens (expert_idx == eng.n_exp = 16)
                // These skip the expert FFN entirely (out = hs, identity).
                if (s->d_expert_counts) {  // always true, keeps compiler happy
                    moe_modskip_passthrough_kernel<<<(B + 255) / 256, 256, 0, s->st>>>(
                        s->d_tmp, s->d_hs, s->d_expert_idx, eng.n_exp, B, eng.h);
                HIP_CHECK(hipGetLastError());
                }
            } else {
                batched_moe_fused_kernel<<<B, 256, batched_fused_smem_bytes(eng.rtr_h,eng.n_exp_t,eng.n_ff), s->st>>>(
                    s->d_hs, s->d_prev_rs + (size_t)il * eng.rtr_h,
                    s->has_eda[il] ? 1 : 0, s->eda_scale[il],
                    l.gdw, l.gdb, l.rfn, l.rf1, l.rf1b,
                    l.rf2, l.rf2b, l.rout, l.bb,
                    l.gu, l.dn,
                    s->d_prev_rs + (size_t)il * eng.rtr_h,
                    s->d_tmp, s->d_expert_idx, s->d_expert_wt,
                    B, eng.h, eng.rtr_h, eng.n_exp_t, eng.n_exp, eng.n_ff);
            HIP_CHECK(hipGetLastError());
            }

            // Post-MLP residual scale (per token)
            for (int b = 0; b < B; b++) {
                __half* hs_b  = s->d_hs + (size_t)b * eng.h;
                __half* tmp_b = s->d_tmp + (size_t)b * eng.h;
                residual_scale_k<<<g1, BLK, 0, s->st>>>(tmp_b, hs_b,
                    l.pmhss, l.pmhsb, l.pmrss, l.pmrsb, eng.h);
                HIP_CHECK(hipGetLastError());
                copy_k<<<g1, BLK, 0, s->st>>>(hs_b, tmp_b, eng.h);
            HIP_CHECK(hipGetLastError());
            }
        }
    }

    // Final RMSNorm (per token)
    for (int b = 0; b < B; b++) {
        __half* hs_b = s->d_hs + (size_t)b * eng.h;
        rmsnorm_k<<<1, BLK, 0, s->st>>>(hs_b, s->d_fnw, eng.h);
    HIP_CHECK(hipGetLastError());
    }

    // lm_head — tiled GEMV for each token; buffer allocated in zaya_init (fixes #63).
    // No sync needed before lm_head: same-stream ordering guarantees RMSNorm completes
    // before the lm_head GEMV launches.
    {
        const size_t max_need = (size_t)8 * eng.vocab * 2;  // B <= 8, allocated in zaya_init
        #if __has_include(<rocwmma/rocwmma.hpp>) && !defined(WMMA_WAVE32_DISABLED)
        if (B >= 2) {
            const int grid_x = (eng.vocab + WMMA_M - 1) / WMMA_M;
            const int grid_y = (B + WMMA_N - 1) / WMMA_N;
            wmma_batched_gemv<<<dim3(grid_x, grid_y, 1), 32, 0, s->st>>>(
                s->d_lm_vocab, s->d_hs, s->d_embed, eng.vocab, eng.h, B);
        HIP_CHECK(hipGetLastError());
        } else {
            moe_tiled_gemv<<<(eng.vocab + WMMA_M - 1) / WMMA_M, WMMA_THREADS, 0, s->st>>>(
                s->d_lm_vocab, s->d_hs, s->d_embed, eng.vocab, eng.h);
        HIP_CHECK(hipGetLastError());
        }
        #else
        for (int b = 0; b < B; b++) {
            __half* hs_b = s->d_hs + (size_t)b * eng.h;
            moe_tiled_gemv<<<(eng.vocab + WMMA_M - 1) / WMMA_M, WMMA_THREADS, 0, s->st>>>(
                s->d_lm_vocab + (size_t)b * eng.vocab, hs_b, s->d_embed, eng.vocab, eng.h);
        HIP_CHECK(hipGetLastError());
        }
        #endif
    }
    HIP_OK_V(hipStreamSynchronize(s->st));

    // Copy logits for all B tokens
    std::vector<__half> lh(B * eng.vocab);
    HIP_OK_V(hipMemcpy(lh.data(), s->d_lm_vocab, (size_t)B * eng.vocab * 2, hipMemcpyDeviceToHost));
    for (int b = 0; b < B; b++)
        for (int v = 0; v < eng.vocab; v++)
            logits_out[b * (size_t)eng.vocab + v] = __half2float(lh[b * (size_t)eng.vocab + v]);
}

// ═══════════════════════════════════════════════════════════════════════
// ── LoRA adapter merge ──
// Reads a .lora file (from merge_lora.py) and merges B*A*scale deltas
// into the GPU-resident weight matrices. Called after zaya_init().
//
// .lora binary format:
//   magic:   b'LORA' (4 bytes)
//   layers:  uint32
//   scale:   float32
//   For each layer:
//     num_mod: uint32
//     For each module:
//       mod_id: uint32 (0=q,1=k,2=v,3=o,4=gate,5=up,6=down)
//       rank:   uint32
//       in_dim: uint32 (input dimension of A = hidden_size)
//       out_dim:uint32 (output dimension of B = projection_size)
//       A[rank][in_dim]:  float32
//       B[out_dim][rank]: float32
// ═══════════════════════════════════════════════════════════════════════

// Per-module LoRA data
struct LoraModule {
    int mod_id;        // 0=q, 1=k, 2=v, 3=o, 4=gate, 5=up, 6=down
    int rank;
    int in_dim;
    int out_dim;
    std::vector<float> A;  // [rank * in_dim]
    std::vector<float> B;  // [out_dim * rank]
};

// Per-layer LoRA data
struct LoraLayer {
    std::vector<LoraModule> modules;
};

// Read a .lora file; returns empty vector on failure
static std::vector<LoraLayer> zaya_read_lora_file(const char* path, float& out_scale) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        fprintf(stderr, "zaya_apply_lora: cannot open %s\n", path);
        return {};
    }
    
    char magic[4];
    f.read(magic, 4);
    if (std::string(magic, 4) != "LORA") {
        fprintf(stderr, "zaya_apply_lora: bad magic in %s\n", path);
        return {};
    }
    
    uint32_t num_layers;
    float scale;
    f.read((char*)&num_layers, 4);
    f.read((char*)&scale, 4);
    out_scale = scale;
    
    std::vector<LoraLayer> layers(num_layers);
    for (uint32_t l = 0; l < num_layers; l++) {
        uint32_t num_mod;
        f.read((char*)&num_mod, 4);
        layers[l].modules.resize(num_mod);
        for (uint32_t m = 0; m < num_mod; m++) {
            LoraModule& mod = layers[l].modules[m];
            f.read((char*)&mod.mod_id, 4);
            f.read((char*)&mod.rank, 4);
            f.read((char*)&mod.in_dim, 4);
            f.read((char*)&mod.out_dim, 4);
            mod.A.resize((size_t)mod.rank * mod.in_dim);
            mod.B.resize((size_t)mod.out_dim * mod.rank);
            f.read((char*)mod.A.data(), mod.A.size() * 4);
            f.read((char*)mod.B.data(), mod.B.size() * 4);
        }
    }
    
    fprintf(stderr, "zaya_apply_lora: read %u layers from %s (scale=%.4f)\n",
            num_layers, path, scale);
    return layers;
}

// Compute delta = B * A * scale for a LoRA module
// Result: [out_dim * in_dim] row-major
static std::vector<float> compute_lora_delta(const LoraModule& mod, float scale) {
    std::vector<float> delta((size_t)mod.out_dim * mod.in_dim, 0.0f);
    // B[out_dim, rank] * A[rank, in_dim] → delta[out_dim, in_dim]
    for (int o = 0; o < mod.out_dim; o++) {
        for (int r = 0; r < mod.rank; r++) {
            float br = mod.B[(size_t)o * mod.rank + r] * scale;
            if (br == 0.0f) continue;
            const float* A_row = mod.A.data() + (size_t)r * mod.in_dim;
            float* delta_row = delta.data() + (size_t)o * mod.in_dim;
            for (int i = 0; i < mod.in_dim; i++) {
                delta_row[i] += br * A_row[i];
            }
        }
    }
    return delta;
}

// Apply LoRA deltas to GPU-resident layer weights
// Supported module -> weight mapping:
//   0 (q_proj)   → lw[].wq   FP16 [QD, H]
//   1 (k_proj)   → lw[].wk   FP16 [KD, H]
//   3 (o_proj)   → lw[].wo   FP16 [H, QD]
//   6 (down)     → lw[].gdw  FP32 [H, RTR_H] (transposed on GPU)
extern "C" int zaya_apply_lora(ZayaState* s, const char* lora_path) {
    if (!s || !lora_path) return -1;
    
    float scale;
    auto layers = zaya_read_lora_file(lora_path, scale);
    if (layers.empty()) {
        fprintf(stderr, "zaya_apply_lora: failed to load %s\n", lora_path);
        return -1;
    }
    
    int total_applied = 0;
    int n_layers = (int)std::min((size_t)eng.n_layers, layers.size());
    
    for (int il = 0; il < n_layers; il++) {
        auto& l = s->lw[il];
        for (auto& mod : layers[il].modules) {
            std::vector<float> delta = compute_lora_delta(mod, scale);
            
            if (mod.mod_id == 0) {  // q_proj → wq [QD, H]
                if ((size_t)mod.out_dim == eng.qd && (size_t)mod.in_dim == eng.h) {
                    // Download GPU weight, add delta, upload back
                    std::vector<__half> gpu_w(eng.qd * eng.h);
                    HIP_OK_R(hipMemcpy(gpu_w.data(), l.wq, eng.qd * eng.h * 2, hipMemcpyDeviceToHost), -1);
                    for (int i = 0; i < eng.qd * eng.h; i++) {
                        float v = __half2float(gpu_w[i]) + delta[i];
                        gpu_w[i] = __float2half(v);
                    }
                    HIP_OK_R(hipMemcpy(l.wq, gpu_w.data(), eng.qd * eng.h * 2, hipMemcpyHostToDevice), -1);
                    total_applied++;
                    fprintf(stderr, "  layer %d q_proj: merged LoRA delta [%dx%d]\n", il, eng.qd, eng.h);
                }
            } else if (mod.mod_id == 1) {  // k_proj → wk [KD, H]
                if ((size_t)mod.out_dim == eng.kd && (size_t)mod.in_dim == eng.h) {
                    std::vector<__half> gpu_w(eng.kd * eng.h);
                    HIP_OK_R(hipMemcpy(gpu_w.data(), l.wk, eng.kd * eng.h * 2, hipMemcpyDeviceToHost), -1);
                    for (int i = 0; i < eng.kd * eng.h; i++) {
                        float v = __half2float(gpu_w[i]) + delta[i];
                        gpu_w[i] = __float2half(v);
                    }
                    HIP_OK_R(hipMemcpy(l.wk, gpu_w.data(), eng.kd * eng.h * 2, hipMemcpyHostToDevice), -1);
                    total_applied++;
                    fprintf(stderr, "  layer %d k_proj: merged LoRA delta [%dx%d]\n", il, eng.kd, eng.h);
                }
            } else if (mod.mod_id == 3) {  // o_proj → wo [H, QD]
                if ((size_t)mod.out_dim == eng.h && (size_t)mod.in_dim == eng.qd) {
                    std::vector<__half> gpu_w(eng.h * eng.qd);
                    HIP_OK_R(hipMemcpy(gpu_w.data(), l.wo, eng.h * eng.qd * 2, hipMemcpyDeviceToHost), -1);
                    for (int i = 0; i < eng.h * eng.qd; i++) {
                        float v = __half2float(gpu_w[i]) + delta[i];
                        gpu_w[i] = __float2half(v);
                    }
                    HIP_OK_R(hipMemcpy(l.wo, gpu_w.data(), eng.h * eng.qd * 2, hipMemcpyHostToDevice), -1);
                    total_applied++;
                    fprintf(stderr, "  layer %d o_proj: merged LoRA delta [%dx%d]\n", il, eng.h, eng.qd);
                }
            } else if (mod.mod_id == 6) {  // down (gate_down_proj) → gdw GPU is [H, RTR_H] transposed
                // File stores raw gate_down_proj weight as [RTR_H, H]. LoRA delta is [RTR_H, H].
                // On GPU, gdw is transposed to [H, RTR_H]. We need to apply delta then retranspose.
                if ((size_t)mod.out_dim == eng.rtr_h && (size_t)mod.in_dim == eng.h) {
                    std::vector<float> gpu_w(eng.h * eng.rtr_h);
                    HIP_OK_R(hipMemcpy(gpu_w.data(), l.gdw, eng.h * eng.rtr_h * 4, hipMemcpyDeviceToHost), -1);
                    // delta is [RTR_H, H]; gpu_w is [H, RTR_H] (transposed)
                    // We add delta^T to gpu_w: gpu_w[j][i] += delta[i][j]
                    for (int i = 0; i < eng.rtr_h; i++) {
                        for (int j = 0; j < eng.h; j++) {
                            gpu_w[(size_t)j * eng.rtr_h + i] += delta[(size_t)i * eng.h + j];
                        }
                    }
                    HIP_OK_R(hipMemcpy(l.gdw, gpu_w.data(), eng.h * eng.rtr_h * 4, hipMemcpyHostToDevice), -1);
                    total_applied++;
                    fprintf(stderr, "  layer %d gate_down: merged LoRA delta [%dx%d]\n", il, eng.rtr_h, eng.h);
                }
            }
        }
    }
    
    HIP_OK_R(hipStreamSynchronize(s->st), -1);
    fprintf(stderr, "zaya_apply_lora: applied %d LoRA deltas\n", total_applied);
    return total_applied > 0 ? 0 : -1;
}

// ── Reset state (new sequence) ──
// ── Set sentinel for expert caching (no previous expert for any layer) ──
__global__ void init_expert_cache_sentinel(float* prev_rs, int n_layers, int rtr_h, int n_exp) {
    // Set prev_rs[layer * rtr_h + rtr_h - 1] = bit-cast n_exp (sentinel)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_layers) return;
    ((int*)&prev_rs[i * rtr_h + rtr_h - 1])[0] = n_exp; // invalid expert, no previous
}

void zaya_reset(ZayaState* s) {
    HIP_OK_V(hipMemsetAsync(s->d_conv,0,(size_t)eng.n_layers*2*eng.qkv*2,s->st));
    HIP_OK_V(hipMemsetAsync(s->d_phs,0,(size_t)eng.n_layers*eng.h*2,s->st));
    HIP_OK_V(hipMemsetAsync(s->d_prev_rs,0,(size_t)eng.n_layers*eng.rtr_h*4,s->st));
    if (s->use_linear_kv) {
        size_t kv_bytes = (size_t)eng.n_layers * s->max_seq * eng.nkv * eng.hd * 2;
        HIP_OK_V(hipMemsetAsync(s->d_kcache, 0, kv_bytes, s->st));
        HIP_OK_V(hipMemsetAsync(s->d_vcache, 0, kv_bytes, s->st));
    } else {
        size_t kv_pool_bytes = (size_t)eng.n_layers * s->kv_pool_pages * s->page_size * eng.nkv * eng.hd * 2;
        HIP_OK_V(hipMemsetAsync(s->d_kcache, 0, kv_pool_bytes, s->st));
        HIP_OK_V(hipMemsetAsync(s->d_vcache, 0, kv_pool_bytes, s->st));
        // Reset page table
        for (int il = 0; il < eng.n_layers; il++) {
            s->page_alloc[il].assign(s->n_kv_pages, false);
            s->page_map[il].assign(s->n_kv_pages, -1);
            s->page_next_evict[il] = 0;
        }
    }
    HIP_OK_V(hipMemsetAsync(s->d_vrec,0,(size_t)eng.n_layers*(eng.kd/2)*2,s->st));
    s->pos=0;
    init_expert_cache_sentinel<<<1, 64, 0, s->st>>>(s->d_prev_rs, eng.n_layers, eng.rtr_h, eng.n_exp);
HIP_CHECK(hipGetLastError());
}

// ── Destroy ──
void zaya_destroy(ZayaState* s) {
    if (!s) return;
    auto safe = [](auto p) { if (p) (void)hipFree(p); };
    safe(s->d_hs); safe(s->d_ao); safe(s->d_tmp); safe(s->d_fnw);
    if (s->graph_exec) { (void)hipGraphExecDestroy(s->graph_exec); s->graph_exec = nullptr; }
    if (s->graph) { (void)hipGraphDestroy(s->graph); s->graph = nullptr; }
    safe(s->d_lm_out); safe(s->d_embed); safe(s->d_ibias); safe(s->d_iscale); safe(s->d_token_id);
    safe(s->d_conv); safe(s->d_phs);
    safe(s->d_prev_rs); safe(s->d_expert_idx); safe(s->d_expert_wt);
    safe(s->d_kcache); safe(s->d_vcache); safe(s->d_vrec);
    safe(s->d_partials);
    safe(s->d_qout); safe(s->d_kout); safe(s->d_vout); safe(s->d_skip_flag);
    safe(s->d_k_gather); safe(s->d_v_gather); safe(s->d_page_map);
    // Bound by s->lw's actual size, not eng.n_layers: zaya_destroy() can be
    // called from an early-exit path in zaya_init() (missing weight files,
    // no GPU, or a model-dimension mismatch) before s->lw is resized to
    // eng.n_layers (that resize happens later, once init has actually
    // committed to loading). Using eng.n_layers here read past the end of
    // an empty/undersized vector, pulled garbage pointers out of it, and
    // handed them to hipFree() — a real segfault, not just UB in theory.
    for (size_t i = 0; i < s->lw.size(); i++) {
        auto& l = s->lw[i];
        safe(l.nw); safe(l.wq); safe(l.wk); safe(l.wv1); safe(l.wv2); safe(l.wo); safe(l.pan);
        safe(l.cdw); safe(l.cdb); safe(l.cgw); safe(l.cgb); safe(l.ks);
        safe(l.pahss); safe(l.pahsb); safe(l.parss); safe(l.parsb);
        safe(l.gdw); safe(l.gdb); safe(l.rfn); safe(l.rf1); safe(l.rf1b);
        safe(l.rf2); safe(l.rf2b); safe(l.rout); safe(l.bb);
        safe(l.gu); safe(l.dn);
        safe(l.pmhss); safe(l.pmhsb); safe(l.pmrss); safe(l.pmrsb);
    }
    if (s->st) {
        // Log rather than HIP_OK_V's early-return here — this is cleanup,
        // the remaining frees and delete s below must still run regardless.
        hipError_t _st = hipStreamDestroy(s->st);
        if (_st != hipSuccess)
            fprintf(stderr, "HIP Error %d at %s:%d — %s\n", _st, __FILE__, __LINE__, hipGetErrorString(_st));
    }
    safe(s->d_lm_vocab); safe(s->d_argmax_idx); safe(s->d_argmax_val);
    safe(s->d_sorted_ids); safe(s->d_expert_counts); safe(s->d_expert_offsets);
    delete s;
}

} // extern "C"
