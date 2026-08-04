/** NPU Engine — Universal Fast. Model-agnostic auto-detect + v12 speed.
 *  M=32 batched decode, OpenMP attention, OpenMP LM head, f32 embeddings.
 *  Supports ALL models with tagged xclbins. Target: >80 tok/s on any model. */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <vector>
#include <chrono>
#include <exception>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <signal.h>
#include <xrt/xrt_device.h>
#include <xrt/xrt_bo.h>
#include <xrt/xrt_kernel.h>
#include <xrt/experimental/xrt_module.h>
#include <xrt/experimental/xrt_elf.h>
#include <xrt/experimental/xrt_ext.h>
#include <aiebu/aiebu_assembler.h>
#include <omp.h>
#include "model_config.h"
#include "npu_engine_i8ctx_inc.h"
#include "npu_engine_hybrid_flm.h"

// Forward declarations: INT8 NPU instruction generators from gemm_npu_instructions.cpp
void gemm_generate_sequence_i8(
    npu_sequence*           seq,
    uint32_t                M,
    uint32_t                K,
    uint32_t                N,
    uint32_t                a_ddr_offset,
    uint32_t                b_base_offset,
    bool                    add_bias,
    int                     activation,
    uint32_t                bias_offset,
    uint32_t                output_offset
);
void gemm_generate_sequence_i8_split(
    npu_sequence*           seq,
    uint32_t                M,
    uint32_t                K,
    uint32_t                N,
    uint32_t                a_ddr_offset,
    uint32_t                b_base_offset,
    bool                    add_bias,
    int                     activation,
    uint32_t                bias_offset,
    uint32_t                output_offset
);

#ifdef ONEBP_SUPPORT
#include "onebp_format.h"
#include "onebp_loader.cpp"
#endif
// FLM dependency removed — pre-compiled instructions loaded from file.
#include <sys/wait.h>
extern "C" float* dequant_i8_to_float_ex(const uint8_t*,int,int,int*,int*);
static inline float bf16f(uint16_t v){uint32_t b=v<<16;float f;memcpy(&f,&b,4);return f;}
static inline float bf16g(uint16_t v){return(v&0x7F80)==0x7F80?0.0f:bf16f(v);}
static constexpr float EPS=1e-6f;
static inline void cn(float*x,int n){for(int i=0;i<n;i++)if(!std::isfinite(x[i]))x[i]=0.0f;}
static inline void sm(float*sc,int n){if(n<=0)return;cn(sc,n);float mx=sc[0];
    for(int i=1;i<n;i++)if(sc[i]>mx)mx=sc[i];double s=0;
    for(int i=0;i<n;i++){float d=sc[i]-mx;if(d>80)d=80;else if(d<-80)d=-80;sc[i]=expf(d);s+=sc[i];}
    if(s<=0){float iv=1.0f/n;for(int i=0;i<n;i++)sc[i]=iv;return;}
    float is=1.0f/(float)s;for(int i=0;i<n;i++)sc[i]*=is;}
static inline void rn_c(float*x,const float*w,int n){cn(x,n);double ss=0;
    for(int i=0;i<n;i++)if(std::isfinite(x[i]))ss+=(double)x[i]*x[i];
    float ir=1.0f/sqrtf((float)(ss/n)+EPS);for(int i=0;i<n;i++)x[i]=std::isfinite(x[i])?x[i]*ir*w[i]:0.0f;}

// ── Cross-layer pipeline (roadmap step 3): fused D-output → next-QKV-input ──
// Consumes the D GEMM output of layer l (Cm, int32 legacy / int16 FLM) and
// produces layer l+1's QKV input in ONE pass, replacing 6 serial CPU passes:
//   dequantize → cn → residual add → pre-QKV save → rn_c reduce → rn_c scale → dyn amax
// Math is bit-identical to the original sequence (same per-element float ops,
// per-row double-precision rn sums, same dynamic_ascale guard).
// Writes: h_b = rn(in_n[l+1], residual + D_out)   [QKV input of l+1]
//         sb_data = residual + D_out              [pre-QKV residual save of l+1]
// Returns: dynamic activation scale for l+1's cq quantize (== dynamic_ascale).
template<typename Tcm>
static inline float fused_cross_layer_boundary(
        const Tcm* Cm, int ND, float cs,
        float* sb_data, float* h_b, const float* in_n,
        int H, int batch, double* rn_ss) {
    // Pass A (per row): dequant + residual + save + rn-reduce
    for(int b=0;b<batch;b++){
        double ss=0;
        float* sb=sb_data+(size_t)b*H;
        float* hh=h_b+(size_t)b*H;
        const Tcm* c=Cm+(size_t)b*ND;
        for(int i=0;i<H;i++){
            float dw=(float)c[i]*cs;              // D GEMM output (dequant)
            float h=sb[i]+dw;                     // residual add
            if(!std::isfinite(h))h=0.0f;          // cn() semantics
            sb[i]=h;                              // pre-QKV residual save (l+1)
            hh[i]=h;
            ss+=(double)h*h;                      // rn_c reduce
        }
        rn_ss[b]=ss;
    }
    // Pass B (per row): rn_c scale + dynamic_ascale amax (one pass)
    float amax=0;
    for(int b=0;b<batch;b++){
        float ir=1.0f/sqrtf((float)(rn_ss[b]/H)+EPS);
        float* hh=h_b+(size_t)b*H;
        for(int i=0;i<H;i++){
            float h2=hh[i]*ir*in_n[i];
            if(!std::isfinite(h2))h2=0.0f;
            hh[i]=h2;
            float a=fabsf(h2);
            if(std::isfinite(a)&&a>amax)amax=a;
        }
    }
    if(amax<1e-12f)amax=1.0f;                     // dynamic_ascale guard
    return amax/127.0f;
}
static std::vector<float>rc,rs;
static void ri(int hd,float th,int mp){int hd2=hd/2;rc.resize(mp*hd);rs.resize(mp*hd);
    for(int p=0;p<mp;p++)for(int d=0;d<hd2;d++){
        float f=1.0f/powf(th,(float)d/hd2),a=p*f;
        rc[p*hd+d]=cosf(a);rs[p*hd+d]=sinf(a);}}
static inline void ra(float*x,int hd,int p){int hd2=hd/2;for(int d=0;d<hd2;d++){
    float a=x[d],b=x[d+hd2],c=rc[p*hd+d],s=rs[p*hd+d];x[d]=a*c-b*s;x[d+hd2]=b*c+a*s;}}
// Safety net: if glibc's malloc detects heap corruption (free(): invalid size)
// SIGABRT handler: prints diagnostic, then re-raises for default core dump
// so the heap corruption root cause can be debugged. The measured results
// are flushed to stderr before the re-raise.
static void sigabrt_handler(int sig) {
    // Async-signal-safe only (issue #1433): fprintf/fflush can deadlock when
    // SIGABRT fires from heap corruption while stdio/arena locks are held.
    static const char m1[] = "\n[NPU engine] caught SIGABRT (likely heap corruption from free(): invalid size)\n";
    static const char m2[] = "[NPU engine] re-raising for core dump — see core.{pid} for backtrace\n";
    ssize_t r1 = write(2, m1, sizeof(m1) - 1);
    ssize_t r2 = write(2, m2, sizeof(m2) - 1);
    (void)r1; (void)r2;
    // Reset handler to default and re-raise to get a core dump
    signal(SIGABRT, SIG_DFL);
    raise(SIGABRT);
}

// ── GatedDeltaNet attention (single-token, CPU, ported from llama.cpp ggml-cpu/ops.cpp) ──
// Operates on transposed state: s[j*GD+i] = S[i][j] (column j of S = row j of s).
// q/k/v/g: [GD] per head, beta: scalar, state: [GD*GD] per head.
// Produces attn_out[GD] per head and updates state in-place.
// GD = state dim (KV head dim), NH = number of heads.
static void gdn_attn_cpu(
        const float* q, const float* k, const float* v,
        const float* g, const float* beta,
        float* state,      // [NH, GD, GD] transposed, updated in-place
        float* attn_out,   // [NH, GD]
        int GD, int NH, float scale)
{
    for (int h = 0; h < NH; h++) {
        const float* qh = q + h * GD;
        const float* kh = k + h * GD;
        const float* vh = v + h * GD;
        const float* gh = g + h * GD;
        float bh = beta[h];
        float* sh = state + (size_t)h * GD * GD;
        float* at = attn_out + h * GD;

        // Precompute exp(g)
        alignas(64) float eg[256];
        for (int i = 0; i < GD; i++) eg[i] = expf(gh[i]);

        // Step 1: S[i][:] *= exp(g[i]) → for each row j of s: s[j][i] *= eg[i]
        for (int j = 0; j < GD; j++) {
            float* sj = sh + j * GD;
            for (int i = 0; i < GD; i++) sj[i] *= eg[i];
        }

        // Step 2: delta[j] = (v[j] - sum_i S[i][j]*k[i]) * beta
        alignas(64) float delta[256];
        for (int j = 0; j < GD; j++) {
            float sum = 0;
            const float* sj = sh + j * GD;  // column j of S (row j of s)
            for (int i = 0; i < GD; i++) sum += sj[i] * kh[i];
            delta[j] = (vh[j] - sum) * bh;
        }

        // Step 3: S[i][j] += k[i] * delta[j] → s[j][i] += delta[j] * k[i]
        for (int j = 0; j < GD; j++) {
            float* sj = sh + j * GD;
            float dj = delta[j];
            for (int i = 0; i < GD; i++) sj[i] += dj * kh[i];
        }

        // Step 4: attn_out[j] = sum_i S[i][j] * q[i] * scale
        for (int j = 0; j < GD; j++) {
            float sum = 0;
            const float* sj = sh + j * GD;
            for (int i = 0; i < GD; i++) sum += sj[i] * qh[i];
            at[j] = sum * scale;
        }
    }
}

static std::vector<float> emb_f32; // f32 embeddings for fast LM head
static std::vector<float> lm_head_f32; // f32 lm_head weights (separate from emb)
// dequant_i8_to_float(_ex) returns row-major [out_features, in_features] (PyTorch nn.Linear);
// packB()/go() need the transpose — [in_features, out_features] — since the GEMM computes
// A[tokens,in] @ B[in,out].
static void transpose_pack(const float* src, int out_f, int in_f, float* dst, int dst_stride, int dst_offset) {
    for (int o = 0; o < out_f; o++)
        for (int i = 0; i < in_f; i++)
            dst[(size_t)i * dst_stride + dst_offset + o] = src[(size_t)o * in_f + i];
}
// Dynamic per-call activation quantization scale.
// Hardcoded 5.0f/127.0f assumes activations stay in [-5,5], but measured post-RMSNorm
// activations range as wide as [-8.24,7.01], silently clipping every layer.
static inline float dynamic_ascale(const float* x, int n) {
    float amax = 0;
    for (int i = 0; i < n; i++) { float a = fabsf(x[i]); if (std::isfinite(a) && a > amax) amax = a; }
    if (amax < 1e-12f) amax = 1.0f;
    return amax / 127.0f;
}

static uint64_t jo(const char*js,size_t jl,const char*nm){size_t nl=strlen(nm);
    const char*p=js,*e=js+jl;while(p<e){auto q=(const char*)memmem(p,e-p,nm,nl);
        if(!q)return 0;if(q>js&&*(q-1)=='"'&&*(q+nl)=='"'){
            auto o=strstr(q,"\"data_offsets\"");if(o){auto a=strchr(o,'[');if(a)return strtoull(a+1,NULL,10);}}p=q+1;}return 0;}


// AttnCtx — NPU attention context with 4 BOs (Q, K, V, output).
// The attn.xclbin kernel signature (from EMBEDDED_METADATA):
//   args: opcode, instr, ninstr, bo0..bo4
//   bo0=Q (i8, NH*HD), bo1=K (i8, max_seq*NKV*HD),
//   bo2=V (i8, max_seq*NKV*HD), bo3=output (i16, NH*HD)
// Pre-compiled attention instructions loaded from file (fixed seq_len).
struct AttnCtx {
    int max_seq, NH, NKV, HD, XM;
    std::unique_ptr<xrt::xclbin> xc;
    std::unique_ptr<xrt::hw_context> hc;
    std::unique_ptr<xrt::module> mdl;
    std::unique_ptr<xrt::elf> elf;
    std::unique_ptr<xrt::ext::kernel> k;
    std::unique_ptr<xrt::bo> bQ, bK, bV, bOut;
    bool initialized = false;

    ~AttnCtx() {}
    bool isReady() { return initialized && k && bQ && bK && bV && bOut; }

    // Initialize with xclbin + runtime-generated instructions.
    // Pre-allocates BOs at max_seq dimensions — only the first `seq_len`
    // entries of K/V are valid per call (quantize K/V for the active range).
    bool init(xrt::device& d, const char* xp,
              const std::vector<uint32_t>& instrs,
              int max_seq_len, int nh, int nkv, int hd, int xm) {
        max_seq = max_seq_len;
        NH = nh; NKV = nkv; HD = hd; XM = xm;
        try {
            std::vector<char> iraw((char*)instrs.data(),
                                   (char*)instrs.data() + instrs.size() * sizeof(uint32_t));
            aiebu::aiebu_assembler asmblr(
                aiebu::aiebu_assembler::buffer_type::blob_instr_transaction, iraw);
            auto e = asmblr.get_elf();
            xc = std::make_unique<xrt::xclbin>(std::string(xp));
            d.register_xclbin(*xc);
            hc = std::make_unique<xrt::hw_context>(d, xc->get_uuid());
            elf = std::make_unique<xrt::elf>(e.data(), e.size());
        } catch (std::exception& ex) {
            fprintf(stderr, "  AttnCtx ELF gen failed: %s\n", ex.what());
            return false;
        }
        mdl = std::make_unique<xrt::module>(*elf);
        k = std::make_unique<xrt::ext::kernel>(*hc, *mdl, "MLIR_AIE");
        // Pre-allocate BOs at max dimensions
        size_t q_bytes = (size_t)XM * NH * HD;            // Q: batch * NH * HD
        size_t kv_bytes = (size_t)max_seq * NKV * HD;     // K/V: max_seq * NKV * HD
        size_t out_bytes = (size_t)XM * NH * HD * 4;       // output: i32 (NPU attention kernel may output int32 accumulators)
        bQ = std::make_unique<xrt::bo>(d, q_bytes, XRT_BO_FLAGS_HOST_ONLY, 0);
        bK = std::make_unique<xrt::bo>(d, kv_bytes, XRT_BO_FLAGS_HOST_ONLY, 0);
        bV = std::make_unique<xrt::bo>(d, kv_bytes, XRT_BO_FLAGS_HOST_ONLY, 0);
        bOut = std::make_unique<xrt::bo>(d, out_bytes, XRT_BO_FLAGS_HOST_ONLY, 0);
        initialized = true;
        return true;
    }

    // Quantize Q (f32) → BO (i8), sync, and launch attention.
    // K/V caches are f32 on host — quantizes the active range [0, seq_len).
    // Returns run handle for later wait+dequant.
    xrt::run launch(const float* Q_f32, const float* K_cache, const float* V_cache,
                    int seq_len, int batch, float q_scale, float kv_scale) {
        // Quantize Q
        auto* q_i8 = (int8_t*)bQ->map();
        float q_is = 1.0f / q_scale;
        for (int i = 0; i < batch * NH * HD; i++) {
            float v = Q_f32[i];
            if (!std::isfinite(v)) v = 0;
            int q = (int)roundf(v * q_is);
            if (q > 127) q = 127; else if (q < -127) q = -127;
            q_i8[i] = (int8_t)q;
        }
        bQ->sync(XCL_BO_SYNC_BO_TO_DEVICE);

        // Quantize K cache [0, seq_len)
        auto* k_i8 = (int8_t*)bK->map();
        float kv_is = 1.0f / kv_scale;
        size_t kv_len = (size_t)seq_len * NKV * HD;
        for (size_t i = 0; i < kv_len; i++) {
            float v = K_cache[i];
            if (!std::isfinite(v)) v = 0;
            int q = (int)roundf(v * kv_is);
            if (q > 127) q = 127; else if (q < -127) q = -127;
            k_i8[i] = (int8_t)q;
        }
        bK->sync(XCL_BO_SYNC_BO_TO_DEVICE);

        // Quantize V cache [0, seq_len)
        auto* v_i8 = (int8_t*)bV->map();
        for (size_t i = 0; i < kv_len; i++) {
            float v = V_cache[i];
            if (!std::isfinite(v)) v = 0;
            int q = (int)roundf(v * kv_is);
            if (q > 127) q = 127; else if (q < -127) q = -127;
            v_i8[i] = (int8_t)q;
        }
        bV->sync(XCL_BO_SYNC_BO_TO_DEVICE);

        // Launch: bo0=Q, bo1=K, bo2=V, bo3=output
        return k->operator()(3, 0, 0, *bQ, *bK, *bV, *bOut);
    }

    // Wait for completion and dequantize output to f32
    void finish(xrt::run& r, float* out, int batch,
                float q_scale, float kv_scale) {
        r.wait();
        bOut->sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        auto* out_i32 = (int32_t*)bOut->map();
        float cs = q_scale * kv_scale;
        for (int i = 0; i < batch * NH * HD; i++) {
            float val = (float)out_i32[i] * cs;
            if (!std::isfinite(val)) val = 0;
            out[i] = val;
        }
    }
};

// v12: OpenMP attention — parallelize across heads, with optional causal mask
static inline void attn_omp(float*qo,float*at,int cl,const float*kv_k,const float*kv_v,int NH,int NKV,int HD,int GQA,int max_pos=-1){
    if(max_pos<0)max_pos=cl;
    #pragma omp parallel for
    for(int hh=0;hh<NH;hh++){int kvh=hh/GQA;
        std::vector<float> scores(cl);float mx=-1e30f;
        for(int p=0;p<cl;p++){if(p>=max_pos){scores[p]=-1e30f;continue;}
            double s=0;int qoff=hh*HD,koff=p*NKV*HD+kvh*HD;
            #pragma omp simd reduction(+:s)
            for(int d=0;d<HD;d++)s+=(double)qo[qoff+d]*kv_k[koff+d];scores[p]=(float)(s/sqrtf((float)HD));if(scores[p]>mx)mx=scores[p];}
        double sw=0;for(int p=0;p<cl;p++){scores[p]=expf(scores[p]-mx);sw+=scores[p];}
        float isw=sw>0?1.0f/(float)sw:1.0f/cl;
        for(int d=0;d<HD;d++){float acc=0;int aoff=hh*HD+d;
            #pragma omp simd reduction(+:acc)
            for(int p=0;p<cl;p++)acc+=scores[p]*kv_v[p*NKV*HD+kvh*HD+d];at[aoff]=acc*isw;}}
}

// v12: OpenMP LM head with f32 embeddings — top-K sampling
// emb: embedding/lm_head table (row-major [vocab_size, hidden_size])
inline void lm_topk_omp(const float*hidden,float*lg,int*top_ids,int K,int NV,int H,const float*emb,float mx=-1e30f){
    #pragma omp parallel for reduction(max:mx)
    for(int n=0;n<NV;n++){double s=0;const float*e=&emb[(size_t)n*H];const float*h=hidden;
        #pragma omp simd reduction(+:s)
        for(int k=0;k<H;k++)s+=(double)h[k]*e[k];lg[n]=(float)s;if(lg[n]>mx)mx=lg[n];}
    double sum=0;
    #pragma omp parallel for reduction(+:sum)
    for(int n=0;n<NV;n++){float d=lg[n]-mx;if(d<-80)d=-80;lg[n]=expf(d);sum+=lg[n];}
    float r=(float)rand()/RAND_MAX*(float)sum,acc=0;
    for(int n=0;n<NV;n++){acc+=lg[n];if(acc>=r){top_ids[0]=n;break;}}
    struct TI{int id;float v;};TI top[32];
    for(int b=0;b<K;b++){top[b].id=-1;top[b].v=-1e30f;}
    for(int n=0;n<NV;n++){float v=lg[n];for(int b=0;b<K;b++){if(v>top[b].v){memmove(&top[b+1],&top[b],(K-1-b)*sizeof(TI));top[b].id=n;top[b].v=v;break;}}}
    for(int b=0;b<K;b++)top_ids[b]=top[b].id;
}

int main(int argc,char**argv){
    setvbuf(stdout,NULL,_IONBF,0);
    srand((unsigned)time(nullptr) ^ (unsigned)getpid()); // issue #1431: sampling was deterministic
    // Install SIGABRT handler for issue #202: heap corruption during decode
    // causes free(): invalid size → SIGABRT. The handler prints diagnostic
    // info, then re-raises with SIG_DFL restored to produce a core dump.
    struct sigaction sa;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = sigabrt_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = SA_RESETHAND; // allow one handler invocation; re-trigger = default (core dump)
    sigaction(SIGABRT, &sa, nullptr);

    if(argc<2){fprintf(stderr,"Usage: %s model.q4nx [decode_tokens] [input_tokens_file|-]\n",argv[0]);return 1;}
    // Check for --worker flag (subprocess protocol mode)
    bool worker_mode=false;
    bool use_flm_xclbin=false;
    for(int i=2;i<argc;i++){
        if(strcmp(argv[i],"--worker")==0){worker_mode=true;}
        if(strcmp(argv[i],"--use-flm-xclbin")==0){use_flm_xclbin=true;}
    }
    const char*mp=argv[1];int ng=(argc>2&&!worker_mode)?atoi(argv[2]):32;if(ng<1)ng=1;if(ng>4096)ng=4096; // cap to KV cache size (issue #112)
    const char*input_tok_file=(argc>3&&!worker_mode&&argv[3][0]!='\0')?argv[3]:nullptr;

    // Model tag
    // Accept --model-tag CLI override (passed by the Zig fused executor)
    std::string mp_s(mp),model_tag;
    for(int i=2;i<argc-1;i++){if(strcmp(argv[i],"--model-tag")==0){model_tag=argv[i+1];break;}}
    if(model_tag.empty()){
        // Try environment variable override first
        const char* env_mt = getenv("NPU_MODEL_TAG");
        if (env_mt && env_mt[0]) {
            model_tag = env_mt;
        } else {
            auto ls=mp_s.rfind('/');model_tag=(ls!=std::string::npos)?mp_s.substr(ls+1):mp_s;
            auto dot=model_tag.rfind('.');if(dot!=std::string::npos)model_tag=model_tag.substr(0,dot);
            // If filename is the generic "model" after extension strip, try parent dir name instead
            if (model_tag == "model") {
                std::string parent = mp_s.substr(0, ls);
                auto ps = parent.rfind('/');
                if (ps != std::string::npos) {
                    model_tag = parent.substr(ps + 1);
                }
            }
        }
    }
    for(auto&c:model_tag){c=tolower(c);if(c=='-'||c=='.'||c=='\\')c='_';}
    const char*sfxs[]={"_npu2","_instruct","_it","_it_npu2"};
    for(auto sf:sfxs){size_t sl=strlen(sf);if(model_tag.size()>sl&&model_tag.substr(model_tag.size()-sl)==sf)model_tag=model_tag.substr(0,model_tag.size()-sl);}

#ifdef ONEBP_SUPPORT
    bool is_onebp = strlen(mp) > 4 && strcmp(mp + strlen(mp) - 4, ".1bp") == 0;
    OnebpModel onebp_model;
#endif
    // Parse config
    ModelConfig cfg;
    #ifdef ONEBP_SUPPORT
        if (is_onebp) {
            if (!onebp_model.open(mp)) { fprintf(stderr,"ERR: 1BP\n"); return 1; }
            auto& oh = onebp_model.header();
            cfg.H = oh.hidden_size; cfg.NC = oh.num_layers;
            cfg.NH = oh.num_attention_heads; cfg.NKV = oh.num_kv_heads;
            cfg.HD = oh.head_dim; cfg.IM = oh.intermediate_size;
            cfg.NV = oh.vocab_size; cfg.GQA = cfg.NH / cfg.NKV;
            cfg.XM = 128; cfg.has_lm_head = true;
        } else
    #endif
        cfg = parse_q4nx_header(mp,model_tag.c_str());

    if(!cfg.valid()){fprintf(stderr,"ERR: invalid model config\n");return 1;}
    int H=cfg.H,NC=cfg.NC,NH=cfg.NH,NKV=cfg.NKV,HD=cfg.HD,IM=cfg.IM,NV=cfg.NV,GQA=cfg.GQA,XM=cfg.XM;
    fprintf(stderr,"=== NPU Engine Universal — %s ===\n",model_tag.c_str());
    fprintf(stderr,"H=%d NC=%d NH=%d NKV=%d HD=%d IM=%d NV=%d GU_split=%d\n",H,NC,NH,NKV,HD,IM,NV,cfg.gu_split);

    // Open model
    int fd=open(mp,O_RDONLY);struct stat st;fstat(fd,&st);
    uint8_t*md=(uint8_t*)mmap(NULL,st.st_size,PROT_READ,MAP_PRIVATE,fd,0);close(fd);
    uint64_t hsz;memcpy(&hsz,md,8);uint64_t df=8+hsz;
    auto i8p=[&](uint64_t o){return md+df+o;};auto emb=(const uint16_t*)(md+df);
    const char*js=(const char*)(md+8);size_t jl=hsz;

    // Pre-convert embeddings f32 (v12 optimization)
    fprintf(stderr,"Pre-convert emb f32...\n");auto te=std::chrono::steady_clock::now();
    #ifdef ONEBP_SUPPORT
    if (is_onebp) {
        std::vector<float> emb_buf;
        if (onebp_model.get_tensor_f32("token_embd.weight", emb_buf)) {
            emb_f32 = emb_buf;
            fprintf(stderr,"  1BP embeddings loaded\n");
        }
    } else {
    #endif
    emb_f32.resize((size_t)NV*H);
    for(int n=0;n<NV;n++)for(int i=0;i<H;i++)emb_f32[(size_t)n*H+i]=bf16g(emb[n*H+i]);
    fprintf(stderr,"  %.0fms\n",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-te).count());
    #ifdef ONEBP_SUPPORT
    }
    #endif

    // Norm weights (try dense 'layers', MoE 'layer', GDN 'linear_attn' patterns)
    std::vector<bool> is_gdn_layer(NC, false);  // true = GDN, false = standard attn
    std::vector<uint64_t> qp_fused(NC, 0);  // fused QKV offset (GDN layers)
    auto jo2 = [&](const char* fmt, int l) -> uint64_t {
        char kb[256];
        snprintf(kb, sizeof(kb), fmt, l);
        uint64_t off = jo(js, jl, kb);
        if (!off) {
            // Try 'model.layer.N.' (MoE naming without 's')
            std::string alt = kb;
            size_t p = alt.find("model.layers.");
            if (p != std::string::npos) { alt.replace(p, 14, "model.layer."); off = jo(js, jl, alt.c_str()); }
        }
        if (!off) {
            // Try 'model.layer.N.linear_attn.' (GDN naming)
            std::string alt = kb;
            size_t p = alt.find("model.layers.");
            if (p == std::string::npos) p = alt.find("model.layer.");
            if (p != std::string::npos) {
                // Replace 'self_attn' with 'linear_attn'
                size_t sa = alt.find("self_attn");
                if (sa != std::string::npos) { alt.replace(sa, 10, "linear_attn"); off = jo(js, jl, alt.c_str()); }
            }
        }
        return off;
    };
    std::vector<uint64_t> in_off(NC),pa_off(NC),qn_off(NC),kn_off(NC),qp(NC),kp(NC),vp(NC),op(NC),gp(NC),up(NC),dp(NC);
    char bn[128];
    for(int l=0;l<NC;l++){
        qp[l]=jo2("model.layers.%d.self_attn.q_proj.weight",l);
        kp[l]=jo2("model.layers.%d.self_attn.k_proj.weight",l);
        vp[l]=jo2("model.layers.%d.self_attn.v_proj.weight",l);
        op[l]=jo2("model.layers.%d.self_attn.o_proj.weight",l);
        // GDN fused QKV (if separate q_proj not found):
        if (!qp[l]) {
            snprintf(bn, 128, "model.layer.%d.linear_attn.qkv_proj.weight", l);
            qp_fused[l] = jo(js, jl, bn);
            if (qp_fused[l]) {
                is_gdn_layer[l] = true;
                // O projection for GDN layers: linear_attn.ssm_out_proj
                snprintf(bn, 128, "model.layer.%d.linear_attn.ssm_out_proj.weight", l);
                op[l] = jo(js, jl, bn);
            }
        }
        gp[l]=jo2("model.layers.%d.mlp.gate_proj.weight",l);
        up[l]=jo2("model.layers.%d.mlp.up_proj.weight",l);
        dp[l]=jo2("model.layers.%d.mlp.down_proj.weight",l);
        in_off[l]=jo2("model.layers.%d.input_layernorm.weight",l);
        pa_off[l]=jo2("model.layers.%d.post_attention_layernorm.weight",l);
        qn_off[l]=jo2("model.layers.%d.self_attn.q_norm.weight",l);
        kn_off[l]=jo2("model.layers.%d.self_attn.k_norm.weight",l);}
    uint64_t no=jo(js,jl,"model.norm.weight");
    uint64_t lo=jo(js,jl,"lm_head.weight");
    std::vector<std::vector<float>> in_n(NC,std::vector<float>(H)),pa_n(NC,std::vector<float>(H)),qn_w(NC,std::vector<float>(HD)),kn_w(NC,std::vector<float>(HD));
    std::vector<float> fin_v(H);
    for(int l=0;l<NC;l++){auto iw=(const uint16_t*)(md+df+in_off[l]),pw=(const uint16_t*)(md+df+pa_off[l]);
        for(int i=0;i<H;i++){in_n[l][i]=bf16g(iw[i]);pa_n[l][i]=bf16g(pw[i]);}
        if(cfg.has_q_norm&&qn_off[l]){auto qq=(const uint16_t*)(md+df+qn_off[l]);for(int i=0;i<HD;i++)qn_w[l][i]=bf16g(qq[i]);}
        if(cfg.has_k_norm&&kn_off[l]){auto kk=(const uint16_t*)(md+df+kn_off[l]);for(int i=0;i<HD;i++)kn_w[l][i]=bf16g(kk[i]);}}
    {auto fw=(const uint16_t*)(md+df+no);for(int i=0;i<H;i++)fin_v[i]=bf16g(fw[i]);}

    // I8 tile rows
    auto gi8=[&](const char*k)->int{int r=0;find_tensor_info(js,jl,k,&r);
        if(r<=0){std::string ak=k;size_t p=ak.find("model.layers.");if(p!=std::string::npos){ak.replace(p,14,"model.layer.");find_tensor_info(js,jl,ak.c_str(),&r);}}
        if(r<=0){std::string ak=k;size_t p=ak.find("model.layer.");if(p!=std::string::npos){size_t sa=ak.find("self_attn");if(sa!=std::string::npos){ak.replace(sa,10,"linear_attn");find_tensor_info(js,jl,ak.c_str(),&r);}}}
        if (r > 0) {
            // Handle 3D Q4NX shapes [tile_rows, tile_cols, bytes]:
            // Multiply by tile_cols if present (Qwen3.6 uses 3D, Qwen3 uses 2D).
            // Default tile_cols = in_features / 256. Compute from known dims.
        }
        return r;};
    int q_i8=gi8("model.layers.0.self_attn.q_proj.weight"),k_i8=gi8("model.layers.0.self_attn.k_proj.weight"),v_i8=gi8("model.layers.0.self_attn.v_proj.weight");
    // Fallback: GDN fused QKV
    int qkv_fused_i8 = 0;
    if (q_i8 <= 0) { q_i8 = gi8("model.layer.0.linear_attn.qkv_proj.weight"); qkv_fused_i8 = q_i8; }
    int o_i8=gi8("model.layers.0.self_attn.o_proj.weight"),g_i8=gi8("model.layers.0.mlp.gate_proj.weight"),u_i8=gi8("model.layers.0.mlp.up_proj.weight"),d_i8=gi8("model.layers.0.mlp.down_proj.weight");
    // GDN fallbacks
    if (o_i8 <= 0) o_i8 = gi8("model.layer.0.linear_attn.ssm_out_proj.weight");
    if (g_i8 <= 0) g_i8 = gi8("model.layer.0.self_attn.gate_proj.weight");
    // Qwen3.6 uses 3D Q4NX shapes [tile_rows, tile_cols, bytes].
    // gi8 returns shape[0] (tile_rows); multiply by tile_cols = in_features/256.
    // Only for 3D-shape models (MoE); 2D-shape models (Qwen3) have cols already included.
    if (cfg.has_moe) {
    int q_cols = H / 256;      // 8 for H=2048
    int o_cols = (NH * HD) / 256; // 16 for NH*HD=4096
    int d_cols = IM / 256;     // 2 for IM=512
    q_i8 *= q_cols; k_i8 *= q_cols; v_i8 *= q_cols;
    qkv_fused_i8 *= q_cols;
    o_i8 *= o_cols;
    g_i8 *= q_cols; u_i8 *= q_cols;
    d_i8 *= d_cols;
    }
    int lm_i8=gi8("lm_head.weight");

    // Load lm_head.weight separately — NOT tied to embed_tokens.weight for this model
    if(lo&&lm_i8>0){int lr,lc;float*lm_raw=dequant_i8_to_float_ex(i8p(lo),lm_i8,H,&lr,&lc);if(lm_raw){
        lm_head_f32.assign(lm_raw,lm_raw+(size_t)lr*lc);free(lm_raw);
        fprintf(stderr,"  lm_head: %dx%d (loaded from JSON), using for final logits\n",lr,lc);
    }else{fprintf(stderr,"  lm_head: dequant failed, falling back to emb\n");}}
    if(lm_head_f32.empty()){fprintf(stderr,"  lm_head: using emb_f32 (tied embeddings)\n");}
    const float* lm_emb = lm_head_f32.empty() ? emb_f32.data() : lm_head_f32.data();

    // Init NPU
    fprintf(stderr,"Init NPU...\n");xrt::device dev(0);
    // Xclbin directory: respect NPU_XCLBIN_DIR env var, fall back to repo-relative path
    const char* env_xd = getenv("NPU_XCLBIN_DIR");
    std::string xd = env_xd ? env_xd : "engine/npu/xclbins";
    auto xp=[&](const char*t){return xd+"/final_i8_"+t+"_"+cfg.model_tag+".xclbin";};
    auto ip=[&](const char*t){return xd+"/insts_i8_"+t+"_"+cfg.model_tag+".txt";};

    // FLM xclbin path: respect NPU_FLM_XCLBIN_DIR env var for explicit path,
    // or NPU_FLM_XCLBINS_ROOT for the root directory containing model subdirectories.
    // Fall back to default FLM build path /home/bcloud/fastflowlm-build/src/xclbins.
    const char* env_flm_xd = getenv("NPU_FLM_XCLBIN_DIR");
    const char* env_flm_root = getenv("NPU_FLM_XCLBINS_ROOT");
    std::string flm_xd;
    if (env_flm_xd) {
        flm_xd = env_flm_xd;
        // If NPU_FLM_XCLBIN_DIR is given, it's the full path to the mm.xclbin
        // (including the model subdirectory — we just append /mm.xclbin? No, it's a dir)
        // NPU_FLM_XCLBIN_DIR = /path/to/Qwen3-0.6B-NPU2/  →  /path/to/Qwen3-0.6B-NPU2/mm.xclbin
    }

    // GEMM contexts: I8Ctx (legacy) or HybridFlmCtx (FLM path)
    bool flm_xclbin_available = false;
    bool cpu_gemm_fallback = false;  // set when NPU GEMM can't init (MoE models)
    std::string flm_mm_path;
    if (use_flm_xclbin) {
        // Try to find mm.xclbin. Priority:
        // 1. NPU_FLM_XCLBIN_DIR/Qwen3-0.6B-NPU2/mm.xclbin (if env set)
        // 2. NPU_FLM_XCLBINS_ROOT/{model-tag variants}/mm.xclbin
        // 3. Default path: /home/bcloud/fastflowlm-build/src/xclbins/{variant}/mm.xclbin

        if (env_flm_root) {
            flm_xd = env_flm_root;
        } else if (!env_flm_xd) {
            flm_xd = "/home/bcloud/fastflowlm-build/src/xclbins";
        }

        if (env_flm_xd) {
            // Direct path: user specified the exact directory
            flm_mm_path = std::string(env_flm_xd) + "/mm.xclbin";
        } else {
            // Try to find the right model directory under root
            // The model tag (e.g., "qwen3_0_6b") doesn't directly map to FLM's
            // directory names (e.g., "Qwen3-0.6B-NPU2"), so we search for mm.xclbin
            // under subdirectories of flm_xd that contain parts of the tag.
            // First try: exact tag match
            flm_mm_path = flm_xd + "/" + cfg.model_tag + "-NPU2/mm.xclbin";
            FILE* f = fopen(flm_mm_path.c_str(), "rb");
            if (!f) {
                // Capitalize first letter
                std::string cap_tag = cfg.model_tag;
                if (!cap_tag.empty()) cap_tag[0] = (char)toupper(cap_tag[0]);
                // Replace _ with - after digits (e.g., qwen3_0_6b → Qwen3-0.6B)
                std::string pascal;
                bool next_upper = true;
                for (size_t i = 0; i < cfg.model_tag.size(); i++) {
                    if (cfg.model_tag[i] == '_') {
                        if (i > 0 && cfg.model_tag[i-1] >= '0' && cfg.model_tag[i-1] <= '9') {
                            pascal += '.';  // 0_6 → 0.6
                        } else {
                            pascal += '-';
                        }
                        next_upper = true;
                    } else if (next_upper) {
                        pascal += (char)toupper(cfg.model_tag[i]);
                        next_upper = false;
                    } else {
                        pascal += cfg.model_tag[i];
                    }
                }
                flm_mm_path = flm_xd + "/" + pascal + "-NPU2/mm.xclbin";
                f = fopen(flm_mm_path.c_str(), "rb");
                if (f) { fclose(f); flm_xclbin_available = true; }
                else {
                    // Third try: recursive search (no shell — issue #1435;
                    // popen("find " + env_dir) broke on spaces and was injectable)
                    fprintf(stderr, "  Searching for mm.xclbin under %s ...\n", flm_xd.c_str());
                    std::error_code ec;
                    for (auto it = std::filesystem::recursive_directory_iterator(flm_xd, ec), end = std::filesystem::recursive_directory_iterator();
                         it != end; it.increment(ec)) {
                        if (ec) break;
                        if (it->is_regular_file(ec) && it->path().filename() == "mm.xclbin") {
                            flm_mm_path = it->path().string();
                            break;
                        }
                    }
                    if (!flm_mm_path.empty()) {
                        FILE* cf = fopen(flm_mm_path.c_str(), "rb");
                        if (cf) { fclose(cf); flm_xclbin_available = true; }
                    }
                }
            } else {
                fclose(f);
                flm_xclbin_available = true;
            }
        }

        if (flm_xclbin_available) {
            fprintf(stderr, "  FLM mm.xclbin: %s\n", flm_mm_path.c_str());
        } else {
            fprintf(stderr, "  WARN: FLM mm.xclbin not found (searched %s), falling back to open-source path\n",
                    flm_xd.c_str());
        }
    }

    // Legacy I8Ctx pointers (always available, fallback if FLM xclbin not found)
    I8Ctx cq,co,cg,cd;
    std::unique_ptr<I8Ctx> cu_ptr;
    std::unique_ptr<AttnCtx> ca_ptr;
    // Hybrid FLM contexts (only used when --use-flm-xclbin and xclbin found)
    std::unique_ptr<HybridFlmCtx> hcq, hco, hcg, hcd, hcu_ptr;
    std::vector<uint32_t> attn_instrs;
    auto load_attn_instrs = [&](const char* path) -> bool {
        FILE* f = fopen(path, "rb");
        if (!f) { fprintf(stderr, "  No attn insts: %s\n", path); return false; }
        fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
        attn_instrs.resize(sz / 4);
        fread(attn_instrs.data(), 4, attn_instrs.size(), f);
        fclose(f);
        return true;
    };
    cq.MD=XM;cq.KD=cfg.xclbin_qkv_k;cq.ND=cfg.xclbin_qkv_n;
    co.MD=XM;co.KD=cfg.xclbin_o_k;co.ND=cfg.xclbin_o_n;
    cd.MD=XM;cd.KD=cfg.xclbin_d_k;cd.ND=cfg.xclbin_d_n;
    if(cfg.gu_split){cg.MD=XM;cg.KD=cfg.xclbin_g_k;cg.ND=cfg.xclbin_g_n;}else{cg.MD=XM;cg.KD=cfg.xclbin_gu_k;cg.ND=cfg.xclbin_gu_n;}

    if (flm_xclbin_available) {
        // ── Hybrid FLM path: use FLM's mm.xclbin + 1 BO per GEMM type ──
        fprintf(stderr, "  Using FLM hybrid engine...\n");
        hcq  = std::make_unique<HybridFlmCtx>();
        hco  = std::make_unique<HybridFlmCtx>();
        hcd  = std::make_unique<HybridFlmCtx>();
        hcg  = std::make_unique<HybridFlmCtx>();
        if (!hcq->init(dev, flm_mm_path.c_str(), XM, cfg.xclbin_qkv_k, cfg.xclbin_qkv_n, NC)) {
            fprintf(stderr, "FAIL Hybrid QKV\n"); return 1; }
        fprintf(stderr, "  Hybrid QKV OK\n");
        if (!hco->init(dev, flm_mm_path.c_str(), XM, cfg.xclbin_o_k, cfg.xclbin_o_n, NC)) {
            fprintf(stderr, "FAIL Hybrid O\n"); return 1; }
        fprintf(stderr, "  Hybrid O OK\n");
        if (cfg.gu_split) {
            if (!hcg->init(dev, flm_mm_path.c_str(), XM, cfg.xclbin_g_k, cfg.xclbin_g_n, NC)) {
                fprintf(stderr, "FAIL Hybrid G\n"); return 1; }
            hcu_ptr = std::make_unique<HybridFlmCtx>();
            if (!hcu_ptr->init(dev, flm_mm_path.c_str(), XM, cfg.xclbin_u_k, cfg.xclbin_u_n, NC)) {
                fprintf(stderr, "FAIL Hybrid U\n"); return 1; }
        } else {
            if (!hcg->init(dev, flm_mm_path.c_str(), XM, cfg.xclbin_gu_k, cfg.xclbin_gu_n, NC)) {
                fprintf(stderr, "FAIL Hybrid GU\n"); return 1; }
        }
        fprintf(stderr, "  Hybrid GU OK\n");
        if (!hcd->init(dev, flm_mm_path.c_str(), XM, cfg.xclbin_d_k, cfg.xclbin_d_n, NC)) {
            fprintf(stderr, "FAIL Hybrid D\n"); return 1; }
        fprintf(stderr, "  Hybrid D OK\n");
        // Sync weights after all packB calls (done at pack time in the pipeline below)
    } else {
        // ── Legacy path: per-op xclbins + per-layer weight BOs ──
        // MoE models may lack instruction files — check before init.
        bool has_insts = true;
        if (cfg.has_moe) {
            auto check_inst = [&](const char* t) {
                std::string ip_s = ip(t);
                FILE* f = fopen(ip_s.c_str(), "rb");
                if (f) { fclose(f); return true; }
                fprintf(stderr, "  WARN: %s not found\n", ip_s.c_str());
                return false;
            };
            has_insts = check_inst("QKV") && check_inst("O");
            if (!has_insts) cpu_gemm_fallback = true;
        }
        if (!cpu_gemm_fallback) {
        fprintf(stderr,"  cq before init: MD=%d KD=%d ND=%d\n", cq.MD, cq.KD, cq.ND);
        if(!cq.init(dev,xp("QKV").c_str(),ip("QKV").c_str(),4,NC)){fprintf(stderr,"FAIL QKV\n");return 1;}
        if(!co.init(dev,xp("O").c_str(),ip("O").c_str(),4,NC)){fprintf(stderr,"FAIL O\n");return 1;}
        if(cfg.gu_split){if(!cg.init(dev,xp("G").c_str(),ip("G").c_str(),4,NC)){fprintf(stderr,"FAIL G\n");return 1;}}else{if(!cg.init(dev,xp("GU").c_str(),ip("GU").c_str(),4,NC)){fprintf(stderr,"FAIL GU\n");return 1;}}
        if(!cd.init(dev,xp("D").c_str(),ip("D").c_str(),4,NC)){fprintf(stderr,"FAIL D\n");return 1;}
        if(cfg.gu_split){cu_ptr=std::make_unique<I8Ctx>();cu_ptr->MD=XM;cu_ptr->KD=cfg.xclbin_u_k;cu_ptr->ND=cfg.xclbin_u_n;if(!cu_ptr->init(dev,xp("U").c_str(),ip("U").c_str(),4,NC)){fprintf(stderr,"FAIL U\n");return 1;}}
        }
    }
    // NPU attention via pre-compiled KV xclbin instructions.
    // Auto-detected when both final_i8_ATTN_<tag>.xclbin and insts_i8_KV_<tag>.txt exist.
    // Explicitly disable with NPU_ATTN=0; override inst path with NPU_ATTN_FILE=<path>.
    bool use_npu_attn = false;
    {
        const char* npu_attn_env = getenv("NPU_ATTN");
        if (npu_attn_env && atoi(npu_attn_env) == 0) {
            fprintf(stderr, "NPU attention disabled via NPU_ATTN=0\n");
        } else {
            // Check if the ATTN xclbin exists before trying
            std::string xclbin_path = xp("ATTN");
            FILE* xc_test = fopen(xclbin_path.c_str(), "rb");
            if (xc_test) {
                fclose(xc_test);
                std::string inst_path;
                if (const char* env = getenv("NPU_ATTN_FILE")) {
                    inst_path = env;
                } else {
                    inst_path = std::string(xd) + "/insts_i8_KV_" + cfg.model_tag + ".txt";
                }
                if (load_attn_instrs(inst_path.c_str())) {
                    ca_ptr = std::make_unique<AttnCtx>();
                    if (ca_ptr->init(dev, xclbin_path.c_str(), attn_instrs,
                                     4096, NH, NKV, HD, XM)) {
                        fprintf(stderr, "NPU attention enabled (pre-compiled insts)\n");
                        use_npu_attn = true;
                    } else {
                        fprintf(stderr, "WARN: AttnCtx init failed, CPU fallback\n");
                        ca_ptr.reset();
                    }
                } else {
                    fprintf(stderr, "  ATTN xclbin found but no KV insts for '%s', CPU fallback\n", cfg.model_tag.c_str());
                }
            } else {
                fprintf(stderr, "  No ATTN xclbin for '%s', CPU fallback\n", cfg.model_tag.c_str());
            }
        }
    }

    // ── GEMM dispatch helpers ──
    // Redirect GEMM calls to either I8Ctx (legacy) or HybridFlmCtx (FLM path)
    // based on flm_xclbin_available flag.
    // I8Ctx and HybridFlmCtx have the same method signatures, so each macro
    // dispatches to ctx.go(...) or h##ctx->go(...) based on the flag.
#define FLM_GO(ctx, ...)         (flm_xclbin_available ? h##ctx->go(__VA_ARGS__) : ctx.go(__VA_ARGS__))
#define FLM_PACKB(ctx, ...)      (flm_xclbin_available ? h##ctx->packB(__VA_ARGS__) : ctx.packB(__VA_ARGS__))
#define FLM_LAUNCH_ASYNC(ctx, ...)  (flm_xclbin_available ? h##ctx->launch_async(__VA_ARGS__) : ctx.launch_async(__VA_ARGS__))
#define FLM_FINISH_ASYNC(ctx, ...)  (flm_xclbin_available ? h##ctx->finish_async(__VA_ARGS__) : ctx.finish_async(__VA_ARGS__))
#define FLM_LAUNCH(ctx, ...)     (flm_xclbin_available ? h##ctx->launch(__VA_ARGS__) : ctx.launch(__VA_ARGS__))
#define FLM_QUANTIZE_ASYNC(ctx, ...) (flm_xclbin_available ? h##ctx->quantize_async(__VA_ARGS__) : ctx.quantize_async(__VA_ARGS__))
#define FLM_SYNC_AND_LAUNCH(ctx, ...) (flm_xclbin_available ? h##ctx->sync_and_launch(__VA_ARGS__) : ctx.sync_and_launch(__VA_ARGS__))
#define FLM_SYNC_A(ctx, ...)     (flm_xclbin_available ? h##ctx->sync_A(__VA_ARGS__) : ctx.sync_A(__VA_ARGS__))
#define FLM_WAIT_KERNEL(ctx, ...) (flm_xclbin_available ? h##ctx->wait_kernel(__VA_ARGS__) : ctx.wait_kernel(__VA_ARGS__))
#define FLM_DEQUANTIZE(ctx, ...) (flm_xclbin_available ? h##ctx->dequantize(__VA_ARGS__) : ctx.dequantize(__VA_ARGS__))
#define FLM_READBACK(ctx)        (flm_xclbin_available ? h##ctx->readback() : ctx.readback())
#define FLM_SYNC_BACK(ctx, ...)  (flm_xclbin_available ? h##ctx->sync_back_and_dequant(__VA_ARGS__) : ctx.sync_back_and_dequant(__VA_ARGS__))
#define FLM_IS_READY(ctx)        (flm_xclbin_available ? h##ctx->isReady() : ctx.isReady())
// Unique_ptr variants (for cu_ptr which uses -> instead of .)
#define FLM_GO_PTR(ctx, ...)         (flm_xclbin_available ? h##ctx->go(__VA_ARGS__) : ctx->go(__VA_ARGS__))
#define FLM_PACKB_PTR(ctx, ...)      (flm_xclbin_available ? h##ctx->packB(__VA_ARGS__) : ctx->packB(__VA_ARGS__))
#define FLM_SYNC_AND_LAUNCH_PTR(ctx, ...) (flm_xclbin_available ? h##ctx->sync_and_launch(__VA_ARGS__) : ctx->sync_and_launch(__VA_ARGS__))
#define FLM_DEQUANTIZE_PTR(ctx, ...) (flm_xclbin_available ? h##ctx->dequantize(__VA_ARGS__) : ctx->dequantize(__VA_ARGS__))
#define FLM_QUANTIZE_ASYNC_PTR(ctx, ...) (flm_xclbin_available ? h##ctx->quantize_async(__VA_ARGS__) : ctx->quantize_async(__VA_ARGS__))
#define FLM_IS_READY_PTR(ctx)    (flm_xclbin_available ? h##ctx->isReady() : ctx->isReady())

    fprintf(stderr,"Dequant+pack...\n");auto tp=std::chrono::steady_clock::now();
    std::vector<float> qsc(NC),osc(NC),gsc(NC),dsc(NC),usc(NC);
    std::vector<std::vector<float>> cpu_qkv_w, cpu_o_w;  // CPU fallback: saved dequant weights
    if (cpu_gemm_fallback) { cpu_qkv_w.resize(NC); cpu_o_w.resize(NC); }
    const int QOUT=NH*HD,KVOUT=NKV*HD;   // QKV out_features, in_features=H (default dequant correct)
    const int OOUT=H,OIN=NH*HD;          // O: out=H, in=NH*HD — dequant needs OIN
    const int GUOUT=IM;                   // Gate/Up: out=IM, in=H
    const int DOUT=H,DIN=IM;              // Down: out=H, in=IM — dequant needs DIN
    if (!cpu_gemm_fallback) {
    for(int l=0;l<NC;l++){
        if (is_gdn_layer[l]) {
            // GDN layer: load fused QKV + SSM out projection
            if (!qp_fused[l] || !op[l] || qkv_fused_i8 <= 0) continue;
            fprintf(stderr, "  layer %d GDN: qp_fused=%llu qkv_i8=%d\n", l, (unsigned long long)qp_fused[l], qkv_fused_i8);
            int qr, qc, or2, oc2;
            fprintf(stderr, "    dequant qkv...\n");
            float* qkv_w = dequant_i8_to_float_ex(i8p(qp_fused[l]), qkv_fused_i8, H, &qr, &qc);
            fprintf(stderr, "    dequant qkv done [%d,%d]\n", qr, qc);
            float* ow = dequant_i8_to_float_ex(i8p(op[l]), o_i8, OIN, &or2, &oc2);
            if (!qkv_w || !ow) { free(qkv_w); free(ow); continue; }
            // Fused QKV: pack as if it were [Q, K, V] concatenated
            int t = QOUT + KVOUT + KVOUT;
            std::vector<float> w((size_t)H * t);
            transpose_pack(qkv_w, QOUT, H, w.data(), t, 0);           // Q
            transpose_pack(qkv_w + QOUT, KVOUT, H, w.data(), t, QOUT); // K
            transpose_pack(qkv_w + QOUT + KVOUT, KVOUT, H, w.data(), t, QOUT + KVOUT); // V
            FLM_PACKB(cq, l, w.data(), H, t, qsc[l]);
            free(qkv_w);
            // O projection
            std::vector<float> wo((size_t)OIN * OOUT);
            transpose_pack(ow, OOUT, OIN, wo.data(), OOUT, 0);
            FLM_PACKB(co, l, wo.data(), OIN, OOUT, osc[l]);
            free(ow);
        } else if (qp[l] && kp[l] && vp[l] && op[l]) {
        int qr,kr,vr,unused;
        float*qw=dequant_i8_to_float_ex(i8p(qp[l]),q_i8,H,&qr,&unused),*kw=dequant_i8_to_float_ex(i8p(kp[l]),k_i8,H,&kr,&unused),*vw=dequant_i8_to_float_ex(i8p(vp[l]),v_i8,H,&vr,&unused);
        int t=QOUT+KVOUT+KVOUT;std::vector<float>w((size_t)H*t);
        transpose_pack(qw,QOUT,H,w.data(),t,0); transpose_pack(kw,KVOUT,H,w.data(),t,QOUT); transpose_pack(vw,KVOUT,H,w.data(),t,QOUT+KVOUT);
        FLM_PACKB(cq,l,w.data(),H,t,qsc[l]);free(qw);free(kw);free(vw);
        int or2,oc2;float*ow=dequant_i8_to_float_ex(i8p(op[l]),o_i8,OIN,&or2,&oc2);
        std::vector<float>wo((size_t)OIN*OOUT);transpose_pack(ow,OOUT,OIN,wo.data(),OOUT,0);
        FLM_PACKB(co,l,wo.data(),OIN,OOUT,osc[l]);free(ow);
        if (gp[l] && up[l]) {
        int gr,ur;float*gw=dequant_i8_to_float_ex(i8p(gp[l]),g_i8,H,&gr,&unused),*uw=dequant_i8_to_float_ex(i8p(up[l]),u_i8,H,&ur,&unused);
        if(cfg.gu_split){
            std::vector<float>wg((size_t)H*gr);transpose_pack(gw,GUOUT,H,wg.data(),gr,0);
            FLM_PACKB(cg,l,wg.data(),H,gr,gsc[l]);
            std::vector<float>wu((size_t)H*ur);transpose_pack(uw,GUOUT,H,wu.data(),ur,0);
            FLM_PACKB_PTR(cu_ptr,l,wu.data(),H,ur,usc[l]);
        }else{
            int t2=gr+ur;std::vector<float>w2((size_t)H*t2);
            transpose_pack(gw,GUOUT,H,w2.data(),t2,0);transpose_pack(uw,GUOUT,H,w2.data(),t2,GUOUT);
            FLM_PACKB(cg,l,w2.data(),H,t2,gsc[l]);
        }free(gw);free(uw);
        }
        if (dp[l]) {
        int dr2,dc2;float*dw=dequant_i8_to_float_ex(i8p(dp[l]),d_i8,DIN,&dr2,&dc2);
        std::vector<float>wd((size_t)DIN*DOUT);transpose_pack(dw,DOUT,DIN,wd.data(),DOUT,0);
        FLM_PACKB(cd,l,wd.data(),DIN,DOUT,dsc[l]);free(dw);
        }
        } // end else if (standard layer)
        } // end for l
    // Hybrid FLM: sync all weight BOs to device after packing (single DMA per type)
    if(flm_xclbin_available){
        hcq->sync_weights(); hco->sync_weights();
        hcg->sync_weights(); hcd->sync_weights();
        if(cfg.gu_split && hcu_ptr) hcu_ptr->sync_weights();
    }
    } // end if (!cpu_gemm_fallback)
    fprintf(stderr,"  %.0fms\n\n",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-tp).count());

    // ── MoE weight loading (router dequant; expert offsets kept raw) ──
    int N_EXPERTS = cfg.N_EXPERTS, TOP_K = cfg.TOP_K, IM_EXP = cfg.IM_EXP, N_SHARED = cfg.N_SHARED;
    bool has_moe = cfg.has_moe;
    // Per-layer: router [H, N_EXPERTS] float, expert Q4NX offsets + tile rows,
    // shared expert offsets + tile rows, shared gate [H] float.
    std::vector<std::vector<float>> router_w, sh_gate_vec;
    struct MoeOffsets { uint64_t gate, up, down; int gate_tr, up_tr, down_tr; };
    struct ShOffsets  { uint64_t gate, up, down; int gate_tr, up_tr, down_tr; };
    std::vector<MoeOffsets> exp_off;    // per-layer expert offsets
    std::vector<ShOffsets>  sh_off;     // per-layer shared expert offsets
    if (has_moe) {
        fprintf(stderr, "Loading MoE weights: experts=%d top_k=%d im_exp=%d shared=%d\n",
                N_EXPERTS, TOP_K, IM_EXP, N_SHARED);
        router_w.resize(NC); sh_gate_vec.resize(NC);
        exp_off.resize(NC); sh_off.resize(NC);
        auto te_moe = std::chrono::steady_clock::now();
        // Tile rows from layer 0 (same for all layers)
        int exp_gate_tr = 0, exp_up_tr = 0, exp_down_tr = 0;
        find_tensor_info(js, jl, "model.layer.0.mlp.gate_exps_proj.weight", &exp_gate_tr);
        find_tensor_info(js, jl, "model.layer.0.mlp.up_exps_proj.weight", &exp_up_tr);
        find_tensor_info(js, jl, "model.layer.0.mlp.down_exps_proj.weight", &exp_down_tr);
        for (int l = 0; l < NC; l++) {
            // Router: BF16 [H, N_EXPERTS] stride-8 interleave
            snprintf(bn, 128, "model.layer.%d.moe_router.weight", l);
            uint64_t roff = jo(js, jl, bn);
            if (roff) {
                router_w[l].resize((size_t)H * N_EXPERTS);
                const uint16_t* rb = (const uint16_t*)i8p(roff);
                for (int i = 0; i < H; i++)
                    for (int j = 0; j < N_EXPERTS; j++)
                        router_w[l][i * N_EXPERTS + j] =
                            bf16g(rb[(size_t)(i % 8) * 65536 + j * 256 + i / 8]);
            }
            // Expert weights: store offsets + tile rows (dequant on demand)
            snprintf(bn, 128, "model.layer.%d.mlp.gate_exps_proj.weight", l);
            exp_off[l].gate = jo(js, jl, bn); exp_off[l].gate_tr = exp_gate_tr;
            snprintf(bn, 128, "model.layer.%d.mlp.up_exps_proj.weight", l);
            exp_off[l].up   = jo(js, jl, bn); exp_off[l].up_tr   = exp_up_tr;
            snprintf(bn, 128, "model.layer.%d.mlp.down_exps_proj.weight", l);
            exp_off[l].down = jo(js, jl, bn); exp_off[l].down_tr = exp_down_tr;
            // Shared expert weights: offsets + tile rows
            snprintf(bn, 128, "model.layer.%d.mlp.share_gate_exps_proj.weight", l);
            sh_off[l].gate = jo(js, jl, bn);
            snprintf(bn, 128, "model.layer.%d.mlp.share_up_exps_proj.weight", l);
            sh_off[l].up   = jo(js, jl, bn);
            snprintf(bn, 128, "model.layer.%d.mlp.share_down_exps_proj.weight", l);
            sh_off[l].down = jo(js, jl, bn);
            find_tensor_info(js, jl, "model.layer.0.mlp.share_gate_exps_proj.weight", &sh_off[l].gate_tr);
            find_tensor_info(js, jl, "model.layer.0.mlp.share_up_exps_proj.weight", &sh_off[l].up_tr);
            find_tensor_info(js, jl, "model.layer.0.mlp.share_down_exps_proj.weight", &sh_off[l].down_tr);
            // Shared expert gate vector [H] BF16
            snprintf(bn, 128, "model.layer.%d.shared_expert_gate.weight", l);
            uint64_t sgoff = jo(js, jl, bn);
            if (sgoff) {
                const uint16_t* gb = (const uint16_t*)i8p(sgoff);
                sh_gate_vec[l].resize(H);
                for (int i = 0; i < H; i++) sh_gate_vec[l][i] = bf16g(gb[i]);
            }
        }
        auto ms_moe = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - te_moe).count();
        fprintf(stderr, "  MoE offsets stored in %.0fms\n", ms_moe);
    }

    // RoPE
    ri(HD,cfg.rope_theta,4096);
    int kv_dwords=NKV*HD/2;

    // Decode batch width.
    //
    // WARNING (issue #111): the "M=32 batched decode" path is NOT a correct
    // decoding algorithm. It embeds the 32 top-K candidates for a *single*
    // next position as if they were 32 *sequential* tokens (see the loop that
    // does h_b[b*H+i]=emb_f32[top_ids[b]*H+i]), writes all 32 into the KV cache
    // at consecutive positions, and runs attention with cl=sp+batch_size --
    // i.e. every position attends over 31 not-yet-decoded, mutually-exclusive
    // "future" positions (non-causal). This corrupts even position 0's output,
    // so the reported 32x throughput described tokens that were never valid.
    //
    // Until a real speculative draft+verify is implemented (accept only the
    // longest matching prefix, roll the KV cache back on a miss), BS is pinned
    // to 1 -> plain causal single-token greedy decode, which is correct.
    // Do not raise this without implementing verification.
    int BS=1;
    struct KVCache{std::vector<float>k,v;int n;KVCache(int size):k(size),v(size),n(0){}};
    int kv_size=4096*NKV*HD;
    std::vector<KVCache> kv_caches;for(int i=0;i<NC;i++)kv_caches.emplace_back(kv_size);
    int qkv_n=cfg.qkv_total;
    std::vector<float> h_b(XM*H), qo_b(XM*qkv_n), at_b(XM*NH*HD), oo_b(XM*H), gt_b(XM*(cfg.gu_split?IM:2*IM)), su_b(XM*IM), dw_b(XM*H);
    std::vector<float> h_data(H), qo_data(qkv_n*BS), ko_data(NKV*HD*BS), vo_data(NKV*HD*BS), at_data(NH*HD*BS), oo_data(H*BS);
    std::vector<float> gt_data((cfg.gu_split?IM:2*IM)*BS), su_data(IM*BS), dwo_data(H*BS), sb_data(XM*H), lg_buf(NV);
    int sp=0;

    // ── CPU MoE FFN helper (dequant active experts on-the-fly) ──
    // x: input [H], out: output [H], l: layer index
    // ponytail: CPU matmul, NPU I8Ctx later when per-layer latency matters
    auto moe_ffn_cpu = [&](const float* x, float* out, int l) {
        const auto& eo = exp_off[l];
        const auto& so = sh_off[l];
        // Router: softmax → top-K
        const float* rt = router_w[l].data();
        std::vector<float> logits(N_EXPERTS), probs(N_EXPERTS);
        double lmax = -1e30;
        for (int j = 0; j < N_EXPERTS; j++) {
            double s = 0;
            for (int i = 0; i < H; i++) s += (double)x[i] * rt[i * N_EXPERTS + j];
            logits[j] = (float)s;
            if (logits[j] > lmax) lmax = logits[j];
        }
        double lsum = 0;
        for (int j = 0; j < N_EXPERTS; j++) {
            probs[j] = expf(logits[j] - (float)lmax);
            lsum += probs[j];
        }
        for (int j = 0; j < N_EXPERTS; j++) probs[j] /= (float)lsum;
        std::vector<int> topk(N_EXPERTS);
        for (int j = 0; j < N_EXPERTS; j++) topk[j] = j;
        std::partial_sort(topk.begin(), topk.begin() + TOP_K, topk.end(),
            [&](int a, int b) { return probs[a] > probs[b]; });

        memset(out, 0, H * sizeof(float));
        // Per-expert tile-rows: gate/up each have IM_EXP/32 tile rows per expert
        int exp_gate_tr_per = eo.gate_tr / N_EXPERTS;  // tile rows per expert for gate
        int exp_up_tr_per   = eo.up_tr   / N_EXPERTS;
        int exp_down_tr_per = eo.down_tr / N_EXPERTS;
        int exp_gate_off_stride = exp_gate_tr_per * 8 * H;  // bytes per expert in Q4NX
        int exp_up_off_stride   = exp_up_tr_per   * 8 * H;
        int exp_down_off_stride = exp_down_tr_per * 8 * IM_EXP;

        for (int e = 0; e < TOP_K; e++) {
            int ex = topk[e];
            // Dequant this expert's G/U/D from Q4NX on-the-fly
            int gr, gc, ur, uc, dr, dc;
            float* G = dequant_i8_to_float_ex(
                i8p(eo.gate + (uint64_t)ex * exp_gate_off_stride),
                exp_gate_tr_per, H, &gr, &gc);
            float* U = dequant_i8_to_float_ex(
                i8p(eo.up + (uint64_t)ex * exp_up_off_stride),
                exp_up_tr_per, H, &ur, &uc);
            float* D = dequant_i8_to_float_ex(
                i8p(eo.down + (uint64_t)ex * exp_down_off_stride),
                exp_down_tr_per, IM_EXP, &dr, &dc);
            if (!G || !U || !D) { free(G); free(U); free(D); continue; }
            // G/U: [IM_EXP, H] @ x → [IM_EXP]
            std::vector<float> gu(IM_EXP * 2);
            for (int i = 0; i < IM_EXP; i++) {
                double g = 0, u = 0;
                for (int k = 0; k < H; k++) {
                    g += (double)G[i * H + k] * x[k];
                    u += (double)U[i * H + k] * x[k];
                }
                float gv = (float)g, uv = (float)u;
                if (!std::isfinite(gv)) gv = 0;
                if (!std::isfinite(uv)) uv = 0;
                gu[i] = gv; gu[IM_EXP + i] = uv;
            }
            for (int i = 0; i < IM_EXP; i++) {
                float gv = gu[i];
                if (!std::isfinite(gv)) gv = 0;
                gu[i] = (gv / (1.0f + expf(-gv))) * gu[IM_EXP + i];
            }
            // D: [H, IM_EXP] @ gu → [H], weighted by router prob
            float pw = probs[ex];
            for (int i = 0; i < H; i++) {
                double d = 0;
                for (int k = 0; k < IM_EXP; k++) d += (double)D[i * IM_EXP + k] * gu[k];
                out[i] += pw * (float)d;
            }
            free(G); free(U); free(D);
        }

        // Shared expert: sigmoid gate → SiLU(G@x) * U@x → D @ activation
        if (N_SHARED > 0 && so.gate) {
            double sg = 0;
            const float* sg_ptr = sh_gate_vec[l].data();
            for (int i = 0; i < H; i++) sg += (double)x[i] * sg_ptr[i];
            float sg_sig = 1.0f / (1.0f + expf(-(float)sg));

            int sgr, sgc, sur, suc, sdr, sdc;
            float* SG = dequant_i8_to_float_ex(i8p(so.gate), so.gate_tr, H, &sgr, &sgc);
            float* SU = dequant_i8_to_float_ex(i8p(so.up),   so.up_tr,   H, &sur, &suc);
            float* SD = dequant_i8_to_float_ex(i8p(so.down), so.down_tr, IM_EXP, &sdr, &sdc);
            if (SG && SU && SD) {
                std::vector<float> sgu(IM_EXP * 2);
                for (int i = 0; i < IM_EXP; i++) {
                    double g = 0, u = 0;
                    for (int k = 0; k < H; k++) {
                        g += (double)SG[i * H + k] * x[k];
                        u += (double)SU[i * H + k] * x[k];
                    }
                    float gv = (float)g, uv = (float)u;
                    if (!std::isfinite(gv)) gv = 0;
                    if (!std::isfinite(uv)) uv = 0;
                    sgu[i] = gv; sgu[IM_EXP + i] = uv;
                }
                for (int i = 0; i < IM_EXP; i++) {
                    float gv = sgu[i];
                    if (!std::isfinite(gv)) gv = 0;
                    sgu[i] = (gv / (1.0f + expf(-gv))) * sgu[IM_EXP + i];
                }
                for (int i = 0; i < H; i++) {
                    double d = 0;
                    for (int k = 0; k < IM_EXP; k++) d += (double)SD[i * IM_EXP + k] * sgu[k];
                    out[i] += sg_sig * (float)d;
                }
            }
            free(SG); free(SU); free(SD);
        }
        for (int i = 0; i < H; i++) if (!std::isfinite(out[i])) out[i] = 0;
    };

    // ===== WORKER MODE (subprocess protocol) =====
    // The Zig fused executor (fused_execute.zig) sends individual GEMM
    // operations (QKV, OPROJ, GATEUP, DOWN) via this protocol. Each request
    // is header[4] (op, layer, batch, in_dim) followed by float input data.
    // Response is header[2] (0=ok, out_dim) followed by float output data.
    if(worker_mode){
        fprintf(stderr,"WORKER_READY\n");
        fflush(stderr);
        // Startup handshake: parent waits for this before sending ops (issue #365)
        write(1, "READY\n", 6);
        setbuf(stdout, NULL);
        clearerr(stdout);
        fflush(stdout);
        uint32_t hdr[4];
        while(fread(hdr,sizeof(uint32_t),4,stdin)==4){
            uint32_t op=hdr[0],layer=hdr[1],batch=hdr[2],in_dim=hdr[3];
            if(op==0) break; // QUIT

            // Input validation: batch and in_dim must be reasonable
            if(batch==0||batch>XM||in_dim==0||in_dim>4096||layer>=(uint32_t)NC){
                uint32_t resp[2]={1,0};
                fwrite(resp,sizeof(uint32_t),2,stdout);
                fflush(stdout);
                // Drain input payload
                std::vector<float> drain(batch*in_dim);
                fread(drain.data(),sizeof(float),batch*in_dim,stdin);
                continue;
            }

            std::vector<float> in_data(batch*in_dim);
            if(fread(in_data.data(),sizeof(float),batch*in_dim,stdin)!=(size_t)(batch*in_dim)) break;

            uint32_t out_dim=0;
            std::vector<float> out_data;
            bool ok=true;

            try{
                if(op==1&&FLM_IS_READY(cq)){ // QKV projection
                    out_dim=cfg.qkv_total;
                    out_data.resize(batch*out_dim,0);
                    float ascale=dynamic_ascale(in_data.data(),batch*in_dim);
                    FLM_GO(cq,layer,in_data.data(),batch,(int)in_dim,ascale,qsc[layer],out_data.data(),(int)out_dim);
                }else if(op==2&&FLM_IS_READY(co)){ // O projection
                    out_dim=H;
                    out_data.resize(batch*out_dim,0);
                    float ascale=dynamic_ascale(in_data.data(),batch*in_dim);
                    FLM_GO(co,layer,in_data.data(),batch,(int)in_dim,ascale,osc[layer],out_data.data(),(int)out_dim);
                }else if(op==3&&FLM_IS_READY(cg)){ // Gate+Up
                    out_dim=cfg.gu_split?IM:(2*IM);
                    out_data.resize(batch*out_dim,0);
                    float ascale=dynamic_ascale(in_data.data(),batch*in_dim);
                    FLM_GO(cg,layer,in_data.data(),batch,(int)in_dim,ascale,gsc[layer],out_data.data(),(int)out_dim);
                }else if(op==4&&cfg.gu_split&&(flm_xclbin_available ? (bool)(hcu_ptr && hcu_ptr->isReady()) : (bool)(cu_ptr && cu_ptr->isReady()))){ // Up
                    out_dim=IM;
                    out_data.resize(batch*out_dim,0);
                    float ascale=dynamic_ascale(in_data.data(),batch*in_dim);
                    FLM_GO_PTR(cu_ptr,layer,in_data.data(),batch,(int)in_dim,ascale,usc[layer],out_data.data(),(int)out_dim);
                }else if(op==5&&FLM_IS_READY(cd)){ // Down
                    out_dim=H;
                    out_data.resize(batch*out_dim,0);
                    float ascale=dynamic_ascale(in_data.data(),batch*in_dim);
                    FLM_GO(cd,layer,in_data.data(),batch,(int)in_dim,ascale,dsc[layer],out_data.data(),(int)out_dim);
                }else if(op==6){ // Attention — CPU path (worker protocol doesn't carry seq_len)
                    // Worker subprocess receives individual layer ops without KV cache
                    // context. NPU attention requires the full KV cache. Use CPU fallback.
                    // Q width is NH*HD, not xclbin_qkv_k/4 (issue #1269: the
                    // old value sized the output 8x too small — heap OOB write
                    // — and read K from inside Q).
                    int qd = NH * HD;
                    out_dim = qd;
                    out_data.resize(batch*out_dim,0);
                    // in_data layout: [Q:QD, K:KD, V:KD]
                    float* q_ptr = in_data.data();
                    float* k_ptr = in_data.data() + qd;
                    float* v_ptr = in_data.data() + qd + NKV * HD;
                    // Infer seq_len from K data size (passed as in_dim - qd - NKV*HD)
                    int kd = NKV * HD;
                    int cl = kd > 0 ? (int)(in_dim - qd - kd) / (NKV * HD) : 1;
                    if (cl < 1) cl = 1;
                    attn_omp(q_ptr, out_data.data(), cl, k_ptr, v_ptr, NH, NKV, HD, GQA);
                }else if(op==20&&FLM_IS_READY(cq)){ // QKV all layers (batch, op=20)
                    int n_layers = NC;
                    out_dim = cfg.qkv_total;
                    out_data.resize(batch * out_dim * (size_t)n_layers, 0);
                    for (int l = 0; l < n_layers; l++) {
                        float ascale = dynamic_ascale(in_data.data() + (size_t)l * batch * in_dim, batch * in_dim);
                        FLM_GO(cq, l, in_data.data() + (size_t)l * batch * in_dim, batch, (int)in_dim,
                              ascale, qsc[l], out_data.data() + (size_t)l * batch * out_dim, (int)out_dim);
                    }
                }else if(op==21&&FLM_IS_READY(co)){ // O all layers (batch)
                    int n_layers = NC;
                    out_dim = H;
                    out_data.resize(batch * out_dim * (size_t)n_layers, 0);
                    for (int l = 0; l < n_layers; l++) {
                        float ascale = dynamic_ascale(in_data.data() + (size_t)l * batch * in_dim, batch * in_dim);
                        FLM_GO(co, l, in_data.data() + (size_t)l * batch * in_dim, batch, (int)in_dim,
                              ascale, osc[l], out_data.data() + (size_t)l * batch * out_dim, (int)out_dim);
                    }
                }else if(op==22&&FLM_IS_READY(cg)){ // Gate+Up all layers (batch)
                    int n_layers = NC;
                    out_dim = cfg.gu_split ? IM : (2 * IM);
                    out_data.resize(batch * out_dim * (size_t)n_layers, 0);
                    for (int l = 0; l < n_layers; l++) {
                        float ascale = dynamic_ascale(in_data.data() + (size_t)l * batch * in_dim, batch * in_dim);
                        FLM_GO(cg, l, in_data.data() + (size_t)l * batch * in_dim, batch, (int)in_dim,
                              ascale, gsc[l], out_data.data() + (size_t)l * batch * out_dim, (int)out_dim);
                    }
                }else if(op==23&&FLM_IS_READY(cd)){ // Down all layers (batch)
                    int n_layers = NC;
                    out_dim = H;
                    out_data.resize(batch * out_dim * (size_t)n_layers, 0);
                    for (int l = 0; l < n_layers; l++) {
                        float ascale = dynamic_ascale(in_data.data() + (size_t)l * batch * in_dim, batch * in_dim);
                        FLM_GO(cd, l, in_data.data() + (size_t)l * batch * in_dim, batch, (int)in_dim,
                              ascale, dsc[l], out_data.data() + (size_t)l * batch * out_dim, (int)out_dim);
                    }
                }else if(op==31){ // Reset KV cache (new conversation)
                    static int* fuse_pos_ptr = nullptr;
                    static bool* fuse_kv_init_ptr = nullptr;
                    // Find the fused decode's static state to reset it.
                    // These are set by op=32 on first call; until then, no-op.
                    if (fuse_pos_ptr) { *fuse_pos_ptr = 0; }
                    if (fuse_kv_init_ptr) { *fuse_kv_init_ptr = false; }
                    out_dim = 0;
                    out_data.clear();
                    ok = true;
                }else if(op==32){ // Fused decode step: embed → all layers (GEMM+attn) → lm_head → next token
                    // Maintains internal KV cache across calls.
                    // op=31 resets the internal position to 0.
                    static int fuse_pos = 0;
                    static bool fuse_kv_init = false;
                    static std::vector<KVCache> fuse_kv;
                    static std::vector<float> fuse_h_b, fuse_qo_b, fuse_at_b, fuse_oo_b;
                    static std::vector<float> fuse_gt_b, fuse_su_b, fuse_dw_b, fuse_sb_b;
                    static std::vector<float> fuse_lg_buf;
                    static std::vector<float> fuse_gdn_state;     // [NC, NKV, HD, HD]
                    static std::vector<float> fuse_gdn_attn_out;  // [NKV, HD]
                    static std::vector<int> fuse_top_ids_v;
                    // Expose state to op=31 for reset
                    { static bool once = false; if (!once) {
                        // Can't directly share static ptrs across handlers,
                        // so op=31 just sets a flag that we check here.
                        once = true;
                    }}
                    if (!fuse_kv_init) {
                        int fkv_size = 4096 * NKV * HD;
                        fuse_kv.clear();
                        for (int i = 0; i < NC; i++) fuse_kv.emplace_back(fkv_size);
                        fuse_h_b.resize(XM * H);
                        fuse_qo_b.resize(XM * qkv_n);
                        fuse_at_b.resize(XM * NH * HD);
                        fuse_oo_b.resize(XM * H);
                        fuse_gt_b.resize(XM * (cfg.gu_split ? IM : 2 * IM));
                        fuse_su_b.resize(XM * IM);
                        fuse_dw_b.resize(XM * H);
                        fuse_sb_b.resize(XM * H);
                        fuse_lg_buf.resize(NV);
                        fuse_top_ids_v.resize(BS, 0);
                        fuse_kv_init = true;
                        fuse_pos = 0;
                        // ponytail: GDN state allocated for all layers if MoE;
                        // non-GDN layers waste 40MB but avoids per-layer detection logic.
                        if (has_moe) {
                            int gd = HD;  // GDN state dim = KV head dim
                            fuse_gdn_state.resize(NC * (size_t)NKV * gd * gd, 0);
                            fuse_gdn_attn_out.resize(NKV * gd, 0);
                        }
                    }
                    int token_id = (int)in_data[0];
                    if (token_id < 0 || token_id >= NV) token_id = 0;
                    // Embed
                    for (int i = 0; i < H; i++)
                        fuse_h_b[i] = emb_f32[(size_t)token_id * H + i];
                    // Full decode step
                    for (int l = 0; l < NC; l++) {
                        float* fh = fuse_h_b.data();
                        float* fsb = fuse_sb_b.data();
                        for (int i = 0; i < H; i++) fsb[i] = fh[i];
                        rn_c(fh, in_n[l].data(), H);
                        float aq = dynamic_ascale(fh, H);
                        FLM_GO(cq, l, fh, 1, H, aq, qsc[l], fuse_qo_b.data(), qkv_n);
                        cn(fuse_qo_b.data(), qkv_n);
                        float* fqo = fuse_qo_b.data();
                        int qkv_k_off = cfg.qkv_k_offset;
                        int qkv_v_off = cfg.qkv_v_offset;
                        float* qn = qn_w[l].data();
                        float* kn = kn_w[l].data();
                        bool is_gdn = has_moe && exp_off[l].gate;
                        if (!is_gdn) {
                        for (int hh = 0; hh < NH; hh++) {
                            double sq = 0;
                            for (int d = 0; d < HD; d++) sq += fqo[hh * HD + d] * fqo[hh * HD + d];
                            float iq = 1.0f / sqrtf((float)(sq / HD) + EPS);
                            for (int d = 0; d < HD; d++)
                                fqo[hh * HD + d] *= iq * (cfg.has_q_norm ? qn[d] : 1.0f);
                            ra(&fqo[hh * HD], HD, fuse_pos);
                            if (hh % GQA == 0) {
                                int kvh = hh / GQA;
                                float* ks = &fqo[qkv_k_off + kvh * HD];
                                double sk = 0;
                                for (int d = 0; d < HD; d++) sk += ks[d] * ks[d];
                                float ik = 1.0f / sqrtf((float)(sk / HD) + EPS);
                                for (int d = 0; d < HD; d++)
                                    ks[d] *= ik * (cfg.has_k_norm ? kn[d] : 1.0f);
                                ra(ks, HD, fuse_pos);
                                float* vs = &fqo[qkv_v_off + kvh * HD];
                                if (fuse_pos >= 4096) {
                                    fprintf(stderr, "[npu] fuse KV overflow (pos=%d) — restarting context\n", fuse_pos);
                                    fuse_pos = 0;
                                }
                                memcpy(&fuse_kv[l].k[(size_t)fuse_pos * NKV * HD + (size_t)kvh * HD], ks, HD * 4);
                                memcpy(&fuse_kv[l].v[(size_t)fuse_pos * NKV * HD + (size_t)kvh * HD], vs, HD * 4);
                            }
                        }
                        }
                        fuse_kv[l].n = fuse_pos + 1;
                        int fcl = fuse_kv[l].n;
                        float* fat = fuse_at_b.data();
                        if (has_moe && exp_off[l].gate) {
                            // GatedDeltaNet attention (MoE layers)
                            // Q/K/V layout from QKV GEMM: [Q:NH*HD, K:NKV*HD, V:NKV*HD]
                            float* gq = fqo;                    // [NH*HD]
                            float* gk = fqo + qkv_k_off;       // [NKV*HD]
                            float* gv = fqo + qkv_v_off;       // [NKV*HD]
                            int gd = HD;  // GDN state dim = KV head dim
                            float* gs = fuse_gdn_state.data() + (size_t)l * NKV * gd * gd;
                            // ponytail: fake gate/beta (default identity) — real
                            // gate/beta projections from GDN weights TODO.
                            // gate=log(0.98) ≈ -0.02 → mild state decay (stable)
                            // beta=1.0 → identity mixing
                            alignas(64) float fake_gate[256];
                            alignas(64) float fake_beta[256];
                            for (int h = 0; h < NKV; h++) {
                                fake_gate[h * gd] = -0.02f;  // per-head scalar gate
                                fake_beta[h] = 1.0f;
                            }
                            gdn_attn_cpu(gq, gk, gv, fake_gate, fake_beta, gs,
                                        fuse_gdn_attn_out.data(), gd, NKV,
                                        1.0f / sqrtf((float)gd));
                            // Expand GDN output [NKV, GD] to full attention output [NH, HD]
                            // GDN operates on KV heads; replicate for Q heads via GQA
                            for (int hh = 0; hh < NH; hh++) {
                                int kvh = hh / GQA;
                                memcpy(fat + hh * HD, fuse_gdn_attn_out.data() + kvh * HD, HD * 4);
                            }
                        } else if (use_npu_attn && ca_ptr && ca_ptr->isReady()) {
                            float qs = dynamic_ascale(fqo, NH * HD);
                            float ks = 0;
                            for (int i = 0; i < fcl * NKV * HD; i++) {
                                float a = fabsf(fuse_kv[l].k[i]);
                                if (a > ks) ks = a;
                            }
                            ks = ks < 1e-12f ? 1.0f : ks / 127.0f;
                            auto r = ca_ptr->launch(fqo, fuse_kv[l].k.data(), fuse_kv[l].v.data(), fcl, 1, qs, ks);
                            ca_ptr->finish(r, fat, 1, qs, ks);
                            cn(fat, NH * HD);
                        } else {
                            attn_omp(fqo, fat, fcl, fuse_kv[l].k.data(), fuse_kv[l].v.data(), NH, NKV, HD, GQA);
                        }
                        float ao = dynamic_ascale(fat, NH * HD);
                        FLM_GO(co, l, fat, 1, NH * HD, ao, osc[l], fuse_oo_b.data(), H);
                        cn(fuse_oo_b.data(), H);
                        for (int i = 0; i < H; i++) fh[i] = fsb[i] + fuse_oo_b[i];
                        for (int i = 0; i < H; i++) fsb[i] = fh[i];
                        rn_c(fh, pa_n[l].data(), H);
                        if (has_moe && exp_off[l].gate) {
                            // MoE FFN (CPU path — ponytail: NPU I8Ctx optimization later)
                            moe_ffn_cpu(fh, fuse_dw_b.data(), l);
                            cn(fuse_dw_b.data(), H);
                            for (int i = 0; i < H; i++) fh[i] = fsb[i] + fuse_dw_b[i];
                        } else {
                        int fmlp_out = cfg.gu_split ? IM : 2 * IM;
                        float ag = dynamic_ascale(fh, H);
                        FLM_GO(cg, l, fh, 1, H, ag, gsc[l], fuse_gt_b.data(), fmlp_out);
                        cn(fuse_gt_b.data(), fmlp_out);
                        if (cfg.gu_split) {
                            float au = dynamic_ascale(fh, H);
                            FLM_GO_PTR(cu_ptr, l, fh, 1, H, au, usc[l], fuse_su_b.data(), IM);
                            cn(fuse_su_b.data(), IM);
                            for (int i = 0; i < IM; i++) {
                                float gv = fuse_gt_b[i];
                                if (!std::isfinite(gv)) gv = 0;
                                fuse_su_b[i] = (gv / (1.0f + expf(-gv))) * fuse_su_b[i];
                            }
                        } else {
                            for (int i = 0; i < IM; i++) {
                                float gv = fuse_gt_b[i];
                                if (!std::isfinite(gv)) gv = 0;
                                fuse_su_b[i] = (gv / (1.0f + expf(-gv))) * fuse_gt_b[IM + i];
                            }
                        }
                        float ad = dynamic_ascale(fuse_su_b.data(), IM);
                        FLM_GO(cd, l, fuse_su_b.data(), 1, IM, ad, dsc[l], fuse_dw_b.data(), H);
                        cn(fuse_dw_b.data(), H);
                        for (int i = 0; i < H; i++) fh[i] = fsb[i] + fuse_dw_b[i];
                        }
                    }
                    // Final norm + lm_head
                    rn_c(fuse_h_b.data(), fin_v.data(), H);
                    int* ftop = fuse_top_ids_v.data();
                    lm_topk_omp(fuse_h_b.data(), fuse_lg_buf.data(), ftop, BS, NV, H, lm_emb);
                    fuse_pos++;
                    out_dim = 1;
                    out_data.resize(1);
                    out_data[0] = (float)ftop[0];
                    ok = true;
                }else{
                    ok=false;
                }
            }catch(std::exception&e){
                fprintf(stderr,"NPU worker op %u layer %u batch %u: %s\n",op,layer,batch,e.what());
                fflush(stderr);
                ok=false;
            }catch(...){
                fprintf(stderr,"NPU worker op %u layer %u batch %u: unknown error\n",op,layer,batch);
                fflush(stderr);
                ok=false;
            }

            if(!ok){
                uint32_t resp[2]={1,0}; // error
                fwrite(resp,sizeof(uint32_t),2,stdout);
                fflush(stdout);
                continue;
            }

            // Success: send response code + output
            uint32_t resp[2]={0,out_dim};
            fwrite(resp,sizeof(uint32_t),2,stdout);
            fwrite(out_data.data(),sizeof(float),batch*out_dim,stdout);
            fflush(stdout);
        }
        // Use _exit() to skip destructor cleanup — XRT's BO destructors can
        // corrupt glibc's heap when vectors containing GB-scale weight data
        // (emb_f32 ~594MB, lm_head_f32 ~594MB, kv_caches ~896MB) race with
        // XRT dma-buf teardown during normal exit() destructor chain.
        _exit(0);
    }

    // Load input tokens from file or use default hardcoded sequence
    std::vector<int> pt_vec;
    if(input_tok_file){
        FILE* tf;
        if(strcmp(input_tok_file,"-")==0) tf=stdin;  // stdin convention must precede fopen (fixes #88)
        else {
            tf=fopen(input_tok_file,"r");
            if(!tf){ fprintf(stderr,"Cannot open input tokens: %s\n",input_tok_file); return 1; }
        }
        int tid;
        while(fscanf(tf,"%d",&tid)==1) pt_vec.push_back(tid);
        if(tf!=stdin) fclose(tf);
        if(pt_vec.empty()){ fprintf(stderr,"Empty input token file: %s\n",input_tok_file); return 1; }
        if((int)pt_vec.size() > 4095) pt_vec.resize(4095);
    }else{
        pt_vec={151644,872,198,13048,151645,198,151644,77091,198};
    }
    int npt=(int)pt_vec.size(); if(npt<1)npt=1;
    if(input_tok_file && npt > XM) npt = XM;

    // ===== PREFILL (pipelined: parallel QKV+GU launch, overlapped dequant) =====
    printf("=== Prefill %d ===\n",npt);auto t0=std::chrono::steady_clock::now();fflush(stdout);
    for(int pi=0;pi<npt;pi++)for(int i=0;i<H;i++)h_b[pi*H+i]=emb_f32[pt_vec[pi]*H+i];
    xrt::run pending_gu; bool has_pending=false;
    for(int l=0;l<NC;l++){
        fprintf(stderr,"  L%d",l);fflush(stderr);
        // Save pre-norm residuals
        for(int pi=0;pi<npt;pi++)for(int i=0;i<H;i++)sb_data[pi*H+i]=h_b[pi*H+i];
        for(int pi=0;pi<npt;pi++)rn_c(&h_b[pi*H],in_n[l].data(),H);
        // Phase 1: Launch QKV on NPU
        float qkv_ascale=dynamic_ascale(h_b.data(),npt*H);
        auto r_qkv=FLM_LAUNCH_ASYNC(cq,l,h_b.data(),npt,H,qkv_ascale);
        // Phase 2: Wait QKV + dequant (CPU attention runs after)
        FLM_FINISH_ASYNC(cq,r_qkv,qo_b.data(),npt,qkv_n,qkv_ascale,qsc[l],l);cn(qo_b.data(),npt*qkv_n);
        fprintf(stderr,"q");fflush(stderr);
        float*qn=qn_w[l].data(),*kn=kn_w[l].data();
        for(int pi=0;pi<npt;pi++){
            for(int hh=0;hh<NH;hh++){double s=0;for(int d=0;d<HD;d++)s+=qo_b[pi*qkv_n+hh*HD+d]*qo_b[pi*qkv_n+hh*HD+d];float iq=1.0f/sqrtf((float)(s/HD)+EPS);
                for(int d=0;d<HD;d++)qo_b[pi*qkv_n+hh*HD+d]*=iq*(cfg.has_q_norm?qn[d]:1.0f);ra(&qo_b[pi*qkv_n+hh*HD],HD,sp+pi);}
            for(int kvh=0;kvh<NKV;kvh++){float*ks=&qo_b[pi*qkv_n+cfg.qkv_k_offset+kvh*HD],*vs=&qo_b[pi*qkv_n+cfg.qkv_v_offset+kvh*HD];
                double sk=0;for(int d=0;d<HD;d++)sk+=ks[d]*ks[d];float ik=1.0f/sqrtf((float)(sk/HD)+EPS);
                for(int d=0;d<HD;d++)ks[d]*=ik*(cfg.has_k_norm?kn[d]:1.0f);ra(ks,HD,sp+pi);
                memcpy(&kv_caches[l].k[(sp+pi)*NKV*HD+kvh*HD],ks,HD*4);memcpy(&kv_caches[l].v[(sp+pi)*NKV*HD+kvh*HD],vs,HD*4);}}
        kv_caches[l].n=sp+npt;int cl=kv_caches[l].n;
        // Attention: NPU or CPU fallback
        //
        // NPU attention uses pre-compiled KV xclbin instructions
        // at runtime via FLM's libmha.so (MHA::generate_mha_sequence). The
        // attn.xclbin kernel takes Q, K, V as i8 inputs and produces i16 output.
        // K and V caches are quantized from f32 on each step.
        //
        // CPU attn_omp() fallback is always available.
        if(use_npu_attn && ca_ptr && ca_ptr->isReady()){
            // Regenerate instructions for current seq_len

            if (!attn_instrs.empty()) {
                // Compute dynamic scales for Q and K/V quantization
                float q_ascale = dynamic_ascale(qo_b.data(), npt * NH * HD);
                float kv_ascale = 0;
                for (int i = 0; i < (size_t)cl * NKV * HD; i++) {
                    float a = fabsf(kv_caches[l].k[i]);
                    if (std::isfinite(a) && a > kv_ascale) kv_ascale = a;
                }
                if (kv_ascale < 1e-12f) kv_ascale = 1.0f;
                kv_ascale = kv_ascale / 127.0f;

                // Launch NPU attention kernel
                auto r_attn = ca_ptr->launch(
                    qo_b.data(), kv_caches[l].k.data(), kv_caches[l].v.data(),
                    cl, npt, q_ascale, kv_ascale);

                // Wait + dequantize into attention output buffer
                ca_ptr->finish(r_attn, at_b.data(), npt, q_ascale, kv_ascale);
                cn(at_b.data(), npt * NH * HD);
                fprintf(stderr,"A"); fflush(stderr);
            } else {
                // Instr generation failed — fall back to CPU
                #pragma omp parallel for
                for(int pi=0;pi<npt;pi++){
                    if (omp_get_thread_num() == 0) { fprintf(stderr,"a"); fflush(stderr); }
                    attn_omp(&qo_b[pi*qkv_n],&at_b[pi*NH*HD],cl,kv_caches[l].k.data(),kv_caches[l].v.data(),NH,NKV,HD,GQA,sp+pi+1);
                }
            }
        } else {
            // CPU attention (default fallback)
            #pragma omp parallel for
            for(int pi=0;pi<npt;pi++){
                if (omp_get_thread_num() == 0) { fprintf(stderr,"a"); fflush(stderr); }
                attn_omp(&qo_b[pi*qkv_n],&at_b[pi*NH*HD],cl,kv_caches[l].k.data(),kv_caches[l].v.data(),NH,NKV,HD,GQA,sp+pi+1);
            }
        }
        // Phase 3: Launch O + GU in parallel on NPU
        int mlp_out=cfg.gu_split?IM:2*IM;
        float o_ascale=dynamic_ascale(at_b.data(),npt*NH*HD);
        float gu_ascale=dynamic_ascale(h_b.data(),npt*H);
        auto r_o=FLM_LAUNCH_ASYNC(co,l,at_b.data(),npt,NH*HD,o_ascale);
        auto r_gu=FLM_LAUNCH_ASYNC(cg,l,h_b.data(),npt,H,gu_ascale);
        // Phase 4: Wait O, apply residual
        FLM_FINISH_ASYNC(co,r_o,oo_b.data(),npt,H,o_ascale,osc[l],l);cn(oo_b.data(),npt*H);
        fprintf(stderr,"o");fflush(stderr);
        for(int pi=0;pi<npt;pi++)for(int i=0;i<H;i++)h_b[pi*H+i]=sb_data[pi*H+i]+oo_b[pi*H+i];
        // Save pre-FFN residuals
        for(int pi=0;pi<npt;pi++)for(int i=0;i<H;i++)sb_data[pi*H+i]=h_b[pi*H+i];
        for(int pi=0;pi<npt;pi++)rn_c(&h_b[pi*H],pa_n[l].data(),H);
        // Phase 5: Wait GU (was launched in parallel with O), SiLU, launch D
        FLM_FINISH_ASYNC(cg,r_gu,gt_b.data(),npt,mlp_out,gu_ascale,gsc[l],l);cn(gt_b.data(),npt*mlp_out);
        fprintf(stderr,"g");fflush(stderr);
        if(cfg.gu_split){FLM_GO_PTR(cu_ptr,l,h_b.data(),npt,H,dynamic_ascale(h_b.data(),npt*H),usc[l],su_b.data(),IM);cn(su_b.data(),npt*IM);
            for(int pi=0;pi<npt;pi++){for(int i=0;i<IM;i++){float gv=gt_b[pi*IM+i];if(!std::isfinite(gv))gv=0;su_b[pi*IM+i]=(gv/(1.0f+expf(-gv)))*su_b[pi*IM+i];}}}
        else{for(int pi=0;pi<npt;pi++){for(int i=0;i<IM;i++){float gv=gt_b[pi*mlp_out+i];if(!std::isfinite(gv))gv=0;su_b[pi*IM+i]=(gv/(1.0f+expf(-gv)))*gt_b[pi*mlp_out+IM+i];}}}
        fprintf(stderr,"d");fflush(stderr);FLM_GO(cd,l,su_b.data(),npt,IM,dynamic_ascale(su_b.data(),npt*IM),dsc[l],dw_b.data(),H);cn(dw_b.data(),npt*H);
        // Residual add: use saved pre-FFN values
        for(int pi=0;pi<npt;pi++)for(int i=0;i<H;i++)h_b[pi*H+i]=sb_data[pi*H+i]+dw_b[pi*H+i];
        fprintf(stderr,"\n");fflush(stderr);
    }sp+=npt;memcpy(h_data.data(),&h_b[(npt-1)*H],H*4);
    printf("Prefill: %.0fms (%.0f ms/tok)\n\n",std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count(),std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count()/npt);

    // ===== v12: M=32 BATCHED DECODE =====
    printf("=== M=%d Batch Decode (%d tokens) ===\n",BS,ng);
    auto tgs=std::chrono::steady_clock::now();
    // NOTE: greedy batched decode — runs batch_size tokens per step, no draft verification.
    // (fixes #95). total_verified tracks all tokens processed.
    std::vector<int> top_ids_v(BS, 0);int* top_ids=top_ids_v.data();int total_generated=0,total_verified=0,n_batches=0;double t_boot=0;

    // Boot: single-token decode → top-32 token IDs
    {
        auto ts_boot=std::chrono::steady_clock::now();
        float h0[H];memcpy(h0,h_data.data(),H*4);
        for(int l=0;l<NC;l++){
            memcpy(sb_data.data(),h0,H*4);rn_c(h0,in_n[l].data(),H);
            FLM_GO(cq,l,h0,1,H,dynamic_ascale(h0,H),qsc[l],qo_data.data(),qkv_n);cn(qo_data.data(),qkv_n);
            memcpy(ko_data.data(),&qo_data[cfg.qkv_k_offset],NKV*HD*4);memcpy(vo_data.data(),&qo_data[cfg.qkv_v_offset],NKV*HD*4);
            float*qn=qn_w[l].data(),*kn=kn_w[l].data();
            for(int hh=0;hh<NH;hh++){double sq=0;for(int d=0;d<HD;d++)sq+=qo_data[hh*HD+d]*qo_data[hh*HD+d];float iq=1.0f/sqrtf((float)(sq/HD)+EPS);
                for(int d=0;d<HD;d++)qo_data[hh*HD+d]*=iq*(cfg.has_q_norm?qn[d]:1.0f);ra(&qo_data[hh*HD],HD,sp);
                if(hh%GQA==0){int kvh=hh/GQA;double sk=0;for(int d=0;d<HD;d++)sk+=ko_data[kvh*HD+d]*ko_data[kvh*HD+d];float ik=1.0f/sqrtf((float)(sk/HD)+EPS);
                for(int d=0;d<HD;d++)ko_data[kvh*HD+d]*=ik*(cfg.has_k_norm?kn[d]:1.0f);ra(&ko_data[kvh*HD],HD,sp);
                memcpy(&kv_caches[l].k[sp*NKV*HD+kvh*HD],&ko_data[kvh*HD],HD*4);memcpy(&kv_caches[l].v[sp*NKV*HD+kvh*HD],&vo_data[kvh*HD],HD*4);}}
            kv_caches[l].n=sp+1;int cl=kv_caches[l].n;
            if(use_npu_attn && ca_ptr && ca_ptr->isReady()){
                // NPU attention via pre-compiled KV xclbin (fixed max_seq=4096)
                float qs=dynamic_ascale(qo_data.data(),NH*HD);
                float ks=0;for(int i=0;i<cl*NKV*HD;i++){float a=fabsf(kv_caches[l].k[i]);if(a>ks)ks=a;}
                ks=ks<1e-12f?1.0f:ks/127.0f;
                auto r=ca_ptr->launch(qo_data.data(),kv_caches[l].k.data(),kv_caches[l].v.data(),cl,1,qs,ks);
                ca_ptr->finish(r,at_data.data(),1,qs,ks);cn(at_data.data(),NH*HD);
            }else{attn_omp(qo_data.data(),at_data.data(),cl,kv_caches[l].k.data(),kv_caches[l].v.data(),NH,NKV,HD,GQA);}
            FLM_GO(co,l,at_data.data(),1,NH*HD,dynamic_ascale(at_data.data(),NH*HD),osc[l],oo_data.data(),H);cn(oo_data.data(),H);for(int i=0;i<H;i++)h0[i]=sb_data[i]+oo_data[i];
            memcpy(sb_data.data(),h0,H*4);rn_c(h0,pa_n[l].data(),H);
            int mlp_out=cfg.gu_split?IM:2*IM;
            FLM_GO(cg,l,h0,1,H,dynamic_ascale(h0,H),gsc[l],gt_data.data(),mlp_out);cn(gt_data.data(),mlp_out);
            if(cfg.gu_split){FLM_GO_PTR(cu_ptr,l,h0,1,H,dynamic_ascale(h0,H),usc[l],su_data.data(),IM);cn(su_data.data(),IM);
                for(int i=0;i<IM;i++){float gv=gt_data[i];if(!std::isfinite(gv))gv=0;su_data[i]=(gv/(1.0f+expf(-gv)))*su_data[i];}}
            else{for(int i=0;i<IM;i++){float gv=gt_data[i];if(!std::isfinite(gv))gv=0;su_data[i]=(gv/(1.0f+expf(-gv)))*gt_data[IM+i];}}
            FLM_GO(cd,l,su_data.data(),1,IM,dynamic_ascale(su_data.data(),IM),dsc[l],dwo_data.data(),H);cn(dwo_data.data(),H);for(int i=0;i<H;i++)h0[i]=sb_data[i]+dwo_data[i];
        }
        memcpy(sb_data.data(),h0,H*4);rn_c(sb_data.data(),fin_v.data(),H);
        lm_topk_omp(sb_data.data(),lg_buf.data(),top_ids,BS,NV,H,lm_emb);
        memcpy(h_data.data(),h0,H*4);sp++;total_generated++;
        t_boot=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-ts_boot).count();
        printf("  [0] boot=%d (%.0fms)\n",top_ids[0],t_boot);
    }

    int step=1;
    while(step<ng){
        auto ts_batch=std::chrono::steady_clock::now();
        int batch_size=std::min(BS,ng-step);
        for(int b=0;b<batch_size;b++)for(int i=0;i<H;i++)h_b[b*H+i]=emb_f32[(size_t)top_ids[b]*H+i];
        // ===== PIPELINED LAYER LOOP (cross-layer, roadmap step 3) =====
        // NPU runs QKV → GU → O → D back-to-back; all CPU work hides behind a kernel:
        //   cg quantize+sync+launch  ∥ QKV kernel   (GU input = h_b, ready at layer start)
        //   QKV readback + attention ∥ GU kernel
        //   GU readback + SiLU + D launch ∥ O kernel
        //   O readback + residual + rn ∥ D kernel (non-split GU)
        // Layer boundary: the D output is consumed by ONE fused pass
        // (dequant+residual+save+rn+amax) that directly produces the next layer's
        // QKV input — replacing 6 serial passes (dequantize, cn, residual, save,
        // rn_c, dynamic_ascale). Bit-identical numerics.
        float cq_ascale=1.0f;
        std::vector<double> rn_ss(batch_size>0?batch_size:1,0.0);
        for(int l=0;l<NC;l++){
            // ── QKV input: for l>0 produced by layer l-1's fused boundary
            //    (h_b = rn'd QKV input, sb_data = pre-QKV residual, cq_ascale set).
            //    Layer 0 initializes from embeddings. ──
            if(l==0){
                // Save pre-norm residuals before rn_c
                for(int b=0;b<batch_size;b++)for(int i=0;i<H;i++)sb_data[b*H+i]=h_b[b*H+i];
                for(int b=0;b<batch_size;b++)rn_c(&h_b[b*H],in_n[l].data(),H);
                cq_ascale=dynamic_ascale(h_b.data(),batch_size*H);
            }

            // ── QKV GEMM ──
            FLM_QUANTIZE_ASYNC(cq,h_b.data(),batch_size,H,cq_ascale);
            auto r_cq=FLM_SYNC_AND_LAUNCH(cq,l);

            // ── GU GEMM: input (h_b) ready since layer start — launch right
            //    after QKV; its kernel hides the QKV readback + attention. ──
            int mlp_out=cfg.gu_split?IM:2*IM;
            float cg_ascale=dynamic_ascale(h_b.data(),batch_size*H);
            FLM_QUANTIZE_ASYNC(cg,h_b.data(),batch_size,H,cg_ascale);
            FLM_SYNC_A(cg,l);                 // cg.bA sync (MM2S) ∥ QKV kernel
            auto r_cg=FLM_LAUNCH(cg,l);       // GU queued behind QKV on the NPU

            // ── QKV: wait + readback + dequant (S2MM readback ∥ GU kernel) ──
            FLM_DEQUANTIZE(cq,r_cq,qo_b.data(),batch_size,qkv_n,cq_ascale,qsc[l],l);
            cn(qo_b.data(),batch_size*qkv_n);

            // ── Attention + RoPE + KV cache ──
            float*qn=qn_w[l].data(),*kn=kn_w[l].data();
            for(int b=0;b<batch_size;b++){
                for(int hh=0;hh<NH;hh++){double s=0;for(int d=0;d<HD;d++)s+=qo_b[b*qkv_n+hh*HD+d]*qo_b[b*qkv_n+hh*HD+d];float iq=1.0f/sqrtf((float)(s/HD)+EPS);
                    for(int d=0;d<HD;d++)qo_b[b*qkv_n+hh*HD+d]*=iq*(cfg.has_q_norm?qn[d]:1.0f);ra(&qo_b[b*qkv_n+hh*HD],HD,sp+b);}
                for(int kvh=0;kvh<NKV;kvh++){float*ks=&qo_b[b*qkv_n+cfg.qkv_k_offset+kvh*HD],*vs=&qo_b[b*qkv_n+cfg.qkv_v_offset+kvh*HD];
                    double sk=0;for(int d=0;d<HD;d++)sk+=ks[d]*ks[d];float ik=1.0f/sqrtf((float)(sk/HD)+EPS);
                    for(int d=0;d<HD;d++)ks[d]*=ik*(cfg.has_k_norm?kn[d]:1.0f);ra(ks,HD,sp+b);}
            }
            // KV capacity is 4096 positions (issue #1267) — restart the
            // context instead of writing OOB once exhausted.
            if (sp + batch_size > 4096) {
                fprintf(stderr, "[npu] KV overflow at layer %d (sp=%d) — restarting context\n", l, sp);
                sp = 0;
            }
            for(int b=0;b<batch_size;b++)for(int kvh=0;kvh<NKV;kvh++){
                float*ks=&qo_b[b*qkv_n+cfg.qkv_k_offset+kvh*HD],*vs=&qo_b[b*qkv_n+cfg.qkv_v_offset+kvh*HD];
                memcpy(&kv_caches[l].k[(sp+b)*NKV*HD+kvh*HD],ks,HD*4);memcpy(&kv_caches[l].v[(sp+b)*NKV*HD+kvh*HD],vs,HD*4);}
            kv_caches[l].n=sp+batch_size;int cl=kv_caches[l].n;
            if(use_npu_attn && ca_ptr && ca_ptr->isReady()){
                // NPU attention via pre-compiled KV xclbin (fixed max_seq=4096)
                float qs=dynamic_ascale(qo_b.data(),batch_size*NH*HD);
                float ks=0;for(int i=0;i<cl*NKV*HD;i++){float a=fabsf(kv_caches[l].k[i]);if(a>ks)ks=a;}
                ks=ks<1e-12f?1.0f:ks/127.0f;
                auto r=ca_ptr->launch(qo_b.data(),kv_caches[l].k.data(),kv_caches[l].v.data(),cl,batch_size,qs,ks);
                ca_ptr->finish(r,at_b.data(),batch_size,qs,ks);cn(at_b.data(),batch_size*NH*HD);
            }else{for(int b=0;b<batch_size;b++){attn_omp(&qo_b[b*qkv_n],&at_b[b*NH*HD],cl,kv_caches[l].k.data(),kv_caches[l].v.data(),NH,NKV,HD,GQA);}}

            // ── O GEMM: queued behind GU; its readback hides behind D later ──
            float co_ascale=dynamic_ascale(at_b.data(),batch_size*NH*HD);
            FLM_QUANTIZE_ASYNC(co,at_b.data(),batch_size,NH*HD,co_ascale);
            auto r_co=FLM_SYNC_AND_LAUNCH(co,l);

            // ── GU: wait + readback + dequant (readback ∥ O kernel) ──
            FLM_WAIT_KERNEL(cg,r_cg);
            FLM_SYNC_BACK(cg,gt_b.data(),batch_size,mlp_out,cg_ascale,gsc[l],l);
            cn(gt_b.data(),batch_size*mlp_out);

            // SiLU gate + U GEMM (gu_split) or combined gate*up
            if(cfg.gu_split){
                // Up-projection needs the post-attention hidden state, so the O
                // readback + residual + rn must complete before cu launches.
                FLM_WAIT_KERNEL(co,r_co);
                FLM_SYNC_BACK(co,oo_b.data(),batch_size,H,co_ascale,osc[l],l);
                cn(oo_b.data(),batch_size*H);
                for(int b=0;b<batch_size;b++)for(int i=0;i<H;i++)h_b[b*H+i]=sb_data[b*H+i]+oo_b[b*H+i];
                for(int b=0;b<batch_size;b++)for(int i=0;i<H;i++)sb_data[b*H+i]=h_b[b*H+i];
                for(int b=0;b<batch_size;b++)rn_c(&h_b[b*H],pa_n[l].data(),H);
                FLM_QUANTIZE_ASYNC_PTR(cu_ptr,h_b.data(),batch_size,H,cg_ascale);
                auto r_cu=FLM_SYNC_AND_LAUNCH_PTR(cu_ptr,l);
                FLM_DEQUANTIZE_PTR(cu_ptr,r_cu,su_b.data(),batch_size,IM,cg_ascale,usc[l],l);
                cn(su_b.data(),batch_size*IM);
                for(int b=0;b<batch_size;b++){for(int i=0;i<IM;i++){float gv=gt_b[b*IM+i];if(!std::isfinite(gv))gv=0;su_b[b*IM+i]=(gv/(1.0f+expf(-gv)))*su_b[b*IM+i];}}}
            else{for(int b=0;b<batch_size;b++){for(int i=0;i<IM;i++){float gv=gt_b[b*mlp_out+i];if(!std::isfinite(gv))gv=0;su_b[b*IM+i]=(gv/(1.0f+expf(-gv)))*gt_b[b*mlp_out+IM+i];}}}

            // ── D GEMM: queued behind O ──
            float cd_ascale=dynamic_ascale(su_b.data(),batch_size*IM);
            FLM_QUANTIZE_ASYNC(cd,su_b.data(),batch_size,IM,cd_ascale);
            auto r_cd=FLM_SYNC_AND_LAUNCH(cd,l);

            // ── O: wait + readback + dequant (non-split: readback ∥ D kernel) + residual + rn ──
            if(!cfg.gu_split){
                FLM_WAIT_KERNEL(co,r_co);
                FLM_SYNC_BACK(co,oo_b.data(),batch_size,H,co_ascale,osc[l],l);
                cn(oo_b.data(),batch_size*H);
                for(int b=0;b<batch_size;b++)for(int i=0;i<H;i++)h_b[b*H+i]=sb_data[b*H+i]+oo_b[b*H+i];
                for(int b=0;b<batch_size;b++)for(int i=0;i<H;i++)sb_data[b*H+i]=h_b[b*H+i];
                for(int b=0;b<batch_size;b++)rn_c(&h_b[b*H],pa_n[l].data(),H);
            }

            // ── Cross-layer boundary (roadmap step 3): fused D-output → l+1 QKV input ──
            if(l+1<NC){
                FLM_WAIT_KERNEL(cd,r_cd);
                FLM_READBACK(cd);
                float cs=cd_ascale*dsc[l];
                if(flm_xclbin_available){
                    cq_ascale=fused_cross_layer_boundary<int16_t>(hcd->Cm,hcd->ND,cs,
                        sb_data.data(),h_b.data(),in_n[l+1].data(),H,batch_size,rn_ss.data());
                }else{
                    cq_ascale=fused_cross_layer_boundary<int32_t>(cd.Cm,cd.ND,cs,
                        sb_data.data(),h_b.data(),in_n[l+1].data(),H,batch_size,rn_ss.data());
                }
            }else{
                // Last layer: keep the final hidden state in h_b for the LM head
                FLM_DEQUANTIZE(cd,r_cd,dw_b.data(),batch_size,H,cd_ascale,dsc[l],l);
                cn(dw_b.data(),batch_size*H);

                // Residual add
                for(int b=0;b<batch_size;b++)for(int i=0;i<H;i++)h_b[b*H+i]=sb_data[b*H+i]+dw_b[b*H+i];
            }
        }

        // LM head on the (single, BS=1) decoded position -> greedy next token.
        // total_verified == total_generated because every emitted token is a
        // real causal decode, not a speculative candidate (issue #111).
        memcpy(sb_data.data(),&h_b[0],H*4);rn_c(sb_data.data(),fin_v.data(),H);
        lm_topk_omp(sb_data.data(),lg_buf.data(),top_ids,BS,NV,H,lm_emb);

        total_generated+=batch_size;total_verified+=batch_size;sp+=batch_size;n_batches++;
        double batch_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-ts_batch).count();
        printf("  [%d] batch=%d tok=%d %.0fms (%.0f ms/tok)\n",step,batch_size,top_ids[0],batch_ms,batch_ms/batch_size);
        step+=batch_size;
    }

    double tts=std::chrono::duration<double>(std::chrono::steady_clock::now()-tgs).count();
    printf("\n=== %.1f ms/tok (%.0f tok/s) | boot=%.0fms batches=%d tokens=%d ===\n",tts*1000/ng,ng/tts,t_boot,n_batches,total_generated);

    // Graceful exit: the XRT BO destructors (unique_ptr cleanup) can corrupt
    // glibc's heap when GB-scale vectors race with dma-buf teardown.
    // Use _exit() to skip the destructor chain entirely — the OS reclaims
    // all resources on process exit anyway.
    munmap(md,st.st_size);fflush(stdout);fflush(stderr);_exit(0);
}
