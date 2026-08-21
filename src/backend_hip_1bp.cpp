// backend_hip_1bp.cpp — Fast GPU inference for 1BP models.
// Custom GEMV kernel replaces rocBLAS. All ops on-device.

#include "backend.h"
#include "gguf_reader.h"
#include "../engine/npu/src/onebp_loader.cpp"
#include "rocm_cpp/ck_gemm.h"
#include <hip/hip_runtime_api.h>
#include <hip/hip_fp16.h>
#include <cstdio>
#include <cmath>
#include <vector>
#include <chrono>
#include <memory>

#include "hip_1bp_kernels.hip"

extern "C" rcpp_status_t rcpp_kv_cache_attn_decode_dpos(
    const void* Q_dev, const void* K_dev, const void* V_dev, void* out_dev,
    int num_q_heads, int num_kv_heads, int head_dim,
    const int* seq_len_dev, float scale, void* stream);

static constexpr float EPS = 1e-6f;

#define HIP_CHECK(call) \
    do { hipError_t _hip_e = (call); \
         if (_hip_e != hipSuccess) { \
             fprintf(stderr, "HIP error %s at %s:%d\n", \
                     hipGetErrorString(_hip_e), __FILE__, __LINE__); \
             std::abort(); } } while(0)
#define HIP_CHECK_D(call) \
    do { hipError_t _hip_e = (call); \
         if (_hip_e != hipSuccess) { \
             fprintf(stderr, "HIP error (dtor) %s at %s:%d\n", \
                     hipGetErrorString(_hip_e), __FILE__, __LINE__); } } while(0)

struct Hip1bpBackend : Backend {
    int H=0,NC=0,NH=0,NKV=0,HD=128,IM=0,VOCAB=0;
    float rope_theta=10000.0f; int max_seq=4096;
    hipStream_t stream=nullptr; bool gpu_ok=false;

    // GPU weights
    float *d_embed=nullptr,*d_final_norm=nullptr,*d_output=nullptr;
    struct GL{float*wq,*wk,*wv,*wo,*w1,*w2,*w3,*pn,*pon,*q_norm,*k_norm; float* bq,*bk,*bv;};
    std::vector<GL> L;

    // Packed 2-bit fast path (TQ2NZ / TQ2NZ_E4M3): tile pointers into the
    // mmap'd 1bp file (kept alive by model_), zero CPU dequant at init.
    struct PL { const uint8_t *pq,*pk,*pv,*po,*p1,*p2,*p3; };
    struct PLD { uint8_t *pq,*pk,*pv,*po,*p1,*p2,*p3; };   // device copies
    std::vector<PLD> PD;
    uint8_t* d_output_packed = nullptr;
    std::vector<PL> P;
    int quant2 = 0;               // 0 = f32 path, 1 = TQ2NZ bf16, 2 = TQ2NZ_E4M3, 3 = Q4NX packed
    std::unique_ptr<NpuOnebpModel> model_;
    std::unique_ptr<GgufReader> gguf_;  // GGUF-direct mode (lossless f32, no 1BP conversion)

    // GPU scratch (persistent, device-only)
    float *dh=nullptr,*datt=nullptr,*dgate=nullptr,*dup=nullptr;
    float *dsilu=nullptr,*doproj=nullptr,*dffn=nullptr,*datt2=nullptr;
    float *dlogits=nullptr; // [VOCAB] — pre-allocated lm_head output
    float* dpart = nullptr;  // packed-gemv partials scratch

    // KV cache (__half device)
    __half *dK=nullptr,*dV=nullptr,*dQ=nullptr,*dAttn=nullptr;
    size_t kvb=0; int pos=0;
    std::vector<float> cpu_final_norm;

    // Phase 1: device argmax scratch (argmax kernel + 4-byte result)
    static constexpr int AMX_MAXB = 512;
    int* d_argmax = nullptr;
    float* d_amx = nullptr;   // [AMX_MAXB]
    int* d_ami = nullptr;     // [AMX_MAXB]

    // Phase 2: hipGraph capture — per-token step captured once, replayed.
    // d_pos/d_token are read by kernels at replay; h_* are pinned host sides.
    int* d_pos = nullptr;   // device pos (written by kernels' *d_pos reads)
    int* d_token = nullptr; // device token id
    int* h_token = nullptr; // pinned host mirrors (updated per replay)
    int* h_pos = nullptr;
    int* h_res = nullptr;
    hipGraph_t graph = nullptr;
    hipGraphExec_t graphExec = nullptr;
    bool graph_ok = false;

    Hip1bpBackend(){type=BackendType::HIP_GPU;name="HIP 1BP GPU";}
    ~Hip1bpBackend()override{destroy();}
    bool can_infer()const override{return true;}

    bool init(const ModelConfig& cfg,const std::string&) override {
        this->cfg=cfg; H=cfg.hidden_size; NC=cfg.num_layers;
        NH=cfg.num_heads; NKV=cfg.num_kv_heads; HD=cfg.head_dim;
        IM=cfg.intermediate_size; VOCAB=cfg.vocab_size;
        rope_theta=cfg.rope_theta>0?cfg.rope_theta:10000.0f;
        { const char* rh=getenv("H1BP_ROPE"); if (rh) rope_theta=(float)atof(rh); }
        if(NKV==0)NKV=NH; if(HD==0)HD=128;
        printf("[hip1bp] H=%d NC=%d NH=%d NKV=%d HD=%d IM=%d V=%d\n",H,NC,NH,NKV,HD,IM,VOCAB);

        int nd=0;
        if(hipGetDeviceCount(&nd)!=hipSuccess||nd==0)return false;
        HIP_CHECK(hipSetDevice(0));
        HIP_CHECK(hipStreamCreate(&stream));

        // Device-only allocations
        // Per-layer KV cache: each layer needs its own max_seq*NKV*HD slots.
        // A shared cache lets layer NC-1 clobber layer 0's keys, and every
        // layer's attention then reads the wrong layer's K/V (fused backend
        // had the same bug; generic CPU keeps k_cache[il][...] per layer).
        kvb=(size_t)NC*max_seq*NKV*HD*sizeof(__half);
        HIP_CHECK(hipMalloc(&dK,kvb));  HIP_CHECK(hipMemset(dK,0,kvb));
        HIP_CHECK(hipMalloc(&dV,kvb));  HIP_CHECK(hipMemset(dV,0,kvb));
        size_t qb=(size_t)NH*HD*sizeof(__half);
        HIP_CHECK(hipMalloc(&dQ,qb));   HIP_CHECK(hipMalloc(&dAttn,qb));
        HIP_CHECK(hipMalloc(&dh,H*4));  HIP_CHECK(hipMalloc(&datt,NH*HD*4));
        HIP_CHECK(hipMalloc(&datt2,NH*HD*4));
        // FFN gate/up buffers must hold IM elements (not just NKV*HD) — the
        // gate/up GEMVs write IM=3072 floats.  Sizing them NKV*HD (1024) was
        // an out-of-bounds write every layer (HW exception on TheRock).
        size_t gb=(size_t)std::max(NKV*HD, IM)*4;
        HIP_CHECK(hipMalloc(&dgate,gb)); HIP_CHECK(hipMalloc(&dup,gb));
        HIP_CHECK(hipMalloc(&dsilu,IM*4)); HIP_CHECK(hipMalloc(&doproj,H*4));
        HIP_CHECK(hipMalloc(&dffn,H*4));
        HIP_CHECK(hipMalloc(&dlogits,VOCAB*4));
        // dpart = packed-gemv partials scratch (32 lanes * wpr per row). The old
        // max(IM,H)*96*4 assumed wpr<=3 and N<=max(IM,H) — the LM head launch
        // (N=VOCAB, K=H) blew past it on real vocabs: illegal memory access on
        // RX 9070 XT (gfx1201) with Qwen2.5-0.5B (V=151936). Worst case across
        // every launch_q4nx/launch_tq2nz call: layer w1/w2 N=IM K=H, w3 N=H
        // K=IM, lm_head N=VOCAB K=H.
        int h_ntc=(H+255)/256, im_ntc=(IM+255)/256;
        int h_wpr=(h_ntc+3)>>2, im_wpr=(im_ntc+3)>>2;
        size_t dpart_rows=(size_t)std::max({IM*h_wpr, H*im_wpr, VOCAB*h_wpr});
        HIP_CHECK(hipMalloc(&dpart, dpart_rows*32*4));  // 32 lanes/row * f32
        HIP_CHECK(hipMalloc(&d_argmax,4));
        HIP_CHECK(hipMalloc(&d_amx,AMX_MAXB*4));
        HIP_CHECK(hipMalloc(&d_ami,AMX_MAXB*4));
        HIP_CHECK(hipMalloc(&d_pos,4));
        HIP_CHECK(hipMalloc(&d_token,4));
        HIP_CHECK(hipHostMalloc(&h_token,sizeof(int),hipHostMallocMapped));
        HIP_CHECK(hipHostMalloc(&h_pos,sizeof(int),hipHostMallocMapped));
        HIP_CHECK(hipHostMalloc(&h_res,sizeof(int),hipHostMallocMapped));

        if(cfg.model_path.empty())return false;
        printf("[hip1bp] Loading: %s\n",cfg.model_path.c_str());
        // GGUF-direct: skip the 1BP conversion entirely — GgufReader gives
        // the exact dequantized weights (j12 Q5_0, matching llama.cpp), so
        // there is no Q4NX/F16 repack to lose precision. F32 path only.
        bool is_gguf = cfg.model_path.size() > 5 &&
                       cfg.model_path.substr(cfg.model_path.size()-5) == ".gguf";
        if (is_gguf) {
            gguf_ = std::make_unique<GgufReader>();
            if(!gguf_->open(cfg.model_path.c_str()))return false;
            quant2 = 0;
        } else {
            if(cfg.format!=ModelFormat::ONEBP)return false;
            model_ = std::make_unique<NpuOnebpModel>();
            if(!model_->open(cfg.model_path.c_str()))return false;
        }
        auto get_w=[&](const char* n,std::vector<float>& v)->bool{
            if(gguf_) return gguf_->get_tensor_f32(n,v);
            return model_->get_tensor_f32(n,v);
        };
        uint32_t q = gguf_ ? (uint32_t)0xFFFFFFFFu : model_->header().quant;
        // #1627: only quants the loader dequantizes (dequant_tile/dequant_tile_tq2)
        // or the packed path (TQ2NZ family) are supported here. TQ1/TQ2BS/I8/F16/F32
        // fall through to the Q4NX-layout dequant -> garbage weights -> NaN logits
        // -> argmax -1 with zero diagnostics. Reject loudly at init instead.
        if (q != ONEBP_Q4NX && q != ONEBP_TQ2 && q != ONEBP_TQ2NZ && q != ONEBP_TQ2NZ_E4M3 &&
            q != ONEBP_F16 && q != ONEBP_F32 && q != 0xFFFFFFFFu) {
            fprintf(stderr, "[hip1bp] unsupported quant %u for GPU backend (Q4NX/TQ2/TQ2NZ/TQ2NZ_E4M3/F16/F32). "
                    "TQ1/TQ2BS/I8 models must be converted first (see gguf_to_onebp --tq2nz).\n", q);
            return false;
        }
        if (q == ONEBP_TQ2NZ) quant2 = 1;
        else if (q == ONEBP_TQ2NZ_E4M3) quant2 = 2;
        else if (q == ONEBP_Q4NX) quant2 = 3;   // #1625: packed Q4NX GEMV
        // F16/F32 stay on the f32 path (quant2 = 0): the packed GEMV kernels
        // are 4-bit-only, and lossless weights must not be re-quantized.
        if (getenv("H1BP_F32")) quant2 = 0;   // debug: force f32 path

        std::vector<float> emb,fn,out;
        auto ld=[&](const char* n,std::vector<float>& v){return get_w(n,v);};
        if(!ld("token_embd.weight",emb))return false;
        if(!ld("output_norm.weight",fn))ld("token_embd_norm.weight",fn);
        if(!ld("output.weight",out))ld("lm_head.weight",out);
        cpu_final_norm=fn;

        auto up=[&](std::vector<float>& c,float*& g){
            if(c.empty()){g=nullptr;return true;}
            if(hipMalloc(&g,c.size()*4)!=hipSuccess)return false;
            HIP_CHECK(hipMemcpy(g,c.data(),c.size()*4,hipMemcpyHostToDevice));
            c.clear();c.shrink_to_fit();return true;
        };
        up(emb,d_embed);up(fn,d_final_norm);up(out,d_output);

        L.resize(NC);char buf[128];
        for(int l=0;l<NC;l++){
            auto& ll=L[l];std::vector<float> w;
            auto gr=[&](const char* bk,const char* lg,float*& gp,int n){
                snprintf(buf,sizeof(buf),"blk.%d.%s",l,bk);w.clear();
                if(!get_w(buf,w)){
                    snprintf(buf,sizeof(buf),"model.layers.%d.%s",l,lg);
                    get_w(buf,w);
                }
                if((int)w.size()==n){HIP_CHECK(hipMalloc(&gp,n*4));HIP_CHECK(hipMemcpy(gp,w.data(),n*4,hipMemcpyHostToDevice));}
                else gp=nullptr;
            };
            gr("attn_q.weight","self_attn.q_proj.weight",ll.wq,H*NH*HD);
            gr("attn_k.weight","self_attn.k_proj.weight",ll.wk,H*NKV*HD);
            gr("attn_v.weight","self_attn.v_proj.weight",ll.wv,H*NKV*HD);
            gr("attn_output.weight","self_attn.o_proj.weight",ll.wo,NH*HD*H);
            gr("ffn_gate.weight","mlp.gate_proj.weight",ll.w1,H*IM);
            gr("ffn_up.weight","mlp.up_proj.weight",ll.w2,H*IM);
            gr("ffn_down.weight","mlp.down_proj.weight",ll.w3,IM*H);
            gr("attn_norm.weight","input_layernorm.weight",ll.pn,H);
            gr("ffn_norm.weight","post_attention_layernorm.weight",ll.pon,H);
            // Q/K/V biases (present in some Qwen2 GGUFs — the graph adds
            // them after the projections, before rope; without them the
            // Q/K gemv outputs are off by up to ±130 → garbage attention).
            gr("attn_q.bias","self_attn.q_proj.bias",ll.bq,NH*HD);
            gr("attn_k.bias","self_attn.k_proj.bias",ll.bk,NKV*HD);
            gr("attn_v.bias","self_attn.v_proj.bias",ll.bv,NKV*HD);
            // Per-head QK-norm (Qwen3/Qwen2.5+): RMSNorm on each head's
            // head_dim slice with a shared [head_dim] weight, before RoPE.
            gr("attn_q_norm.weight","self_attn.q_norm.weight",ll.q_norm,HD);
            gr("attn_k_norm.weight","self_attn.k_norm.weight",ll.k_norm,HD);
        }
        // Packed fast-path pointers (TQ2NZ family only)
        P.resize(NC);
        if (quant2) {
            NpuOnebpModel& mdl=*model_;
            for(int l=0;l<NC;l++){
                auto& pp=P[l];
                auto gt=[&](const char* bk,const char* lg,const uint8_t*& gp){
                    snprintf(buf,sizeof(buf),"blk.%d.%s",l,bk);
                    gp = mdl.get_tile_ptr(buf,0,0);
                    if(!gp){ snprintf(buf,sizeof(buf),"model.layers.%d.%s",l,lg); gp=mdl.get_tile_ptr(buf,0,0); }
                };
                gt("attn_q.weight","self_attn.q_proj.weight",pp.pq);
                gt("attn_k.weight","self_attn.k_proj.weight",pp.pk);
                gt("attn_v.weight","self_attn.v_proj.weight",pp.pv);
                gt("attn_output.weight","self_attn.o_proj.weight",pp.po);
                gt("ffn_gate.weight","mlp.gate_proj.weight",pp.p1);
                gt("ffn_up.weight","mlp.up_proj.weight",pp.p2);
                gt("ffn_down.weight","mlp.down_proj.weight",pp.p3);
            }
            // lm_head packed path only when it shares the TQ2NZ-family quant
            auto* out_t = mdl.find_tensor("output.weight");
            if (!out_t) out_t = mdl.find_tensor("lm_head.weight");
            if (out_t && (out_t->quant == ONEBP_TQ2NZ || out_t->quant == ONEBP_TQ2NZ_E4M3 || out_t->quant == ONEBP_Q4NX))
                d_output_packed = (uint8_t*)mdl.get_tile_ptr(out_t->name.c_str(),0,0);
            // Device copies of the packed tiles (kernel cannot read the mmap)
            PD.resize(NC);
            for (int l = 0; l < NC; l++) {
                auto& pp = P[l]; auto& pd = PD[l];
                auto cp=[&](const uint8_t* srcp, uint8_t*& dst, const char* nm){
                    auto* te = mdl.find_tensor(nm);
                    if (!te || !srcp) { dst = nullptr; return; }
                    HIP_CHECK(hipMalloc((void**)&dst, te->total_bytes));
                    HIP_CHECK(hipMemcpy(dst, srcp, te->total_bytes, hipMemcpyHostToDevice));
                };
                char nm[128];
                snprintf(nm, sizeof(nm), "blk.%d.attn_q.weight", l);   cp(pp.pq, pd.pq, nm);
                snprintf(nm, sizeof(nm), "blk.%d.attn_k.weight", l);   cp(pp.pk, pd.pk, nm);
                snprintf(nm, sizeof(nm), "blk.%d.attn_v.weight", l);   cp(pp.pv, pd.pv, nm);
                snprintf(nm, sizeof(nm), "blk.%d.attn_output.weight", l); cp(pp.po, pd.po, nm);
                snprintf(nm, sizeof(nm), "blk.%d.ffn_gate.weight", l); cp(pp.p1, pd.p1, nm);
                snprintf(nm, sizeof(nm), "blk.%d.ffn_up.weight", l);   cp(pp.p2, pd.p2, nm);
                snprintf(nm, sizeof(nm), "blk.%d.ffn_down.weight", l); cp(pp.p3, pd.p3, nm);
            }
            if (d_output_packed) {
                auto* te = mdl.find_tensor("output.weight");
                if (!te) te = mdl.find_tensor("lm_head.weight");
                uint8_t* dc = nullptr;
                if (te) { HIP_CHECK(hipMalloc((void**)&dc, te->total_bytes));
                          HIP_CHECK(hipMemcpy(dc, d_output_packed, te->total_bytes, hipMemcpyHostToDevice)); }
                d_output_packed = dc;
            }
            if (quant2 == 2) printf("[hip1bp] packed fast path: TQ2NZ_E4M3 (%d layers)\n", NC);
            else if (quant2 == 3) printf("[hip1bp] packed fast path: Q4NX (%d layers)\n", NC);
            else printf("[hip1bp] packed fast path: TQ2NZ bf16 (%d layers)\n", NC);
        }

        // #1624: architecture validation — hip_1bp implements the dense GQA
        // transformer path only (QKV + gated FFN). Hybrid models (SSM/Mamba
        // layers, CCA, MoE: ZAYA1-8B has cca_*/ssm_conv1d/res_scale and NO
        // ffn_gate/up/down) leave null weight pointers here and fault in the
        // first packed launch or silently produce garbage on the f32 path.
        {
            bool ok_q = (L[0].wq || (quant2 && PD[0].pq)), ok_v = (L[0].wv || (quant2 && PD[0].pv)),
                 ok_f1 = (L[0].w1 || (quant2 && PD[0].p1)), ok_f2 = (L[0].w2 || (quant2 && PD[0].p2)),
                 ok_f3 = (L[0].w3 || (quant2 && PD[0].p3));
            if (!ok_q || !ok_v || !ok_f1 || !ok_f2 || !ok_f3) {
                fprintf(stderr, "[hip1bp] unsupported architecture: missing required tensors "
                        "(attn_v/ffn_gate/ffn_up/ffn_down). Hybrid SSM/MoE models (e.g. ZAYA1-8B) "
                        "are not supported by the GPU backend — use a dense GQA transformer "
                        "(Llama/Qwen family) model.\n");
                return false;
            }
        }

        // Phase 2: capture the full per-token step (embed → layers → final
        // norm → lm_head → device argmax → 4B result copy) into a hipGraph.
        // Kernels read *d_pos / *d_token at replay time, so one capture serves
        // every decode step; the KV cache writes land at the replayed *d_pos.
        gpu_ok = true;   // forward_dev() checks this — needed during capture
        *h_token = 0; *h_pos = 0; *h_res = -1;
        HIP_CHECK(hipMemcpy(d_token, h_token, sizeof(int), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_pos, h_pos, sizeof(int), hipMemcpyHostToDevice));
        // #1626: H1BP_GRAPH=0 or H1BP_DUMP set -> eager mode (no graph capture),
        // so H1BP_DUMP host I/O fires and packed-path debugging is possible.
        bool want_graph = (d_output || d_embed);
        if (const char* g = getenv("H1BP_GRAPH")) want_graph = want_graph && (atoi(g) != 0);
        if (getenv("H1BP_DUMP")) want_graph = false;  // dumps do host I/O — not capturable
        if (want_graph) {
            hipError_t ce = hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal);
            if (ce == hipSuccess) {
                bool ok = forward_dev(0, false);
                if (ok) {
                    if (quant2 && d_output_packed) {
                        if (quant2 == 3) launch_q4nx(d_output_packed, dh, dlogits, VOCAB, H);
                        else launch_tq2nz(d_output_packed, dh, dlogits, VOCAB, H);
                    } else h1bp_gemv_kernel<<<VOCAB,256,0,stream>>>(dlogits, d_output?d_output:d_embed, dh, VOCAB, H);
                    int nblk = std::min(AMX_MAXB, (VOCAB + 255) / 256);
                    h1bp_argmax_pass1_kernel<<<nblk,256,0,stream>>>(dlogits, VOCAB, d_amx, d_ami);
                    h1bp_argmax_pass2_kernel<<<1,256,0,stream>>>(d_amx, d_ami, nblk, d_argmax);
                    HIP_CHECK(hipMemcpyAsync(h_res, d_argmax, sizeof(int), hipMemcpyDeviceToHost, stream));
                }
                ce = hipStreamEndCapture(stream, &graph);
                if (ce == hipSuccess && ok) {
                    hipError_t ie = hipGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);
                    if (ie == hipSuccess) {
                        graph_ok = true;
                        printf("[hip1bp] Phase 2: decode graph captured OK\n");
                    } else {
                        fprintf(stderr, "[hip1bp] graph instantiate failed: %s\n", hipGetErrorString(ie));
                    }
                } else {
                    fprintf(stderr, "[hip1bp] capture failed: ok=%d err=%s\n", ok ? 1 : 0, hipGetErrorString(ce));
                }
            } else {
                fprintf(stderr, "[hip1bp] capture begin failed: %s\n", hipGetErrorString(ce));
            }
        } else if (getenv("H1BP_DUMP")) {
            printf("[hip1bp] H1BP_DUMP set: graph capture skipped so dumps fire (eager mode)\n");
        } else if (d_output || d_embed) {
            printf("[hip1bp] H1BP_GRAPH=0: graph capture disabled (eager mode)\n");
        }

        initialized=true;
        printf("[hip1bp] ✅ GPU 1BP ready\n");
        return true;
    }

    bool reset()override{pos=0;HIP_CHECK(hipMemset(dK,0,kvb));HIP_CHECK(hipMemset(dV,0,kvb));return true;}

    void launch_tq2nz(const uint8_t* w, const float* x, float* out, int N, int K) {
        int ntc = (K + 255) / 256;                    // 256-col tiles per row
        int wpr = (ntc + 3) >> 2;
        int blocks = (N * wpr + 3) / 4;               // 4 warps per block
        if (quant2 == 2)
            h1bp_tq2nz_part_kernel<true><<<blocks,128,0,stream>>>(w,x,dpart,N,ntc);
        else
            h1bp_tq2nz_part_kernel<false><<<blocks,128,0,stream>>>(w,x,dpart,N,ntc);
        h1bp_tq2nz_sum_kernel<<<(N + 255) / 256,256,0,stream>>>(dpart,out,N,wpr);
    }

    void launch_q4nx(const uint8_t* w, const float* x, float* out, int N, int K) {
        int ntc = (K + 255) / 256;
        int wpr = (ntc + 3) >> 2;
        int blocks = (N * wpr + 3) / 4;
        h1bp_q4nx_part_kernel<<<blocks,128,0,stream>>>(w,x,dpart,N,ntc);
        h1bp_tq2nz_sum_kernel<<<(N + 255) / 256,256,0,stream>>>(dpart,out,N,wpr);
    }

    // Device-resident layer loop: runs the full forward, leaves the final
    // hidden state in dh. No D2H copies, no pos++ — caller decides.
    // with_dumps=false when capturing (H1BP_DUMP does host I/O — not capturable).
    bool forward_dev(int token_id, bool with_dumps){
        if(!gpu_ok)return false;
        // KV cache holds max_seq positions (issue #1267)
        if (pos >= max_seq) {
            fprintf(stderr, "[hip_1bp] KV overflow: pos=%d >= max_seq=%d\n", pos, max_seq);
            return false;
        }
        // Phase 2: pos/token live in device memory so the captured graph
        // re-reads them every replay. Pinned host → async 4B copies.
        *h_token = token_id; *h_pos = pos;
        HIP_CHECK(hipMemcpyAsync(d_token, h_token, sizeof(int), hipMemcpyHostToDevice, stream));
        HIP_CHECK(hipMemcpyAsync(d_pos, h_pos, sizeof(int), hipMemcpyHostToDevice, stream));
        int H_=H,NH_=NH,NKV_=NKV,HD_=HD,IM_=IM,NC_=NC;
        int block=256;
        const char* dump_dir = with_dumps ? getenv("H1BP_DUMP") : nullptr;

        // Embedding (token guard moved into the kernel — reads *d_token)
        if(d_embed)
            h1bp_embed_copy_kernel<<<(H_+block-1)/block,block,0,stream>>>(dh,d_embed,d_token,H_,VOCAB);
        else
            HIP_CHECK(hipMemset(dh,0,H_*4));

        for(int l=0;l<NC_;l++){
            auto& ll=L[l];

            // 1. Pre-attention RMSNorm — save pre-norm input for the residual
            //    (rmsnorm is in-place; adding the normed value instead of the
            //    original input breaks the residual stream — same bug the
            //    fused backend had). dsilu is free during attention.
            h1bp_copy_kernel<<<(H_+255)/256,256,0,stream>>>(dsilu,dh,H_);
            if(ll.pn)h1bp_rmsnorm_kernel<<<1,256,0,stream>>>(dh,ll.pn,H_,EPS);
            else h1bp_rmsnorm_kernel<<<1,256,0,stream>>>(dh,nullptr,H_,EPS);

            // 2. QKV via custom GEMV (device-only, no CPU copies)
            if(quant2&&PD[l].pq){
                if (quant2 == 3) {
                    launch_q4nx(PD[l].pq,dh,datt,NH_*HD_,H_);
                    launch_q4nx(PD[l].pk,dh,dgate,NKV_*HD_,H_);
                    launch_q4nx(PD[l].pv,dh,dup,NKV_*HD_,H_);
                } else {
                    launch_tq2nz(PD[l].pq,dh,datt,NH_*HD_,H_);
                    launch_tq2nz(PD[l].pk,dh,dgate,NKV_*HD_,H_);
                    launch_tq2nz(PD[l].pv,dh,dup,NKV_*HD_,H_);
                }
            } else {
                if(ll.wq) h1bp_gemv_kernel<<<NH_*HD_,256,0,stream>>>(datt,ll.wq,dh,NH_*HD_,H_);
                if(ll.wk) h1bp_gemv_kernel<<<NKV_*HD_,256,0,stream>>>(dgate,ll.wk,dh,NKV_*HD_,H_);
                if(ll.wv) h1bp_gemv_kernel<<<NKV_*HD_,256,0,stream>>>(dup,ll.wv,dh,NKV_*HD_,H_);
                // Q/K/V biases (llama.cpp adds them post-projection, pre-rope)
                if(ll.bq) h1bp_add_kernel<<<(NH_*HD_+255)/256,256,0,stream>>>(datt,ll.bq,NH_*HD_);
                if(ll.bk) h1bp_add_kernel<<<(NKV_*HD_+255)/256,256,0,stream>>>(dgate,ll.bk,NKV_*HD_);
                if(ll.bv) h1bp_add_kernel<<<(NKV_*HD_+255)/256,256,0,stream>>>(dup,ll.bv,NKV_*HD_);
            }

            // 2b. Per-head QK-norm (Qwen3/Qwen2.5+): RMSNorm each head's
            // head_dim slice with the shared [head_dim] weight, before RoPE.
            if(ll.q_norm) h1bp_head_rmsnorm_kernel<<<NH_,256,0,stream>>>(datt,ll.q_norm,HD_,EPS);
            if(ll.k_norm) h1bp_head_rmsnorm_kernel<<<NKV_,256,0,stream>>>(dgate,ll.k_norm,HD_,EPS);

            // 3. RoPE (pos read from *d_pos)
            if(ll.wq) h1bp_rope_kernel<<<NH_,HD_/2,0,stream>>>(datt,HD_,d_pos,rope_theta,NH_);
            if(ll.wk) h1bp_rope_kernel<<<NKV_,HD_/2,0,stream>>>(dgate,HD_,d_pos,rope_theta,NKV_);

            // QKV post-RoPE dump (bit-perfect bisection): H1BP_DUMP=<dir>
            if (dump_dir) {
                static std::vector<float> tq, tk, tv;
                int qn = NH_*HD_, kn = NKV_*HD_;
                tq.resize(qn); tk.resize(kn); tv.resize(kn);
                hipMemcpy(tq.data(), datt, qn*4, hipMemcpyDeviceToHost);
                hipMemcpy(tk.data(), dgate, kn*4, hipMemcpyDeviceToHost);
                hipMemcpy(tv.data(), dup, kn*4, hipMemcpyDeviceToHost);
                char fn[512];
                snprintf(fn, sizeof fn, "%s/hip_L%02d_T%05d_Q.f32", dump_dir, l, pos);
                FILE* f = fopen(fn, "wb"); if (f) { fwrite(tq.data(),4,qn,f); fclose(f); }
                snprintf(fn, sizeof fn, "%s/hip_L%02d_T%05d_K.f32", dump_dir, l, pos);
                f = fopen(fn, "wb"); if (f) { fwrite(tk.data(),4,kn,f); fclose(f); }
                snprintf(fn, sizeof fn, "%s/hip_L%02d_T%05d_V.f32", dump_dir, l, pos);
                f = fopen(fn, "wb"); if (f) { fwrite(tv.data(),4,kn,f); fclose(f); }
            }

            // 4. Attention — all on stream, no syncs needed
            if(ll.wo){
                h1bp_f2h_kernel<<<(NH_*HD_+255)/256,256,0,stream>>>(dQ,datt,NH_*HD_);
                // Per-layer KV: layer l owns [l*max_seq*NKV*HD, (l+1)*...)
                __half* lk=dK+(size_t)l*max_seq*NKV_*HD_;
                __half* lv=dV+(size_t)l*max_seq*NKV_*HD_;
                h1bp_kv_store_kernel<<<NKV_,HD_,0,stream>>>(lk,lv,dgate,dup,d_pos,NKV_,HD_,max_seq);

                float scl=1.0f/sqrtf((float)HD_);
                rcpp_kv_cache_attn_decode_dpos(dQ,lk,lv,dAttn,NH_,NKV_,HD_,d_pos,scl,stream);

                // Use separate datt2 for attn output — avoids RAW hazard with datt (used by Q GEMV next layer)
                h1bp_h2f_kernel<<<(NH_*HD_+255)/256,256,0,stream>>>(datt2,dAttn,NH_*HD_);
                if (dump_dir) {
                    static std::vector<float> ta;
                    int qn = NH_*HD_;
                    ta.resize(qn);
                    hipMemcpy(ta.data(), datt2, qn*4, hipMemcpyDeviceToHost);
                    char fn[512];
                    snprintf(fn, sizeof fn, "%s/hip_L%02d_T%05d_ATTN.f32", dump_dir, l, pos);
                    FILE* f = fopen(fn, "wb"); if (f) { fwrite(ta.data(),4,qn,f); fclose(f); }
                }
                if(quant2&&PD[l].po){
                    if (quant2 == 3) launch_q4nx(PD[l].po,datt2,doproj,H_,NH_*HD_);
                    else launch_tq2nz(PD[l].po,datt2,doproj,H_,NH_*HD_);
                } else {
                    h1bp_gemv_kernel<<<H_,256,0,stream>>>(doproj,ll.wo,datt2,H_,NH_*HD_);
                }
                if (dump_dir) {
                    static std::vector<float> t1;
                    t1.resize(H_);
                    hipMemcpy(t1.data(), doproj, H_*4, hipMemcpyDeviceToHost);
                    char fn[512]; snprintf(fn, sizeof fn, "%s/hip_L%02d_T%05d_AO.f32", dump_dir, l, pos);
                    FILE* f = fopen(fn, "wb"); if (f) { fwrite(t1.data(),4,H_,f); fclose(f); }
                }
                // Residual: dh = saved pre-norm input + attn_out
                h1bp_copy_kernel<<<(H_+255)/256,256,0,stream>>>(dh,dsilu,H_);
                h1bp_add_kernel<<<(H_+255)/256,256,0,stream>>>(dh,doproj,H_);
            }

            // 5. Post-attention RMSNorm — save pre-norm input for the residual
            //    (doproj is free during FFN).
            h1bp_copy_kernel<<<(H_+255)/256,256,0,stream>>>(doproj,dh,H_);
            if(ll.pon)h1bp_rmsnorm_kernel<<<1,256,0,stream>>>(dh,ll.pon,H_,EPS);
            else h1bp_rmsnorm_kernel<<<1,256,0,stream>>>(dh,nullptr,H_,EPS);

            // 6. FFN (all on-device)
            if(ll.w1&&ll.w2&&ll.w3){
                if(quant2&&PD[l].p1){
                    if (quant2 == 3) {
                        launch_q4nx(PD[l].p1,dh,dgate,IM_,H_);
                        launch_q4nx(PD[l].p2,dh,dup,IM_,H_);
                        h1bp_silu_kernel<<<(IM_+255)/256,256,0,stream>>>(dsilu,dgate,dup,IM_);
                        launch_q4nx(PD[l].p3,dsilu,dffn,H_,IM_);
                    } else {
                        // K=1024: 1 warp/row, no atomics; K=3072 (down): 3 warps/row + atomics
                        launch_tq2nz(PD[l].p1,dh,dgate,IM_,H_);
                        launch_tq2nz(PD[l].p2,dh,dup,IM_,H_);
                        h1bp_silu_kernel<<<(IM_+255)/256,256,0,stream>>>(dsilu,dgate,dup,IM_);
                        launch_tq2nz(PD[l].p3,dsilu,dffn,H_,IM_);
                    }
                } else {
                    h1bp_gemv_kernel<<<IM_,256,0,stream>>>(dgate,ll.w1,dh,IM_,H_);
                    h1bp_gemv_kernel<<<IM_,256,0,stream>>>(dup,ll.w2,dh,IM_,H_);

                    h1bp_silu_kernel<<<(IM_+255)/256,256,0,stream>>>(dsilu,dgate,dup,IM_);
                    h1bp_gemv_kernel<<<H_,256,0,stream>>>(dffn,ll.w3,dsilu,H_,IM_);

                }
                if (dump_dir) {
                    static std::vector<float> t2;
                    t2.resize(H_);
                    hipMemcpy(t2.data(), dffn, H_*4, hipMemcpyDeviceToHost);
                    char fn[512]; snprintf(fn, sizeof fn, "%s/hip_L%02d_T%05d_DD.f32", dump_dir, l, pos);
                    FILE* f = fopen(fn, "wb"); if (f) { fwrite(t2.data(),4,H_,f); fclose(f); }
                }
                // Residual: dh = saved pre-FFN input + ffn_out (doproj held
                // the pre-norm input from step 5).
                h1bp_copy_kernel<<<(H_+255)/256,256,0,stream>>>(dh,doproj,H_);
                h1bp_add_kernel<<<(H_+255)/256,256,0,stream>>>(dh,dffn,H_);
            }

            // Layer-output dump (bit-perfect bisection): H1BP_DUMP=<dir>
            if (dump_dir) {
                static std::vector<float> tmp;
                tmp.resize(H_);
                hipMemcpy(tmp.data(), dh, H_*4, hipMemcpyDeviceToHost);
                char fn[512]; snprintf(fn, sizeof fn, "%s/hip_L%02d_T%05d.f32", dump_dir, l, pos);
                FILE* f = fopen(fn, "wb");
                if (f) { fwrite(tmp.data(), 4, H_, f); fclose(f); }
            }
        }

        // Final RMSNorm
        h1bp_rmsnorm_kernel<<<1,256,0,stream>>>(dh,d_final_norm,H_,EPS);
        return true;
    }

    bool forward(int token_id,float* hidden_out)override{
        if(!forward_dev(token_id,true))return false;
        HIP_CHECK(hipMemcpy(hidden_out,dh,H*4,hipMemcpyDeviceToHost));
        pos++;
        return true;
    }

    // Phase 1 fused fast path: forward + lm_head + argmax all on device,
    // single 4-byte D2H copy of the token id per step (kills the H-sized
    // hidden round-trip and the VOCAB-sized logits copy). Bit-identical to
    // generate() == forward() + lm_head() — argmax semantics unchanged.
    // Phase 2: when graph_ok, the whole step replays from the captured graph.
    int generate_fast(int token_id){
        if (graph_ok) {
            *h_token = token_id; *h_pos = pos;
            HIP_CHECK(hipGraphLaunch(graphExec, stream));
            HIP_CHECK(hipStreamSynchronize(stream));
            pos++;
            return *h_res;
        }
        if(!forward_dev(token_id, getenv("H1BP_DUMP") ? true : false))return -1;  // #1626: dumps in eager mode
        if(!d_output&&!d_embed){pos++;return 0;}
        if(quant2&&d_output_packed){
            if (quant2 == 3) launch_q4nx(d_output_packed,dh,dlogits,VOCAB,H);
            else launch_tq2nz(d_output_packed,dh,dlogits,VOCAB,H);
        } else {
            h1bp_gemv_kernel<<<VOCAB,256,0,stream>>>(dlogits,d_output?d_output:d_embed,dh,VOCAB,H);
        }
        int nblk=std::min(AMX_MAXB,(VOCAB+255)/256);
        h1bp_argmax_pass1_kernel<<<nblk,256,0,stream>>>(dlogits,VOCAB,d_amx,d_ami);
        h1bp_argmax_pass2_kernel<<<1,256,0,stream>>>(d_amx,d_ami,nblk,d_argmax);
        int n=-1;
        HIP_CHECK(hipMemcpy(&n,d_argmax,sizeof(int),hipMemcpyDeviceToHost));
        if (n < 0) {  // #1627: NaN/garbage logits surface as argmax -1 — never silent
            fprintf(stderr, "[hip1bp] generate(): argmax returned %d — logits NaN/garbage? "
                    "(unsupported quant or corrupt model)\n", n);
        }
        pos++;
        return n;
    }

    bool lm_head(const float* hidden,float* logits,int* argmax)override{
        // Tied-embedding models (e.g. Qwen3) have no output.weight — the LM
        // head is the embedding matrix itself.
        if(!d_output&&!d_embed){memset(logits,0,VOCAB*4);logits[0]=1;if(argmax)*argmax=0;return true;}
        HIP_CHECK(hipMemcpy(dh,hidden,H*4,hipMemcpyHostToDevice));
        if(quant2&&d_output_packed){
            if (quant2 == 3) launch_q4nx(d_output_packed,dh,dlogits,VOCAB,H);
            else launch_tq2nz(d_output_packed,dh,dlogits,VOCAB,H);
        } else {
            h1bp_gemv_kernel<<<VOCAB,256,0,stream>>>(dlogits,d_output?d_output:d_embed,dh,VOCAB,H);
        }
        HIP_CHECK(hipMemcpy(logits,dlogits,VOCAB*4,hipMemcpyDeviceToHost));
        if (const char* dd = getenv("H1BP_DUMP")) {
            char fn[512]; snprintf(fn, sizeof fn, "%s/hip_logits_T%05d.f32", dd, pos);
            FILE* f = fopen(fn, "wb");
            if (f) { fwrite(logits, 4, VOCAB, f); fclose(f); }
        }
        if(argmax){*argmax=0;for(int v=1;v<VOCAB;v++)if(logits[v]>logits[*argmax])*argmax=v;}
        return true;
    }

    int generate(int token_id)override{
        return generate_fast(token_id);
    }

    float benchmark(int tokens)override{
        if(!initialized)return-1;reset();
        auto t0=std::chrono::steady_clock::now();int tok=1;
        for(int i=0;i<tokens;i++){tok=generate(tok);if(tok<0)break;}
        auto t1=std::chrono::steady_clock::now();
        return std::chrono::duration<float,std::milli>(t1-t0).count()/tokens;
    }

    void destroy()override{
        auto hf=[](auto*&p){if(p){HIP_CHECK_D(hipFree(p));p=nullptr;}};
        hf(d_embed);hf(d_final_norm);hf(d_output);
        for(auto&ll:L){hf(ll.wq);hf(ll.wk);hf(ll.wv);hf(ll.wo);hf(ll.w1);hf(ll.w2);hf(ll.w3);hf(ll.pn);hf(ll.pon);hf(ll.q_norm);hf(ll.k_norm);hf(ll.bq);hf(ll.bk);hf(ll.bv);}
        for (auto& pd : PD) { hf(pd.pq); hf(pd.pk); hf(pd.pv); hf(pd.po); hf(pd.p1); hf(pd.p2); hf(pd.p3); }
        PD.clear(); hf(d_output_packed);
        L.clear();P.clear();model_.reset();
        hf(dh);hf(datt);hf(dgate);hf(dup);hf(dsilu);hf(doproj);hf(dffn);hf(dlogits);hf(dpart);
        hf(datt2);hf(dK);hf(dV);hf(dQ);hf(dAttn);
        hf(d_argmax);hf(d_amx);hf(d_ami);
        hf(d_pos);hf(d_token);
        if (graphExec) { HIP_CHECK_D(hipGraphExecDestroy(graphExec)); graphExec = nullptr; }
        if (graph) { HIP_CHECK_D(hipGraphDestroy(graph)); graph = nullptr; }
        graph_ok = false;
        if (h_token) { HIP_CHECK_D(hipHostFree(h_token)); h_token = nullptr; }
        if (h_pos) { HIP_CHECK_D(hipHostFree(h_pos)); h_pos = nullptr; }
        if (h_res) { HIP_CHECK_D(hipHostFree(h_res)); h_res = nullptr; }
        if(stream){HIP_CHECK_D(hipStreamDestroy(stream));stream=nullptr;}
        gpu_ok=false;initialized=false;
    }
};

extern"C" Backend* create_hip_1bp_backend(){return static_cast<Backend*>(new Hip1bpBackend());}
