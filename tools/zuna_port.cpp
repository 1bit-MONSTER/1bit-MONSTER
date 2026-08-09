// zuna_port.cpp — standalone C++ CPU port of ZUNA1.1 (masked diffusion autoencoder)
// Numerical parity vs the reference `zuna` Python package.
//
// CANONICAL HOME: github.com/1bit-systems/eeg-medical (tools/zuna_port.cpp).
// This copy is vendored into the 1bit engine build (ONE_BIN_SOURCES); sync
// changes back to eeg-medical first, then copy into tools/ here.
//
// Hyperparams (resolved from Zyphra/ZUNA1.1 config.json + package source):
//   dim=1024 head_dim=64 n_heads=8 n_kv_heads=8 ffn_hidden=2816 n_layers=16
//   rope_dim=4 rope_theta=1.0e4 max_seqlen=256 norm_eps=1e-5 (QK-norms 1e-5)
//   input_dim=32 t_dim=64 global_sigma=0.1 sample_steps=50 cfg=1.0
//   register_tok_idx="mean_all" encoder_latent_downsample_factor df=1
//
// Weight format: weights.bin (row-major fp32) + weights.json (list of
//   {name,shape,offset,bytes}) emitted by tools/export_weights.py (safetensors
//   naming). Weights stored [OUT,IN]; y = x@W^T.
//   Plain RMSNorm weight  -> "<path>.norm.weight"        (weight 1D [dim])
//   AdaRMSNorm (Linear)   -> "<path>.weight.weight" + "_weight.bias" ([dim,64],[dim])
//
// I/O: tokens.bin (fp32,[S,32]) + tok_idx.bin (int32,[S,4]) -> enc_out.bin + recon.bin

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <cstdint>
typedef float f32;
static inline f32 silu(f32 x){ return x/(1.0f+std::exp(-x)); }
// ---------------- weight manifest ----------------
struct WT { std::string name; std::vector<uint32_t> shape; long off; long bytes; };
static std::vector<WT> parse_manifest(const std::string& s){
    std::vector<WT> out; size_t i=0;
    while ((i=s.find("\"name\"",i))!=std::string::npos){
        WT t; size_t c=s.find(':',i); size_t a=s.find('\"',c+1); size_t b=s.find('\"',a+1);
        t.name=s.substr(a+1,b-a-1);
        size_t sh=s.find("shape",b); size_t lb=s.find('[',sh); size_t rb=s.find(']',lb);
        std::string ss=s.substr(lb+1,rb-lb-1); size_t p=0;
        while((p=ss.find_first_of("0123456789",p))!=std::string::npos){ size_t e=ss.find_first_not_of("0123456789",p); t.shape.push_back((uint32_t)strtol(ss.substr(p,e-p).c_str(),nullptr,10)); p=e; }
        size_t o=s.find("\"offset\"",rb); c=s.find(':',o); size_t c2=s.find_first_of(",}",c); t.off=strtol(s.substr(c+1,c2-c-1).c_str(),nullptr,10);
        size_t by=s.find("\"bytes\"",c2); c=s.find(':',by); c2=s.find_first_of(",}",c); t.bytes=strtol(s.substr(c+1,c2-c-1).c_str(),nullptr,10);
        out.push_back(std::move(t)); i=c2;
    }
    return out;
}
struct Weights {
    std::vector<f32> data; std::vector<WT> man;
    auto find(const char* n) const { for(auto it=man.begin();it!=man.end();++it) if(it->name==n) return it; return man.end(); }
    const f32* ptr(const WT& t) const { return data.data()+t.off/4; }
    bool w1(const char* n, std::vector<f32>& w, size_t& sz) const { auto it=find(n); if(it==man.end())return false; sz=it->bytes/4; const f32* p=ptr(*it); w.assign(p,p+sz); return true; }
    bool w2(const char* n, std::vector<f32>& w, size_t& out, size_t& in) const { auto it=find(n); if(it==man.end())return false; out=it->shape[0]; in=it->shape[1]; const f32* p=ptr(*it); w.assign(p,p+it->bytes/4); return true; }
};
// ---------------- math ----------------
static void linear(const f32* x, const f32* W, const f32* bias, f32* y, int R, int in, int out){
    for(int r=0;r<R;r++){ const f32* xr=x+(size_t)r*in; f32* yr=y+(size_t)r*out;
        for(int o=0;o<out;o++){ double s=0; const f32* wr=W+(size_t)o*in;
            for(int i=0;i<in;i++) s+=(double)xr[i]*wr[i];
            yr[o]=(f32)s + (bias?bias[o]:0.f); } }
}
static void rmsnorm(f32* y,const f32* x,const f32* w,int n,f32 eps){
    double m=1e-8; for(int i=0;i<n;i++) m+=(double)x[i]*x[i]; m=1.0/std::sqrt(m/n+eps);
    for(int i=0;i<n;i++) y[i]=(f32)(x[i]*m)*w[i];
}
static void adarnorm(f32* y,const f32* x,const f32* cond,const f32* W,const f32* bias,int dim,int emb,f32 eps){
    double m=1e-8; for(int i=0;i<dim;i++) m+=(double)x[i]*x[i]; m=1.0/std::sqrt(m/dim+eps);
    for(int i=0;i<dim;i++){ double s=0; for(int j=0;j<emb;j++) s+=(double)cond[j]*W[(size_t)i*emb+j]; s+=bias[i];
        y[i]=(f32)(x[i]*m)*(f32)s; }
}
static void adarnorm_rows(f32* y,const f32* x,const f32* cond,const f32* W,const f32* bias,int S,int D,int emb,f32 eps){
    for(int s=0;s<S;s++) adarnorm(y+(size_t)s*D, x+(size_t)s*D, cond, W, bias, D, emb, eps);
}
// ---------------- 4D RoPE ----------------
struct Freq {
    std::vector<f32> ct,st; int mslen=0,pairs=0;
    Freq(){}
    void build(int m,int dimaxis,f32 th){ mslen=m; pairs=dimaxis/2; ct.assign((size_t)m*pairs,0.f); st.assign((size_t)m*pairs,0.f);
        for(int p=0;p<m;p++) for(int j=0;j<pairs;j++){ f32 f=1.0f/std::pow(th,(2.0f*j)/(f32)dimaxis); f32 a=(f32)p*f; ct[(size_t)p*pairs+j]=cos(a); st[(size_t)p*pairs+j]=sin(a);} }
    void apply(f32* x,int S,int H,int HD,const std::vector<int>& ti) const {
        for(int s=0;s<S;s++){ const int* tok=&ti[(size_t)s*4];
            for(int h=0;h<H;h++){ f32* xh=x+((size_t)s*H+h)*HD;
                for(int axis=0;axis<4;axis++){ int pos=tok[axis]; if(pos<0)pos=0; if(pos>=mslen)pos=mslen-1;
                    for(int j=0;j<pairs;j++){ int p=axis*pairs+j; f32 c=ct[(size_t)pos*pairs+j],ss=st[(size_t)pos*pairs+j];
                        f32 a0=xh[2*p],a1=xh[2*p+1]; xh[2*p]=a0*c-a1*ss; xh[2*p+1]=a0*ss+a1*c; } } } }
    }
};
// ---------------- attention ----------------
static void sdpa(const f32* q,int S,const f32* k,const f32* v,int T,int H,int HD,f32* out){
    f32 scale=1.0f/std::sqrt((f32)HD);
    for(int s=0;s<S;s++) for(int h=0;h<H;h++){
        const f32* qh=q+((size_t)s*H+h)*HD; f32 mx=-1e30f;
        std::vector<f32> sc((size_t)T);
        for(int t=0;t<T;t++){ const f32* kh=k+((size_t)t*H+h)*HD; double a=0; for(int j=0;j<HD;j++) a+=(double)qh[j]*kh[j]; sc[t]=(f32)(a*scale); mx=std::max(mx,sc[t]); }
        f32 sum=0; for(int t=0;t<T;t++){ sc[t]=(f32)exp(sc[t]-mx); sum+=sc[t]; }
        f32* oh=out+((size_t)s*H+h)*HD;
        for(int j=0;j<HD;j++){ double acc=0; for(int t=0;t<T;t++) acc+=(double)(sc[t]/sum)*v[((size_t)t*H+h)*HD+j]; oh[j]=(f32)acc; }
    }
}
static void attention(const f32* xq,int S,const f32* xkv,int T,int D,int H,int HD,
                      const f32* wq,const f32* wk,const f32* wv,const f32* wo,
                      const f32* qn,const f32* kn,const Freq& freq,
                      const std::vector<int>& tq,const std::vector<int>& tk,f32* out){
    int O=H*HD;
    std::vector<f32> q((size_t)S*O),k((size_t)T*O),v((size_t)T*O);
    linear(xq,wq,nullptr,q.data(),S,D,O);
    linear(xkv,wk,nullptr,k.data(),T,D,O);
    linear(xkv,wv,nullptr,v.data(),T,D,O);
    std::vector<f32> qq((size_t)S*O),kk((size_t)T*O);
    for(int s=0;s<S;s++) for(int h=0;h<H;h++) rmsnorm(qq.data()+((size_t)s*H+h)*HD, q.data()+((size_t)s*H+h)*HD, qn, HD, 1e-5f);
    for(int t=0;t<T;t++) for(int h=0;h<H;h++) rmsnorm(kk.data()+((size_t)t*H+h)*HD, k.data()+((size_t)t*H+h)*HD, kn, HD, 1e-5f);
    freq.apply(qq.data(),S,H,HD,tq);
    freq.apply(kk.data(),T,H,HD,tk);
    sdpa(qq.data(),S,kk.data(),v.data(),T,H,HD,out);
    // output projection O->D
    { std::vector<f32> att((size_t)S*D);
      linear(out,wo,nullptr,att.data(),S,O,D);
      memcpy(out,att.data(),(size_t)S*D*sizeof(f32)); }
}
// ---------------- model ----------------
struct ZModel {
    Weights* W; Freq freqs;
    const int D=1024, HD=64, H=8, FH=2816, NL=16, ID=32, TD=64;
    const f32 eps=1e-5f;
    explicit ZModel(Weights* w){ W=w; freqs.build(256,16,10000.0f); }

    void encode(const f32* tokens,int S,const f32* regmat,const std::vector<int>& tok_idx,
                f32* enc_out){
        int seq=2*S;
        std::vector<f32> xin((size_t)seq*ID);
        for(int g=0;g<S;g++) for(int j=0;j<ID;j++){ xin[((size_t)2*g)*ID+j]=regmat[j]; xin[((size_t)(2*g+1))*ID+j]=tokens[g*ID+j]; }
        std::vector<f32> te,teb; size_t no,ni,tn;
        W->w2("encoder.tok_embeddings.weight",te,no,ni); W->w1("encoder.tok_embeddings.bias",teb,tn);
        std::vector<f32> h((size_t)seq*D);
        linear(xin.data(),te.data(),teb.data(),h.data(),seq,ID,D);
        std::vector<int> itok((size_t)2*S*4);
        for(int g=0;g<S;g++) for(int a=0;a<4;a++){ itok[((size_t)2*g)*4+a]=tok_idx[g*4+a]; itok[((size_t)(2*g+1))*4+a]=tok_idx[g*4+a]; }
        for(int l=0;l<NL;l++){
            char bf[256];
            std::vector<f32> anw,anp,ffnw,fp; size_t nn;
            snprintf(bf,sizeof(bf),"encoder.layers.%d.attention_norm.norm.weight",l); W->w1(bf,anw,nn);
            snprintf(bf,sizeof(bf),"encoder.layers.%d.attention_norm_post.norm.weight",l); W->w1(bf,anp,nn);
            snprintf(bf,sizeof(bf),"encoder.layers.%d.ffn_norm.norm.weight",l); W->w1(bf,ffnw,nn);
            snprintf(bf,sizeof(bf),"encoder.layers.%d.ffn_norm_post.norm.weight",l); W->w1(bf,fp,nn);
            std::vector<f32> wq,wk,wv,wo,qn,kn;
            snprintf(bf,sizeof(bf),"encoder.layers.%d.attention.wq.weight",l); W->w2(bf,wq,no,ni);
            snprintf(bf,sizeof(bf),"encoder.layers.%d.attention.wk.weight",l); W->w2(bf,wk,no,ni);
            snprintf(bf,sizeof(bf),"encoder.layers.%d.attention.wv.weight",l); W->w2(bf,wv,no,ni);
            snprintf(bf,sizeof(bf),"encoder.layers.%d.attention.wo.weight",l); W->w2(bf,wo,no,ni);
            snprintf(bf,sizeof(bf),"encoder.layers.%d.attention.q_norm.norm.weight",l); W->w1(bf,qn,nn);
            snprintf(bf,sizeof(bf),"encoder.layers.%d.attention.k_norm.norm.weight",l); W->w1(bf,kn,nn);
            std::vector<f32> fn1,fn2,fn3;
            snprintf(bf,sizeof(bf),"encoder.layers.%d.feed_forward.w1.weight",l); W->w2(bf,fn1,no,ni);
            snprintf(bf,sizeof(bf),"encoder.layers.%d.feed_forward.w2.weight",l); W->w2(bf,fn2,no,ni);
            snprintf(bf,sizeof(bf),"encoder.layers.%d.feed_forward.w3.weight",l); W->w2(bf,fn3,no,ni);
            std::vector<f32> hnorm((size_t)seq*D), aatt((size_t)seq*D);
            for(int s=0;s<seq;s++) rmsnorm(hnorm.data()+s*D, h.data()+s*D, anw.data(), D, eps);
            attention(hnorm.data(),seq,hnorm.data(),seq,D,H,HD,wq.data(),wk.data(),wv.data(),wo.data(),qn.data(),kn.data(),freqs,itok,itok,aatt.data());
            for(int s=0;s<seq;s++){ f32* hh=h.data()+s*D; const f32* aa=aatt.data()+s*D; std::vector<f32> post((size_t)D); rmsnorm(post.data(),aa,anp.data(),D,eps); for(int j=0;j<D;j++) hh[j]+=post[j]; }
            std::vector<f32> hn2((size_t)seq*D);
            for(int s=0;s<seq;s++) rmsnorm(hn2.data()+s*D, h.data()+s*D, ffnw.data(), D, eps);
            for(int s=0;s<seq;s++){ const f32* x=hn2.data()+s*D; f32* hh=h.data()+s*D;
                f32 g1[4096],g3[4096],mid[4096],ffb[4096];
                linear(x,fn1.data(),nullptr,g1,1,D,FH); linear(x,fn3.data(),nullptr,g3,1,D,FH);
                for(int j=0;j<FH;j++) mid[j]=silu(g1[j])*g3[j];
                linear(mid,fn2.data(),nullptr,ffb,1,FH,D);
                std::vector<f32> post((size_t)D); rmsnorm(post.data(),ffb,fp.data(),D,eps);
                for(int j=0;j<D;j++) hh[j]+=post[j];
            }
        }
        std::vector<f32> enw,eo; size_t enn,eoo,eoi;
        W->w1("encoder.norm.norm.weight",enw,enn); W->w2("encoder.output.weight",eo,eoo,eoi);
        for(int g=0;g<S;g++){ const f32* hh=h.data()+((size_t)2*g)*D; f32 normv[1024]; rmsnorm(normv,hh,enw.data(),D,eps);
            linear(normv,eo.data(),nullptr,enc_out+g*ID,1,D,ID); }
    }

    void velocity(const f32* z,int S,const f32* cross_enc,int T, f32 tval,
                  const std::vector<int>& tok_idx,f32* out){
        std::vector<f32> tmw,tmp,tmpb; size_t n,no,ni;
        W->w1("decoder.t_embedder.weight",tmw,n);
        W->w2("decoder.t_embedder.proj.weight",tmp,no,ni); W->w1("decoder.t_embedder.proj.bias",tmpb,n);
        f32 f32e[32]; for(int j=0;j<32;j++) f32e[j]=(f32)(2.0*M_PI*tval*tmw[j]);
        f32 fe[64]; for(int j=0;j<32;j++) fe[j]=cos(f32e[j]); for(int j=0;j<32;j++) fe[32+j]=sin(f32e[j]);
        f32 c[64]; linear(fe,tmp.data(),tmpb.data(),c,1,64,64);
        std::vector<f32> teb,tebb,ep,epb,ow; size_t tn;
        W->w2("decoder.tok_embeddings.weight",teb,no,ni); W->w1("decoder.tok_embeddings.bias",tebb,tn);
        W->w2("decoder.encoder_proj.weight",ep,no,ni); W->w1("decoder.encoder_proj.bias",epb,tn);
        W->w2("decoder.output.weight",ow,no,ni);
        std::vector<f32> h((size_t)S*D), cross((size_t)T*D);
        linear(z,teb.data(),tebb.data(),h.data(),S,ID,D);
        linear(cross_enc,ep.data(),epb.data(),cross.data(),T,ID,D);
        for(int l=0;l<NL;l++){
            char bf[256];
            std::vector<f32> cxw,cxb,cyw,cyb,cq,ck,cv,co,cqn,ckn,cpost; size_t nn;
            snprintf(bf,sizeof(bf),"decoder.layers.%d.cross_attention_x_norm.weight.weight",l); W->w2(bf,cxw,no,ni);
            snprintf(bf,sizeof(bf),"decoder.layers.%d.cross_attention_x_norm.weight.bias",l); W->w1(bf,cxb,nn);
            snprintf(bf,sizeof(bf),"decoder.layers.%d.cross_attention_y_norm.weight.weight",l); W->w2(bf,cyw,no,ni);
            snprintf(bf,sizeof(bf),"decoder.layers.%d.cross_attention_y_norm.weight.bias",l); W->w1(bf,cyb,nn);
            snprintf(bf,sizeof(bf),"decoder.layers.%d.cross_attention.wq.weight",l); W->w2(bf,cq,no,ni);
            snprintf(bf,sizeof(bf),"decoder.layers.%d.cross_attention.wk.weight",l); W->w2(bf,ck,no,ni);
            snprintf(bf,sizeof(bf),"decoder.layers.%d.cross_attention.wv.weight",l); W->w2(bf,cv,no,ni);
            snprintf(bf,sizeof(bf),"decoder.layers.%d.cross_attention.wo.weight",l); W->w2(bf,co,no,ni);
            snprintf(bf,sizeof(bf),"decoder.layers.%d.cross_attention.q_norm.norm.weight",l); W->w1(bf,cqn,nn);
            snprintf(bf,sizeof(bf),"decoder.layers.%d.cross_attention.k_norm.norm.weight",l); W->w1(bf,ckn,nn);
            snprintf(bf,sizeof(bf),"decoder.layers.%d.cross_attention_norm_post.norm.weight",l); W->w1(bf,cpost,nn);
            std::vector<f32> sann,sanb,sq,sk,sv,so,sqn,skn,sanp;
            snprintf(bf,sizeof(bf),"decoder.layers.%d.attention_norm.weight.weight",l); W->w2(bf,sann,no,ni);
            snprintf(bf,sizeof(bf),"decoder.layers.%d.attention_norm.weight.bias",l); W->w1(bf,sanb,nn);
            snprintf(bf,sizeof(bf),"decoder.layers.%d.attention.wq.weight",l); W->w2(bf,sq,no,ni);
            snprintf(bf,sizeof(bf),"decoder.layers.%d.attention.wk.weight",l); W->w2(bf,sk,no,ni);
            snprintf(bf,sizeof(bf),"decoder.layers.%d.attention.wv.weight",l); W->w2(bf,sv,no,ni);
            snprintf(bf,sizeof(bf),"decoder.layers.%d.attention.wo.weight",l); W->w2(bf,so,no,ni);
            snprintf(bf,sizeof(bf),"decoder.layers.%d.attention.q_norm.norm.weight",l); W->w1(bf,sqn,nn);
            snprintf(bf,sizeof(bf),"decoder.layers.%d.attention.k_norm.norm.weight",l); W->w1(bf,skn,nn);
            snprintf(bf,sizeof(bf),"decoder.layers.%d.attention_norm_post.norm.weight",l); W->w1(bf,sanp,nn);
            std::vector<f32> fann,fanb,fn1,fn2,fn3,fp;
            snprintf(bf,sizeof(bf),"decoder.layers.%d.ffn_norm.weight.weight",l); W->w2(bf,fann,no,ni);
            snprintf(bf,sizeof(bf),"decoder.layers.%d.ffn_norm.weight.bias",l); W->w1(bf,fanb,nn);
            snprintf(bf,sizeof(bf),"decoder.layers.%d.feed_forward.w1.weight",l); W->w2(bf,fn1,no,ni);
            snprintf(bf,sizeof(bf),"decoder.layers.%d.feed_forward.w2.weight",l); W->w2(bf,fn2,no,ni);
            snprintf(bf,sizeof(bf),"decoder.layers.%d.feed_forward.w3.weight",l); W->w2(bf,fn3,no,ni);
            snprintf(bf,sizeof(bf),"decoder.layers.%d.ffn_norm_post.norm.weight",l); W->w1(bf,fp,nn);
            std::vector<f32> xnx(S*D), ynx(T*D), ca(S*D);
            adarnorm_rows(xnx.data(),h.data(),c,cxw.data(),cxb.data(),S,D,TD,eps);
            adarnorm_rows(ynx.data(),cross.data(),c,cyw.data(),cyb.data(),T,D,TD,eps);
            attention(xnx.data(),S,ynx.data(),T,D,H,HD,cq.data(),ck.data(),cv.data(),co.data(),cqn.data(),ckn.data(),freqs,tok_idx,tok_idx,ca.data());
            for(int s=0;s<S;s++){ f32* hh=h.data()+s*D; const f32* aa=ca.data()+s*D; std::vector<f32> post((size_t)D); rmsnorm(post.data(),aa,cpost.data(),D,eps); for(int j=0;j<D;j++) hh[j]+=post[j]; }
            std::vector<f32> snx(S*D), sa(S*D);
            adarnorm_rows(snx.data(),h.data(),c,sann.data(),sanb.data(),S,D,TD,eps);
            attention(snx.data(),S,snx.data(),S,D,H,HD,sq.data(),sk.data(),sv.data(),so.data(),sqn.data(),skn.data(),freqs,tok_idx,tok_idx,sa.data());
            for(int s=0;s<S;s++){ f32* hh=h.data()+s*D; const f32* aa=sa.data()+s*D; std::vector<f32> post((size_t)D); rmsnorm(post.data(),aa,sanp.data(),D,eps); for(int j=0;j<D;j++) hh[j]+=post[j]; }
            std::vector<f32> fnx(S*D);
            adarnorm_rows(fnx.data(),h.data(),c,fann.data(),fanb.data(),S,D,TD,eps);
            for(int s=0;s<S;s++){ const f32* x=fnx.data()+s*D; f32* hh=h.data()+s*D;
                f32 g1[4096],g3[4096],mid[4096],ffb[4096];
                linear(x,fn1.data(),nullptr,g1,1,D,FH); linear(x,fn3.data(),nullptr,g3,1,D,FH);
                for(int j=0;j<FH;j++) mid[j]=silu(g1[j])*g3[j];
                linear(mid,fn2.data(),nullptr,ffb,1,FH,D);
                std::vector<f32> post((size_t)D); rmsnorm(post.data(),ffb,fp.data(),D,eps);
                for(int j=0;j<D;j++) hh[j]+=post[j];
            }
        }
        std::vector<f32> dnw,dnb; size_t dnn;
        W->w2("decoder.norm.weight.weight",dnw,no,ni); W->w1("decoder.norm.weight.bias",dnb,dnn);
        for(int s=0;s<S;s++){ f32 hn[1024]; adarnorm(hn,h.data()+s*D,c,dnw.data(),dnb.data(),D,TD,eps);
            linear(hn,ow.data(),nullptr,out+s*ID,1,D,ID); }
    }
};
// ---------------- full sampling ----------------
struct ZRunner {
    Weights* W; ZModel M;
    explicit ZRunner(Weights* w):W(w),M(w){}
    // z may be null -> generate from mt19937(seed); otherwise use provided initial noise
    void sample(const f32* tokens,int S,const std::vector<int>& tok_idx,unsigned int seed,
                f32* z_input,f32* enc_out,f32* recon){
        std::vector<f32> regmat(32), enc((size_t)S*M.ID);
        size_t rn; W->w1("encoder.registers",regmat,rn); regmat.resize(M.ID);
        M.encode(tokens,S,regmat.data(),tok_idx,enc.data());
        if(enc_out) memcpy(enc_out,enc.data(),(size_t)S*M.ID*sizeof(f32));
        std::vector<f32> z((size_t)S*M.ID);
        if(z_input){ memcpy(z.data(),z_input,(size_t)S*M.ID*sizeof(f32)); }
        else { std::mt19937 rng(seed); std::normal_distribution<f32> nd(0.f,1.f);
            for(size_t i=0;i<(size_t)S*M.ID;i++) z[i]=0.1f*nd(rng); }
        f32 dt=1.0f/50.0f;
        std::vector<f32> v((size_t)S*M.ID);
        for(int step=50; step>=1; step--){
            f32 t = dt*(f32)step;
            M.velocity(z.data(),S,enc.data(),S,t,tok_idx,v.data());
            for(size_t i=0;i<(size_t)S*M.ID;i++) z[i] -= dt*v[i];
        }
        memcpy(recon,z.data(),(size_t)S*M.ID*sizeof(f32));
    }
};
static bool load_f32_file(const char* path, std::vector<f32>& v){ FILE* f=fopen(path,"rb"); if(!f) return false; fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET); v.resize(n/4); if(n) fread(v.data(),4,v.size(),f); fclose(f); return true; }
#ifdef ONE_BIN_DISPATCH
int zuna_main(int argc, char** argv){
#else
int main(int argc, char** argv){
#endif
    if(argc<5){ fprintf(stderr,"usage: %s <weights_dir> <tokens.bin> <tok_idx.bin> <out_recon.bin> [out_enc.bin] [seed]\n", argv[0]); return 2; }
    std::string dir=argv[1];
    FILE* fb=fopen((dir+"/weights.bin").c_str(),"rb"); if(!fb){fprintf(stderr,"no weights.bin\n");return 1;}
    fseek(fb,0,SEEK_END); long fs=ftell(fb); fseek(fb,0,SEEK_SET);
    Weights W; W.data.resize((size_t)fs/4); if(fs) fread(W.data.data(),1,(size_t)fs,fb); fclose(fb);
    FILE* fm=fopen((dir+"/weights.json").c_str(),"rb"); if(!fm){fprintf(stderr,"no weights.json\n");return 1;}
    std::string js; char ch[8192]; size_t rr; while((rr=fread(ch,1,sizeof(ch),fm))>0) js.append(ch,rr); fclose(fm);
    W.man=parse_manifest(js);
    ZRunner R(&W);
    std::vector<f32> tokens; if(!load_f32_file(argv[2],tokens)){fprintf(stderr,"no tokens.bin\n");return 1;}
    const int S = (int)(tokens.size()/R.M.ID);
    FILE* fidx=fopen(argv[3],"rb"); if(!fidx){fprintf(stderr,"no tok_idx.bin\n");return 1;}
    std::vector<int32_t> ti(S*4); fread(ti.data(),4,S*4,fidx); fclose(fidx);
    std::vector<int> tok(S*4); for(int i=0;i<S*4;i++) tok[i]=(int)ti[i];
    unsigned int seed = argc>=7 ? (unsigned int)atoi(argv[6]) : 0;
    std::vector<f32> enc((size_t)S*R.M.ID), recon((size_t)S*R.M.ID);
    // optional initial-noise file as argv[7] (else generate from mt19937(seed))
    std::vector<f32> z_in;
    if(argc>=8 && argv[7][0]!='-') load_f32_file(argv[7], z_in);
    R.sample(tokens.data(),S,tok,seed,z_in.empty()?nullptr:z_in.data(),enc.data(),recon.data());

    FILE* fo=fopen(argv[4],"wb"); fwrite(recon.data(),4,recon.size(),fo); fclose(fo);
    if(argc>=6 && argv[5][0]!='-'){ FILE* e=fopen(argv[5],"wb"); fwrite(enc.data(),4,enc.size(),e); fclose(e); }
    printf("wrote recon S=%d -> %s (enc %s)\n", S, argv[4], argc>=6?argv[5]:"<skip>");
    return 0;
}
