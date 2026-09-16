// test_fk3_rope.cpp — verify the host RoPE pass against an INDEPENDENT transcription
// of the engine's ra2 convention (half-split pairs). Build:
//   g++ -std=c++17 -O2 -I engine/npu/src -o /tmp/rope_test engine/npu/tests/test_fk3_rope.cpp && /tmp/rope_test
#include "npu_fk3_rope.h"
#include <cstdio>
#include <vector>
#include <cmath>
// independent reference: a direct transcription of the engine's ra2 loop
int main(){
  const int M=5, NH=2, NKV=1, HD=8;   // small but full-rotary (rope_dim=HD)
  const float th=1e6f;
  const int NQKV=(NH+2*NKV)*HD, KOFF=NH*HD;
  std::vector<uint16_t> a((size_t)M*NQKV), b((size_t)M*NQKV);
  auto rne=[](float f){uint32_t u;memcpy(&u,&f,4);uint32_t l=(u>>16)&1;return (uint16_t)((u+0x7FFFu+l)>>16);};
  for(size_t i=0;i<a.size();i++) a[i]=rne((float)((int)(i%37)-18)*0.05f);
  b=a;
  fk3::rope_qk_bf16(a.data(),M,NH,NKV,HD,th,0);
  // reference: rotate only Q and K, half-split
  for(int i=0;i<M;i++){
    uint16_t*row=b.data()+(size_t)i*NQKV;
    int cols[3]; for(int h=0;h<NH;h++) cols[h]=h*HD;
    int n=0; int cidx[8];
    for(int h=0;h<NH;h++) cidx[n++]=h*HD;
    for(int kh=0;kh<NKV;kh++) cidx[n++]=KOFF+kh*HD;
    for(int ci=0;ci<n;ci++){
      uint16_t*x=row+cidx[ci];
      float v[HD]; for(int d=0;d<HD;d++){uint32_t u=(uint32_t)x[d]<<16;memcpy(&v[d],&u,4);}
      for(int d=0;d<HD/2;d++){
        double f=1.0/pow((double)th,(double)d/(HD/2)), an=(double)i*f;
        float c=cosf(an), s=sinf(an), x0=v[d], x1=v[d+HD/2];
        v[d]=x0*c-x1*s; v[d+HD/2]=x1*c+x0*s;
      }
      for(int d=0;d<HD;d++) x[d]=rne(v[d]);
    }
  }
  int bad=0;
  for(size_t i=0;i<a.size();i++) if(a[i]!=b[i]) bad++;
  printf("rope_qk vs independent reference: %d/%zu differ -> %s\n", bad, a.size(), bad?"MISMATCH":"MATCH");
  // and confirm V is untouched
  int vbad=0;
  for(int i=0;i<M;i++) for(int kh=0;kh<NKV;kh++) for(int d=0;d<HD;d++){
    size_t idx=(size_t)i*NQKV+KOFF+NKV*HD+(size_t)kh*HD+d;
    if(a[idx]!=b[idx]) vbad++;
  }
  printf("V untouched: %s\n", vbad?"NO":"yes");
  return 0;
}
