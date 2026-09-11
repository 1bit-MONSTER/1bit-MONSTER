// replay_attn.cpp — run FLM's captured attention ELF (elf_0012) with the
// captured act (Q) + kv (32MB) BOs, and compare output vs captured attn_out.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include "npu_utils/npu_utils_xrt.hpp"
#include "lm_config.hpp"
namespace utils { std::string find_xclbin_path() { return "/home/bcloud/amd-oss/fastflowlm/src/xclbins"; } }
static float bff32(uint16_t v){uint32_t b=v<<16;float f;memcpy(&f,&b,4);return f;}
int main(int argc, char** argv) {
    std::string xd = "/home/bcloud/amd-oss/fastflowlm/src/xclbins/Qwen3-0.6B-NPU2";
    std::string elf = argc>1?argv[1]:"/tmp/attn_elf.bin";
    std::string actf = argc>2?argv[2]:"/tmp/attnio/attn_act_0000.bin";
    std::string kvf  = argc>3?argv[3]:"/tmp/attnio/attn_kv_0000.bin";
    std::string outf = argc>4?argv[4]:"/tmp/attnio/attn_out_0001.bin";
    xrt::device dev(0);
    auto xc=std::make_unique<xrt::xclbin>(xd+"/attn.xclbin"); dev.register_xclbin(*xc);
    auto hc=std::make_unique<xrt::hw_context>(dev, xc->get_uuid());
    npu_app app(device_npu2,&dev,hc.get(),"MLIR_AIE");
    app.load_elf(elf);
    auto bOut=app.create_bo_buffer<uint16_t>(256*2048);
    auto bAct=app.create_bo_buffer<uint16_t>(256*2048);
    auto bKv=app.create_bo_buffer<uint16_t>(33554432/2);
    // load captured act + kv
    { FILE* f=fopen(actf.c_str(),"rb"); if(!f){fprintf(stderr,"no act\n");return 1;} fread(bAct.data(),1,1048576,f); fclose(f); }
    { FILE* f=fopen(kvf.c_str(),"rb"); if(!f){fprintf(stderr,"no kv\n");return 1;} fread(bKv.data(),1,33554432,f); fclose(f); }
    memset(bOut.data(),0,256*2048*2);
    app.safe_run(bOut, bAct, bKv);
    // compare vs captured out
    std::vector<uint16_t> ref(256*2048);
    { FILE* f=fopen(outf.c_str(),"rb"); if(!f){fprintf(stderr,"no out\n");return 1;} fread(ref.data(),1,1048576,f); fclose(f); }
    long match=0, nz=0;
    for(int i=0;i<256*2048;i++){ if(bOut.data()[i]==ref[i]) match++; if(bOut.data()[i]!=0) nz++; }
    printf("match=%ld/524288 (%.4f%%) nz=%ld\n", match, 100.0*match/524288, nz);
    // also correlation
    double s=0,sa=0,sb=0;
    for(int i=0;i<256*2048;i++){float a=bff32(bOut.data()[i]), b=bff32(ref[i]); s+=a*b; sa+=a*a; sb+=b*b;}
    printf("corr=%.6f\n", s/(sqrt(sa)*sqrt(sb)));
    return 0;
}
