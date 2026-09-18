// test_attn3.cpp — minimal: gen_mha_engine_seq(0,256) ALONE (no dlsym send),
// uniform Q=K=V=1.0, run attn.xclbin, check output.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include "npu_utils/npu_utils_xrt.hpp"
#include "models/qwen3/qwen3_npu_sequence.hpp"
#include "lm_config.hpp"
namespace utils { std::string find_xclbin_path() { return "/home/bcloud/amd-oss/fastflowlm/src/xclbins"; } }
static uint16_t f32bf(float f){uint32_t b;memcpy(&b,&f,4);return(uint16_t)((b+0x7FFF+((b>>16)&1))>>16);}
static float bff32(uint16_t v){uint32_t b=v<<16;float f;memcpy(&f,&b,4);return f;}
int main(int argc, char** argv) {
    std::string md = argc>1?argv[1]:"/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2";
    std::string xd = argc>2?argv[2]:"/home/bcloud/amd-oss/fastflowlm/src/xclbins/Qwen3-0.6B-NPU2";
    LM_Config cfg; cfg.from_pretrained(md);
    int NH=cfg.get<u32>("num_attention_heads"), NKV=cfg.get<u32>("num_key_value_heads"), HD=cfg.get<u32>("head_dim");
    int QD=NH*HD;
    xrt::device dev(0);
    auto xc=std::make_unique<xrt::xclbin>(xd+"/attn.xclbin"); dev.register_xclbin(*xc);
    auto hc=std::make_unique<xrt::hw_context>(dev, xc->get_uuid());
    npu_app app(device_npu2,&dev,hc.get(),"MLIR_AIE");
    qwen3_npu_sequence qseq(cfg, 8192);
    qseq.gen_mha_engine_seq(app.seq(), 0, 256);
    app.update_ctrl_seq();
    auto bOut=app.create_bo_buffer<uint16_t>(256*QD);
    auto bAct=app.create_bo_buffer<uint16_t>(256*QD);
    auto bKv=app.create_bo_buffer<uint16_t>(33554432/2);
    for(int i=0;i<256*QD;i++) bAct.data()[i]=f32bf(1.0f);
    for(int i=0;i<33554432/2;i++) bKv.data()[i]=f32bf(1.0f);
    memset(bOut.data(),0,256*QD*2);
    app.safe_run(bOut,bAct,bKv);
    float mn=1e9,mx=-1e9; int nz=0;
    for(int i=0;i<256*QD;i++){float v=bff32(bOut.data()[i]); if(v<mn)mn=v; if(v>mx)mx=v; if(v!=0)nz++;}
    printf("uniform-1.0: out range=[%.4f,%.4f] nz=%d/%d\n", mn,mx,nz,256*QD);
    printf("out[0..8]: "); for(int i=0;i<8;i++) printf("%.4f ",bff32(bOut.data()[i])); printf("\n");
    return 0;
}
