// test_attn2.cpp — full-pipeline attention harness: replicate FLM's runtime
// sequence generation (Impl::_send_rope_rms_weights + _send_rms_weights +
// gen_mha_engine_seq) then run attn.xclbin via npu_app.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>
#include <dlfcn.h>
#include "npu_utils/npu_utils_xrt.hpp"
#include "models/qwen3/qwen3_npu_sequence.hpp"
#include "lm_config.hpp"

namespace utils { std::string find_xclbin_path() { return "/home/bcloud/amd-oss/fastflowlm/src/xclbins"; } }

static uint16_t f32bf(float f){uint32_t b;memcpy(&b,&f,4);return(uint16_t)((b+0x7FFF+((b>>16)&1))>>16);}
static float bff32(uint16_t v){uint32_t b=v<<16;float f;memcpy(&f,&b,4);return f;}

int main(int argc, char** argv) {
    std::string md = argc>1?argv[1]:"/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2";
    std::string xd = argc>2?argv[2]:"/home/bcloud/amd-oss/fastflowlm/src/xclbins/Qwen3-0.6B-NPU2";
    int npt = argc>3?atoi(argv[3]):256;
    LM_Config cfg; cfg.from_pretrained(md);
    int NH = cfg.get<u32>("num_attention_heads");
    int NKV = cfg.get<u32>("num_key_value_heads");
    int HD = cfg.get<u32>("head_dim");
    int QD = NH*HD;
    fprintf(stderr, "NH=%d NKV=%d HD=%d QD=%d npt=%d\n", NH, NKV, HD, QD, npt);

    // dlsym private Impl methods (exact FLM runtime pipeline)
    auto* h = dlopen("/home/bcloud/amd-oss/fastflowlm/src/lib/xrt/libqwen3_npu.so", RTLD_LAZY|RTLD_GLOBAL);
    auto sym = [&](const char* n){ void* s = dlsym(RTLD_DEFAULT, n); if(!s) fprintf(stderr,"MISS %s\n", n); return s; };
    auto send_rope_rms = (void(*)(void*,void*))sym("_ZN18qwen3_npu_sequence4Impl22_send_rope_rms_weightsEP12npu_sequence");
    auto send_rms      = (void(*)(void*,void*))sym("_ZN18qwen3_npu_sequence4Impl17_send_rms_weightsEP12npu_sequence");
    if (!send_rope_rms || !send_rms) { fprintf(stderr,"symbols missing\n"); return 1; }

    xrt::device dev(0);
    auto xc = std::make_unique<xrt::xclbin>(xd + "/attn.xclbin");
    dev.register_xclbin(*xc);
    auto hc = std::make_unique<xrt::hw_context>(dev, xc->get_uuid());
    npu_app app(device_npu2, &dev, hc.get(), "MLIR_AIE");

    qwen3_npu_sequence qseq(cfg, 8192);
    void* impl = *(void**)&qseq;
    fprintf(stderr, "impl=%p\n", impl);

    send_rope_rms(impl, app.seq());
    send_rms(impl, app.seq());
    qseq.gen_mha_engine_seq(app.seq(), 0, (uint32_t)npt);
    app.update_ctrl_seq();

    auto bOut = app.create_bo_buffer<uint16_t>(256*QD);
    auto bAct = app.create_bo_buffer<uint16_t>(256*QD);
    auto bKv  = app.create_bo_buffer<uint16_t>(33554432/2);

    for (int t=0;t<256;t++) for (int h=0;h<NH;h++) for (int d=0;d<HD;d++)
        bAct.data()[t*QD + h*HD + d] = f32bf((float)(t + h + d) * 0.001f);
    int regionBytes = 8*1024*1024;
    for (int t=0;t<256;t++) for (int r=0;r<4;r++) for (int h=0;h<4;h++) for (int d=0;d<HD;d++) {
        uint16_t* base = bKv.data() + (r*regionBytes)/2 + t*512 + h*HD + d;
        *base = f32bf((r<2) ? (float)(t+d)*0.001f : (float)(t+d)*0.002f);
    }
    memset(bOut.data(), 0, 256*QD*2);

    app.safe_run(bOut, bAct, bKv);

    printf("out[0..15]:");
    for (int i=0;i<16;i++) printf(" %.4f", bff32(bOut.data()[i]));
    printf("\n");
    float mn=1e9, mx=-1e9; int nz=0;
    for (int i=0;i<256*QD;i++){float v=bff32(bOut.data()[i]); if(v<mn)mn=v; if(v>mx)mx=v; if(v!=0)nz++;}
    printf("out range=[%.4f, %.4f] nz=%d/%d\n", mn, mx, nz, 256*QD);
    return 0;
}
