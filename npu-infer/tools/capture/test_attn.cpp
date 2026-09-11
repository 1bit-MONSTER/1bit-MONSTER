// test_attn.cpp — standalone harness: run FLM's attn.xclbin via npu_app +
// qwen3_npu_sequence::gen_mha_engine_seq with synthetic Q/K/V, and dump the
// output. This pins the exact act/kv/out BO layout + whether the kernel
// applies q_norm/k_norm/RoPE internally.
//
// Build (same libs as run_qwen3_prefill + qwen3_npu):
//   g++ -O2 -std=c++17 -include climits test_attn.cpp -o test_attn \
//     -I/home/bcloud/amd-oss/fastflowlm/src/include \
//     -I/home/bcloud/amd-oss/fastflowlm/src/include/npu_utils \
//     -L/home/bcloud/amd-oss/fastflowlm/src/lib/xrt -lqwen3_npu -lgemm -lmha -lq4_npu_eXpress \
//     -L/usr/local/lib -laiebu -lxrt_coreutil -lxrt_core \
//     -Wl,-rpath,/home/bcloud/amd-oss/fastflowlm/src/lib/xrt
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
    int npt = argc>3?atoi(argv[3]):2;
    LM_Config cfg; cfg.from_pretrained(md);
    int H = cfg.get<u32>("hidden_size");
    int NH = cfg.get<u32>("num_attention_heads");
    int NKV = cfg.get<u32>("num_key_value_heads");
    int HD = cfg.get<u32>("head_dim");
    int QD = NH*HD, KVD = NKV*HD;
    fprintf(stderr, "H=%d NH=%d NKV=%d HD=%d QD=%d KVD=%d npt=%d\n", H, NH, NKV, HD, QD, KVD, npt);

    xrt::device dev(0);
    // attn.xclbin via npu_app
    auto xc = std::make_unique<xrt::xclbin>(xd + "/attn.xclbin");
    dev.register_xclbin(*xc);
    auto hc = std::make_unique<xrt::hw_context>(dev, xc->get_uuid());
    npu_app app(device_npu2, &dev, hc.get(), "MLIR_AIE");

    // MHA sequence (MAX_L=8192 -> 4x8MB regions in the 32MB kv BO)
    qwen3_npu_sequence qseq(cfg, 8192);
    qseq.gen_mha_engine_seq(app.seq(), 0, (uint32_t)npt);
    app.update_ctrl_seq();

    // BOs: out (1MB), act (1MB = Q 256x2048 bf16), kv (32MB)
    auto bOut = app.create_bo_buffer<uint16_t>(256*QD);
    auto bAct = app.create_bo_buffer<uint16_t>(256*QD);
    auto bKv  = app.create_bo_buffer<uint16_t>(33554432/2);  // 32MB as bf16 elems

    // Q: token t, head h, dim d sentinel (small, no overflow)
    for (int t=0;t<256;t++) for (int h=0;h<NH;h++) for (int d=0;d<HD;d++)
        bAct.data()[t*QD + h*HD + d] = f32bf((float)(t + h + d) * 0.001f);
    // K/V: token t in region r (K0-3,K4-7,V0-3,V4-7), 4 heads x 128 dims per token
    int regionBytes = 8*1024*1024; // 8MB per region
    for (int t=0;t<npt;t++) for (int r=0;r<4;r++) for (int h=0;h<4;h++) for (int d=0;d<HD;d++) {
        uint16_t* base = bKv.data() + (r*regionBytes)/2 + t*512 + h*HD + d;
        *base = f32bf((r<2) ? (float)(t+d)*0.001f : (float)(t+d)*0.002f);
    }
    memset(bOut.data(), 0, 256*QD*2);

    app.safe_run(bOut, bAct, bKv);

    // dump output
    printf("out[0..31]:");
    for (int i=0;i<32;i++) printf(" %.4f", bff32(bOut.data()[i]));
    printf("\n");
    float mn=1e9, mx=-1e9; int nz=0;
    for (int i=0;i<256*QD;i++){float v=bff32(bOut.data()[i]); if(v<mn)mn=v; if(v>mx)mx=v; if(v!=0)nz++;}
    printf("out range=[%.4f, %.4f] nz=%d/%d\n", mn, mx, nz, 256*QD);
    return 0;
}
