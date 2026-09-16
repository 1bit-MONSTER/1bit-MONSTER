// dump_lin5_weights.cpp — drive the vendor's own weight-prep entry point to
// answer addendum 148's open question: where do the linear-attn conv1d/SSM
// weights (ssm_conv1d/ssm_norm/ssm_a/ssm_dt = 66048 B) actually live, given the
// engine packs them at norms-BO @0..66048 where the layer ELF S2MM-writes its
// scratch output and never reads.
//
// Calls qwen3_6_moe_desc::load_linear_weights(int, Q4NX&,
//   buffer<unsigned char>& pool, buffer<bf16>& b1, buffer<bf16>& b2)
// (exported at _ZN16qwen3_6_moe_desc19load_linear_weightsEiR4Q4NXR6bufferIhERS2_IN8biovault10bfloat16_tEES8_)
// and dumps b1/b2 + a pool head, so npu_pack_moe_linear5_bo can be diffed
// against the vendor's actual layout byte-for-byte.
//
// Build (mirrors gen_layer_elfs_moe.cpp):
//   g++ -O2 -std=c++20 -include climits dump_lin5_weights.cpp -o dump_lin5_weights \
//     -I/home/bcloud/amd-oss/fastflowlm/src/include \
//     -I/home/bcloud/amd-oss/fastflowlm/src/include/npu_utils \
//     -L/home/bcloud/amd-oss/fastflowlm/src/lib/xrt \
//     -lqwen3_6_moe_npu -lq4_npu_eXpress -lgemm -ldequant -lmha -llm_head \
//     -L/usr/local/lib -laiebu -lxrt_coreutil -lxrt_core \
//     -Wl,-rpath,/home/bcloud/amd-oss/fastflowlm/src/lib/xrt \
//     /home/bcloud/amd-oss/fastflowlm/src/common/utils.cpp \
//     -DCMAKE_INSTALL_PREFIX="/home/bcloud/amd-oss/fastflowlm" \
//     -DCMAKE_XCLBIN_PREFIX="/home/bcloud/amd-oss/fastflowlm/src/xclbins"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include "npu_utils/npu_instr_utils.hpp"
#include "lm_config.hpp"
#include "tensor_utils/q4_npu_eXpress.hpp"   // Q4NX
#include "buffer.hpp"                        // buffer<T>
#include "typedef.hpp"                       // bf16 = biovault::bfloat16_t
#include "device_runtime.hpp"                // flm_rt = xrt (device-backed buffers)

// Minimal utils::find_xclbin_path stub (build-only; the dump never loads an xclbin).
namespace utils {
std::string find_xclbin_path() {
    const char* env = std::getenv("FLM_XCLBIN_PATH");
    if (env && *env) return std::string(env);
    return "/home/bcloud/.local/flm-v0946";
}
}

// ---- opaque ABI decl for the binary-only qwen3_6_moe_desc ----
struct qwen3_6_moe_desc {
    unsigned char _pad[0x2000];
    void build(LM_Config& cfg);
    // _ZN16qwen3_6_moe_desc19load_linear_weightsEiR4Q4NXR6bufferIhERS2_IN8biovault10bfloat16_tEES8_
    void load_linear_weights(int L, Q4NX& q, buffer<unsigned char>& pool,
                             buffer<bf16>& b1, buffer<bf16>& b2);
};

static int dump(const char* path, const void* p, size_t n) {
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 1; }
    size_t w = fwrite(p, 1, n, f);
    fclose(f);
    fprintf(stderr, "wrote %s (%zu/%zu)\n", path, w, n);
    return w == n ? 0 : 1;
}

int main(int argc, char** argv) {
    std::string model_dir = (argc > 1) ? argv[1]
        : "/home/bcloud/.config/flm/models/Qwen3.6-35B-A3B-NPU2";
    std::string outdir = (argc > 2) ? argv[2] : "/tmp/lin5dump";
    int L = (argc > 3) ? atoi(argv[3]) : 1;

    LM_Config config;
    config.from_pretrained(model_dir);
    fprintf(stderr, "config loaded; building desc...\n");

    const size_t DESC_CAP = 0x20000;
    void* desc_mem = malloc(DESC_CAP);
    if (!desc_mem) { fprintf(stderr, "desc malloc failed\n"); return 1; }
    memset(desc_mem, 0, DESC_CAP);
    qwen3_6_moe_desc* desc = new (desc_mem) qwen3_6_moe_desc;
    desc->build(config);
    fprintf(stderr, "desc built\n");

    Q4NX q4nx(model_dir);   // model DIRECTORY (Q4NX resolves model.q4nx/safetensors inside)
    fprintf(stderr, "Q4NX loaded from %s\n", model_dir.c_str());

    // load_linear_weights syncs its buffers to device at the end, so they must
    // be device-backed. data() is the host mapping and stays valid after the
    // call (sync_to_device pushes host->device but leaves the host copy).
    flm_rt::device dev(0);

    // pool = 512 MB expert pool; b1/b2 = the two 5 MB linear-attn BOs.
    buffer<unsigned char> pool(dev, 536870912ull);
    buffer<bf16> b1(dev, 2621440);   // 5 MB / 2
    buffer<bf16> b2(dev, 2621440);
    fprintf(stderr, "buffers ready; calling load_linear_weights(L=%d)...\n", L);

    desc->load_linear_weights(L, q4nx, pool, b1, b2);
    fprintf(stderr, "load_linear_weights returned\n");

    char p[512];
    snprintf(p, sizeof(p), "%s/lin5_b1_L%d.bin", outdir.c_str(), L);
    dump(p, b1.data(), 5242880);
    snprintf(p, sizeof(p), "%s/lin5_b2_L%d.bin", outdir.c_str(), L);
    dump(p, b2.data(), 5242880);
    snprintf(p, sizeof(p), "%s/pool_L%d_full.bin", outdir.c_str(), L);
    dump(p, pool.data(), 536870912ull);   // full 512 MB pool

    free(desc_mem);
    return 0;
}
