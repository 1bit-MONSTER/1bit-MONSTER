// verify_moe_reorder_bf16.cpp — call the vendor's qwen3_6_reorder_cpy on the
// bf16/f32 linear-attn tensors (alpha/beta/conv1d/iln/paln/sg/router/norm/a/dt)
// at DIFFERENT dtype codes, and search each output in the load_linear_weights
// dumps (b1/b2). Goal: identify the reorder the vendor applies so
// npu_pack_moe_linear5_bo / npu_pack_moe_router_bo can be corrected.
//
// Build (same as verify_moe_reorder_qkv.cpp):
//   g++ -O2 -std=c++20 verify_moe_reorder_bf16.cpp -o verify_moe_reorder_bf16 \
//     -L/home/bcloud/amd-oss/fastflowlm/src/lib/xrt -lqwen3_6_moe_npu \
//     -lq4_npu_eXpress -L/usr/local/lib -laiebu -lxrt_coreutil -lxrt_core \
//     -Wl,-rpath,/home/bcloud/amd-oss/fastflowlm/src/lib/xrt -ldl
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <dlfcn.h>
#include <vector>
#include <string>

struct ByteBuf { void* vtable; uint64_t unk; uint8_t* data; size_t cap; };
using ReorderFn = void (*)(uint8_t*, ByteBuf&, int, int, int);

static bool read_json_tensor(const char* path, const char* key,
                             uint64_t* out_off, uint64_t* out_size, uint64_t* out_db) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    uint64_t hdr; fread(&hdr, 8, 1, f);
    std::vector<char> js(hdr); fread(js.data(), 1, hdr, f);
    fclose(f);
    *out_db = 8 + hdr;
    std::string s(js.data(), js.size());
    std::string needle = "\"" + std::string(key) + "\"";
    size_t kp = s.find(needle);
    if (kp == std::string::npos) return false;
    const char* doff = strstr(s.c_str() + kp, "\"data_offsets\"");
    if (!doff) return false;
    const char* br = strchr(doff, '[');
    if (!br) return false;
    *out_off = strtoull(br + 1, nullptr, 10);
    const char* comma = strchr(br + 1, ',');
    if (comma) *out_size = strtoull(comma + 1, nullptr, 10) - *out_off;
    return true;
}

static std::vector<uint8_t> read_all(const char* path) {
    FILE* f = fopen(path, "rb");
    std::vector<uint8_t> v;
    if (!f) return v;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    v.resize(sz); fread(v.data(), 1, sz, f); fclose(f);
    return v;
}

// search `needle` (the first `use` bytes of reordered output) in buf; return offset or -1
static long find_in(const std::vector<uint8_t>& buf, const std::vector<uint8_t>& needle) {
    if (needle.empty() || needle.size() > buf.size()) return -1;
    for (size_t i = 0; i + needle.size() <= buf.size(); i++) {
        if (memcmp(&buf[i], needle.data(), needle.size()) == 0) return (long)i;
    }
    return -1;
}

int main(int argc, char** argv) {
    const char* model = argc > 1 ? argv[1]
        : "/home/bcloud/.config/flm/models/Qwen3.6-35B-A3B-NPU2/model.q4nx";
    const char* b1path = argc > 2 ? argv[2] : "/tmp/lin5dump/lin5_b1_L1.bin";
    const char* b2path = argc > 3 ? argv[3] : "/tmp/lin5dump/lin5_b2_L1.bin";

    void* dep = dlopen("/home/bcloud/amd-oss/fastflowlm/src/lib/xrt/libq4_npu_eXpress.so",
                       RTLD_NOW | RTLD_GLOBAL);
    void* h = dlopen("/home/bcloud/amd-oss/fastflowlm/src/lib/xrt/libqwen3_6_moe_npu.so",
                     RTLD_NOW | RTLD_GLOBAL);
    if (!dep || !h) { fprintf(stderr, "dlopen failed\n"); return 1; }
    auto* g = (void (*)(void*, void*, unsigned, bool, bool))dlsym(
        h, "_ZN24qwen3_6_moe_npu_sequence13gen_layer_seqEP12npu_sequencejbb");
    if (!g) { fprintf(stderr, "dlsym gen_layer_seq failed\n"); return 1; }
    uintptr_t base = (uintptr_t)g - 0x97ad0;
    ReorderFn reorder = (ReorderFn)(base + 0x68b80);

    std::vector<uint8_t> b1 = read_all(b1path), b2 = read_all(b2path);
    fprintf(stderr, "b1=%zu B b2=%zu B\n", b1.size(), b2.size());

    struct T { const char* key; int n; int flag; int dtype; };
    T tens[] = {
        {"model.layer.1.linear_attn.ssm_alpha_proj.weight", 2048, 64, 2},  // bf16 [2048,32]
        {"model.layer.1.linear_attn.ssm_beta_proj.weight",  2048, 64, 2},
        {"model.layer.1.linear_attn.ssm_conv1d.weight",     4,    64, 2},  // bf16 [4,8192]
        {"model.layer.1.input_layernorm.weight",            2048, 64, 2},  // bf16 [2048]
        {"model.layer.1.post_attention_layernorm.weight",   2048, 64, 2},
        {"model.layer.1.shared_expert_gate.weight",         2048, 64, 2},
        {"model.layer.1.moe_router.weight",                 2048, 64, 2},  // bf16 [2048,256]
    };

    for (auto& t : tens) {
        uint64_t off, size, db;
        if (!read_json_tensor(model, t.key, &off, &size, &db)) {
            fprintf(stderr, "%s: NOT FOUND\n", t.key); continue;
        }
        std::vector<uint8_t> src(size);
        FILE* f = fopen(model, "rb");
        fseek(f, (long)(db + off), SEEK_SET);
        size_t br = fread(src.data(), 1, size, f); fclose(f);
        if (br != size) { fprintf(stderr, "%s: short read\n", t.key); continue; }

        // dst cap: generous (n rows x width + pad). width = size/n bytes.
        size_t width = size / t.n;
        size_t dst_cap = size + 65536;
        std::vector<uint8_t> dst(dst_cap, 0xEE);
        ByteBuf bb; bb.vtable = nullptr; bb.unk = 0; bb.data = src.data(); bb.cap = size;
        reorder(dst.data(), bb, t.n, t.dtype, t.flag);

        // find first non-0xEE to know written extent
        size_t written = 0;
        for (size_t i = 0; i < dst_cap; i++) if (dst[i] != 0xEE) { written = i; break; }
        long i1 = find_in(b1, dst), i2 = find_in(b2, dst);
        // also search a prefix in case only part matches
        std::vector<uint8_t> prefix(dst.begin(), dst.begin() + (written > 0 ? written : dst_cap));
        long p1 = find_in(b1, prefix), p2 = find_in(b2, prefix);
        fprintf(stderr, "%s (n=%d dtype=%d flag=%d size=%zu width=%zu written=%zu): raw in b1=%ld b2=%ld; reordered full b1=%ld b2=%ld\n",
                t.key, t.n, t.dtype, t.flag, size, width, written, i1, i2, p1, p2);
        // also check if raw src itself is in b1/b2
        long r1 = find_in(b1, src), r2 = find_in(b2, src);
        fprintf(stderr, "    RAW src in b1=%ld b2=%ld\n", r1, r2);
    }
    return 0;
}
