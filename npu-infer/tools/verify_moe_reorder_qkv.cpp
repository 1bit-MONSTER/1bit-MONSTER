// verify_moe_reorder_qkv.cpp — probe qwen3_6_reorder_cpy (constprop.2 clone,
// dtype=8) on the 8704-row linear-attn tensors (qkv / ssm_out / share_* /
// gate_proj) with the args captured from the live load (R47 gdb):
//   qkv        n=2048  flag=64
//   ssm_out    n=2048  flag=64
//   share_*    n=512   flag=64
//   gate_proj  n=4096  flag=64
//
// Goal: determine whether reorder_cpy produces the region-B weight-BO content
// (the A/B interleave of 4736-trimmed rows) for the 8704-row tensors, which
// would unblock the layer ELF's arg-0 weight BO packing without the runtime.
//
// Build (same as verify_moe_reorder.cpp):
//   g++ -O2 -std=c++20 verify_moe_reorder_qkv.cpp -o verify_moe_reorder_qkv \
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
                             uint64_t* out_off, uint64_t* out_size,
                             uint64_t* out_db) {
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

static int run_one(ReorderFn reorder, const char* path, const char* key,
                   int n, int flag, const char* outfile) {
    uint64_t off, size, db;
    if (!read_json_tensor(path, key, &off, &size, &db)) {
        fprintf(stderr, "%s: tensor not found\n", key);
        return -1;
    }
    fprintf(stderr, "%s: size=%llu n=%d flag=%d\n", key,
            (unsigned long long)size, n, flag);
    std::vector<uint8_t> src(size);
    FILE* f = fopen(path, "rb");
    fseek(f, (long)(db + off), SEEK_SET);
    size_t br = fread(src.data(), 1, size, f);
    fclose(f);
    if (br != size) { fprintf(stderr, "%s: short read %zu\n", key, br); return -1; }

    // dst: n rows x 4736 (the dtype=8 elsize). Allocate generously.
    size_t dst_cap = (size_t)n * 4736 + 65536;
    std::vector<uint8_t> dst(dst_cap, 0xEE);
    ByteBuf b; b.vtable = nullptr; b.unk = 0;
    b.data = src.data(); b.cap = src.size();
    reorder(dst.data(), b, n, 8, flag);

    // how much of dst was written (first non-0xEE byte marks the end)?
    size_t written = dst_cap;
    for (size_t i = 0; i < dst_cap; i++) {
        bool all_ee = true;
        for (size_t k = 0; k < 16 && i + k < dst_cap; k++)
            if (dst[i+k] != 0xEE) { all_ee = false; break; }
        if (!all_ee) { written = i; break; }
    }
    // write the first 1 MB (or dst_cap) to outfile
    FILE* of = fopen(outfile, "wb");
    if (of) { fwrite(dst.data(), 1, dst_cap < (1u<<20) ? dst_cap : (1u<<20), of); fclose(of); }
    fprintf(stderr, "%s: first non-0xEE at byte %zu; wrote %s\n", key, written, outfile);
    return 0;
}

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1]
        : "/home/bcloud/.config/flm/models/Qwen3.6-35B-A3B-NPU2/model.q4nx";

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

    struct { const char* key; int n; int flag; const char* out; } tens[] = {
        {"model.layer.0.linear_attn.qkv_proj.weight", 2048, 64, "/tmp/reorder_qkv.bin"},
        {"model.layer.0.linear_attn.ssm_out_proj.weight", 2048, 64, "/tmp/reorder_ssmout.bin"},
        {"model.layer.0.mlp.share_up_exps_proj.weight", 512, 64, "/tmp/reorder_shareup.bin"},
        {"model.layer.0.self_attn.gate_proj.weight", 4096, 64, "/tmp/reorder_gateproj.bin"},
    };
    for (auto& t : tens) {
        fprintf(stderr, "\n=== %s (n=%d flag=%d) ===\n", t.key, t.n, t.flag);
        int rc = run_one(reorder, path, t.key, t.n, t.flag, t.out);
        fprintf(stderr, "rc=%d\n", rc);
    }
    return 0;
}
