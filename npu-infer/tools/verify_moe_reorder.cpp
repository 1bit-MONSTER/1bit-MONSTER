// verify_moe_reorder.cpp — verify the 35B MoE expert weight-BO reorder
// formula against the runtime's own qwen3_6_reorder_cpy (libqwen3_6_moe_npu.so,
// constprop.2 clone, dtype=8). Round 38: the runtime row layout is
//
//   trimmed_tile[i] = file_tile[i][0:4736]          (drop the last 384 B)
//   out[o] = trimmed[o/2 + 8*(o%2)]  per 16-row block (A/B half interleave)
//
// Verified byte-exact on real up_exps tiles (and again here for the record):
// the block structure matches the layer TXN weight BDs (18944-B reads at
// 75776-B strides == 4 rows / 16-row block).
//
// Build:
//   g++ -O2 -std=c++20 verify_moe_reorder.cpp -o verify_moe_reorder \
//     -L/home/bcloud/amd-oss/fastflowlm/src/lib/xrt -lqwen3_6_moe_npu \
//     -lq4_npu_eXpress -L/usr/local/lib -laiebu -lxrt_coreutil -lxrt_core \
//     -Wl,-rpath,/home/bcloud/amd-oss/fastflowlm/src/lib/xrt -ldl
// Run:
//   ./verify_moe_reorder <model.q4nx> [n_tiles]
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
                             uint64_t* out_off, uint64_t* out_size) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    uint64_t hdr; fread(&hdr, 8, 1, f);
    std::vector<char> js(hdr); fread(js.data(), 1, hdr, f);
    fclose(f);
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

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1]
        : "/home/bcloud/.config/flm/models/Qwen3.6-35B-A3B-NPU2/model.q4nx";
    int n_tiles = argc > 2 ? atoi(argv[2]) : 32;   // 2 blocks of 16

    void* dep = dlopen("/home/bcloud/amd-oss/fastflowlm/src/lib/xrt/libq4_npu_eXpress.so",
                       RTLD_NOW | RTLD_GLOBAL);
    void* h = dlopen("/home/bcloud/amd-oss/fastflowlm/src/lib/xrt/libqwen3_6_moe_npu.so",
                     RTLD_NOW | RTLD_GLOBAL);
    auto* g = (void (*)(void*, void*, unsigned, bool, bool))dlsym(
        h, "_ZN24qwen3_6_moe_npu_sequence13gen_layer_seqEP12npu_sequencejbb");
    uintptr_t base = (uintptr_t)g - 0x97ad0;
    ReorderFn reorder = (ReorderFn)(base + 0x68b80);

    uint64_t off = 0, size = 0;
    if (!read_json_tensor(path, "model.layer.0.mlp.up_exps_proj.weight", &off, &size)) {
        fprintf(stderr, "tensor not found\n");
        return 1;
    }
    // data_base = 8 + json_len; parse again to get it
    FILE* f = fopen(path, "rb");
    uint64_t hdr; fread(&hdr, 8, 1, f);
    uint64_t db = 8 + hdr;
    fclose(f);
    fprintf(stderr, "up_exps off=%llu size=%llu db=%llu n_tiles=%d\n",
            (unsigned long long)off, (unsigned long long)size,
            (unsigned long long)db, n_tiles);

    std::vector<uint8_t> tiles((size_t)n_tiles * 5120);
    f = fopen(path, "rb");
    fseek(f, (long)(db + off), SEEK_SET);
    fread(tiles.data(), 1, tiles.size(), f);
    fclose(f);

    // 1) runtime call on a trimmed 75776-B src (16 rows of 4736)
    std::vector<uint8_t> trimmed(16 * 4736);
    for (int i = 0; i < 16; i++)
        memcpy(trimmed.data() + i * 4736, tiles.data() + i * 5120, 4736);
    std::vector<uint8_t> actual(16 * 4736, 0xEE);
    ByteBuf b; b.vtable = nullptr; b.unk = 0;
    b.data = trimmed.data(); b.cap = trimmed.size();
    reorder(actual.data(), b, 2048, 8, 0);

    // 2) our formula: out[o] = trimmed[o/2 + 8*(o%2)]
    std::vector<uint8_t> expected(16 * 4736);
    for (int o = 0; o < 16; o++) {
        int ti = o / 2 + 8 * (o % 2);
        memcpy(expected.data() + o * 4736, trimmed.data() + ti * 4736, 4736);
    }
    bool match = (actual == expected);
    printf("reorder formula (trim 5120->4736 + A/B interleave): %s\n",
           match ? "PASS — byte-exact vs runtime qwen3_6_reorder_cpy" : "FAIL");

    // 3) block structure vs the layer TXN geometry (18944 = 4 rows,
    //    75776 = 16-row block)
    printf("row 0 == tile 0 [0:4736]:  %s\n",
           memcmp(actual.data(), tiles.data(), 4736) == 0 ? "yes" : "no");
    printf("row 1 == tile 8 [0:4736]:  %s\n",
           memcmp(actual.data() + 4736, tiles.data() + 8 * 5120, 4736) == 0 ? "yes" : "no");
    printf("row 2 == tile 1 [0:4736]:  %s\n",
           memcmp(actual.data() + 9472, tiles.data() + 5120, 4736) == 0 ? "yes" : "no");
    printf("per-expert runtime size check: 32768 x 4736 = %d (desc gate_exps offset delta 155189248)\n",
           32768 * 4736);
    return match ? 0 : 1;
}
