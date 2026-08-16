// dedup_loader_check.cpp — verify a v4 1BP dedup alias resolves correctly.
// Expects a .1bp produced from Testing/make_mini_gguf.py's GGUF:
//   blk.0.attn_q.weight == blk.1.attn_q.weight (alias, byte-identical),
//   blk.2.attn_q.weight different. Both loaders are exercised:
//   NpuOnebpModel (engine) and OnebpModel (onebp_model.cpp).
//
// Run: g++ -std=c++17 -Iinclude -Isrc -Iengine/npu/src \
//        Testing/dedup_loader_check.cpp src/onebp_model.cpp -o /tmp/dc && /tmp/dc mini.1bp
#include <cstdio>
#include <cstring>
#include <vector>
#include <string>
#include "onebp_format.h"
#include "../engine/npu/src/onebp_loader.cpp"  // NpuOnebpModel
#include "../include/onebp_loader.h"  // OnebpModel (other loader)

static int checks = 0, fails = 0;
#define CHECK(cond, msg) do { ++checks; if (!(cond)) { std::printf("FAIL %s\n", msg); ++fails; } } while (0)

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "/tmp/mini.1bp";

    // ── Loader 1: NpuOnebpModel (engine/npu unity loader) ──
    NpuOnebpModel m;
    CHECK(m.open(path), "NpuOnebpModel open");
    std::vector<float> a, b, c;
    CHECK(m.get_tensor_f32("blk.0.attn_q.weight", a) && a.size() == 256 * 64, "blk.0 loads");
    CHECK(m.get_tensor_f32("blk.1.attn_q.weight", b) && b.size() == a.size(), "blk.1 (alias) loads");
    CHECK(m.get_tensor_f32("blk.2.attn_q.weight", c) && c.size() == a.size(), "blk.2 loads");
    int same01 = 0, same02 = 0;
    for (size_t i = 0; i < a.size(); i++) { same01 += (a[i] == b[i]); same02 += (a[i] == c[i]); }
    CHECK(same01 == (int)a.size(), "blk.0 == blk.1 (alias exact)");
    CHECK(same02 < (int)a.size() / 10, "blk.2 differs from blk.0 (not aliased)");

    // ── Loader 2: OnebpModel (onebp_model.cpp, used by vision/tq2_to_q4nx) ──
    {
        OnebpModel m2;
        CHECK(m2.load(path), "OnebpModel open");
        const OnebpTensor* ta = nullptr, *tb = nullptr;
        for (auto& t : m2.tensors) {
            if (t.name == "blk.0.attn_q.weight") ta = &t;
            if (t.name == "blk.1.attn_q.weight") tb = &t;
        }
        CHECK(ta && tb, "both tensors present in OnebpModel index");
        CHECK(ta && tb && ta->offset == tb->offset && ta->bytes == tb->bytes && ta->bytes > 0,
              "OnebpModel alias resolves to same (offset, bytes)");
    }

    std::printf("dedup_loader_check: %d checks, %d fails\n", checks, fails);
    return fails ? 1 : 0;
}
