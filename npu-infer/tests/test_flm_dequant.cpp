// test_flm_dequant.cpp — call FLM's own Q4NX dequant via dlopen, verify vs original weights.
#include "buffer.hpp"
#include <dlfcn.h>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <vector>

static float bf16_to_f(uint16_t u) { uint32_t b = (uint32_t)u << 16; float f; memcpy(&f, &b, 4); return f; }

// Q4NX::q4nx_dequantize<biovault::bfloat16_t>(bytes& in, bytes& out, int n)
typedef void (*dequant_fn)(bytes&, bytes&, int);

int main(int argc, char** argv) {
    const char* lib = "/opt/fastflowlm/lib/libq4_npu_eXpress.so";
    void* h = dlopen(lib, RTLD_LAZY | RTLD_GLOBAL);
    if (!h) { fprintf(stderr, "dlopen failed: %s\n", dlerror()); return 1; }
    const char* sym = "_ZN4Q4NX15q4nx_dequantizeIN8biovault10bfloat16_tEEEvR5bytesS4_i";
    dequant_fn deq = (dequant_fn)dlsym(h, sym);
    if (!deq) { fprintf(stderr, "dlsym failed: %s\n", dlerror()); return 1; }
    fprintf(stderr, "dequant fn at %p\n", (void*)deq);

    // Load q_proj layer0 raw bytes from the model
    const char* model = argc > 1 ? argv[1] : "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2/model.q4nx";
    FILE* f = fopen(model, "rb");
    uint64_t hs; fread(&hs, 8, 1, f);
    std::vector<uint8_t> hdr(hs); fread(hdr.data(), 1, hs, f);
    // find q_proj offset in JSON (simple search)
    std::string js((char*)hdr.data(), hs);
    const char* key = "\"model.layers.0.self_attn.q_proj.weight\"";
    auto pos = js.find(key);
    fprintf(stderr, "q_proj key pos: %zu\n", pos);
    // crude: search data_offsets after the key
    auto offpos = js.find("data_offsets", pos);
    auto lb = js.find("[", offpos);
    auto rb = js.find("]", lb);
    uint64_t o0 = strtoull(js.substr(lb+1, rb-lb-1).c_str(), nullptr, 10);
    fprintf(stderr, "q_proj data_offsets[0] = %llu\n", (unsigned long long)o0);
    uint64_t base = 8 + hs;
    fseek(f, base + o0, SEEK_SET);
    // q_proj is [256,5120] = 1310720 bytes
    std::vector<uint8_t> raw(1310720);
    fread(raw.data(), 1, raw.size(), f);
    fclose(f);

    // Try different n values; output should be bf16
    for (int n : {1310720, 2097152, 2048, 8192, 65536}) {
        bytes in(raw.data(), raw.size());
        bytes out;  // function resizes
        try {
            deq(in, out, n);
        } catch (const std::exception& e) {
            fprintf(stderr, "n=%d threw: %s\n", n, e.what());
            continue;
        }
        const uint16_t* bf = (const uint16_t*)out.data();
        size_t cnt = out.size() / 2;
        fprintf(stderr, "n=%d -> out %zu bytes (%zu bf16 vals); first8: %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",
                n, out.size(), cnt, bf16_to_f(bf[0]), bf16_to_f(bf[1]), bf16_to_f(bf[2]), bf16_to_f(bf[3]),
                bf16_to_f(bf[4]), bf16_to_f(bf[5]), bf16_to_f(bf[6]), bf16_to_f(bf[7]));
    }
    dlclose(h);
    return 0;
}
