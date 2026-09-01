// dump_reorder_win.cpp — Windows reorder-dumper for the closed FastFlowLM
// q4nx reorder. Links q4_npu_eXpress.dll (import lib q4_npu_eXpress.lib) and
// calls Q4NX::q4nx_dequantize<float>(weight, q, scale, zp, columns) on the
// q_proj tensor of model.q4nx, dumping the q/scale/zp buffers (the layout the
// mm.xclbin kernel's weight BO expects).
//
// Build (mingw):
//   x86_64-w64-mingw32-g++ -O2 -std=c++17 -I <fflm>/src/include \
//       dump_reorder_win.cpp -o dump_reorder_win.exe \
//       <fflm>/src/lib/hrx/q4_npu_eXpress.lib
// Run (Windows guest): dump_reorder_win.exe model.q4nx [outdir]
#include "tensor_utils/q4_npu_eXpress.hpp"
#include "biovault_bfloat16.h"
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <fstream>
#include <cstdlib>
#include <windows.h>

static std::string find_tensor_offset(const std::string& json, const char* name) {
    // crude: locate "name": {...} then "data_offsets": [N, ...]
    size_t p = json.find(name);
    while (p != std::string::npos) {
        // ensure it is a tensor key (preceded by quote)
        if (p > 0 && json[p-1] == '"') break;
        p = json.find(name, p + 1);
    }
    if (p == std::string::npos) return "";
    size_t q = json.find("data_offsets", p);
    if (q == std::string::npos) return "";
    size_t b = json.find('[', q);
    if (b == std::string::npos) return "";
    return json.substr(b + 1, json.find(',', b) - b - 1);
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s model.q4nx [outdir]\n", argv[0]); return 2; }
    const char* model_path = argv[1];
    const char* outdir = argc > 2 ? argv[2] : ".";
    std::ifstream f(model_path, std::ios::binary | std::ios::ate);
    if (!f) { fprintf(stderr, "cannot open %s\n", model_path); return 1; }
    size_t fsz = (size_t)f.tellg();
    f.seekg(0);
    std::vector<uint8_t> file(fsz);
    f.read((char*)file.data(), fsz);
    uint64_t hsz; memcpy(&hsz, file.data(), 8);
    std::string json((const char*)file.data() + 8, hsz);
    std::string offs = find_tensor_offset(json, "model.layers.0.self_attn.q_proj.weight");
    if (offs.empty()) { fprintf(stderr, "q_proj not found\n"); return 1; }
    uint64_t off = strtoull(offs.c_str(), nullptr, 10);
    const uint8_t* data = file.data() + 8 + hsz + off;
    fprintf(stderr, "q_proj tiles at %llu (256 I8 rows x 5120 B)\n", (unsigned long long)(8 + hsz + off));

    // full-tensor counts: 256 tiles x 32 rows x 8 groups
    const size_t SN = 256 * 32 * 8;   // 65536 scale/zp entries
    const size_t WN = SN * 32;        // 2097152 (function demands weight == scale*32)
    const size_t QN = 2048 * 128;     // 262144
    const size_t ZN = SN;               // 65536

    bytes q4nx(const_cast<uint8_t*>(data), 256 * 5120);
    // NON-OWNER (external) buffers: the DLL writes into them, then its
    // bytes::resize() throws for non-owners — the write survives in our
    // vectors when we catch the exception.
    std::vector<float> wv(WN, -999.0f);
    std::vector<uint32_t> qv(QN, 0xDEADBEEFu);
    std::vector<biovault::bfloat16_t> sv(SN);
    std::vector<int32_t> zv(ZN, -12345);
    buffer<float> wb(wv.data(), wv.size());
    buffer<uint32_t> qb(qv.data(), qv.size());
    buffer<biovault::bfloat16_t> sb(sv.data(), sv.size());
    buffer<int32_t> zb(zv.data(), zv.size());
    const float* wd = wv.data();
    const uint32_t* qd = qv.data();
    const auto* sd = sv.data();
    const int32_t* zd = zv.data();
    size_t wn = wv.size(), qn = qv.size(), sn = sv.size(), zn = zv.size();
    // Load the DLL and call the exported MSVC-mangled symbol directly (the
    // .lib import names are MSVC-decorated; Win64 x64 ABI is uniform so a
    // GetProcAddress + typed call is safe).
    HMODULE dll = LoadLibraryA("q4_npu_eXpress.dll");
    if (!dll) { fprintf(stderr, "LoadLibrary q4_npu_eXpress.dll failed (%lu)\n", GetLastError()); return 3; }
    typedef void (__stdcall *deq_fn)(buffer<float>&, buffer<unsigned>&,
                                     buffer<biovault::bfloat16_t>&, buffer<int>&, int);
    const char* sym = "??$q4nx_dequantize@M@Q4NX@@SAXAEAV?$buffer@M@@AEAV?$buffer@I@@AEAV?$buffer@Vbfloat16_t@biovault@@@@AEAV?$buffer@H@@H@Z";
    deq_fn fn = (deq_fn)GetProcAddress(dll, sym);
    if (!fn) { fprintf(stderr, "GetProcAddress failed (%lu)\n", GetLastError()); return 4; }
    fprintf(stderr, "q4nx_dequantize<float>(q,scale,zp) found\n");
    try {
        fn(wb, qb, sb, zb, 1024);
        fprintf(stderr, "call OK\n");
    } catch (const std::exception& e) {
        fprintf(stderr, "exception: %s\n", e.what());
    }
    fprintf(stderr, "buffers: weight=%zu q=%zu scale=%zu zp=%zu (my allocations)\n", wn, qn, sn, zn);
    fprintf(stderr, "final sizes reported: %zu %zu %zu %zu\n",
            wb.size()/4, qb.size()/4, sb.size()/2, zb.size()/4);
    fprintf(stderr, "weight[0:4]: %.4f %.4f %.4f %.4f\n", wd[0], wd[1], wd[2], wd[3]);
    fprintf(stderr, "q[0:8]:      %u %u %u %u %u %u %u %u\n", qd[0], qd[1], qd[2], qd[3], qd[4], qd[5], qd[6], qd[7]);
    fprintf(stderr, "scale[0:4]:  %.4f %.4f %.4f %.4f\n", (float)sd[0], (float)sd[1], (float)sd[2], (float)sd[3]);
    fprintf(stderr, "zp[0:4]:     %d %d %d %d\n", zd[0], zd[1], zd[2], zd[3]);
    auto dump = [&](const char* name, const void* p, size_t n) {
        std::string path = std::string(outdir) + "/" + name;
        std::ofstream o(path, std::ios::binary);
        o.write((const char*)p, n);
        fprintf(stderr, "wrote %s (%zu B)\n", path.c_str(), n);
    };
    dump("reordered_weight.bin", wd, wn * 4);
    dump("reordered_q.bin", qd, qn * 4);
    dump("reordered_scale.bin", sd, sn * 2);
    dump("reordered_zp.bin", zd, zn * 4);
    return 0;
}
