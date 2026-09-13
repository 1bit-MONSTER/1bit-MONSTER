// npu_q4nx_layout_probe.cpp — HOST-ONLY probe (no NPU, no device): for each
// artifact, what the pre-serve check in src/backend_npu.cpp concludes, and how
// many layers the old 64 KB find_offset window could even see.
// Temporary diagnostic; not wired into any build.
//
//   g++ -std=c++17 -Iinclude -Isrc -O2 Testing/npu_q4nx_layout_probe.cpp \
//       src/q4nx_reader.cpp -o /tmp/layout_probe
#include "q4nx_reader.h"
#include "npu_key_contract.h"
#include <cstdio>
#include <cstdlib>
#include <map>
#include <set>
#include <string>
#include <vector>

static const size_t OLD_WINDOW = 65536;

int main(int argc, char** argv) {
    for (int a = 1; a < argc; a++) {
        Q4nxReader r;
        if (!r.open(argv[a])) { std::fprintf(stderr, "open failed: %s\n", argv[a]); continue; }
        const size_t hdr = r.data_start >= 8 ? r.data_start - 8 : 0;
        std::string head(r.data + 8, hdr);
        const std::string mt = r.model_type();

        // Layer count: the declared one, else the highest model.layers.N. seen.
        int layers = 0;
        size_t p = head.find("\"num_hidden_layers\"");
        if (p != std::string::npos) layers = std::atoi(head.c_str() + head.find(':', p) + 1);
        std::map<std::string, int> suffix;
        for (size_t i = 0; (i = head.find("\"model.layers.", i)) != std::string::npos; i++) {
            size_t dot = head.find('.', i + 15);
            size_t q = head.find('"', i + 1);
            if (dot == std::string::npos || q == std::string::npos || q < dot) continue;
            int l = std::atoi(head.c_str() + i + 14);
            if (l + 1 > layers) layers = l + 1;
            suffix[head.substr(dot + 1, q - dot - 1)]++;
        }

        std::printf("=== %s\n", argv[a]);
        std::printf("    header %zu B%s   layers %d   model_type '%s'\n", hdr,
                    hdr > OLD_WINDOW ? "  <-- header over the old 64 KB window" : "",
                    layers, mt.c_str());

        const NpuKeyContract c = npu_contract_for(mt);
        // How many layers had their contract keys outside the old window?
        int invisible = 0;
        for (int l = 0; l < layers && c.layout; l++) {
            for (const char* s : c.suffixes) {
                char key[256];
                std::snprintf(key, sizeof(key), "model.layers.%d.%s", l, s);
                std::string q = std::string("\"") + key + "\"";
                size_t pos = head.find(q);
                if (pos != std::string::npos && 8 + pos >= OLD_WINDOW) { invisible++; break; }
            }
        }
        std::printf("    layers the OLD window could not see: %d/%d\n", invisible, layers);

        const NpuKeyCheck ck = npu_verify_layer_keys(r, layers, c);
        if (!c.layout)
            std::printf("    NEW: 'no GEMM key contract for family \"%s\"' (skipped, not faked)\n",
                        mt.c_str());
        else if (ck.status == NpuKeyCheck::Verified)
            std::printf("    NEW: verified %d layers (layout: %s)\n", layers, c.layout);
        else if (ck.status == NpuKeyCheck::MissingKey)
            std::printf("    NEW: 'GEMM weights missing' — layer %d, '%s' (layout: %s)\n",
                        ck.miss_layer, ck.miss_key.c_str(), c.layout);
        else
            std::printf("    NEW: 'names this check does not know' — check skipped, not called "
                        "missing (layout: %s)\n", c.layout);

        if (ck.status == NpuKeyCheck::MissingKey && c.layout) {
            std::printf("    vocabulary (layer suffixes, first 10):\n");
            int n = 0;
            for (const auto& kv : suffix) {
                if (++n > 10) break;
                std::printf("      %4d  %s\n", kv.second, kv.first.c_str());
            }
        }
        r.close();
    }
    return 0;
}
