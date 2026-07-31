// vit_dump.cpp — dump mage_vit_forward inputs/outputs for numeric validation
// against the torch reference (tools/vit_check_reference.py).
//
// Usage:
//   vit_dump <Mage-ViT.1bp> <image.jpg> <pixels.bin> <embeds.bin>   full forward
//   vit_dump <Mage-ViT.1bp> <image.jpg> <pixels.bin> <embeds.bin> t tower only
//   vit_dump <Mage-ViT.1bp> <image.jpg> <pixels.bin> <dir>        L per-layer dumps
#include "vision_encoder.h"
#include "vl_processor.h"
#include "onebp_loader.h"
#include <cstdio>
#include <vector>

static const int IMG = 224;
static const float MEAN[3] = {0.48145467f, 0.45782750f, 0.40821072f};
static const float STD[3]  = {0.26862954f, 0.26130258f, 0.27577711f};

static void dump_bin(const char* path, const float* data, size_t n) {
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "cannot write %s\n", path); exit(1); }
    fwrite(data, sizeof(float), n, f);
    fclose(f);
}

#include <sys/stat.h>
// Run forward on every layer-count prefix 1..24 (validates per-layer numerics).
static void dump_per_layer(VisionWeights& vw, const float* px, const char* dir) {
    mkdir(dir, 0755);
    for (int k = 1; k <= 24; k++) {
        VisionWeights w2 = vw;
        w2.config.num_layers = k;
        w2.layers.resize(k);
        w2.mm0_w.clear(); w2.mm1_w.clear(); w2.mm2_w.clear();
        w2.mm0_b.clear(); w2.mm1_b.clear(); w2.mm2_b.clear();
        std::vector<float> e = mage_vit_forward(w2, px, 3, 1, IMG, IMG, 4);
        char p[512]; snprintf(p, sizeof(p), "%s/layer%02d.bin", dir, k);
        dump_bin(p, e.data(), e.size());
    }
    fprintf(stderr, "per-layer dumps done\n");
}

int main(int argc, char** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <tower.1bp> <img> <pixels.bin> <embeds.bin> [t|L <dir>]\n", argv[0]);
        return 1;
    }
    VisionWeights vw;
    if (!mage_vit_load_weights_1bp(argv[1], vw)) return 1;
    if (argc > 5 && (argv[5][0] == 't' || argv[5][0] == 'L')) {  // tower-only modes
        vw.mm0_w.clear(); vw.mm1_w.clear(); vw.mm2_w.clear();
        vw.mm0_b.clear(); vw.mm1_b.clear(); vw.mm2_b.clear();
    }
    VlProcessor proc;
    if (!proc.load(argv[2], IMG, IMG, MEAN, STD)) { fprintf(stderr, "image load failed\n"); return 1; }
    dump_bin(argv[3], proc.pixels(), (size_t)IMG * IMG * 3);
    if (argc > 5 && argv[5][0] == 'L') {
        dump_per_layer(vw, proc.pixels(), argv[6]);
        return 0;
    }
    std::vector<float> embs = mage_vit_forward(vw, proc.pixels(), 3, 1, IMG, IMG, 4);
    if (embs.empty()) { fprintf(stderr, "forward failed\n"); return 1; }
    dump_bin(argv[4], embs.data(), embs.size());
    fprintf(stderr, "dumped %zu pixels, %zu embeddings\n", (size_t)IMG * IMG * 3, embs.size());
    return 0;
}
