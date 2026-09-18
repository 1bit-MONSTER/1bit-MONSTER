// moe_smoke.cpp — NPU smoke test for MoERuntimeLayerEngine: load the 35B
// layer.xclbin + task-1 ELFs, pack the MoE weight BOs, run one layer + lm_head
// in a single xrt::runlist, and report whether it executes (non-crash) and
// whether the logits are non-NaN.
//
// Build (runlist XRT 2.26.0):
//   g++ -std=c++17 -O2 -I npu-infer/include moe_smoke.cpp npu-infer/src/model.c \
//       npu-infer/src/runtime_layer_moe.cpp -o moe_smoke \
//       -L/usr/local/xrt-runlist/lib -l:libxrt_coreutil.so.2 -l:libxrt_core.so.2 \
//       -Wl,-rpath,/usr/local/xrt-runlist/lib -laiebu -luuid -lm -ldl -pthread
#include "runtime_layer_moe.h"
#include "model.h"
#include "common.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <xrt/xrt_device.h>

int main(int argc, char** argv) {
    const char* model_path = argc > 1 ? argv[1]
        : "/home/bcloud/.config/flm/models/Qwen3.6-35B-A3B-NPU2/model.q4nx";
    const char* elf_dir = argc > 2 ? argv[2]
        : "/home/bcloud/1bit-MONSTER/npu-infer/captures/txn-elfs-moe35b";
    const char* lm_elf = argc > 3 ? argv[3]
        : "/home/bcloud/1bit-MONSTER/npu-infer/captures/txn-elfs-moe35b/moe_lm_head.elf";
    const char* xclbin = argc > 4 ? argv[4]
        : "/home/bcloud/amd-oss/fastflowlm/src/xclbins/Qwen3.6-35B-A3B-NPU2/layer.xclbin";

    // Which model layer to pack and run. This is a LAYER INDEX (0..39), not a
    // context length — moe_layer_ctx<N>.elf is layer N. The two must be the same
    // layer: the engine packs one layer's weights into the BOs and refuses an ELF
    // for any other. Default 1 = the layer this harness has always loaded; the
    // reference token below is only meaningful for the layer it was derived on,
    // so re-derive it if you change this.
    int layer = argc > 5 ? atoi(argv[5]) : (getenv("MOE_LAYER") ? atoi(getenv("MOE_LAYER")) : 1);

    ModelConfig cfg = QWEN3_0_6B_CONFIG;
    ModelWeights* mw = model_load(model_path, cfg);
    if (!mw) { fprintf(stderr, "model_load failed\n"); return 1; }
    fprintf(stderr, "model: %d layers, hidden %d, vocab %d\n",
            mw->config.num_layers, mw->config.hidden_size, mw->config.vocab_size);
    if (layer < 0 || layer >= mw->config.num_layers) {
        fprintf(stderr, "layer %d out of range (model has %d layers)\n", layer, mw->config.num_layers);
        return 1;
    }
    fprintf(stderr, "layer: %d (moe_layer_ctx%d.elf), set via argv[5] or MOE_LAYER\n", layer, layer);

    xrt::device dev(0);
    fprintf(stderr, "device opened\n");

    MoERuntimeLayerEngine eng;
    if (!eng.init(dev, mw, mw->config, elf_dir, lm_elf, xclbin, layer)) {
        fprintf(stderr, "MoERuntimeLayerEngine::init FAILED\n");
        return 1;
    }
    fprintf(stderr, "engine init OK (packed layer %d)\n", eng.packed_layer());

    // token 151644 (the reference prompt's first token)
    if (!eng.embed(151644)) { fprintf(stderr, "embed failed\n"); return 1; }
    eng.dump_act("/tmp/moe_act_pre.bin");
    fprintf(stderr, "embed done (pre-embed act dumped); running forward(%d)...\n", layer);

    bool ok = eng.forward(layer);
    fprintf(stderr, "forward(%d): %s\n", layer, ok ? "EXECUTED" : "FAILED");
    if (!ok) return 1;

    int vocab = mw->config.vocab_size;
    std::vector<float> logits(vocab);
    // Prefer the host lm_head when the model's lm_head is 3-D: the device path
    // cannot express that format and silently skips the lm_head, which leaves the
    // logits BO all-zero and reports it as a successful run.
    if (eng.logits_host(logits.data(), vocab))
        fprintf(stderr, "lm_head: HOST path (3-D/Q8_0 source, device lm_head skipped)\n");
    else
        eng.get_logits(logits.data(), vocab);
    int nan = 0; int argmax = 0; int nz = 0, nz_finite = 0;
    float mx = -INFINITY;
    for (int i = 0; i < vocab; i++) {
        bool fin = std::isfinite(logits[i]);
        if (!fin) nan++;
        if (logits[i] != 0.0f) nz++;                 // NB: NaN != 0.0f is true, so this
        if (fin && logits[i] != 0.0f) nz_finite++;   // counts NaN as "nonzero" -- use nz_finite
        if (fin && logits[i] > mx) { mx = logits[i]; argmax = i; }
    }
    if (nan == vocab)
        fprintf(stderr, "logits: ALL NaN (of %d) -- the lm_head ran; the NaN comes from upstream "
                        "(the act BO). Check the layer, not the head.\n", vocab);
    else if (nz_finite == 0)
        fprintf(stderr, "logits: ALL ZERO (of %d) -- no lm_head produced anything; treat as NO RESULT\n", vocab);
    else
        fprintf(stderr, "logits: argmax=%d max=%.4f NaN=%d nonzero=%d (of %d)\n",
                argmax, mx, nan, nz_finite, vocab);
    fprintf(stderr, "reference: greedy next token = 76740 (for a SINGLE layer this is informational only)\n");
    if (getenv("NPU_DUMP_BOS")) eng.dump_bos("/tmp/bo");
    // The dump above reads only the first 4096 B of a 1 MB act BO. If the layer
    // writes its result at an offset, that dump shows scratch and "all NaN" is an
    // artifact of the dump, not a defect. Dump the whole BO and scan it page by
    // page: a NaN confined to page 0 and finite data later is a dump-offset
    // artifact; NaN in every page is a real upstream failure.
    size_t act_bytes = 1048576;
    if (const char* e = getenv("MOE_ACT_DUMP_BYTES")) {
        long v = atol(e);
        if (v > 0 && (size_t)v <= act_bytes) act_bytes = (size_t)v;
    }
    eng.dump_act("/tmp/moe_act.bin", act_bytes);
    fprintf(stderr, "act dumped to /tmp/moe_act.bin (%zu bytes)\n", act_bytes);
    {
        FILE* f = fopen("/tmp/moe_act.bin", "rb");
        if (f) {
            std::vector<uint8_t> buf(act_bytes);
            size_t got = fread(buf.data(), 1, act_bytes, f);
            fclose(f);
            const size_t PAGE = 4096;
            size_t pages = got / PAGE, nan_pages = 0, fin_pages = 0, fin_total = 0;
            size_t zero_pages = 0, nz_total = 0;
            size_t first_fin_page = (size_t)-1;
            // bf16 -> f32 for the scan (the act BO is bf16 hidden state).
            auto b2f = [](uint16_t u) { uint32_t w = (uint32_t)u << 16;
                                        float r; memcpy(&r, &w, 4); return r; };
            for (size_t p = 0; p < pages; p++) {
                const uint8_t* base = buf.data() + p * PAGE;
                size_t fin = 0, nz = 0, n = PAGE / 2;
                for (size_t i = 0; i < n; i++) {
                    uint16_t u = (uint16_t)(base[2 * i] | (base[2 * i + 1] << 8));
                    float v = b2f(u);
                    if (std::isfinite(v)) { fin++; if (v != 0.0f) nz++; }
                }
                // NB: isfinite(0.0f) is TRUE, so counting "finite" alone reports the
                // init memset as if it were data. That mistake was made once here and
                // nearly reported as "finite data past page 0". Count NON-ZERO too.
                if (fin == 0) nan_pages++;
                else if (nz == 0) zero_pages++;
                else { fin_pages++; nz_total += nz;
                    if (first_fin_page == (size_t)-1) first_fin_page = p; }
                fin_total += fin;
            }
            fprintf(stderr, "ACT BO SCAN: %zu pages of %zu B -- %zu all-NaN, %zu all-zero "
                            "(untouched), %zu with real data (%zu non-zero values, first at page=%zd)\n",
                    pages, PAGE, nan_pages, zero_pages, fin_pages, nz_total,
                    first_fin_page == (size_t)-1 ? (ssize_t)-1 : (ssize_t)first_fin_page);
            if (nan_pages == pages)
                fprintf(stderr, "  -> NaN spans the WHOLE act BO: not a dump-offset artifact\n");
            else if (fin_pages == 0)
                fprintf(stderr, "  -> no real data anywhere in the act BO: the NaN slot is the "
                                "output and the rest is untouched memset -- the layer wrote "
                                "NOTHING but NaN\n");
            else if (fin_pages == 1 && first_fin_page == 0)
                fprintf(stderr, "  -> page 0 is the only page with data: output written in place "
                                "at the hidden slot, which is where logits_host reads -- OK\n");
            else
                fprintf(stderr, "  -> real data exists outside the NaN pages: CHECK THE DUMP OFFSET\n");
        }
    }
    fprintf(stderr, "DONE\n");
    model_free(mw);
    return 0;
}
