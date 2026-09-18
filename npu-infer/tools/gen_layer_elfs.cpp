// gen_layer_elfs.cpp — build per-context layer ELFs exactly like the runtime's
// _setup_kernel: gen_layer_seq(ctx+1) -> aiebu_assembler_get_elf -> ELF file.
//
// Usage: gen_layer_elfs <model_dir> <outdir> [L_begin] [L_end] [MAX_L] [family]
//
// FAMILY (added 2026-09-13): every family's sequence class exposes the SAME API —
//   Seq(LM_Config, uint32_t MAX_L);  void gen_layer_seq(npu_sequence*, uint32_t L);
// so the per-layer loop is a template and the family only selects which class to
// instantiate. That matters because the runlist decode needs per-context layer ELFs and,
// until now, only Qwen3's could be generated — which is why Llama-3.1-8B's native decode
// produced no timing line at all, and why the four families whose prefill is wrong had no
// reference pipeline to diff against. Family defaults to qwen3, so the existing call sites
// behave exactly as before.
//
// Verified present under ~/.local/flm-v0946/include/models/: qwen3, llama, nanbeige, phi4,
// gemma_text (and gemma4e, gpt_oss, lfm2, qwen2, ...).
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <dlfcn.h>
#include "npu_utils/npu_instr_utils.hpp"
#include "models/qwen3/qwen3_npu_sequence.hpp"
#include "models/llama/llama_npu_sequence.hpp"
#include "models/nanbeige/nanbeige_npu_sequence.hpp"
#include "models/phi4/phi4_npu_sequence.hpp"
#include "models/gemma_text/gemma_text_npu_sequence.hpp"
#include "lm_config.hpp"
#include "aiebu/aiebu.h"

namespace utils { std::string find_xclbin_path() { return "/home/bcloud/amd-oss/fastflowlm/src/xclbins"; } }

// Per-layer ELFs + the raw TXN next to each one. Takes a caller rather than calling
// gen_layer_seq directly, because gemma_text's signature carries an extra is_sliding_window
// bool (`gen_layer_seq(npu_sequence*, uint32_t, bool)`) while qwen3/llama/nanbeige/phi4 are
// all `gen_layer_seq(npu_sequence*, uint32_t)`. The generation body is otherwise identical.
template <class Gen>
static void emit_layers(Gen gen, int L0, int L1, const std::string& outdir) {
    for (int L = L0; L <= L1; L++) {
        npu_sequence seq(device_npu2);
        gen(&seq, L);
        seq.cmds2seq();
        auto [ptr, nw] = seq.dump();
        char* elf_buf = nullptr;
        uint32_t elf_size = aiebu_assembler_get_elf(
            aiebu_assembler_buffer_type_blob_instr_transaction,
            (const char*)ptr, (size_t)(nw * sizeof(uint32_t)), NULL, 0,
            (void**)&elf_buf, NULL, 0, "", "", NULL, 0);
        if (elf_size == 0) { fprintf(stderr, "L=%d aiebu failed\n", L); continue; }
        // raw TXN dump alongside the ELF (for runtime A/B)
        {
            char rname[256];
            snprintf(rname, sizeof(rname), "%s/layer_ctx%d.txn", outdir.c_str(), L);
            FILE* fr = fopen(rname, "wb");
            if (fr) { fwrite(ptr, 4, nw, fr); fclose(fr); }
        }
        char fname[256];
        snprintf(fname, sizeof(fname), "%s/layer_ctx%d.elf", outdir.c_str(), L);
        FILE* f = fopen(fname, "wb");
        if (f) { fwrite(elf_buf, 1, elf_size, f); fclose(f); }
        printf("ctx=%d txn_words=%zu elf=%u -> %s\n", L, nw, elf_size, fname);
        free(elf_buf);
    }
}

// The lm_head sequence is exported from the family's shared object but NOT declared in the
// shipped sequence headers, so the ABI entry is resolved directly. The mangled name embeds
// the class-name length: qwen3_npu_sequence (18) -> _ZN18qwen3_npu_sequence15gen_lm_head_seqEP12npu_sequence.
template <class Seq>
static void emit_lm_head(Seq& qseq, const char* class_name, const std::string& outdir) {
    char mangled[160];
    snprintf(mangled, sizeof(mangled),
             "_ZN%zu%s15gen_lm_head_seqEP12npu_sequence", strlen(class_name), class_name);
    typedef void (*gen_lm_head_t)(Seq*, npu_sequence*);
    gen_lm_head_t gen_lm_head = (gen_lm_head_t)dlsym(RTLD_DEFAULT, mangled);
    if (!gen_lm_head) {
        // Not fatal: the layer ELFs are what the decode needs, and the engine has its own
        // lm_head path. Report and continue rather than aborting the whole run.
        fprintf(stderr, "lm_head: dlsym %s failed (%s) — layer ELFs still written\n",
                mangled, dlerror());
        return;
    }
    npu_sequence seq(device_npu2);
    gen_lm_head(&qseq, &seq);
    seq.cmds2seq();
    auto [ptr, nw] = seq.dump();
    char* elf_buf = nullptr;
    uint32_t elf_size = aiebu_assembler_get_elf(
        aiebu_assembler_buffer_type_blob_instr_transaction,
        (const char*)ptr, (size_t)(nw * sizeof(uint32_t)), NULL, 0,
        (void**)&elf_buf, NULL, 0, "", "", NULL, 0);
    if (elf_size) {
        char fname[256];
        snprintf(fname, sizeof(fname), "%s/elf_0002_lmhead.bin", outdir.c_str());
        FILE* f = fopen(fname, "wb");
        if (f) { fwrite(elf_buf, 1, elf_size, f); fclose(f); }
        printf("lm_head txn_words=%zu elf=%u -> %s\n", nw, elf_size, fname);
    } else {
        fprintf(stderr, "lm_head aiebu failed\n");
    }
    free(elf_buf);
}

int main(int argc, char** argv) {
    std::string model_dir = (argc > 1) ? argv[1] : "/home/bcloud/.config/flm/models/Qwen3-0.6B-NPU2";
    std::string outdir = (argc > 2) ? argv[2] : ".";
    int L0 = (argc > 3) ? atoi(argv[3]) : 1;
    int L1 = (argc > 4) ? atoi(argv[4]) : 2;
    // MAX_L bounds gen_layer_seq (assert L <= MAX_L+1) AND bakes the KV REGION
    // STRIDE into the instruction stream. It must equal the stride the runtime
    // writes, which is NOT the BO capacity: RuntimeLayerEngine allocates
    // npu_kv_cache_bo_size (128MB) as a ceiling but lays the 4 regions out with
    // region_stride_u16 = token_u16 * 8192, i.e. an 8MB stride -> MAX_L = 8192.
    //
    // Getting this wrong is silent: max_l=32768 yields an ELF of the SAME byte
    // size that runs at the SAME speed and returns WRONG tokens. Every committed
    // ELF in npu-infer/captures/txn-elfs* is 8192 (verified by regenerating a
    // known ctx and comparing sha256). Override via argv only to match a changed
    // stride. See benchmarks/SESSION-FINDINGS-2026-09-14.md section 4b.
    uint32_t max_l = (argc > 5) ? (uint32_t)atoi(argv[5]) : 8192;
    std::string family = (argc > 6) ? argv[6] : "qwen3";
    // gemma_text only: which layers are sliding-window is a property of the model, so the
    // caller states it rather than this tool guessing. Ignored by every other family.
    bool sliding = (argc > 7) && atoi(argv[7]) != 0;

    LM_Config config;
    config.from_pretrained(model_dir);

    if (family == "qwen3") {
        qwen3_npu_sequence q(config, max_l);
        emit_layers([&](npu_sequence* s, int L) { q.gen_layer_seq(s, (uint32_t)L); }, L0, L1, outdir);
        emit_lm_head(q, "qwen3_npu_sequence", outdir);
    } else if (family == "llama") {
        llama_npu_sequence q(config, max_l);
        emit_layers([&](npu_sequence* s, int L) { q.gen_layer_seq(s, (uint32_t)L); }, L0, L1, outdir);
        emit_lm_head(q, "llama_npu_sequence", outdir);
    } else if (family == "nanbeige") {
        nanbeige_npu_sequence q(config, max_l);
        emit_layers([&](npu_sequence* s, int L) { q.gen_layer_seq(s, (uint32_t)L); }, L0, L1, outdir);
        emit_lm_head(q, "nanbeige_npu_sequence", outdir);
    } else if (family == "phi4") {
        phi4_npu_sequence q(config, max_l);
        emit_layers([&](npu_sequence* s, int L) { q.gen_layer_seq(s, (uint32_t)L); }, L0, L1, outdir);
        emit_lm_head(q, "phi4_npu_sequence", outdir);
    } else if (family == "gemma_text" || family == "gemma3") {
        gemma_text_npu_sequence q(config, max_l);
        emit_layers([&](npu_sequence* s, int L) { q.gen_layer_seq(s, (uint32_t)L, sliding); },
                    L0, L1, outdir);
        emit_lm_head(q, "gemma_text_npu_sequence", outdir);
    } else {
        fprintf(stderr, "unknown family '%s' — known: qwen3 llama nanbeige phi4 gemma_text\n",
                family.c_str());
        return 1;
    }
    return 0;
}
