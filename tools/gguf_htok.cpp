// gguf_htok — export a GGUF's tokenizer to the .htok v2 binary the rcpp
// tokenizer loads (src/tokenizer.cpp). Replaces the deleted
// halo-1bit/scripts/export_tokenizer.py. Used by the reconvert pipeline
// (issue #1243) to build per-vocab ppl gate sets for non-Qwen families.
//
// Usage: gguf_htok <model.gguf> <out.htok>
//
// GGUF byte-level BPE vocabs (gpt2/llama/qwen2) store tokens already
// GPT-2 byte-mapped and merges as "A B" pairs — pass-through to .htok.
// SentencePiece-type vocabs (gemma) export too; the regex pre-tokenizer may
// not match SP piece boundaries, so gate the ppl output with a sanity check.

#include "gguf_reader.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

static void wr(FILE* f, const void* p, size_t n) { fwrite(p, 1, n, f); }
static void wr_u32(FILE* f, uint32_t v) { wr(f, &v, 4); }
static void wr_u16(FILE* f, uint16_t v) { wr(f, &v, 2); }

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s model.gguf out.htok\n", argv[0]); return 1; }
    GgufReader r;
    if (!r.open(argv[1])) { fprintf(stderr, "gguf_htok: cannot open %s\n", argv[1]); return 1; }

    std::vector<std::string> toks;
    if (!r.get_string_array("tokenizer.ggml.tokens", toks) || toks.empty()) {
        fprintf(stderr, "gguf_htok: no tokenizer.ggml.tokens\n"); return 1;
    }
    const uint32_t vocab_size = (uint32_t)toks.size();

    std::unordered_map<std::string, uint32_t> id_of;
    id_of.reserve(vocab_size);
    for (uint32_t i = 0; i < vocab_size; ++i)
        if (!toks[i].empty()) id_of.emplace(toks[i], i);

    std::vector<std::string> merges;
    r.get_string_array("tokenizer.ggml.merges", merges);

    struct Merge { uint32_t a, b, merged; };
    std::vector<Merge> out;
    out.reserve(merges.size());
    for (const std::string& m : merges) {
        size_t sp = m.find(' ');
        if (sp == std::string::npos) continue;
        auto ita = id_of.find(m.substr(0, sp)), itb = id_of.find(m.substr(sp + 1));
        if (ita == id_of.end() || itb == id_of.end()) continue;
        auto itm = id_of.find(m.substr(0, sp) + m.substr(sp + 1));
        if (itm == id_of.end()) continue;
        out.push_back({ita->second, itb->second, itm->second});
    }

    uint32_t bos = 128000, eos = 128001;
    r.get_u32("tokenizer.ggml.bos_token_id", bos);
    r.get_u32("tokenizer.ggml.eos_token_id", eos);

    FILE* f = fopen(argv[2], "wb");
    if (!f) { fprintf(stderr, "gguf_htok: cannot write %s\n", argv[2]); return 1; }
    const uint32_t version = 2;
    wr(f, "HTOK", 4);
    wr_u32(f, version);
    wr_u32(f, vocab_size);
    wr_u32(f, (uint32_t)out.size());
    wr_u32(f, bos);
    wr_u32(f, eos);
    for (const std::string& t : toks) {
        if (t.size() > 65535) { fprintf(stderr, "gguf_htok: token %zu bytes, cap 65535\n", t.size()); fclose(f); return 1; }
        wr_u16(f, (uint16_t)t.size());
        wr(f, t.data(), t.size());
    }
    for (const Merge& m : out) { wr_u32(f, m.a); wr_u32(f, m.b); wr_u32(f, m.merged); }
    wr_u32(f, 0);  // num_special (v2; gate corpus is plain text)
    fclose(f);
    fprintf(stderr, "gguf_htok: %u tokens, %zu merges (%zu dropped)\n",
            vocab_size, out.size(), merges.size() - out.size());
    return 0;
}
