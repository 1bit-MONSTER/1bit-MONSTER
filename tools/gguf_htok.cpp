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

int main(int argc, char** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s model.gguf out.htok\n", argv[0]); return 1; }
    GgufReader r;
    if (!r.open(argv[1])) { fprintf(stderr, "gguf_htok: cannot open %s\n", argv[1]); return 1; }
    if (!r.write_htok(argv[2])) {
        fprintf(stderr, "gguf_htok: no tokenizer.ggml.tokens in %s\n", argv[1]);
        return 1;
    }
    return 0;
}
